"""
E1 · ¿La multimodalidad aporta?

Contrasta el modelo multimodal contra los tres unimodales sobre EXACTAMENTE los
mismos datos y la misma partición. Es el experimento que el asesor pidió separar
del resto: mezclarlo con la tolerancia a la ausencia de ramas impedía atribuir el
resultado a ninguno de los dos efectos.

Se restringe al subconjunto con las tres modalidades y se desactivan el centinela
y el descarte, para que la comparación mida la fusión y no el manejo de huecos.

Concluye que hay sinergia si el multimodal supera al mejor unimodal más allá de la
dispersión entre semillas Y la prueba de McNemar es significativa. Ambas
condiciones: la primera sin la segunda puede ser ruido de inicialización, y la
segunda sin la primera puede ser un efecto real pero irrelevante --con miles de
observaciones McNemar declara significativa casi cualquier diferencia--.

    python scripts/experimentos/e1_multimodalidad.py [--epocas 3] [--rapido]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clasicos import CLASICOS, ejecutar_clasico
from comun import (SEMILLAS, agregar, barras_con_error, cargar_corpus, emitir,
                   equivalencia, intervalo_bootstrap, particion_agrupada,
                   subconjunto_trimodal, tpr_a_fpr)
from entrenar import ETIQUETAS, mcnemar
from ejecutor import ejecutar_variante

# El orden importa para la figura: la propuesta primero, los unimodales después.
VARIANTES = ("atencion_cruzada_token", "solo_texto", "solo_estructura", "solo_red")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epocas", type=int, default=3)
    p.add_argument("--rapido", action="store_true",
                   help="una semilla y pocos pasos, solo para comprobar que corre")
    p.add_argument("--reusar", action="store_true",
                   help="no reentrena: reconstruye el informe desde las predicciones "
                        "ya guardadas. Sin GPU.")
    args = p.parse_args()

    semillas = SEMILLAS[:1] if args.rapido else SEMILLAS
    corpus = subconjunto_trimodal(cargar_corpus())
    particion = particion_agrupada(corpus)

    print(f"E1 · subconjunto trimodal: {len(corpus):,} correos, "
          f"prevalencia {corpus.label.mean():.4f}")
    print(f"     partición {particion.resumen()}")

    resultados: dict[str, dict] = {}
    predicciones: dict[str, tuple] = {}
    probabilidades: dict[str, object] = {}
    for variante in VARIANTES:
        por_semilla = []
        for semilla in semillas:
            r = ejecutar_variante(variante, corpus, particion, semilla,
                                  epocas=args.epocas, sin_centinela=True,
                                  rapido=args.rapido, reusar=args.reusar, experimento="e1")
            por_semilla.append(r["metricas"])
            predicciones.setdefault(variante, (r["y_true"], r["y_pred"]))
            probabilidades.setdefault(variante, r.get("y_proba"))
            print(f"  {variante:28s} semilla {semilla}  F1={r['metricas']['f1']:.4f}")
        resultados[variante] = {"por_semilla": por_semilla, **agregar(por_semilla)}

    # Piso clásico. Entra como filas de pleno derecho --la investigación es
    # exploratoria y si una línea base gana, se reporta y ya-- pero NO decide el
    # veredicto de sinergia, y la razón no es defensiva sino de atribución: si el
    # multimodal le gana a B1, la diferencia mezcla la modalidad con la familia de
    # modelo (transformador contra bolsa de palabras) y no se puede repartir entre
    # las dos. La comparación atribuible es contra el unimodal NEURONAL, que solo
    # se diferencia en qué entra.
    print("")
    print("  piso clasico (CPU)")
    for nombre in CLASICOS:
        corridas = [ejecutar_clasico(nombre, corpus, particion, semilla=s_)
                    for s_ in semillas]
        por_semilla = [c["metricas"] for c in corridas]
        r0 = corridas[0]  # se reutiliza el primero: reajustar seria el mismo modelo
        resultados[nombre] = {"familia": "clasica",
                              "modalidades": list(CLASICOS[nombre]["modalidades"]),
                              "por_semilla": por_semilla, **agregar(por_semilla)}
        predicciones[nombre] = (r0["y_true"], r0["y_pred"])
        probabilidades[nombre] = r0["y_proba"]
        print(f"  {CLASICOS[nombre]['etiqueta']:34s} F1={resultados[nombre]['f1']:.4f}")

    # Punto de operación e incertidumbre de muestreo, para toda fila con
    # probabilidades. Ver la nota de `tpr_a_fpr` sobre por qué se fijan a priori.
    for nombre, (yt, yp) in predicciones.items():
        resultados[nombre]["punto_de_operacion"] = tpr_a_fpr(yt, probabilidades.get(nombre))
        resultados[nombre]["ic_bootstrap_f1"] = intervalo_bootstrap(yt, yp)

    # Contraste pareado contra el mejor unimodal, que es la comparación que
    # decide. Compararlo contra el promedio de los tres sería más favorable y
    # menos informativo.
    unimodales = [v for v in VARIANTES if v != "atencion_cruzada_token"]
    mejor = max(unimodales, key=lambda v: resultados[v]["f1"])
    y_true, pred_multi = predicciones["atencion_cruzada_token"]
    _, pred_uni = predicciones[mejor]
    prueba = mcnemar(y_true, pred_multi, pred_uni)

    multi, uni = resultados["atencion_cruzada_token"], resultados[mejor]
    supera = multi["f1"] - uni["f1"]
    dispersión = max(multi.get("f1_desv", 0), uni.get("f1_desv", 0))

    informe = {
        "experimento": "E1",
        "pregunta": "¿La multimodalidad aporta sobre los unimodales?",
        "generado_en": datetime.now(timezone.utc).isoformat(),
        "datos": {
            "n": int(len(corpus)),
            "prevalencia": round(float(corpus.label.mean()), 4),
            "particion": particion.resumen(),
            "sin_centinela": True,
        },
        "semillas": list(semillas),
        "resultados": resultados,
        "contraste": {
            "multimodal": "atencion_cruzada_token",
            "mejor_unimodal": mejor,
            "diferencia_f1": round(supera, 4),
            "dispersion_entre_semillas": round(dispersión, 4),
            "supera_la_dispersion": bool(supera > dispersión),
            "mcnemar": prueba,
        },
        "equivalencia_con_el_piso_clasico": {
            n: equivalencia(predicciones["atencion_cruzada_token"][0],
                            predicciones["atencion_cruzada_token"][1],
                            predicciones[n][1])
            for n in CLASICOS if n in predicciones
        },
        "nota_de_atribucion": (
            "Las filas clásicas se informan como comparadores de pleno derecho. El "
            "veredicto de sinergia se calcula contra el mejor unimodal NEURONAL "
            "porque es la única comparación atribuible a la modalidad: frente a un "
            "clásico, la diferencia mezcla modalidad y familia de modelo."
        ),
        "veredicto": (
            "hay sinergia" if supera > dispersión and prueba["significativo_005"]
            else "no se observa sinergia"
        ),
        "advertencia": (
            "El legítimo del subconjunto trimodal procede casi enteramente de una "
            "comunidad. La cifra absoluta NO es rendimiento de detección de phishing; "
            "la comparación sí es válida, porque los cuatro modelos ven los mismos datos."
        ),
    }

    def figura(destino: Path) -> None:
        # La figura incluye el piso clasico: una lamina que solo muestre las
        # variantes neuronales deja al lector sin la referencia con la que
        # juzgar si la diferencia entre ellas significa algo.
        filas = list(VARIANTES) + [n for n in CLASICOS if n in resultados]
        etiquetas = [ETIQUETAS.get(v) or CLASICOS[v]["etiqueta"] for v in filas]
        medias = [resultados[v]["f1"] for v in filas]
        desv = [resultados[v].get("f1_desv", 0.0) for v in filas]
        barras_con_error(
            destino, "e1_multimodalidad",
            "E1 - La multimodalidad frente a unimodales y al piso clasico. "
            "F1 sobre el subconjunto trimodal, media de tres semillas",
            etiquetas, medias, desv, resaltar=0)

    emitir("e1", informe, figura, corpus=corpus, particion=particion)
    print(f"\n  veredicto: {informe['veredicto']}  "
          f"(Δ={supera:+.4f}, dispersión {dispersión:.4f}, p={prueba['p_valor']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
