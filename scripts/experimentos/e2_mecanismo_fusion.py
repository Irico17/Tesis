"""
E2 · ¿La atención cruzada aporta sobre una fusión simple?

Contrasta cuatro maneras de combinar las mismas tres ramas, sobre EXACTAMENTE los
mismos datos y la misma partición que E1. La línea base es el fusor MLP, que es lo
que el asesor propuso: *«puede ser el fusor que tú propones, que es cross
attention, pero puede ser algo más simple, tipo MLP»*.

**E2 depende de E1 y su lectura cambia según el resultado de aquel.** Si E1
concluye que la multimodalidad no aporta, E2 compara cuatro maneras de no aportar
nada; el experimento sigue valiendo, porque descarta que el resultado nulo proceda
de una atención mal implementada o mal parametrizada, pero no puede leerse como
una comparación de mecanismos útiles. El informe lo declara de forma explícita
leyendo el veredicto de E1.

    python scripts/experimentos/e2_mecanismo_fusion.py [--epocas 3] [--rapido]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clasicos import CLASICOS, ejecutar_clasico
from comun import (SALIDA, SEMILLAS, agregar, barras_con_error, cargar_corpus,
                   emitir, particion_agrupada, subconjunto_trimodal,
                   tpr_a_fpr, intervalo_bootstrap, equivalencia)
from ejecutor import ejecutar_variante
from entrenar import ETIQUETAS, mcnemar

# La mezcla de expertos entra aqui ademas de en E3. Es un MECANISMO DE FUSION, y
# esta es la pregunta por el mecanismo: dejarla solo en E3 --donde se evalua como
# tolera la ausencia de ramas-- respondia a la mitad de lo que es.
VARIANTES = ("atencion_cruzada_token", "atencion_cruzada_modalidad",
             "fusor_mlp", "concatenacion_tardia", "mezcla_de_expertos")
BASE_DE_COMPARACION = "fusor_mlp"


def veredicto_e1() -> str | None:
    """Lee la conclusión de E1, que condiciona cómo debe leerse este experimento."""
    ruta = SALIDA / "e1" / "e1.json"
    if not ruta.exists():
        return None
    return json.loads(ruta.read_text(encoding="utf-8")).get("veredicto")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epocas", type=int, default=3)
    p.add_argument("--rapido", action="store_true")
    p.add_argument("--reusar", action="store_true",
                   help="reconstruye el informe desde predicciones guardadas")
    args = p.parse_args()

    semillas = SEMILLAS[:1] if args.rapido else SEMILLAS
    corpus = subconjunto_trimodal(cargar_corpus())
    particion = particion_agrupada(corpus)

    print(f"E2 · {len(corpus):,} correos, misma partición que E1")

    resultados: dict[str, dict] = {}
    predicciones: dict[str, tuple] = {}
    probabilidades: dict[str, object] = {}
    for variante in VARIANTES:
        por_semilla = []
        for semilla in semillas:
            r = ejecutar_variante(variante, corpus, particion, semilla,
                                  epocas=args.epocas, sin_centinela=True,
                                  rapido=args.rapido, reusar=args.reusar, experimento="e2")
            por_semilla.append(r["metricas"])
            predicciones.setdefault(variante, (r["y_true"], r["y_pred"]))
            probabilidades.setdefault(variante, r.get("y_proba"))
            print(f"  {variante:28s} semilla {semilla}  F1={r['metricas']['f1']:.4f}")
        resultados[variante] = {"por_semilla": por_semilla, **agregar(por_semilla)}

    # B5 es el piso de FUSION SIN ATENCION: las tres modalidades concatenadas y
    # una regresion logistica encima. Es lo que separa "sirve fusionar" de "sirve
    # esta manera de fusionar", que es exactamente la pregunta de E2.
    print("")
    print("  piso de fusion clasica (CPU)")
    corridas = [ejecutar_clasico("b5_multimodal_clasico", corpus, particion, semilla=s_)
                for s_ in semillas]
    por_semilla = [c["metricas"] for c in corridas]
    r0 = corridas[0]
    resultados["b5_multimodal_clasico"] = {
        "familia": "clasica", "por_semilla": por_semilla, **agregar(por_semilla),
        "punto_de_operacion": tpr_a_fpr(r0["y_true"], r0["y_proba"]),
        "ic_bootstrap_f1": intervalo_bootstrap(r0["y_true"], r0["y_pred"]),
    }
    print(f"  {CLASICOS['b5_multimodal_clasico']['etiqueta']:34s} "
          f"F1={resultados['b5_multimodal_clasico']['f1']:.4f}")

    # Punto de operacion e incertidumbre de muestreo para TODAS las filas. Antes
    # solo los llevaba la fila clasica, y era justo la columna donde E1 mostro que
    # los modelos si se separan: al umbral 0.5 empatan y a FPR 0.1% no.
    for nombre_v, (yt, yp) in predicciones.items():
        resultados[nombre_v]["punto_de_operacion"] = tpr_a_fpr(yt, probabilidades.get(nombre_v))
        resultados[nombre_v]["ic_bootstrap_f1"] = intervalo_bootstrap(yt, yp)

    # Pruebas de EQUIVALENCIA entre todos los mecanismos y el mas simple. Es la
    # conclusion que corresponde aqui: con los cinco separados por 0.0015 y McNemar
    # no significativo, "no se demostro diferencia" se lee como que no se supo
    # medir. Acotar |A-B| < delta permite AFIRMAR equivalencia, que es una
    # conclusion positiva y falsable, y ademas la honesta sobre un corpus donde una
    # bolsa de palabras alcanza 0.9945.
    yt_base, yp_base = predicciones[BASE_DE_COMPARACION]
    equivalencias = {
        v: equivalencia(yt_base, predicciones[v][1], yp_base)
        for v in predicciones if v != BASE_DE_COMPARACION
    }

    # Diferencias pareadas frente al fusor simple, semilla a semilla. La media de
    # las diferencias dice menos que su consistencia: una ventaja de 0.003 que
    # aparece en las tres semillas es más informativa que una de 0.01 que aparece
    # en una sola.
    base = resultados[BASE_DE_COMPARACION]
    comparaciones = {}
    y_true, pred_base = predicciones[BASE_DE_COMPARACION]
    for v in VARIANTES:
        if v == BASE_DE_COMPARACION:
            continue
        diffs = [a["f1"] - b["f1"] for a, b in
                 zip(resultados[v]["por_semilla"], base["por_semilla"])]
        comparaciones[v] = {
            "diferencia_f1_por_semilla": [round(d, 4) for d in diffs],
            "gana_en": f"{sum(d > 0 for d in diffs)} de {len(diffs)}",
            "mcnemar": mcnemar(y_true, predicciones[v][1], pred_base),
        }

    v_e1 = veredicto_e1()
    informe = {
        "experimento": "E2",
        "pregunta": "¿La atención cruzada aporta sobre una fusión simple?",
        "generado_en": datetime.now(timezone.utc).isoformat(),
        "linea_base": BASE_DE_COMPARACION,
        "datos": {"n": int(len(corpus)), "particion": particion.resumen()},
        "semillas": list(semillas),
        "resultados": resultados,
        "equivalencia_frente_a_la_base": equivalencias,
        "comparaciones_frente_a_la_base": comparaciones,
        "dependencia_de_e1": {
            "veredicto_e1": v_e1,
            "lectura": (
                "E1 no encontró sinergia multimodal: este experimento compara "
                "maneras de incorporar modalidades que no aportan, y sirve para "
                "descartar que el resultado nulo proceda de una atención mal "
                "implementada, no para ordenar mecanismos útiles."
                if v_e1 == "no se observa sinergia" else
                "E1 encontró sinergia: la comparación entre mecanismos es "
                "directamente interpretable."
                if v_e1 else
                "E1 no se ha ejecutado; ejecútelo antes de interpretar este informe."
            ),
        },
    }

    def figura(destino: Path) -> None:
        etiquetas = [ETIQUETAS[v] for v in VARIANTES]
        medias = [resultados[v]["f1"] for v in VARIANTES]
        desv = [resultados[v].get("f1_desv", 0.0) for v in VARIANTES]
        barras_con_error(
            destino, "e2_mecanismo_fusion",
            "E2. ¿Importa el mecanismo de fusión?\nF1 con las mismas ramas y la "
            "misma partición, media de tres semillas",
            etiquetas, medias, desv,
            resaltar=VARIANTES.index(BASE_DE_COMPARACION))

    emitir("e2", informe, figura, corpus=corpus, particion=particion)
    orden = sorted(VARIANTES, key=lambda v: -resultados[v]["f1"])
    print(f"\n  orden: {' > '.join(orden)}")
    print(f"  intervalo entre la mejor y la peor: "
          f"{resultados[orden[0]]['f1'] - resultados[orden[-1]]['f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
