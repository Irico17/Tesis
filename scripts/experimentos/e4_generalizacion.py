"""
E4 · ¿Generaliza el modelo a correo no visto?

Sobre el CORPUS COMPLETO, no solo el subconjunto trimodal: la pregunta aquí no es
si la fusión aprovecha las modalidades sino si el sistema funciona con correo que
no vio, y restringirlo a las filas completas respondería a otra cosa.

Cuatro mediciones, ninguna de las cuales requiere un eje de partición nuevo:

1. **Agrupada frente a no agrupada.** La diferencia mide cuánto infla la métrica
   permitir que una misma campaña caiga a ambos lados, que es lo que hace la mayor
   parte de la literatura del área. Es el resultado con mayor valor metodológico
   propio de este experimento.
2. **Por tamaño de conglomerado.** ¿Detecta campañas vistas una sola vez tan bien
   como las masivas? Generalización real, medible dentro de la partición existente.
3. **Curva de aprendizaje.** ¿Está limitado por datos o ya saturó?
4. **Varias semillas de partición.** Estabilidad de la estimación.

**Lo que NO se mide, y por qué.** Los tres ejes de cambio de distribución están
confundidos con la clase en este corpus, y se comprobó uno por uno: la partición
temporal excluiría el 40% de los correos porque Kaggle no conserva fechas; la
partición por remitente excluiría lo mismo y además el 99.5% de los dominios es de
clase única; y la partición por colección tiene clase idéntica a procedencia en
cinco de ocho colecciones. Se declara como propiedad medida del corpus público
disponible, con las cifras en `doc/DECISIONES_CORPUS_Y_PROTOCOLO.md`.

    python scripts/experimentos/e4_generalizacion.py [--epocas 3] [--rapido]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clasicos import CLASICOS, ejecutar_clasico
from comun import (SEMILLAS, agregar, barras_con_error, cargar_corpus, emitir,
                   metricas, particion_agrupada, tpr_a_fpr,
                   intervalo_bootstrap, equivalencia)
from entrenar import etiqueta_de, mcnemar
from ejecutor import ejecutar_variante

# La variante ya no esta fija. Antes E4 media siempre la atencion cruzada aunque
# E2 hubiera elegido otra, de modo que la generalizacion se informaba de un modelo
# distinto del que E6 cuantizaba y del que el trabajo propone desplegar.
# TODAS las arquitecturas entran en el cuadro comparativo. No se hereda ninguna
# eleccion de E2: alli el corpus es el subconjunto trimodal, donde la clase
# coincide con la procedencia, y una decision tomada en esas condiciones no puede
# arrastrarse al corpus que el trabajo describe.
ARQUITECTURAS = ("atencion_cruzada_token", "atencion_cruzada_modalidad",
                 "mezcla_de_expertos", "fusor_mlp", "concatenacion_tardia",
                 "solo_texto", "solo_estructura", "solo_red")

# La REFERENCIA es la arquitectura que el trabajo propone. Se usa para los
# analisis de PROTOCOLO --agrupar o no, tamano de campana, curva de aprendizaje,
# estabilidad de la particion--, que miden el protocolo y no la arquitectura, y
# como base de las pruebas de equivalencia pareadas. No se presenta como ganadora:
# el cuadro informa las doce filas.
REFERENCIA = "atencion_cruzada_token"
FRACCIONES = (0.25, 0.50, 0.75, 1.00)
SEMILLAS_PARTICION = (7, 17, 27)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epocas", type=int, default=3)
    p.add_argument("--rapido", action="store_true")
    p.add_argument("--reusar", action="store_true",
                   help="reconstruye el informe desde predicciones guardadas")
    args = p.parse_args()

    semillas = SEMILLAS[:1] if args.rapido else SEMILLAS
    corpus = cargar_corpus()
    print(f"E4 - corpus completo: {len(corpus):,} correos, "
          f"prevalencia {corpus.label.mean():.4f}")

    # (1) CUADRO COMPARATIVO sobre el corpus completo, con TODAS las arquitecturas.
    #
    # Es el nucleo del experimento y la tabla que encabeza el capitulo. No se trae
    # aqui una ganadora elegida en E2: E1 y E2 viven en el subconjunto trimodal,
    # donde la clase coincide con la procedencia y una bolsa de palabras alcanza
    # 0.9945. Dar por buena alli una eleccion y arrastrarla al corpus completo
    # seria decidir sobre datos que no representan el problema. Cada arquitectura
    # se entrena y se mide aqui, sobre la misma particion agrupada.
    particion = particion_agrupada(corpus, agrupar=True)
    print(f"  particion agrupada: {particion.resumen()}")
    print("  entrenando las arquitecturas neuronales")

    comparativa: dict[str, dict] = {}
    predicciones: dict[str, tuple] = {}
    for variante in ARQUITECTURAS:
        por_semilla = []
        primera = None
        for semilla in semillas:
            r = ejecutar_variante(variante, corpus, particion, semilla,
                                  epocas=args.epocas, rapido=args.rapido,
                                  reusar=args.reusar, experimento="e4",
                                  sufijo="agr")
            por_semilla.append(r["metricas"])
            if primera is None:
                primera = r
        comparativa[variante] = {"familia": "neuronal", "por_semilla": por_semilla,
                                 **agregar(por_semilla)}
        predicciones[variante] = (primera["y_true"], primera["y_pred"],
                                  primera.get("y_proba"))
        print(f"    {variante:30s} F1={comparativa[variante]['f1']:.4f}")

    print("  lineas base clasicas (CPU)")
    for nombre in CLASICOS:
        corridas = [ejecutar_clasico(nombre, corpus, particion, semilla=s_,
                                     marca=f"e4_{nombre}_s{s_}_agr")
                    for s_ in semillas]
        por_s = [c["metricas"] for c in corridas]
        comparativa[nombre] = {"familia": "clasica",
                               "modalidades": list(CLASICOS[nombre]["modalidades"]),
                               "por_semilla": por_s, **agregar(por_s)}
        predicciones[nombre] = (corridas[0]["y_true"], corridas[0]["y_pred"],
                                corridas[0]["y_proba"])
        print(f"    {CLASICOS[nombre]['etiqueta']:30s} F1={comparativa[nombre]['f1']:.4f}")

    # Punto de operacion e intervalo para TODAS las filas, y equivalencia pareada
    # frente a la arquitectura de referencia. La referencia se DECLARA --es la que
    # el trabajo propone-- y no se presenta como ganadora: la tabla informa las
    # doce filas y el lector juzga.
    for nombre, (yt, yp, pr) in predicciones.items():
        comparativa[nombre]["punto_de_operacion"] = tpr_a_fpr(yt, pr)
        comparativa[nombre]["ic_bootstrap_f1"] = intervalo_bootstrap(yt, yp)
    yt_ref, yp_ref, _pr_ref = predicciones[REFERENCIA]
    for nombre, (yt, yp, _pr) in predicciones.items():
        if nombre != REFERENCIA:
            comparativa[nombre]["equivalencia_con_la_referencia"] = equivalencia(
                yt_ref, yp_ref, yp)

    # McNemar pareado de cada fila contra la referencia. La equivalencia acota
    # cuanto pueden diferir; McNemar dice si la diferencia observada es atribuible
    # al azar. Son preguntas distintas y el cuadro principal necesita las dos.
    for nombre, (yt, yp, _pr) in predicciones.items():
        if nombre != REFERENCIA:
            comparativa[nombre]["mcnemar_contra_la_referencia"] = mcnemar(
                yt_ref, yp_ref, yp)

    orden = sorted(comparativa, key=lambda k: -comparativa[k]["f1"])
    print("  orden por F1: " + " > ".join(orden[:4]) + " ...")

    # (2) agrupada frente a NO agrupada. Mide el PROTOCOLO, no la arquitectura, y
    # por eso se hace sobre una sola --la de referencia-- y se declara: repetirlo
    # con las doce multiplicaria el coste sin cambiar lo que se mide.
    reparto: dict[str, dict] = {"agrupada": {"particion": particion.resumen(),
                                             **comparativa[REFERENCIA]}}
    part_sin = particion_agrupada(corpus, agrupar=False)
    por_semilla = []
    for semilla in semillas:
        r = ejecutar_variante(REFERENCIA, corpus, part_sin, semilla,
                              epocas=args.epocas, rapido=args.rapido,
                              reusar=args.reusar, experimento="e4",
                              sufijo="sinagr")
        por_semilla.append(r["metricas"])
    reparto["sin agrupar"] = {"particion": part_sin.resumen(),
                              "por_semilla": por_semilla, **agregar(por_semilla)}
    inflado = reparto["sin agrupar"]["f1"] - reparto["agrupada"]["f1"]
    print(f"  agrupada F1={reparto['agrupada']['f1']:.4f}  "
          f"sin agrupar F1={reparto['sin agrupar']['f1']:.4f}")

    # (3) DECISION SOBRE LA PONDERACION DE CLASES. El indicador de R1.4 exige
    # evaluarla empiricamente y no asumirla: se contrastan las dos condiciones
    # sobre la misma particion y la misma arquitectura, y la decision se toma con
    # la medicion delante. La condicion sin ponderar ya esta entrenada arriba.
    print("")
    print("  decision sobre la ponderacion de clases")
    por_semilla_p = []
    for semilla in semillas:
        r = ejecutar_variante(REFERENCIA, corpus, particion, semilla,
                              epocas=args.epocas, rapido=args.rapido,
                              reusar=args.reusar, experimento="e4",
                              sufijo="agr", ponderar_clases=True)
        por_semilla_p.append(r["metricas"])
    sin_p, con_p = comparativa[REFERENCIA], agregar(por_semilla_p)
    ponderacion = {
        "sin ponderar": {k: sin_p.get(k) for k in
                         ("f1", "f1_desv", "recall", "specificity", "precision")},
        "con ponderacion": {k: con_p.get(k) for k in
                            ("f1", "f1_desv", "recall", "specificity", "precision")},
        "diferencia_f1": round((con_p.get("f1") or 0) - (sin_p.get("f1") or 0), 4),
        "prevalencia_del_corpus": round(float(corpus.label.mean()), 4),
        "decision": (
            "No se aplica ponderación de clases. Con prevalencia {:.4f} ninguna de "
            "las dos clases es minoritaria en sentido operativo, y la exhaustividad "
            "de la clase positiva ({}) y la especificidad ({}) son equiparables sin "
            "ponderar. La ponderación {} el F1 en {:+.4f}, dentro de la dispersión "
            "entre semillas.".format(
                float(corpus.label.mean()), sin_p.get("recall"), sin_p.get("specificity"),
                "mejora" if (con_p.get("f1") or 0) > (sin_p.get("f1") or 0) else "reduce",
                (con_p.get("f1") or 0) - (sin_p.get("f1") or 0))),
    }
    print(f"    sin ponderar    F1={sin_p.get('f1'):.4f}  recall={sin_p.get('recall')}  "
          f"especificidad={sin_p.get('specificity')}")
    print(f"    con ponderacion F1={con_p.get('f1'):.4f}  recall={con_p.get('recall')}  "
          f"especificidad={con_p.get('specificity')}")

    piso = comparativa  # nombre que ya usan el informe y las figuras


    # (2) rendimiento por tamaño de conglomerado, sobre la partición agrupada
    y_true, y_pred, y_proba = predicciones[REFERENCIA]
    prueba = corpus.loc[particion.prueba]
    tam = prueba["template_cluster_id"].map(
        corpus.groupby("template_cluster_id").size()).fillna(1).to_numpy()
    por_tamano = {}
    for etiqueta, mascara in (
        ("campañas únicas (1)", tam == 1),
        ("pequeñas (2–5)", (tam >= 2) & (tam <= 5)),
        ("masivas (>5)", tam > 5),
    ):
        if mascara.sum() >= 20 and len(np.unique(y_true[mascara])) > 1:
            m = metricas(y_true[mascara], y_pred[mascara],
                         None if y_proba is None else y_proba[mascara])
            por_tamano[etiqueta] = {"n": int(mascara.sum()), "f1": m["f1"],
                                    "roc_auc": m.get("roc_auc"), "mcc": m["mcc"]}
            print(f"  {etiqueta:22s} n={mascara.sum():6,}  F1={m['f1']:.4f}")

    # (3) curva de aprendizaje
    curva = {}
    rng = np.random.default_rng(11)
    base = particion_agrupada(corpus, agrupar=True)
    for f in FRACCIONES:
        idx = base.entrenamiento if f >= 1.0 else rng.choice(
            base.entrenamiento, size=int(f * len(base.entrenamiento)), replace=False)
        recorte = type(base)(idx, base.validacion, base.prueba)
        r = ejecutar_variante(REFERENCIA, corpus, recorte, semillas[0],
                              epocas=args.epocas, rapido=args.rapido, experimento="e4",
                              reusar=args.reusar, sufijo=f"frac{int(f * 100)}")
        curva[f"{int(f*100)}%"] = {"n_entrenamiento": int(len(idx)), **{
            k: r["metricas"][k] for k in ("f1", "roc_auc", "mcc") if k in r["metricas"]}}
        print(f"  entrenamiento al {int(f*100):3d}%  F1={curva[f'{int(f*100)}%']['f1']:.4f}")

    # (4) estabilidad entre semillas de partición
    estabilidad = []
    for sp in (SEMILLAS_PARTICION[:1] if args.rapido else SEMILLAS_PARTICION):
        pp = particion_agrupada(corpus, semilla=sp, agrupar=True)
        r = ejecutar_variante(REFERENCIA, corpus, pp, semillas[0],
                              epocas=args.epocas, rapido=args.rapido, experimento="e4",
                              reusar=args.reusar, sufijo=f"psem{sp}")
        estabilidad.append({"semilla_particion": sp, "f1": r["metricas"]["f1"]})

    informe = {
        "experimento": "E4",
        "pregunta": "¿Generaliza el modelo a correo no visto?",
        "generado_en": datetime.now(timezone.utc).isoformat(),
        "datos": {"n": int(len(corpus)), "prevalencia": round(float(corpus.label.mean()), 4)},
        "arquitecturas_comparadas": list(ARQUITECTURAS) + list(CLASICOS),
        "arquitectura_de_referencia": REFERENCIA,
        "nota_sobre_la_referencia": (
            "La referencia se usa para los analisis de protocolo y como base de "
            "las equivalencias pareadas. No implica que sea la mejor: el cuadro "
            "comparativo informa todas las filas y ninguna se selecciona."),
        "ponderacion_de_clases_r1_4": ponderacion,
        "reparto": reparto,
        "cuadro_comparativo_corpus_completo": comparativa,
        "inflado_por_no_agrupar": round(inflado, 4),
        "por_tamano_de_conglomerado": por_tamano,
        "curva_de_aprendizaje": curva,
        "estabilidad_entre_particiones": {
            "corridas": estabilidad,
            "desviacion": round(float(np.std([e["f1"] for e in estabilidad])), 4),
        },
        "ejes_no_medibles": {
            "temporal": "Kaggle no conserva fechas: 0 columnas, 3.46% recuperable del cuerpo",
            "remitente": "Kaggle sin sender; 99.5% de los 7,002 dominios es de clase única",
            "coleccion": "clase idéntica a procedencia en cinco de ocho colecciones",
            "referencia": "doc/DECISIONES_CORPUS_Y_PROTOCOLO.md",
        },
    }

    def figura(destino: Path) -> None:
        # El cuadro completo, con las doce filas. Es la lamina que encabeza el
        # capitulo: la de E1 compara sobre el subconjunto trimodal, donde la clase
        # coincide con la procedencia, y no representa el rendimiento del sistema.
        orden_fig = sorted(comparativa, key=lambda k: -comparativa[k]["f1"])
        etiquetas = [CLASICOS[n]["etiqueta"] if n in CLASICOS else etiqueta_de(n)
                     for n in orden_fig]
        barras_con_error(
            destino, "e4_cuadro_comparativo",
            "E4. Todas las arquitecturas sobre el corpus completo\n"
            "F1 con partición agrupada por campaña, media de tres semillas",
            etiquetas,
            [comparativa[n]["f1"] for n in orden_fig],
            [comparativa[n].get("f1_desv", 0.0) for n in orden_fig],
            resaltar=orden_fig.index(REFERENCIA))


        barras_con_error(
            destino, "e4_agrupacion",
            "E4. Cuánto infla la métrica no agrupar por campaña",
            list(reparto), [v["f1"] for v in reparto.values()],
            [v.get("f1_desv", 0.0) for v in reparto.values()], resaltar=1)
        if por_tamano:
            barras_con_error(
                destino, "e4_por_tamano",
                "E4. ¿Detecta las campañas raras tan bien como las masivas?",
                list(por_tamano), [v["f1"] for v in por_tamano.values()],
                [0.0] * len(por_tamano))
        barras_con_error(
            destino, "e4_curva_aprendizaje",
            "E4. Curva de aprendizaje: ¿limitado por datos o saturado?",
            list(curva), [v["f1"] for v in curva.values()], [0.0] * len(curva))

    emitir("e4", informe, figura, corpus=corpus,
           particion=particion_agrupada(corpus))
    print(f"\n  no agrupar infla el F1 en {inflado:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
