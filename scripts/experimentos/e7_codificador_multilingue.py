"""
E7 · ¿Conviene un codificador multilingüe? Y sobre todo, ¿qué mediría si mejora?

E5 midió que el 16.37% de los mensajes con idioma determinado no está en inglés,
mientras el codificador textual del trabajo es monolingüe. La pregunta de si
convendría uno multilingüe es legítima, pero su respuesta NO puede leerse de
forma directa, y esa es la razón de que este experimento exista y no sea una
simple sustitución.

**El problema.** E5 midió también que, en este corpus, estar escrito en un idioma
distinto del inglés equivale casi a ser phishing: la prevalencia es 0.3959 en
inglés y alcanza o supera 0.95 en siete de los nueve idiomas restantes con
muestra suficiente. Un codificador multilingüe representa mejor esos mensajes, de
modo que casi con seguridad subirá la cifra global. Esa subida sería compatible
con dos explicaciones incompatibles entre sí: que entiende mejor el fenómeno, o
que explota con más eficacia un atajo del material. Una comparación global no
puede distinguirlas.

**La salida.** Se evalúa la misma corrida por separado sobre dos subconjuntos de
la prueba:

  Inglés     El idioma no aporta nada sobre la clase, porque todo está en inglés.
             Una diferencia aquí es atribuible al codificador y a nada más. Es el
             contraste que decide.
  No inglés  El idioma casi determina la clase. Una diferencia aquí es
             ininterpretable, y se informa como tal.

La comparación limpia ideal --entrenar y evaluar solo sobre mensajes no ingleses
de ambas clases-- no es posible con este material: la única colección con las dos
clases es Kaggle y contiene 275 mensajes no ingleses, 112 legítimos y 163 de
phishing. Con esa muestra un conjunto de prueba tendría unas decenas de mensajes
y cualquier diferencia quedaría dentro del ruido. Se declara la imposibilidad en
lugar de fabricar un resultado sin respaldo.

Se ensayan dos arquitecturas por una razón de atribución: el unimodal de texto
aísla el codificador, y la propuesta dice si el efecto sobrevive a la fusión.

    python scripts/experimentos/e7_codificador_multilingue.py
    python scripts/experimentos/e7_codificador_multilingue.py --rapido
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from comun import (SEMILLAS, agregar, cargar_corpus, emitir, metricas,
                   particion_agrupada, tpr_a_fpr)
from ejecutor import ejecutar_variante
from entrenar import etiqueta_de, mcnemar

# El codificador vigente y el multilingüe comparable. Se elige la variante
# `cased` porque es la única multilingüe de la familia: distinguir mayúsculas es
# necesario en idiomas donde marcan sustantivos, y no existe versión `uncased`.
CODIFICADORES = {
    "monolingue": None,  # el del trabajo, distilbert-base-uncased
    "multilingue": "distilbert-base-multilingual-cased",
}

# El unimodal de texto aísla el codificador; la propuesta dice si el efecto
# sobrevive a la fusión con las modalidades no textuales.
ARQUITECTURAS = ("solo_texto", "atencion_cruzada_token")


def idioma_de_la_prueba(corpus: pd.DataFrame, indices) -> pd.Series:
    """Idioma de cada mensaje del conjunto de prueba.

    Se detecta una sola vez y se reutiliza para las cuatro condiciones: hacerlo
    dentro de cada evaluación gastaría el mismo cómputo cuatro veces y, si el
    detector no fuera determinista, repartiría los mensajes de forma distinta en
    cada una, que es peor que gastar el cómputo.
    """
    from e5_validez import _detectar

    return corpus.loc[indices, "clean_text"].map(_detectar)


def _por_subconjunto(y_true, y_pred, y_proba, mascara) -> dict:
    """Métricas restringidas a un subconjunto de la prueba."""
    if mascara.sum() < 30 or len(np.unique(np.asarray(y_true)[mascara])) < 2:
        return {"n": int(mascara.sum()), "evaluable": False,
                "motivo": ("menos de 30 mensajes" if mascara.sum() < 30
                           else "una sola clase en el subconjunto")}
    m = metricas(np.asarray(y_true)[mascara], np.asarray(y_pred)[mascara],
                 np.asarray(y_proba)[mascara])
    return {"n": int(mascara.sum()), "evaluable": True,
            **{k: m[k] for k in ("f1", "accuracy", "precision", "recall", "roc_auc")
               if k in m}}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epocas", type=int, default=3)
    p.add_argument("--rapido", action="store_true",
                   help="una semilla y pocos pasos, solo para comprobar que corre")
    args = p.parse_args()

    corpus = cargar_corpus()
    particion = particion_agrupada(corpus, agrupar=True)
    print(f"E7 · corpus completo: {len(corpus):,} correos")
    print(f"  partición agrupada: {particion.resumen()}")

    semillas = SEMILLAS[:1] if args.rapido else SEMILLAS

    print("  detectando el idioma del conjunto de prueba…")
    idiomas = idioma_de_la_prueba(corpus, particion.prueba)
    es_ingles = (idiomas == "en").to_numpy()
    no_ingles = (~es_ingles) & (idiomas != "indeterminado").to_numpy()
    print(f"    inglés {es_ingles.sum():,}  ·  no inglés {no_ingles.sum():,}  "
          f"·  indeterminado {(len(idiomas) - es_ingles.sum() - no_ingles.sum()):,}")

    resultados: dict[str, dict] = {}
    predicciones: dict[str, tuple] = {}

    for arquitectura in ARQUITECTURAS:
        for condicion, codificador in CODIFICADORES.items():
            clave = f"{arquitectura}__{condicion}"
            corridas = []
            for semilla in semillas:
                r = ejecutar_variante(
                    arquitectura, corpus, particion, semilla,
                    epocas=args.epocas, rapido=args.rapido, sin_centinela=True,
                    experimento="e7", codificador=codificador,
                )
                corridas.append(r)
                print(f"  {clave:44s} semilla {semilla}  "
                      f"F1={r['metricas']['f1']:.4f}")

            base = corridas[0]
            y_true, y_pred = base["y_true"], base["y_pred"]
            y_proba = base["y_proba"]
            predicciones[clave] = (y_true, y_pred, y_proba)

            resultados[clave] = {
                "arquitectura": arquitectura,
                "condicion": condicion,
                "codificador": codificador or "distilbert-base-uncased",
                **agregar([c["metricas"] for c in corridas]),
                "punto_de_operacion": tpr_a_fpr(y_true, y_proba),
                "por_idioma": {
                    "ingles": _por_subconjunto(y_true, y_pred, y_proba, es_ingles),
                    "no_ingles": _por_subconjunto(y_true, y_pred, y_proba, no_ingles),
                },
            }

    # --- el contraste que decide, sobre el subconjunto en inglés ---
    contrastes = {}
    for arquitectura in ARQUITECTURAS:
        mono = f"{arquitectura}__monolingue"
        multi = f"{arquitectura}__multilingue"
        if mono not in predicciones or multi not in predicciones:
            continue
        yt, yp_mono, _ = predicciones[mono]
        _, yp_multi, _ = predicciones[multi]
        entrada = {}
        for nombre, mascara in (("ingles", es_ingles), ("no_ingles", no_ingles),
                                ("global", np.ones(len(yt), dtype=bool))):
            if mascara.sum() < 30:
                entrada[nombre] = {"n": int(mascara.sum()), "evaluable": False}
                continue
            mc = mcnemar(np.asarray(yt)[mascara], np.asarray(yp_mono)[mascara],
                         np.asarray(yp_multi)[mascara])
            f1_mono = resultados[mono]["por_idioma"].get(nombre, {}).get("f1")
            f1_multi = resultados[multi]["por_idioma"].get(nombre, {}).get("f1")
            if nombre == "global":
                f1_mono, f1_multi = resultados[mono]["f1"], resultados[multi]["f1"]
            entrada[nombre] = {
                "n": int(mascara.sum()),
                "evaluable": True,
                "f1_monolingue": f1_mono,
                "f1_multilingue": f1_multi,
                "diferencia": (round(f1_multi - f1_mono, 4)
                               if None not in (f1_mono, f1_multi) else None),
                "mcnemar": mc,
            }
        contrastes[arquitectura] = entrada

    # La lectura se construye a partir del contraste sobre el subconjunto en
    # inglés, que es el único interpretable, y se dice por qué el otro no lo es.
    decisiva = contrastes.get("solo_texto", {}).get("ingles", {})
    dif_ingles = decisiva.get("diferencia")
    significativo = (decisiva.get("mcnemar") or {}).get("significativo_005")

    if dif_ingles is None:
        lectura = ("El contraste sobre el subconjunto en inglés no resultó "
                   "evaluable, de modo que la pregunta queda sin respuesta.")
    elif not significativo:
        lectura = (
            f"Sobre el subconjunto en inglés, donde el idioma no puede predecir la "
            f"clase, la diferencia entre ambos codificadores es de {dif_ingles:+.4f} "
            "en F1 y la prueba de McNemar no la distingue del azar. El codificador "
            "multilingüe no representa mejor el material en la parte donde su "
            "ventaja podría atribuirse a la calidad de la representación.")
    elif dif_ingles > 0:
        lectura = (
            f"Sobre el subconjunto en inglés la diferencia es de {dif_ingles:+.4f} en "
            "F1 y resulta significativa: el codificador multilingüe representa mejor "
            "el material incluso donde el idioma no aporta información sobre la "
            "clase, de modo que la ventaja no se explica por el atajo.")
    else:
        lectura = (
            f"Sobre el subconjunto en inglés la diferencia es de {dif_ingles:+.4f} en "
            "F1 y resulta significativa en contra del multilingüe, que es lo "
            "esperable de un modelo con la misma capacidad repartida entre muchos "
            "idiomas. Cualquier ganancia global procede entonces del subconjunto no "
            "inglés, donde el idioma casi determina la clase y la mejora no es "
            "interpretable como comprensión del fenómeno.")

    informe = {
        "experimento": "E7",
        "pregunta": ("Conviene un codificador multilingue, y que mediria una "
                     "mejora si la hubiera?"),
        "naturaleza": (
            "Contraste de diseño con un confusor declarado. La comparación global "
            "no es interpretable porque en este corpus el idioma distinto del "
            "inglés casi determina la clase; el contraste que decide es el del "
            "subconjunto en inglés."),
        "generado_en": datetime.now(timezone.utc).isoformat(),
        "codificadores": {k: (v or "distilbert-base-uncased")
                          for k, v in CODIFICADORES.items()},
        "arquitecturas": list(ARQUITECTURAS),
        "semillas": list(semillas),
        "composicion_de_la_prueba": {
            "ingles": int(es_ingles.sum()),
            "no_ingles": int(no_ingles.sum()),
            "indeterminado": int(len(idiomas) - es_ingles.sum() - no_ingles.sum()),
        },
        "limite_del_material": (
            "La comparacion limpia ideal, entrenando y evaluando solo sobre "
            "mensajes no ingleses de ambas clases, no es posible: la unica "
            "coleccion con las dos clases es Kaggle y contiene 275 mensajes no "
            "ingleses, 112 legitimos y 163 de phishing. Se declara la "
            "imposibilidad en lugar de fabricar un resultado sin respaldo."),
        "resultados": resultados,
        "contrastes": contrastes,
        "lectura": lectura,
    }

    def figura(destino: Path) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        grupos = ["global", "ingles", "no_ingles"]
        titulos = {"global": "Toda la prueba", "ingles": "Solo en inglés",
                   "no_ingles": "Solo fuera del inglés"}
        fig, ejes = plt.subplots(1, len(ARQUITECTURAS),
                                 figsize=(5.6 * len(ARQUITECTURAS), 4.8), sharey=True)
        ejes = np.atleast_1d(ejes)
        x = np.arange(len(grupos))
        for eje, arquitectura in zip(ejes, ARQUITECTURAS):
            for i, condicion in enumerate(CODIFICADORES):
                clave = f"{arquitectura}__{condicion}"
                if clave not in resultados:
                    continue
                alturas = []
                for g in grupos:
                    if g == "global":
                        alturas.append(resultados[clave]["f1"] or 0)
                    else:
                        alturas.append(
                            resultados[clave]["por_idioma"][g].get("f1") or 0)
                eje.bar(x + i * 0.38, alturas, 0.36,
                        label="monolingüe" if condicion == "monolingue"
                        else "multilingüe",
                        color="#4C72B0" if condicion == "monolingue" else "#DD8452",
                        edgecolor="#22303F", linewidth=0.6)
                for j, a in enumerate(alturas):
                    eje.text(x[j] + i * 0.38, a + 0.006, f"{a:.4f}",
                             ha="center", fontsize=8)
            eje.set_xticks(x + 0.19)
            eje.set_xticklabels([titulos[g] for g in grupos], fontsize=9)
            # `etiqueta_de` devuelve nombres partidos en dos lineas para los
            # rotulos del eje; como titulo de subgrafico se quieren en una.
            eje.set_title(" ".join(etiqueta_de(arquitectura).split()), fontsize=10)
            eje.grid(axis="y", alpha=0.3)
        ejes[0].set_ylabel("F1")
        ejes[0].legend(fontsize=9, loc="lower left")
        fig.suptitle("E7. Codificador monolingüe frente a multilingüe\n"
                     "el contraste que decide es el del subconjunto en inglés, "
                     "donde el idioma no predice la clase", fontsize=11)
        fig.tight_layout()
        fig.savefig(destino / "e7_codificador.png", dpi=160)
        plt.close(fig)

    emitir("e7", informe, figura, corpus=corpus, particion=particion)
    print(f"\n  {lectura}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
