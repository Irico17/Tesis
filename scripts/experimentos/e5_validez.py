"""
E5 · Validez: ¿cuánto de lo medido es procedencia y no fenómeno?

**No es un experimento de rendimiento y no sostiene ninguna afirmación de
desempeño.** Califica los cuatro anteriores: si sus cifras se explican en buena
parte por la colección de la que procede cada correo, deben leerse como tales.

Tres mediciones, ninguna de las cuales entrena el modelo propuesto:

1. **Auditoría por característica.** Para cada una de las dieciocho no textuales,
   la información que aporta sobre la clase condicionada a la colección frente a
   la que aporta sobre la colección misma. El cociente dice si describe el
   fenómeno o su procedencia.
2. **Control de procedencia.** Un clasificador que ve ÚNICAMENTE rasgos
   identificadores de la colección, sin acceso al contenido. Su desempeño es el
   suelo del confusor: cualquier modelo debe superarlo con holgura para que su
   cifra signifique algo.
3. **Pliegues por colección, como diagnóstico.** Dejar fuera colecciones enteras y
   observar dónde se desploman las modalidades no textuales. No mide
   generalización, porque la colección es un artefacto del ensamblado y no una
   propiedad del fenómeno, sino dónde la clase coincide con el origen.
4. **Composición por idioma.** Qué idiomas contiene cada colección y si el idioma
   predice la clase. Responde a dos preguntas distintas: si el idioma es otra vía
   por la que la clase coincide con la procedencia, y si el codificador textual
   empleado, monolingüe en inglés, resulta adecuado al material.

    python scripts/experimentos/e5_validez.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from comun import cargar_corpus, emitir, metricas, particion_agrupada


def auditoria_de_caracteristicas(df: pd.DataFrame) -> list[dict]:
    """Información mutua sobre la clase (dada la colección) frente a la colección."""
    from sklearn.feature_selection import mutual_info_classif
    from phishing_pipeline.features.vectorizer import (NETWORK_FEATURE_COLS,
                                                       STRUCTURAL_FEATURE_COLS)

    columnas = [c for c in list(STRUCTURAL_FEATURE_COLS) + list(NETWORK_FEATURE_COLS)
                if c in df.columns]
    X = df[columnas].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
    fuente = pd.factorize(df["source_dataset"])[0]

    # La información sobre la clase se calcula SOLO donde la colección tiene las
    # dos clases: en una de clase única es nula por construcción y promediarla
    # con las demás la diluiría hasta hacerla ilegible.
    mixtas = df.groupby("source_dataset")["label"].nunique()
    mascara = df["source_dataset"].isin(mixtas[mixtas > 1].index).to_numpy()

    mi_clase = (mutual_info_classif(X[mascara], df["label"].to_numpy()[mascara],
                                    random_state=0)
                if mascara.sum() > 50 else np.zeros(len(columnas)))
    mi_fuente = mutual_info_classif(X, fuente, random_state=0)

    filas = []
    for i, c in enumerate(columnas):
        cociente = float(mi_clase[i] / mi_fuente[i]) if mi_fuente[i] > 1e-9 else 0.0
        filas.append({
            "caracteristica": c,
            "mi_clase_dada_la_coleccion": round(float(mi_clase[i]), 4),
            "mi_coleccion": round(float(mi_fuente[i]), 4),
            "cociente": round(cociente, 4),
        })
    return sorted(filas, key=lambda f: f["cociente"])


def control_de_procedencia(df: pd.DataFrame) -> dict:
    """Clasificador que solo ve rasgos de procedencia, sin acceso al contenido.

    Es el suelo del confusor. Se le dan exclusivamente las banderas de
    disponibilidad de modalidad y la presencia de campos de cabecera: nada de
    texto, ni de URL, ni de estructura. Lo que consiga es lo que puede lograrse
    sabiendo únicamente de qué colección viene el correo.
    """
    from sklearn.ensemble import RandomForestClassifier

    rasgos = pd.DataFrame({
        "tiene_estructura": df["has_structure_modality"].astype(int),
        "tiene_red": df["has_network_modality"].astype(int),
        "tiene_spf": df["spf_result"].notna().astype(int),
        "tiene_dkim": df["dkim_result"].notna().astype(int),
        "tiene_dmarc": df["dmarc_result"].notna().astype(int),
        "tiene_remitente": df["sender"].notna().astype(int) if "sender" in df else 0,
        "tiene_fecha": df["sent_date"].notna().astype(int) if "sent_date" in df else 0,
    })
    p = particion_agrupada(df, agrupar=True)
    modelo = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
    modelo.fit(rasgos.loc[p.entrenamiento], df.loc[p.entrenamiento, "label"])
    y_true = df.loc[p.prueba, "label"].to_numpy()
    y_pred = modelo.predict(rasgos.loc[p.prueba])
    y_proba = modelo.predict_proba(rasgos.loc[p.prueba])[:, 1]
    m = metricas(y_true, y_pred, y_proba)
    return {
        "rasgos": list(rasgos.columns),
        "n_prueba": int(len(y_true)),
        **{k: m[k] for k in ("f1", "accuracy", "roc_auc", "mcc", "balanced_accuracy")
           if k in m},
    }


def clase_por_coleccion(df: pd.DataFrame) -> dict:
    """Hasta qué punto conocer la colección equivale a conocer la clase."""
    t = df.groupby("source_dataset")["label"].agg(["size", "mean", "nunique"])
    return {
        "por_coleccion": {
            str(i): {"n": int(r["size"]), "prevalencia": round(float(r["mean"]), 4),
                     "clases": int(r["nunique"])}
            for i, r in t.iterrows()
        },
        "colecciones_de_clase_unica": int((t["nunique"] == 1).sum()),
        "colecciones_totales": int(len(t)),
        "fraccion_del_corpus_en_colecciones_de_clase_unica": round(
            float(t.loc[t["nunique"] == 1, "size"].sum() / t["size"].sum()), 4),
    }


# Muestra por colección para la detección de idioma. `langdetect` tarda unos
# milisegundos por mensaje: sobre las cuarenta y cuatro mil filas del corpus serían
# varios minutos en cada ejecución de E5, y la composición por idioma de una
# colección se estima igual de bien con una muestra. La semilla es fija, de modo
# que la muestra no cambia entre corridas.
MUESTRA_POR_COLECCION = 1500
SEMILLA_MUESTRA = 7

# Por debajo de este número de caracteres la detección es ruido: el detector
# devuelve un idioma con la misma seguridad para un asunto de cinco palabras que
# para un cuerpo completo, y se equivoca mucho más.
MINIMO_CARACTERES = 60


def _detectar(texto: str) -> str:
    """Idioma de un texto, o una etiqueta explícita cuando no puede determinarse."""
    from langdetect import DetectorFactory, detect
    from langdetect.lang_detect_exception import LangDetectException

    DetectorFactory.seed = 0  # el detector es estocástico si no se fija la semilla
    limpio = " ".join(str(texto or "").split())
    if len(limpio) < MINIMO_CARACTERES:
        return "indeterminado"
    try:
        return detect(limpio[:2000])
    except LangDetectException:
        return "indeterminado"


def composicion_por_idioma(df: pd.DataFrame) -> dict:
    """Qué idiomas hay en cada colección y si el idioma predice la clase.

    La pregunta tiene dos partes que conviene no mezclar.

    La primera es de validez, y es la misma que atraviesa el resto de E5: si cada
    colección estuviera escrita en un idioma distinto y la clase coincidiera con la
    colección, el idioma sería otra vía por la que la clase se predice desde la
    procedencia. Se mide con el mismo criterio que las demás características: la
    información que el idioma aporta sobre la clase dentro de cada colección frente
    a la que aporta sobre la colección misma.

    La segunda es de diseño y afecta al modelo: el codificador textual empleado es
    monolingüe en inglés. Si una fracción apreciable del corpus no lo fuera, la
    representación textual de esos mensajes sería peor de lo necesario y convendría
    un codificador multilingüe. Esa comparación no se hace aquí; lo que se hace es
    medir la fracción, que es lo que decide si la pregunta llega a plantearse.
    """
    from sklearn.feature_selection import mutual_info_classif

    partes = []
    for _, g in df.groupby("source_dataset"):
        partes.append(g.sample(min(len(g), MUESTRA_POR_COLECCION),
                               random_state=SEMILLA_MUESTRA))
    muestra = pd.concat(partes)
    idiomas = muestra["clean_text"].map(_detectar)

    por_coleccion = {}
    for coleccion, g in muestra.groupby("source_dataset"):
        cuenta = idiomas.loc[g.index].value_counts()
        total = int(cuenta.sum())
        por_coleccion[str(coleccion)] = {
            "n_muestreado": total,
            "idioma_predominante": str(cuenta.index[0]),
            "fraccion_del_predominante": round(float(cuenta.iloc[0] / total), 4),
            "fraccion_ingles": round(float(cuenta.get("en", 0) / total), 4),
            "idiomas_distintos": int(len(cuenta)),
            "distribucion": {str(k): int(v) for k, v in cuenta.head(6).items()},
        }

    globales = idiomas.value_counts()
    n = int(globales.sum())
    determinados = int(n - globales.get("indeterminado", 0))
    no_ingles = determinados - int(globales.get("en", 0))

    # Mismo criterio que la auditoría de características: la información sobre la
    # clase se calcula solo donde la colección tiene las dos clases.
    codigo = pd.factorize(idiomas)[0].reshape(-1, 1)
    fuente = pd.factorize(muestra["source_dataset"])[0]
    mixtas = muestra.groupby("source_dataset")["label"].nunique()
    mascara = muestra["source_dataset"].isin(mixtas[mixtas > 1].index).to_numpy()
    mi_clase = (float(mutual_info_classif(
        codigo[mascara], muestra["label"].to_numpy()[mascara],
        discrete_features=True, random_state=0)[0])
        if mascara.sum() > 50 else 0.0)
    mi_fuente = float(mutual_info_classif(codigo, fuente, discrete_features=True,
                                          random_state=0)[0])
    cociente = mi_clase / mi_fuente if mi_fuente > 1e-9 else 0.0

    # Prevalencia por idioma: si el phishing se concentrara en un idioma, bastaría
    # con detectarlo para acertar, y eso es un atajo y no una capacidad del modelo.
    prevalencia = {}
    for idioma, g in muestra.groupby(idiomas.values):
        if len(g) >= 30:
            prevalencia[str(idioma)] = {
                "n": int(len(g)),
                "prevalencia": round(float(g["label"].mean()), 4),
            }

    fraccion_no_ingles = round(no_ingles / determinados, 4) if determinados else 0.0
    return {
        "muestra_por_coleccion": MUESTRA_POR_COLECCION,
        "minimo_de_caracteres": MINIMO_CARACTERES,
        "n_muestreado": n,
        "n_con_idioma_determinado": determinados,
        "fraccion_indeterminada": round(1 - determinados / n, 4) if n else 0.0,
        "fraccion_no_ingles": fraccion_no_ingles,
        "distribucion_global": {str(k): int(v) for k, v in globales.head(12).items()},
        "por_coleccion": por_coleccion,
        "prevalencia_por_idioma": prevalencia,
        "mi_clase_dada_la_coleccion": round(mi_clase, 4),
        "mi_coleccion": round(mi_fuente, 4),
        "cociente": round(cociente, 4),
        "lectura": (
            "El {:.2%} de los mensajes con idioma determinado no está en inglés. El "
            "idioma informa {:.4f} veces más sobre la clase que sobre la colección "
            "de procedencia: ".format(fraccion_no_ingles, cociente)
            + ("por debajo de uno, de modo que describe mejor el origen del mensaje "
               "que el fenómeno y se suma a las vías por las que la clase coincide "
               "con la procedencia."
               if cociente < 1.0 else
               "por encima de uno, de modo que aporta sobre el fenómeno información "
               "que no se explica por la procedencia.")),
        "lectura_sobre_el_codificador": (
            "La fracción no inglesa es marginal, de modo que un codificador "
            "monolingüe en inglés es adecuado al material y sustituirlo por uno "
            "multilingüe no está justificado por los datos."
            if fraccion_no_ingles < 0.05 else
            "Una fracción apreciable del corpus, el {:.2%}, no está en inglés. El "
            "codificador empleado es monolingüe, de modo que la representación "
            "textual de esos mensajes es peor de lo necesario y conviene contrastar "
            "un codificador multilingüe. Se declara como medición y la comparación "
            "queda como trabajo futuro.".format(fraccion_no_ingles)),
    }


def pliegues_por_coleccion(df) -> dict:
    """Deja fuera cada colección entera y observa qué ocurre. DIAGNÓSTICO.

    No mide generalización, y conviene decirlo antes que nada: la colección es un
    artefacto de cómo se ensambló el corpus, no una propiedad del fenómeno del
    phishing. Lo que mide es DÓNDE la clase coincide con el origen, que es la
    limitación principal del trabajo, y hasta ahora se declaraba sin exhibirla.

    Se usa una bolsa de palabras con regresión logística como sonda: es barata, no
    necesita tarjeta gráfica y responde a la pregunta que importa aquí, que no es
    cuánto se acierta sino si el pliegue es siquiera evaluable. Un pliegue cuya
    colección tiene una sola clase NO es evaluable --no hay nada que discriminar--
    y se informa como tal en vez de con una cifra que parecería un resultado.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    filas = []
    for coleccion in sorted(df["source_dataset"].unique()):
        fuera = df["source_dataset"] == coleccion
        prueba, entrenamiento = df[fuera], df[~fuera]
        clases_en_prueba = int(prueba["label"].nunique())
        fila = {
            "coleccion": str(coleccion),
            "n": int(len(prueba)),
            "clases_presentes": clases_en_prueba,
            "prevalencia": round(float(prueba["label"].mean()), 4),
        }
        if clases_en_prueba < 2 or entrenamiento["label"].nunique() < 2:
            fila["evaluable"] = False
            fila["motivo"] = ("la colección contiene una sola clase, de modo que no "
                              "hay nada que discriminar dentro de ella")
            filas.append(fila)
            continue
        vec = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), sublinear_tf=True)
        X = vec.fit_transform(entrenamiento["clean_text"].fillna("").astype(str))
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        clf.fit(X, entrenamiento["label"].to_numpy())
        Xp = vec.transform(prueba["clean_text"].fillna("").astype(str))
        m = metricas(prueba["label"].to_numpy(), clf.predict(Xp), clf.predict_proba(Xp)[:, 1])
        fila["evaluable"] = True
        fila.update({k: m.get(k) for k in ("f1", "accuracy", "recall", "roc_auc")})
        filas.append(fila)

    evaluables = [f for f in filas if f.get("evaluable")]
    return {
        "naturaleza": ("Diagnóstico de dónde la clase coincide con el origen. NO mide "
                       "generalización: la colección es un artefacto del ensamblado del "
                       "corpus y no una propiedad del fenómeno."),
        "sonda": "TF-IDF con regresión logística sobre el texto",
        "pliegues": filas,
        "n_evaluables": len(evaluables),
        "n_no_evaluables": len(filas) - len(evaluables),
        "lectura": (
            "de las {} colecciones del corpus, {} no son evaluables porque contienen "
            "una sola clase. Esa es, medida, la limitación principal del corpus: en la "
            "mayor parte de él la clase coincide con la procedencia, de modo que dejar "
            "una colección fuera equivale a dejar fuera una clase entera.".format(
                len(filas), len(filas) - len(evaluables))),
    }


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    df = cargar_corpus()
    print(f"E5 · corpus completo: {len(df):,} correos")

    print("  auditando características…")
    caracteristicas = auditoria_de_caracteristicas(df)
    print("  entrenando el control de procedencia…")
    control = control_de_procedencia(df)
    estructura = clase_por_coleccion(df)
    print("  pliegues por colección (diagnóstico)…")
    pliegues = pliegues_por_coleccion(df)
    print("  composición por idioma…")
    idioma = composicion_por_idioma(df)

    mayor = max(f["cociente"] for f in caracteristicas)
    informe = {
        "experimento": "E5",
        "pregunta": "¿Cuánto de lo medido es procedencia y no fenómeno?",
        "naturaleza": (
            "Auditoría de validez. No mide rendimiento y no sostiene ninguna "
            "afirmación de desempeño: califica los resultados de E1 a E4."
        ),
        "generado_en": datetime.now(timezone.utc).isoformat(),
        "auditoria_de_caracteristicas": caracteristicas,
        "mayor_cociente": round(mayor, 4),
        "lectura_de_la_auditoria": (
            f"Ninguna de las {len(caracteristicas)} características no textuales "
            f"alcanza un cociente de uno; la mayor llega a {mayor:.4f}. Toda "
            "característica no textual informa más sobre de dónde procede el correo "
            "que sobre si es phishing."
            if mayor < 1.0 else
            "Al menos una característica informa más sobre la clase que sobre la "
            "procedencia; revisar cuál y por qué."
        ),
        "control_de_procedencia": control,
        "estructura_del_corpus": estructura,
        "pliegues_por_coleccion": pliegues,
        "composicion_por_idioma": idioma,
    }

    def figura(destino: Path) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        top = caracteristicas[-14:]
        fig, ax = plt.subplots(figsize=(8.5, 5.6))
        y = np.arange(len(top))
        ax.barh(y, [f["mi_coleccion"] for f in top], 0.4,
                label="sobre la colección", color="#C44E52")
        ax.barh(y + 0.42, [f["mi_clase_dada_la_coleccion"] for f in top], 0.4,
                label="sobre la clase, dada la colección", color="#4C72B0")
        ax.set_yticks(y + 0.21)
        ax.set_yticklabels([f["caracteristica"] for f in top], fontsize=9)
        ax.set_xlabel("información mutua (nats)")
        ax.set_title("E5 · Qué informa cada característica no textual")
        ax.legend(fontsize=9)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(destino / "e5_auditoria_caracteristicas.png", dpi=160)
        plt.close(fig)

    emitir("e5", informe, figura, corpus=df)
    print(f"\n  mayor cociente entre las 18: {mayor:.4f}")
    print(f"  control de procedencia: F1={control['f1']:.4f} "
          f"(suelo del confusor; todo modelo debe superarlo con holgura)")
    print(f"  idioma: {idioma['fraccion_no_ingles']:.2%} no inglés, "
          f"cociente {idioma['cociente']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
