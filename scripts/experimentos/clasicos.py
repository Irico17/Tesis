"""
Líneas base clásicas (B1, B3, B4, B5) sobre la partición de los experimentos.

Existe como módulo único porque las consumen cuatro experimentos --E1, E2, E3 y
E4-- y si cada uno se escribiera su propio adaptador acabarían ajustando el
vectorizador de forma distinta, o partiendo distinto, y las filas dejarían de ser
comparables entre tablas sin que nada avisara.

Los guiones `run_bN` originales no sirven aquí: esperan las columnas `holdout`,
`stratified_cv` y `group_loso_cv`, que son el esquema de partición del corpus
ANTERIOR. Lo que sí se reutiliza --y es lo que importa-- son las tuberías de
modelo y la extracción de rasgos, de modo que estas líneas base son las mismas
que las del capítulo previo y no unas nuevas escritas para la ocasión.

Qué es cada una:

  b1_tfidf_lr           TF-IDF + regresión logística.  Unimodal de TEXTO.
  b3_rf_estructura      Bosque aleatorio sobre rasgos DOM.  Unimodal de ESTRUCTURA.
  b4_red                Bosque aleatorio sobre autenticación y URL.  Unimodal de RED.
  b5_multimodal_clasico TF-IDF + numéricos concatenados + regresión logística.
                        Las TRES modalidades fusionadas SIN atención.

B2 no está a propósito: es DistilBERT afinado, que es exactamente la variante
`solo_texto` de la red. Tenerlas las dos sería la misma fila con dos nombres.

Se ejecutan en CPU. Ninguna necesita GPU, de modo que pueden correr en paralelo
a la cola de entrenamiento sin disputarle la tarjeta.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from comun import Particion, metricas  # noqa: E402

# Nombre visible -> (módulo, familia). La familia decide si la fila puede entrar
# en un veredicto de sinergia: ver la nota de `es_unimodal`.
CLASICOS: dict[str, dict] = {
    "b1_tfidf_lr": {"modalidades": ("texto",),
                    "etiqueta": "B1: TF-IDF y regresión logística\n(texto)"},
    "b3_rf_estructura": {"modalidades": ("estructura",),
                         "etiqueta": "B3: bosque aleatorio\n(estructura)"},
    "b4_red": {"modalidades": ("red",),
               "etiqueta": "B4: bosque aleatorio\n(red)"},
    "b5_multimodal_clasico": {"modalidades": ("texto", "estructura", "red"),
                              "etiqueta": "B5: fusión clásica\n(las tres modalidades)"},
}


def es_unimodal(nombre: str) -> bool:
    return len(CLASICOS[nombre]["modalidades"]) == 1


def _enmascarar_prueba(prueba: pd.DataFrame, enmascarar: tuple[str, ...]) -> pd.DataFrame:
    """Retira ramas SOLO del conjunto de prueba, como hace E3 con la red.

    Es la manipulación que da sentido a comparar: B5 no tiene centinela ni se
    entrenó con descarte modal, así que cuando una rama desaparece en despliegue
    recibe ceros sin saber que faltan. Medir cuánto se degrada frente a cuánto se
    degrada la red es lo que muestra si el mecanismo de ausencia hace algo.
    """
    from phishing_pipeline.features.vectorizer import (
        NETWORK_FEATURE_COLS, STRUCTURAL_FEATURE_COLS)
    from phishing_model.config import NETWORK_CATEGORICAL_COLS

    prueba = prueba.copy()
    if "estructura" in enmascarar:
        for c in STRUCTURAL_FEATURE_COLS:
            if c in prueba.columns:
                prueba[c] = 0
    if "red" in enmascarar:
        for c in NETWORK_FEATURE_COLS:
            if c in prueba.columns:
                prueba[c] = 0
        for c in NETWORK_CATEGORICAL_COLS:
            if c in prueba.columns:
                prueba[c] = None
    return prueba


def _ajustar_b1(tr: pd.DataFrame, semilla: int):
    from phishing_baseline.b1_tfidf_lr import _make_pipeline

    pipe = _make_pipeline()
    pipe.set_params(clf__random_state=semilla)
    # El vectorizador se ajusta SOLO sobre entrenamiento: ajustarlo sobre todo
    # filtraría el vocabulario de la prueba, que es fuga de la más silenciosa.
    pipe.fit(tr["clean_text"].fillna("").astype(str), tr["label"].to_numpy())
    return pipe


def _b1(modelo, te: pd.DataFrame):
    x = te["clean_text"].fillna("").astype(str)
    return modelo.predict(x), modelo.predict_proba(x)[:, 1]


def _ajustar_b3(tr: pd.DataFrame, semilla: int):
    from phishing_baseline.b3_rf_structural import _extract_features, _make_pipeline

    Xtr = _extract_features(tr)
    pipe = _make_pipeline(Xtr.shape[1])
    pipe.set_params(clf__random_state=semilla)
    pipe.fit(Xtr, tr["label"].to_numpy())
    return pipe


def _b3(modelo, te: pd.DataFrame):
    from phishing_baseline.b3_rf_structural import _extract_features

    X = _extract_features(te)
    return modelo.predict(X), modelo.predict_proba(X)[:, 1]


def _ajustar_b4(tr: pd.DataFrame, semilla: int):
    from phishing_baseline.b4_network import _extract_features, _make_pipeline

    Xtr = _extract_features(tr)
    pipe = _make_pipeline(Xtr.shape[1])
    pipe.set_params(clf__random_state=semilla)
    pipe.fit(Xtr, tr["label"].to_numpy())
    return pipe


def _b4(modelo, te: pd.DataFrame):
    from phishing_baseline.b4_network import _extract_features

    X = _extract_features(te)
    return modelo.predict(X), modelo.predict_proba(X)[:, 1]


def _ajustar_b5(tr: pd.DataFrame, semilla: int):
    from scipy.sparse import csr_matrix, hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from phishing_baseline.b5_classical_multimodal import MAX_TFIDF_FEATURES, _numeric_matrix

    vec = TfidfVectorizer(max_features=MAX_TFIDF_FEATURES, ngram_range=(1, 2))
    Xt = vec.fit_transform(tr["clean_text"].fillna("").astype(str))
    esc = StandardScaler()
    Xn = esc.fit_transform(_numeric_matrix(tr))
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=semilla)
    clf.fit(hstack([Xt, csr_matrix(Xn)]).tocsr(), tr["label"].to_numpy())
    return (vec, esc, clf)


def _b5(modelo, te: pd.DataFrame):
    from scipy.sparse import csr_matrix, hstack

    from phishing_baseline.b5_classical_multimodal import _numeric_matrix

    vec, esc, clf = modelo
    X = hstack([vec.transform(te["clean_text"].fillna("").astype(str)),
                csr_matrix(esc.transform(_numeric_matrix(te)))]).tocsr()
    return clf.predict(X), clf.predict_proba(X)[:, 1]


_AJUSTE = {"b1_tfidf_lr": _ajustar_b1, "b3_rf_estructura": _ajustar_b3,
           "b4_red": _ajustar_b4, "b5_multimodal_clasico": _ajustar_b5}
_IMPL = {"b1_tfidf_lr": _b1, "b3_rf_estructura": _b3,
         "b4_red": _b4, "b5_multimodal_clasico": _b5}

# Memoria del ajuste. La rejilla de degradación de E3 evalúa la misma línea base
# sobre veinticuatro conjuntos de prueba distintos, y el ENTRENAMIENTO es idéntico
# en las veinticuatro: reajustarlo cada vez multiplicaba por veinticuatro un coste
# que no cambia el resultado. La clave incluye las filas de entrenamiento, de modo
# que un corpus o una partición distintos nunca reutilizan un ajuste ajeno.
_MEMORIA: dict[tuple, object] = {}


def _modelo(nombre: str, tr: pd.DataFrame, semilla: int):
    clave = (nombre, int(semilla), len(tr), int(pd.util.hash_pandas_object(
        tr.index.to_series(), index=False).sum()))
    if clave not in _MEMORIA:
        _MEMORIA[clave] = _AJUSTE[nombre](tr, semilla)
    return _MEMORIA[clave]


PREDICCIONES = BASE / "data" / "predictions"


def _guardar_predicciones(marca: str, prueba: pd.DataFrame, y_true, y_pred,
                          y_proba) -> None:
    """Deja en disco la prediccion fila por fila de una linea base.

    Las redes ya lo hacian y las lineas base no, de modo que el medio de
    verificacion de R2.2 --curvas ROC y matrices de confusion de TODOS los
    modelos-- solo podia construirse para la mitad del cuadro.
    """
    PREDICCIONES.mkdir(parents=True, exist_ok=True)
    columna_id = "email_id" if "email_id" in prueba.columns else None
    pd.DataFrame({
        "email_id": (prueba[columna_id].to_numpy() if columna_id
                     else prueba.index.to_numpy()),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "model_name": nombre_de(marca),
        "split_name": marca,
    }).to_parquet(PREDICCIONES / f"{marca}_preds.parquet", index=False)


def nombre_de(marca: str) -> str:
    """El identificador de modelo dentro de una marca de corrida."""
    return marca.split("_s4")[0].split("_", 1)[-1] if "_" in marca else marca


def ejecutar_clasico(nombre: str, corpus: pd.DataFrame, particion: Particion,
                     semilla: int = 42,
                     enmascarar: tuple[str, ...] | None = None,
                     prueba_alternativa: pd.DataFrame | None = None,
                     marca: str | None = None) -> dict:
    """Ajusta una línea base sobre la misma partición que usa la red y evalúa.

    `semilla` alimenta la aleatoriedad del modelo --el remuestreo del bosque, en
    los dos que lo usan--. Se varía sobre las mismas semillas que la red no
    porque haga falta para el ajuste, sino para que la fila clásica tenga una
    dispersión estimada igual que las neuronales: comparar una media de tres
    contra un valor único invita a leer como diferencia lo que es ruido.
    """
    if nombre not in _IMPL:
        raise KeyError(f"línea base desconocida: {nombre}")

    entrenamiento = corpus.loc[particion.entrenamiento]
    # `prueba_alternativa` recibe el conjunto de prueba ya transformado --texto
    # degradado, en E3--. El ENTRENAMIENTO sale siempre del corpus limpio: el
    # modelo se ajusta en condiciones normales y se le cambia el mundo debajo,
    # que es la situación que se quiere medir.
    prueba = prueba_alternativa if prueba_alternativa is not None else corpus.loc[particion.prueba]
    if enmascarar:
        prueba = _enmascarar_prueba(prueba, enmascarar)

    y_pred, y_proba = _IMPL[nombre](_modelo(nombre, entrenamiento, semilla), prueba)
    y_true = prueba["label"].to_numpy()
    if marca:
        _guardar_predicciones(marca, prueba, y_true, np.asarray(y_pred),
                              np.asarray(y_proba))
    return {
        "linea_base": nombre,
        "semilla": semilla,
        "metricas": metricas(y_true, y_pred, y_proba),
        "y_true": y_true,
        "y_pred": np.asarray(y_pred),
        "y_proba": np.asarray(y_proba),
    }
