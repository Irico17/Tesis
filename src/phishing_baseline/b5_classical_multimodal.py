"""
B5: modelo de referencia CLÁSICO sobre las tres modalidades concatenadas.

Motivación metodológica. Los modelos B1 a B4 son unimodales, de modo que el
conjunto permite sostener que la fusión supera a cada modalidad por separado.
Ahora bien, ese resultado no basta para atribuir la mejora al mecanismo de
atención cruzada propuesto, porque deja abierta una explicación rival evidente:
que el desempeño provenga simplemente de *disponer* de las tres modalidades, y
no de la forma en que se las combina. Formulada como pregunta de un evaluador:
«¿por qué no basta con un modelo clásico alimentado con todas las
características?».

B5 responde esa pregunta. Concatena la representación textual de B1 —TF-IDF de
uni y bigramas— con las características estructurales y de red empleadas por B3
y B4, y ajusta un clasificador clásico sobre el vector resultante. Constituye,
por tanto, una fusión temprana ingenua sin ningún mecanismo de atención.

La comparación que habilita es la decisiva del trabajo:

- Si el modelo multimodal propuesto supera a B5 de forma estadísticamente
  significativa, la mejora es atribuible al mecanismo de fusión y no a la mera
  disponibilidad de las modalidades, que es la contribución que la tesis
  reivindica.
- Si no lo supera, se trata de un hallazgo legítimo que debe reportarse: la
  complejidad arquitectónica adicional no estaría justificada por la evidencia
  sobre este corpus, y la conclusión honesta sería que la multimodalidad aporta
  pero la atención cruzada no añade valor por encima de una concatenación.

Omitir este control dejaría la afirmación central del trabajo sin respaldo
frente a la alternativa más simple y, previsiblemente, sería el primer
cuestionamiento en la sustentación.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from phishing_baseline.b4_network import _encode_categorical
from phishing_baseline.evaluation import save_predictions
from phishing_baseline.group_cv import (
    run_group_loso_cv,
    run_holdout_eval,
    run_stratified_cv,
)
from phishing_model.config import NETWORK_CATEGORICAL_COLS
from phishing_pipeline.config import RANDOM_STATE
from phishing_pipeline.features.vectorizer import NETWORK_FEATURE_COLS, STRUCTURAL_FEATURE_COLS

BASELINE_NAME = "B5_Classical_Multimodal"

MAX_TFIDF_FEATURES = 50_000  # mismo presupuesto léxico que B1, para que la comparación sea limpia
NUMERIC_COLS = list(STRUCTURAL_FEATURE_COLS) + list(NETWORK_FEATURE_COLS)
CATEGORICAL_COLS = list(NETWORK_CATEGORICAL_COLS)


def _numeric_matrix(df: pd.DataFrame) -> np.ndarray:
    numeric = df[NUMERIC_COLS].fillna(0).astype(float).values
    categorical = np.stack([_encode_categorical(df, c) for c in CATEGORICAL_COLS], axis=1)
    return np.hstack([numeric, categorical.astype(float)])


def _fit_predict(
    train_texts: pd.Series,
    train_numeric: np.ndarray,
    y_train: np.ndarray,
    test_texts: pd.Series,
    test_numeric: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Ajusta el vectorizador y el escalador SOLO sobre entrenamiento y predice."""
    vectorizer = TfidfVectorizer(max_features=MAX_TFIDF_FEATURES, ngram_range=(1, 2))
    X_train_text = vectorizer.fit_transform(train_texts)
    X_test_text = vectorizer.transform(test_texts)

    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(train_numeric)
    X_test_num = scaler.transform(test_numeric)

    X_train = hstack([X_train_text, csr_matrix(X_train_num)]).tocsr()
    X_test = hstack([X_test_text, csr_matrix(X_test_num)]).tocsr()

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    return clf.predict(X_test), clf.predict_proba(X_test)[:, 1]


def _make_cv_fit_predict(df: pd.DataFrame):
    """Adapta `_fit_predict` a la interfaz de `group_cv`, que opera sobre índices.

    Las funciones de validación cruzada entregan matrices de características ya
    materializadas, pero aquí el vectorizador debe ajustarse dentro de cada
    pliegue para no filtrar vocabulario del conjunto de evaluación. Se pasa por
    tanto una matriz de ÍNDICES y se resuelven las filas correspondientes en cada
    llamada.
    """
    texts = df["clean_text"].fillna("").astype(str).reset_index(drop=True)
    numeric = _numeric_matrix(df)

    def fit_predict(
        X_train_idx: np.ndarray, y_train: np.ndarray, X_test_idx: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        tr = X_train_idx[:, 0].astype(int)
        te = X_test_idx[:, 0].astype(int)
        return _fit_predict(texts.iloc[tr], numeric[tr], y_train, texts.iloc[te], numeric[te])

    return fit_predict


def run_b5(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    full_df: pd.DataFrame | None = None,
    save_preds: bool = True,
) -> dict[str, Any]:
    train_texts = train_df["clean_text"].fillna("").astype(str)
    train_numeric = _numeric_matrix(train_df)
    y_train = train_df["label"].values

    results: dict[str, Any] = {
        "baseline": BASELINE_NAME,
        "description": (
            "TF-IDF + características estructurales y de red concatenadas, "
            "Regresión Logística (fusión temprana sin atención)"
        ),
        "modality": "texto + estructura + red (concatenación)",
        "n_numeric_features": len(NUMERIC_COLS) + len(CATEGORICAL_COLS),
    }

    holdout: dict[str, Any] = {}
    for split_name, split_df in (("val", val_df), ("test", test_df)):
        y_true = split_df["label"].values
        y_pred, y_proba = _fit_predict(
            train_texts,
            train_numeric,
            y_train,
            split_df["clean_text"].fillna("").astype(str),
            _numeric_matrix(split_df),
        )
        holdout[split_name] = run_holdout_eval(y_true, y_pred, y_proba, split_name)
        if save_preds:
            save_predictions(
                split_df["email_id"].values, y_true, y_pred, y_proba, BASELINE_NAME, split_name
            )
    results["holdout"] = holdout

    if full_df is not None:
        full_reset = full_df.reset_index(drop=True)
        indices = np.arange(len(full_reset)).reshape(-1, 1)
        y_full = full_reset["label"].values
        cv_fit_predict = _make_cv_fit_predict(full_reset)
        results["stratified_cv"] = run_stratified_cv(indices, y_full, cv_fit_predict)
        results["group_loso_cv"] = run_group_loso_cv(
            full_reset, indices, y_full, cv_fit_predict
        )

    return results
