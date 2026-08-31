"""
B4: modelo de referencia unimodal sobre la modalidad de RED (metadatos de
autenticación y características léxicas de URL).

Motivación metodológica. La investigación declara tres modalidades —texto,
estructura y red— y sostiene que su fusión supera a los enfoques unimodales.
Sin embargo, los modelos de referencia implementados cubrían únicamente dos de
ellas: B1 y B2 sobre texto, B3 sobre estructura. La modalidad de red carecía de
referencia unimodal, de modo que no existía forma de responder qué desempeño
alcanza esa modalidad por sí sola ni, en consecuencia, de sostener que la fusión
supera a *cada* una de sus partes.

Sin este modelo, la comparación multimodal se apoya en un argumento incompleto:
puede afirmarse que la fusión supera al texto y a la estructura, pero no que
supere a la modalidad restante. B4 cierra ese hueco y permite formular la
comparación en la forma canónica de la literatura de fusión multimodal —cada
modalidad por separado, después la fusión ingenua, y finalmente la fusión
propuesta—, verificando que cada escalón supere al anterior.

Advertencia de interpretación. Los metadatos de autenticación presentan una
cobertura desigual entre fuentes (documentada en `doc/INFORME_PIPELINE_DATOS.md`),
por lo que un desempeño elevado de B4 en la partición agrupada debe contrastarse
siempre con su resultado en la validación por fuente no observada: la brecha
entre ambos cuantifica en qué medida el modelo se apoya en la disponibilidad del
metadato en lugar de en su valor.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from phishing_baseline.evaluation import compute_metrics, save_predictions
from phishing_baseline.group_cv import run_group_loso_cv, run_holdout_eval, run_stratified_cv
from phishing_model.config import NETWORK_CATEGORICAL_COLS, NETWORK_CATEGORICAL_VOCABS
from phishing_pipeline.config import RANDOM_STATE
from phishing_pipeline.features.vectorizer import NETWORK_FEATURE_COLS

BASELINE_NAME = "B4_Network"

# Se importan de la fuente canónica, por la misma razón que en B3: una copia
# local se desincroniza en silencio del esquema y termina evaluando una
# modalidad distinta de la declarada.
NETWORK_CONTINUOUS_COLS = list(NETWORK_FEATURE_COLS)
NETWORK_CATEGORICAL = list(NETWORK_CATEGORICAL_COLS)


def _encode_categorical(df: pd.DataFrame, col: str) -> np.ndarray:
    """Codifica una columna categórica de autenticación a índices de vocabulario.

    Se emplea el MISMO vocabulario que consume el modelo multimodal, incluida la
    categoría explícita "missing", de modo que B4 y la rama de red de la
    propuesta operen sobre exactamente la misma representación de la ausencia.
    """
    vocab = NETWORK_CATEGORICAL_VOCABS[col]
    missing_idx = vocab["missing"]
    out = np.full(len(df), missing_idx, dtype="int64")
    for i, value in enumerate(df[col].tolist()):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        out[i] = vocab.get(str(value).strip().lower(), missing_idx)
    return out


def _extract_features(df: pd.DataFrame) -> np.ndarray:
    continuous = df[NETWORK_CONTINUOUS_COLS].fillna(0).astype(float).values
    categorical = np.stack([_encode_categorical(df, c) for c in NETWORK_CATEGORICAL], axis=1)
    return np.hstack([continuous, categorical.astype(float)])


def _make_pipeline(n_features: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def _fit_predict(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    pipeline = _make_pipeline(X_train.shape[1])
    pipeline.fit(X_train, y_train)
    return pipeline.predict(X_test), pipeline.predict_proba(X_test)[:, 1]


def run_b4(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    full_df: pd.DataFrame | None = None,
    save_preds: bool = True,
) -> dict[str, Any]:
    X_train, y_train = _extract_features(train_df), train_df["label"].values
    feature_names = NETWORK_CONTINUOUS_COLS + NETWORK_CATEGORICAL

    results: dict[str, Any] = {
        "baseline": BASELINE_NAME,
        "description": "Random Forest sobre metadatos de red y características léxicas de URL",
        "modality": "red",
        "features_used": feature_names,
    }

    holdout: dict[str, Any] = {}
    for split_name, split_df in (("val", val_df), ("test", test_df)):
        y_true = split_df["label"].values
        y_pred, y_proba = _fit_predict(X_train, y_train, _extract_features(split_df))
        holdout[split_name] = run_holdout_eval(y_true, y_pred, y_proba, split_name)
        if save_preds:
            save_predictions(
                split_df["email_id"].values, y_true, y_pred, y_proba, BASELINE_NAME, split_name
            )
    results["holdout"] = holdout

    if full_df is not None:
        X_full, y_full = _extract_features(full_df), full_df["label"].values
        results["stratified_cv"] = run_stratified_cv(X_full, y_full, _fit_predict)
        results["group_loso_cv"] = run_group_loso_cv(
            full_df, X_full, y_full, _fit_predict, nombre_modelo=BASELINE_NAME
        )

    return results
