"""B3: Random Forest on structural/DOM features."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from phishing_pipeline.config import RANDOM_STATE
from phishing_pipeline.features.vectorizer import STRUCTURAL_FEATURE_COLS
from phishing_baseline.evaluation import save_predictions
from phishing_baseline.group_cv import run_group_loso_cv, run_holdout_eval, run_stratified_cv

BASELINE_NAME = "B3_RF_Structural"

# Se importa la lista CANÓNICA de la rama estructural en lugar de mantener una
# copia local. La copia anterior se desincronizó silenciosamente: conservaba
# `has_form`, `has_iframe` y `has_javascript` -- constantes en las 110,152 filas
# del corpus, y por tanto tres de sus nueve entradas eran ruido puro -- y no
# incorporaba los estadísticos de complejidad del DOM añadidos posteriormente.
# Además incluía `has_ip_link` y `num_urls_metadata`, que pertenecen a la rama de
# RED: su presencia hacía que B3 dejara de ser un modelo de referencia puramente
# estructural, debilitando la comparación por modalidad del Capítulo 4.
# Importando la lista canónica, B3 evalúa exactamente la misma modalidad que la
# rama estructural del modelo propuesto y no puede volver a divergir.
STRUCTURAL_COLS = list(STRUCTURAL_FEATURE_COLS)


def _extract_features(df: pd.DataFrame) -> np.ndarray:
    cols = [c for c in STRUCTURAL_COLS if c in df.columns]
    return df[cols].fillna(0).astype(float).values


def _make_pipeline(n_features: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=20,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def _fit_predict_rf(
    pipeline: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return y_pred, y_proba


def run_b3(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    full_df: pd.DataFrame | None = None,
    run_cv: bool = True,
) -> dict[str, Any]:
    """Entrena RF sobre features estructurales/DOM."""
    pipeline = _make_pipeline(len(STRUCTURAL_COLS))

    X_train = _extract_features(train_df)
    y_train = train_df["label"].astype(int).values
    X_val = _extract_features(val_df)
    y_val = val_df["label"].astype(int).values
    X_test = _extract_features(test_df)
    y_test = test_df["label"].astype(int).values

    y_pred_val, y_proba_val = _fit_predict_rf(pipeline, X_train, y_train, X_val)
    y_pred_test, y_proba_test = _fit_predict_rf(pipeline, X_train, y_train, X_test)

    # Guardado de predicciones fila por fila (R2.2 — insumo de McNemar/t-test).
    # _extract_features(df) usa df[cols].values sin reordenar filas, así que
    # email_id se puede tomar directamente en el mismo orden que X_test/X_val.
    if "email_id" in test_df.columns:
        save_predictions(test_df["email_id"].values, y_test, y_pred_test, y_proba_test, BASELINE_NAME, "test")
    if "email_id" in val_df.columns:
        save_predictions(val_df["email_id"].values, y_val, y_pred_val, y_proba_val, BASELINE_NAME, "val")

    feature_cols = [c for c in STRUCTURAL_COLS if c in train_df.columns]
    results: dict[str, Any] = {
        "baseline": BASELINE_NAME,
        "description": "Random Forest on structural/DOM features",
        "features_used": feature_cols,
        "holdout": {
            "val": run_holdout_eval(y_val, y_pred_val, y_proba_val, "val"),
            "test": run_holdout_eval(y_test, y_pred_test, y_proba_test, "test"),
        },
    }

    if run_cv and full_df is not None:
        X_all = _extract_features(full_df)
        y_all = full_df["label"].astype(int).values

        def fit_predict_fn(X_tr, y_tr, X_te):
            p = _make_pipeline(X_tr.shape[1])
            return _fit_predict_rf(p, X_tr, y_tr, X_te)

        results["stratified_cv"] = run_stratified_cv(X_all, y_all, fit_predict_fn)
        results["group_loso_cv"] = run_group_loso_cv(
            full_df, X_all, y_all, fit_predict_fn, nombre_modelo=BASELINE_NAME
        )

    return results
