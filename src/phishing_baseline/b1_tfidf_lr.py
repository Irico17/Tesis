"""B1: TF-IDF + Logistic Regression baseline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from phishing_pipeline.config import RANDOM_STATE
from phishing_baseline.evaluation import compute_metrics, save_predictions
from phishing_baseline.group_cv import run_group_loso_cv, run_holdout_eval, run_stratified_cv


BASELINE_NAME = "B1_TFIDF_LR"
MAX_FEATURES = 50000


def _make_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1, 2), sublinear_tf=True)),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def _fit_predict_lr(
    pipeline: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return y_pred, y_proba


def run_b1(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    full_df: pd.DataFrame | None = None,
    run_cv: bool = True,
) -> dict[str, Any]:
    """Entrena y evalúa B1 en hold-out + CV dual."""
    pipeline = _make_pipeline()
    X_train = train_df["clean_text"].fillna("").astype(str).values
    y_train = train_df["label"].astype(int).values
    X_val = val_df["clean_text"].fillna("").astype(str).values
    y_val = val_df["label"].astype(int).values
    X_test = test_df["clean_text"].fillna("").astype(str).values
    y_test = test_df["label"].astype(int).values

    pipeline.fit(X_train, y_train)
    y_pred_test, y_proba_test = _fit_predict_lr(pipeline, X_train, y_train, X_test)
    y_pred_val, y_proba_val = _fit_predict_lr(pipeline, X_train, y_train, X_val)

    # Guardado de predicciones fila por fila (R2.2 — insumo de McNemar/t-test).
    # X_test/X_val/y_test/y_val se derivan de test_df/val_df sin reordenar
    # (mismo .values sobre el DataFrame original), así que email_id se puede
    # tomar directamente en el mismo orden.
    if "email_id" in test_df.columns:
        save_predictions(test_df["email_id"].values, y_test, y_pred_test, y_proba_test, BASELINE_NAME, "test")
    if "email_id" in val_df.columns:
        save_predictions(val_df["email_id"].values, y_val, y_pred_val, y_proba_val, BASELINE_NAME, "val")

    results: dict[str, Any] = {
        "baseline": BASELINE_NAME,
        "description": "TF-IDF (1-2 grams, 50k) + Logistic Regression (balanced)",
        "holdout": {
            "val": run_holdout_eval(y_val, y_pred_val, y_proba_val, "val"),
            "test": run_holdout_eval(y_test, y_pred_test, y_proba_test, "test"),
        },
    }

    if run_cv and full_df is not None:
        X_all = full_df["clean_text"].fillna("").astype(str).values
        y_all = full_df["label"].astype(int).values

        def fit_predict_fn(X_tr, y_tr, X_te):
            p = _make_pipeline()
            return _fit_predict_lr(p, X_tr, y_tr, X_te)

        results["stratified_cv"] = run_stratified_cv(X_all, y_all, fit_predict_fn)
        results["group_loso_cv"] = run_group_loso_cv(full_df, X_all, y_all, fit_predict_fn)

    return results
