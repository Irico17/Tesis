"""Validación cruzada con GroupKFold (Leave-One-Source-Out)."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold

from phishing_pipeline.config import RANDOM_STATE, get_source_family
from phishing_baseline.evaluation import aggregate_cv_results, compute_metrics


def run_stratified_cv(
    X: np.ndarray,
    y: np.ndarray,
    fit_predict_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray | None]],
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Validación estratificada K-Fold."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_results = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        y_pred, y_proba = fit_predict_fn(X[train_idx], y[train_idx], X[test_idx])
        metrics = compute_metrics(y[test_idx], y_pred, y_proba)
        metrics["fold"] = fold_idx
        fold_results.append(metrics)
    return aggregate_cv_results(fold_results)


def run_group_loso_cv(
    df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    fit_predict_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray | None]],
) -> dict[str, Any]:
    """
    Leave-One-Source-Out con GroupKFold por source_family.

    Simula generalización a fuentes no vistas.
    """
    groups = df["source_dataset"].apply(get_source_family).values
    unique_groups = np.unique(groups)
    n_splits = len(unique_groups)
    gkf = GroupKFold(n_splits=n_splits)

    fold_results = []
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        held_out = np.unique(groups[test_idx]).tolist()
        y_pred, y_proba = fit_predict_fn(X[train_idx], y[train_idx], X[test_idx])
        metrics = compute_metrics(y[test_idx], y_pred, y_proba)
        metrics["fold"] = fold_idx
        metrics["held_out_source"] = held_out
        fold_results.append(metrics)
    return aggregate_cv_results(fold_results)


def run_holdout_eval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    split_name: str = "test",
) -> dict[str, Any]:
    """Evaluación en hold-out split."""
    metrics = compute_metrics(y_true, y_pred, y_proba)
    metrics["split"] = split_name
    return metrics
