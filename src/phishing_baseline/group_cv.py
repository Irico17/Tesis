"""Validación cruzada con GroupKFold (Leave-One-Source-Out)."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold

from phishing_baseline.evaluation import save_predictions
from phishing_pipeline.config import RANDOM_STATE, asignar_pliegues
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
    nombre_modelo: str | None = None,
) -> dict[str, Any]:
    """
    Leave-One-Source-Out con GroupKFold por pliegue.

    Simula generalización a fuentes no vistas.

    `nombre_modelo`, si se indica, guarda las predicciones fila a fila de cada
    pliegue. **Hace falta para la prueba de significancia**: McNemar es una prueba
    PAREADA y necesita saber qué acertó cada modelo sobre CADA correo, no solo sus
    métricas agregadas. Sin estas predicciones, la comparación entre el modelo
    propuesto y los de referencia solo puede hacerse comparando medias, que no
    permite decidir si una diferencia es atribuible a varianza aleatoria.

    Se comprobó que faltaban: el modelo profundo sí las guardaba por pliegue y los
    de referencia no, de modo que no había con qué emparejar.
    """
    # Se agrupa por PLIEGUE, no por familia de fuente. En el corpus de correo real
    # cinco de las seis fuentes son de clase única, de modo que retener una sola
    # produce un conjunto de prueba sin negativos ni positivos y las métricas dejan
    # de estar definidas. `asignar_pliegues` devuelve la familia cuando el corpus no
    # necesita composición, así que las corridas sobre el corpus histórico no cambian.
    groups = asignar_pliegues(df).values
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

        if nombre_modelo:
            # El identificador del pliegue va en el nombre del archivo, igual que
            # hace el modelo profundo, para que el emparejamiento posterior sea
            # una simple unión por `email_id` dentro del mismo pliegue.
            etiqueta_pliegue = "_".join(str(h) for h in held_out)
            save_predictions(
                df.iloc[test_idx]["email_id"].values,
                y[test_idx],
                y_pred,
                y_proba,
                f"{nombre_modelo}_LOSO",
                f"loso_test_{etiqueta_pliegue}",
            )
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
