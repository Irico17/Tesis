"""Métricas de evaluación para baselines R2.1 y comparación estadística R2.2."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest, ttest_rel
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from phishing_pipeline.config import BASE_DIR

# Directorio por defecto donde cada baseline/modelo guarda sus predicciones fila
# por fila (email_id, y_true, y_pred, y_proba). Formato genérico: no asume que
# solo existen B1/B2/B3 — phishing_model.evaluate escribirá aquí también.
DEFAULT_PREDICTIONS_DIR = BASE_DIR / "data" / "predictions"


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None) -> dict[str, Any]:
    """Calcula métricas estándar de clasificación binaria."""
    metrics: dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "support": {"negative": int((y_true == 0).sum()), "positive": int((y_true == 1).sum())},
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_proba)), 4)
        except ValueError:
            metrics["roc_auc"] = None
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics["classification_report"] = report
    return metrics


def aggregate_cv_results(fold_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Promedia métricas de múltiples folds."""
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    agg: dict[str, Any] = {"n_folds": len(fold_results), "folds": fold_results}
    for key in keys:
        values = [f[key] for f in fold_results if f.get(key) is not None]
        if values:
            agg[f"{key}_mean"] = round(float(np.mean(values)), 4)
            agg[f"{key}_std"] = round(float(np.std(values)), 4)
    return agg


def save_results(results: dict[str, Any], path: Path) -> Path:
    """Guarda resultados JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    results["generated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def _as_1d_array(values: Any, name: str) -> np.ndarray:
    """Convierte a array 1D y falla con un mensaje claro si no es posible."""
    try:
        arr = np.asarray(values)
    except Exception as exc:  # pragma: no cover - defensivo
        raise ValueError(f"'{name}' no se pudo convertir a array: {exc}") from exc
    if arr.ndim != 1:
        raise ValueError(f"'{name}' debe ser 1D, tiene forma {arr.shape}")
    return arr


def save_predictions(
    email_ids: Any,
    y_true: Any,
    y_pred: Any,
    y_proba: Any,
    model_name: str,
    split_name: str,
    output_dir: Path | None = None,
) -> Path:
    """
    Guarda las predicciones fila por fila de un modelo (baseline o el modelo
    multimodal final) para permitir pruebas de significancia estadística
    (McNemar, t-test pareado) más adelante, incluso retroactivamente.

    Escribe un parquet en `{output_dir}/{model_name}_{split_name}_preds.parquet`
    con columnas `email_id, y_true, y_pred, y_proba` (más `model_name`/`split_name`
    como metadatos, útiles para `compare_models.build_comparison_table`).

    Formato genérico: no asume que el modelo es uno de los baselines B1/B2/B3 —
    cualquier modelo (incl. `phishing_model.evaluate`) puede reutilizar esta
    función siempre que tenga `email_id, y_true, y_pred, y_proba` alineados.
    """
    ids_arr = _as_1d_array(email_ids, "email_ids")
    true_arr = _as_1d_array(y_true, "y_true")
    pred_arr = _as_1d_array(y_pred, "y_pred")
    proba_arr = _as_1d_array(y_proba, "y_proba")

    lengths = {
        "email_ids": len(ids_arr),
        "y_true": len(true_arr),
        "y_pred": len(pred_arr),
        "y_proba": len(proba_arr),
    }
    if len(set(lengths.values())) > 1:
        raise ValueError(f"save_predictions: longitudes inconsistentes entre arrays: {lengths}")
    if lengths["email_ids"] == 0:
        raise ValueError("save_predictions: los arrays están vacíos")

    if not model_name or not isinstance(model_name, str):
        raise ValueError(f"save_predictions: 'model_name' inválido: {model_name!r}")
    if not split_name or not isinstance(split_name, str):
        raise ValueError(f"save_predictions: 'split_name' inválido: {split_name!r}")

    try:
        true_int = true_arr.astype(int)
        pred_int = pred_arr.astype(int)
        proba_float = proba_arr.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"save_predictions: y_true/y_pred/y_proba deben ser numéricos: {exc}") from exc

    if np.isnan(proba_float).any():
        raise ValueError("save_predictions: y_proba contiene valores NaN")

    output_dir = output_dir or DEFAULT_PREDICTIONS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "email_id": ids_arr,
            "y_true": true_int,
            "y_pred": pred_int,
            "y_proba": proba_float,
            "model_name": model_name,
            "split_name": split_name,
        }
    )
    path = output_dir / f"{model_name}_{split_name}_preds.parquet"
    df.to_parquet(path, index=False)
    return path


def mcnemar_test(y_true: Any, y_pred_a: Any, y_pred_b: Any) -> dict[str, Any]:
    """
    Test de McNemar EXACTO (binomial, no la aproximación chi-cuadrado) para
    comparar dos modelos pareados sobre las mismas filas.

    Construye la tabla de contingencia 2x2 de aciertos/errores entre el
    modelo A y el modelo B (mismo `y_true`), cuenta los pares discordantes:
    - n01: A acierta y B falla.
    - n10: A falla y B acierta.
    y aplica `scipy.stats.binomtest(min(n01, n10), n01 + n10, 0.5)`.

    Devuelve `n01`, `n10`, `p_value`, `significant_at_0.05` y, si
    `n01 + n10` es pequeño (<10), una `warning` explícita de poca potencia
    estadística en vez de fallar silenciosamente.
    """
    true_arr = _as_1d_array(y_true, "y_true")
    pred_a_arr = _as_1d_array(y_pred_a, "y_pred_a")
    pred_b_arr = _as_1d_array(y_pred_b, "y_pred_b")

    lengths = {"y_true": len(true_arr), "y_pred_a": len(pred_a_arr), "y_pred_b": len(pred_b_arr)}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"mcnemar_test: longitudes inconsistentes entre arrays: {lengths}")
    if lengths["y_true"] == 0:
        raise ValueError("mcnemar_test: arrays vacíos")

    try:
        true_f = true_arr.astype(float)
        pred_a_f = pred_a_arr.astype(float)
        pred_b_f = pred_b_arr.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"mcnemar_test: entradas no numéricas: {exc}") from exc

    if np.isnan(true_f).any() or np.isnan(pred_a_f).any() or np.isnan(pred_b_f).any():
        raise ValueError("mcnemar_test: se encontraron valores NaN en las entradas")

    correct_a = pred_a_f == true_f
    correct_b = pred_b_f == true_f

    n01 = int(np.sum(correct_a & ~correct_b))  # A acierta, B falla
    n10 = int(np.sum(~correct_a & correct_b))  # A falla, B acierta
    n_discordant = n01 + n10

    result: dict[str, Any] = {"n01": n01, "n10": n10}

    if n_discordant == 0:
        result["p_value"] = 1.0
        result["significant_at_0.05"] = False
        result["warning"] = (
            "n01+n10=0: A y B coinciden en todas las predicciones (no hay pares "
            "discordantes), McNemar no aporta información en este caso."
        )
        return result

    test_result = binomtest(min(n01, n10), n_discordant, 0.5)
    p_value = float(test_result.pvalue)
    # NO se redondea: con miles de pares discordantes el p-value real es del orden
    # de 1e-300, y redondear a seis decimales lo convertía en `0.0`. Un valor p
    # nunca es exactamente cero, y publicarlo así en una tabla de resultados es
    # incorrecto. Se conserva el valor completo y se añade una forma ya
    # formateada para su presentación directa en el documento.
    result["p_value"] = p_value
    result["p_value_display"] = "< 0.001" if p_value < 0.001 else f"{p_value:.4f}"
    result["significant_at_0.05"] = bool(p_value < 0.05)

    if n_discordant < 10:
        result["warning"] = (
            f"n01+n10={n_discordant} < 10: poca potencia estadística, el p-value "
            "de este McNemar puede no ser confiable."
        )

    return result


def paired_ttest_across_folds(metric_values_a: list[float], metric_values_b: list[float]) -> dict[str, Any]:
    """
    T-test pareado (`scipy.stats.ttest_rel`) sobre una métrica (ej. F1) por
    fold entre dos modelos.

    Con pocos folds (p.ej. 3 folds LOSO) este test tiene muy poca potencia
    estadística; el resultado incluye una `warning` explícita si
    `n_folds < 5` en vez de dejar que el texto de la tesis lo insinúe como
    concluyente.
    """
    a_arr = _as_1d_array(metric_values_a, "metric_values_a")
    b_arr = _as_1d_array(metric_values_b, "metric_values_b")

    if len(a_arr) != len(b_arr):
        raise ValueError(
            f"paired_ttest_across_folds: longitudes distintas (a={len(a_arr)}, b={len(b_arr)})"
        )
    if len(a_arr) == 0:
        raise ValueError("paired_ttest_across_folds: listas vacías")

    try:
        a_f = a_arr.astype(float)
        b_f = b_arr.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"paired_ttest_across_folds: valores no numéricos: {exc}") from exc

    if np.isnan(a_f).any() or np.isnan(b_f).any():
        raise ValueError("paired_ttest_across_folds: valores NaN en las métricas por fold")
    if len(a_f) < 2:
        raise ValueError(
            "paired_ttest_across_folds: se requieren al menos 2 folds para un t-test pareado"
        )

    n_folds = len(a_f)
    warning = None
    if n_folds < 5:
        warning = (
            f"Solo {n_folds} folds: la potencia estadística es baja, este resultado "
            "debe interpretarse con cautela."
        )

    if np.allclose(a_f, b_f):
        # Diferencia nula -> varianza cero, ttest_rel devolvería NaN. Se reporta
        # explícitamente en vez de propagar un NaN silencioso.
        return {
            "t_statistic": 0.0,
            "p_value": 1.0,
            "significant_at_0.05": False,
            "n_folds": n_folds,
            "warning": warning,
        }

    t_stat, p_value = ttest_rel(a_f, b_f)
    return {
        "t_statistic": round(float(t_stat), 6),
        "p_value": round(float(p_value), 6),
        "significant_at_0.05": bool(p_value < 0.05),
        "n_folds": n_folds,
        "warning": warning,
    }
