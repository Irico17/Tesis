"""Tabla comparativa R2.2: significancia estadística (McNemar) entre modelos.

Lee las predicciones fila por fila guardadas por `evaluation.save_predictions`
(baselines B1/B2/B3 hoy; `phishing_model.evaluate` más adelante — el formato
es genérico, no asume qué modelos existen) y calcula McNemar exacto para
cada par de modelos que comparte split y filas (mismo `email_id`).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from phishing_pipeline.config import BASE_DIR
from phishing_baseline.evaluation import DEFAULT_PREDICTIONS_DIR, mcnemar_test

COMPARISON_OUTPUT_PATH = BASE_DIR / "data" / "reports" / "csv" / "model_comparison_mcnemar.csv"

_COMPARISON_COLUMNS = ["modelo_a", "modelo_b", "split", "n01", "n10", "p_value", "significativo"]


def _load_predictions(predictions_dir: Path) -> list[dict]:
    """Carga cada `*_preds.parquet` con su model_name/split_name resueltos."""
    loaded = []
    for path in sorted(predictions_dir.glob("*_preds.parquet")):
        df = pd.read_parquet(path)
        required_cols = {"email_id", "y_true", "y_pred", "y_proba"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{path.name}: faltan columnas requeridas {missing}")

        if "model_name" in df.columns and "split_name" in df.columns and len(df) > 0:
            model_name = str(df["model_name"].iloc[0])
            split_name = str(df["split_name"].iloc[0])
        else:
            # Fallback: parsear del nombre de archivo "{model_name}_{split_name}_preds.parquet"
            stem = path.stem
            stem = stem[: -len("_preds")] if stem.endswith("_preds") else stem
            model_name, _, split_name = stem.rpartition("_")
            if not model_name:
                raise ValueError(f"{path.name}: no se pudo inferir model_name/split_name del nombre de archivo")

        loaded.append({"model_name": model_name, "split_name": split_name, "df": df, "file": path.name})
    return loaded


def build_comparison_table(predictions_dir: Path | None = None) -> pd.DataFrame:
    """
    Construye la tabla comparativa de McNemar entre todos los pares de
    modelos que comparten split y el mismo conjunto de `email_id`.

    Devuelve un DataFrame con columnas:
    `modelo_a, modelo_b, split, n01, n10, p_value, significativo`.

    Si `data/predictions/` no existe o está vacío, devuelve una tabla vacía
    con las columnas esperadas (no falla).
    """
    predictions_dir = predictions_dir or DEFAULT_PREDICTIONS_DIR

    if not predictions_dir.exists():
        return pd.DataFrame(columns=_COMPARISON_COLUMNS)

    loaded = _load_predictions(predictions_dir)
    if not loaded:
        return pd.DataFrame(columns=_COMPARISON_COLUMNS)

    rows = []
    for i in range(len(loaded)):
        for j in range(i + 1, len(loaded)):
            a, b = loaded[i], loaded[j]
            if a["split_name"] != b["split_name"]:
                continue
            if a["model_name"] == b["model_name"]:
                continue

            df_a, df_b = a["df"], b["df"]
            if set(df_a["email_id"]) != set(df_b["email_id"]):
                # No comparables directamente: distinto conjunto de filas
                # (ej. B2 corrido sobre un subsample vs B1/B3 sobre el split
                # completo). Se omite en vez de comparar filas distintas.
                continue

            merged = df_a[["email_id", "y_true", "y_pred"]].merge(
                df_b[["email_id", "y_true", "y_pred"]], on="email_id", suffixes=("_a", "_b")
            )
            if merged.empty:
                continue
            if (merged["y_true_a"] != merged["y_true_b"]).any():
                raise ValueError(
                    f"y_true inconsistente entre '{a['model_name']}' y '{b['model_name']}' "
                    f"(split={a['split_name']}) para las mismas email_id — revisar alineación."
                )

            mcnemar_result = mcnemar_test(
                merged["y_true_a"].values, merged["y_pred_a"].values, merged["y_pred_b"].values
            )
            rows.append(
                {
                    "modelo_a": a["model_name"],
                    "modelo_b": b["model_name"],
                    "split": a["split_name"],
                    "n01": mcnemar_result["n01"],
                    "n10": mcnemar_result["n10"],
                    "p_value": mcnemar_result["p_value"],
                    "significativo": mcnemar_result["significant_at_0.05"],
                }
            )

    return pd.DataFrame(rows, columns=_COMPARISON_COLUMNS)


if __name__ == "__main__":
    table = build_comparison_table()
    if table.empty:
        print("Sin predicciones comparables en", DEFAULT_PREDICTIONS_DIR)
    else:
        print(table.to_string(index=False))

    COMPARISON_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(COMPARISON_OUTPUT_PATH, index=False)
    print(f"\nGuardado en: {COMPARISON_OUTPUT_PATH}")
