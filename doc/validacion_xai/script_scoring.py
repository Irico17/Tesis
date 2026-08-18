"""
script_scoring.py — Scoring del panel de validación de usabilidad/comprensibilidad
del módulo XAI (Fase E0/E2, resultado de tesis R3.3).

Contexto
--------
Este script procesa las respuestas del panel de 10 evaluadores recolectadas con
`instrumento_evaluacion.md` (SUS adaptado + PSSUQ adaptado + sección de
comprensión narrativa) y calcula:
  1. El score SUS (0-100) por evaluador y el promedio del panel.
  2. Las 3 subescalas PSSUQ (SysUse, InfoQual, IntQual) + el score "Overall"
     por evaluador y el promedio del panel.
  3. El % de comprensión narrativa por evaluador y el promedio del panel,
     comparado explícitamente contra el umbral del 80% exigido por R3.3.

Formato de CSV esperado (una fila por evaluador)
-------------------------------------------------
Columnas requeridas:

  evaluador_id            : str, identificador anónimo (ej. "EVAL_01")

  sus_q1 ... sus_q10       : int, 1-5. Respuesta a cada ítem del SUS adaptado
                              (Sección A de instrumento_evaluacion.md), en el
                              mismo orden/numeración que el instrumento
                              (1=totalmente en desacuerdo, 5=totalmente de
                              acuerdo). Los ítems impares (1,3,5,7,9) son
                              afirmaciones positivas; los pares (2,4,6,8,10)
                              son negativas — igual que el SUS original de
                              Brooke (1996); el scoring ya tiene esto en cuenta.

  pssuq_q1 ... pssuq_q16   : int, 1-5. Respuesta a cada ítem del PSSUQ
                              adaptado (Sección B de instrumento_evaluacion.md).
                              Subescalas (versión de 16 ítems, Lewis 2002,
                              adaptada a escala 1-5 con mismo sentido que SUS,
                              mayor = mejor):
                                - SysUse   (Utilidad del Sistema)   : q1-q6
                                - InfoQual (Calidad de la Información): q7-q12
                                - IntQual  (Calidad de la Interfaz) : q13-q15
                                - Overall  (satisfacción general)  : q1-q16
                                  (q16 es además el ítem individual de
                                  satisfacción general del instrumento)

  comprension_q1 ... comprension_q5      : str/opcional. La respuesta cruda
                              que dio el evaluador en la Sección C (letra de
                              opción elegida, o "V"/"F"). Se conserva para
                              trazabilidad/auditoría, pero NO se usa para el
                              cálculo de comprensión — ver siguiente columna.

  comprension_correcta_q1 ... comprension_correcta_q5 : int, 0 o 1. Ya
                              codifica si esa respuesta fue correcta (1) o
                              incorrecta (0), comparándola contra la clave de
                              respuestas de instrumento_evaluacion.md en el
                              momento de transcribir el cuestionario. Esta es
                              la columna que efectivamente usa el script para
                              calcular el % de comprensión.

Uso
---
  py script_scoring.py --input ruta/a/respuestas.csv
  py script_scoring.py --input ruta/a/respuestas.csv --output data/reports/xai_validation_results.json
  py script_scoring.py --input ruta/a/respuestas.csv --no-save

Si se ejecuta sin argumentos (`py script_scoring.py`), corre sobre un panel
sintético de 5 evaluadores de ejemplo, solo para verificar que el script
funciona correctamente antes de tener datos reales del panel.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configuración de columnas / escalas
# ---------------------------------------------------------------------------

SUS_N_ITEMS = 10
SUS_ODD_ITEMS = [1, 3, 5, 7, 9]     # afirmaciones positivas
SUS_EVEN_ITEMS = [2, 4, 6, 8, 10]   # afirmaciones negativas

PSSUQ_N_ITEMS = 16
PSSUQ_SYSUSE_ITEMS = list(range(1, 7))     # q1-q6
PSSUQ_INFOQUAL_ITEMS = list(range(7, 13))  # q7-q12
PSSUQ_INTQUAL_ITEMS = list(range(13, 16))  # q13-q15
PSSUQ_OVERALL_ITEMS = list(range(1, 17))   # q1-q16

COMPRENSION_N_ITEMS = 5
COMPRENSION_UMBRAL_R3_3 = 80.0  # % exigido por R3.3

DEFAULT_OUTPUT_PATH = Path("data/reports/xai_validation_results.json")

LIKERT_MIN, LIKERT_MAX = 1, 5


# ---------------------------------------------------------------------------
# Validación
# ---------------------------------------------------------------------------

def _expected_columns() -> list[str]:
    cols = ["evaluador_id"]
    cols += [f"sus_q{i}" for i in range(1, SUS_N_ITEMS + 1)]
    cols += [f"pssuq_q{i}" for i in range(1, PSSUQ_N_ITEMS + 1)]
    cols += [f"comprension_correcta_q{i}" for i in range(1, COMPRENSION_N_ITEMS + 1)]
    return cols


def validate_dataframe(df: pd.DataFrame) -> None:
    missing = [c for c in _expected_columns() if c not in df.columns]
    if missing:
        raise ValueError(
            "Faltan columnas requeridas en el CSV de respuestas: "
            f"{missing}. Revisa el formato documentado al inicio de este script."
        )
    if df.empty:
        raise ValueError("El CSV de respuestas está vacío.")

    sus_cols = [f"sus_q{i}" for i in range(1, SUS_N_ITEMS + 1)]
    pssuq_cols = [f"pssuq_q{i}" for i in range(1, PSSUQ_N_ITEMS + 1)]
    for col in sus_cols + pssuq_cols:
        out_of_range = ~df[col].between(LIKERT_MIN, LIKERT_MAX)
        if out_of_range.any():
            bad_rows = df.loc[out_of_range, "evaluador_id"].tolist()
            raise ValueError(
                f"Valores fuera de rango [{LIKERT_MIN}-{LIKERT_MAX}] en '{col}' "
                f"para evaluador(es): {bad_rows}"
            )

    correcta_cols = [f"comprension_correcta_q{i}" for i in range(1, COMPRENSION_N_ITEMS + 1)]
    for col in correcta_cols:
        invalid = ~df[col].isin([0, 1])
        if invalid.any():
            bad_rows = df.loc[invalid, "evaluador_id"].tolist()
            raise ValueError(
                f"Valores inválidos en '{col}' (deben ser 0 o 1) para "
                f"evaluador(es): {bad_rows}"
            )


# ---------------------------------------------------------------------------
# Scoring por evaluador
# ---------------------------------------------------------------------------

def compute_sus_score(row: pd.Series) -> float:
    """SUS estándar (Brooke, 1996), adaptado a escala 1-5. Rango resultado: 0-100."""
    total = 0.0
    for i in SUS_ODD_ITEMS:
        total += row[f"sus_q{i}"] - 1
    for i in SUS_EVEN_ITEMS:
        total += 5 - row[f"sus_q{i}"]
    return total * 2.5


def compute_pssuq_subscales(row: pd.Series) -> dict[str, float]:
    """Subescalas PSSUQ (Lewis, 2002; versión 16 ítems adaptada). Rango: 1-5, mayor = mejor."""
    def _mean(items: list[int]) -> float:
        vals = [row[f"pssuq_q{i}"] for i in items]
        return sum(vals) / len(vals)

    return {
        "sysuse": _mean(PSSUQ_SYSUSE_ITEMS),
        "infoqual": _mean(PSSUQ_INFOQUAL_ITEMS),
        "intqual": _mean(PSSUQ_INTQUAL_ITEMS),
        "overall": _mean(PSSUQ_OVERALL_ITEMS),
    }


def compute_comprehension_pct(row: pd.Series) -> float:
    """% de respuestas correctas en la Sección C (comprensión narrativa), 0-100."""
    correctas = sum(
        int(row[f"comprension_correcta_q{i}"]) for i in range(1, COMPRENSION_N_ITEMS + 1)
    )
    return (correctas / COMPRENSION_N_ITEMS) * 100.0


# ---------------------------------------------------------------------------
# Scoring del panel completo
# ---------------------------------------------------------------------------

def score_panel(df: pd.DataFrame) -> dict:
    validate_dataframe(df)

    per_evaluator = []
    for _, row in df.iterrows():
        sus = compute_sus_score(row)
        pssuq = compute_pssuq_subscales(row)
        comprension = compute_comprehension_pct(row)
        per_evaluator.append(
            {
                "evaluador_id": row["evaluador_id"],
                "sus_score": round(sus, 2),
                "pssuq_sysuse": round(pssuq["sysuse"], 2),
                "pssuq_infoqual": round(pssuq["infoqual"], 2),
                "pssuq_intqual": round(pssuq["intqual"], 2),
                "pssuq_overall": round(pssuq["overall"], 2),
                "comprension_pct": round(comprension, 2),
            }
        )

    per_df = pd.DataFrame(per_evaluator)

    panel_summary = {
        "n_evaluadores": int(len(per_df)),
        "sus_score_promedio": round(per_df["sus_score"].mean(), 2),
        "sus_score_referencia_promedio_literatura": 68.0,
        "sus_por_encima_del_promedio": bool(per_df["sus_score"].mean() > 68.0),
        "pssuq_sysuse_promedio": round(per_df["pssuq_sysuse"].mean(), 2),
        "pssuq_infoqual_promedio": round(per_df["pssuq_infoqual"].mean(), 2),
        "pssuq_intqual_promedio": round(per_df["pssuq_intqual"].mean(), 2),
        "pssuq_overall_promedio": round(per_df["pssuq_overall"].mean(), 2),
        "comprension_pct_promedio": round(per_df["comprension_pct"].mean(), 2),
        "umbral_r3_3": COMPRENSION_UMBRAL_R3_3,
        "umbral_r3_3_alcanzado": bool(
            per_df["comprension_pct"].mean() >= COMPRENSION_UMBRAL_R3_3
        ),
    }

    return {
        "por_evaluador": per_evaluator,
        "resumen_panel": panel_summary,
    }


# ---------------------------------------------------------------------------
# Presentación de resultados
# ---------------------------------------------------------------------------

def print_summary(results: dict) -> None:
    per_evaluator = results["por_evaluador"]
    panel = results["resumen_panel"]

    print("=" * 72)
    print("VALIDACIÓN DE USABILIDAD/COMPRENSIBILIDAD DEL MÓDULO XAI — R3.3")
    print("=" * 72)

    print(f"\nEvaluadores procesados: {panel['n_evaluadores']}\n")

    header = (
        f"{'evaluador_id':<14}{'SUS':>8}{'SysUse':>10}{'InfoQual':>10}"
        f"{'IntQual':>10}{'Overall':>10}{'Compr.%':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in per_evaluator:
        print(
            f"{row['evaluador_id']:<14}"
            f"{row['sus_score']:>8.1f}"
            f"{row['pssuq_sysuse']:>10.2f}"
            f"{row['pssuq_infoqual']:>10.2f}"
            f"{row['pssuq_intqual']:>10.2f}"
            f"{row['pssuq_overall']:>10.2f}"
            f"{row['comprension_pct']:>10.1f}"
        )

    print("\n" + "-" * 72)
    print("PROMEDIOS DEL PANEL")
    print("-" * 72)
    print(f"SUS promedio            : {panel['sus_score_promedio']:.2f} / 100"
          f"  ({'por encima' if panel['sus_por_encima_del_promedio'] else 'por debajo'}"
          f" del punto de referencia habitual de 68)")
    print(f"PSSUQ SysUse promedio    : {panel['pssuq_sysuse_promedio']:.2f} / 5")
    print(f"PSSUQ InfoQual promedio  : {panel['pssuq_infoqual_promedio']:.2f} / 5")
    print(f"PSSUQ IntQual promedio   : {panel['pssuq_intqual_promedio']:.2f} / 5")
    print(f"PSSUQ Overall promedio   : {panel['pssuq_overall_promedio']:.2f} / 5")

    print("\n" + "-" * 72)
    print("CRITERIO PRIMARIO R3.3 — COMPRENSIÓN NARRATIVA")
    print("-" * 72)
    print(f"Comprensión promedio del panel : {panel['comprension_pct_promedio']:.2f}%")
    print(f"Umbral exigido por R3.3        : {panel['umbral_r3_3']:.1f}%")
    if panel["umbral_r3_3_alcanzado"]:
        print(">>> UMBRAL ALCANZADO: el panel satisface el criterio primario de "
              "comprensibilidad de R3.3. <<<")
    else:
        print(">>> UMBRAL NO ALCANZADO: el panel NO satisface el criterio primario "
              "de comprensibilidad de R3.3. Reportar como hallazgo y evaluar "
              "iteración sobre el módulo de narrativa (R3.2) antes de una nueva "
              "ronda de validación. <<<")
    print("=" * 72)


def save_results_json(results: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResultados agregados guardados en: {output_path}")


# ---------------------------------------------------------------------------
# Datos sintéticos (solo para probar el script antes de tener datos reales)
# ---------------------------------------------------------------------------

def build_synthetic_panel(n: int = 5) -> pd.DataFrame:
    """Genera un panel sintético de `n` evaluadores ficticios para poder correr
    y verificar este script antes de que existan datos reales del panel."""
    rows = []
    # Patrones variados a propósito: uno "óptimo", uno "mediocre", uno con
    # comprensión por debajo del umbral, y el resto intermedios, para que el
    # promedio del panel sintético no sea trivialmente 100%.
    sus_patterns = [
        [5, 1, 5, 1, 5, 1, 5, 1, 5, 1],  # evaluador muy positivo -> SUS alto
        [4, 2, 4, 2, 4, 2, 4, 2, 4, 2],  # evaluador moderadamente positivo
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],  # evaluador neutro
        [4, 1, 5, 2, 4, 1, 4, 2, 5, 1],  # evaluador positivo con matices
        [2, 4, 2, 4, 3, 3, 2, 4, 2, 4],  # evaluador crítico -> SUS bajo
    ]
    pssuq_base = [5, 4, 4, 3, 4, 4, 4, 3, 4, 4, 3, 4, 4, 4, 3, 4]
    comprension_correcta_patterns = [
        [1, 1, 1, 1, 1],  # 100%
        [1, 1, 1, 1, 0],  # 80%
        [1, 1, 0, 1, 0],  # 60% (por debajo del umbral individual)
        [1, 1, 1, 0, 1],  # 80%
        [1, 1, 1, 1, 1],  # 100%
    ]

    for idx in range(n):
        evaluador_id = f"EVAL_SINTETICO_{idx + 1:02d}"
        sus_vals = sus_patterns[idx % len(sus_patterns)]
        # pequeña variación por evaluador para no repetir filas idénticas
        pssuq_vals = [max(1, min(5, v - (idx % 2))) for v in pssuq_base]
        comprension_correcta = comprension_correcta_patterns[idx % len(comprension_correcta_patterns)]

        row = {"evaluador_id": evaluador_id}
        row.update({f"sus_q{i + 1}": sus_vals[i] for i in range(SUS_N_ITEMS)})
        row.update({f"pssuq_q{i + 1}": pssuq_vals[i] for i in range(PSSUQ_N_ITEMS)})
        row.update(
            {f"comprension_q{i + 1}": ["A", "B", "C", "V", "F"][i] for i in range(COMPRENSION_N_ITEMS)}
        )
        row.update(
            {
                f"comprension_correcta_q{i + 1}": comprension_correcta[i]
                for i in range(COMPRENSION_N_ITEMS)
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scoring del panel de validación SUS/PSSUQ/comprensión narrativa (R3.3)."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Ruta al CSV de respuestas del panel. Si se omite, corre sobre datos sintéticos de prueba.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"Ruta del JSON de salida con resultados agregados (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="No guardar el JSON de resultados, solo imprimir el resumen en consola.",
    )
    args = parser.parse_args()

    if args.input is None:
        print(
            "No se especificó --input: usando panel sintético de 5 evaluadores "
            "de prueba (solo para verificar que el script funciona).\n"
        )
        df = build_synthetic_panel(n=5)
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo de entrada: {input_path}")
        df = pd.read_csv(input_path)

    results = score_panel(df)
    print_summary(results)

    if not args.no_save:
        save_results_json(results, Path(args.output))


if __name__ == "__main__":
    main()
