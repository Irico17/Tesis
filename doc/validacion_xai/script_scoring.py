"""
script_scoring.py — Scoring del panel de validación de usabilidad/comprensibilidad
del módulo XAI (Fase E0/E2, resultado de tesis R3.3).

Contexto
--------
Este script procesa las respuestas del panel recolectadas con el formulario de
`instrumento_evaluacion.md` y calcula:
  1. El índice de claridad y utilidad de las explicaciones, criterio primario de
     R3.3: el % de valoraciones de 4 o 5 en los ítems E1 a E8, que acompañan a
     cada correo (E impares: «entendí por qué el sistema tomó esta decisión»;
     E pares: «esta explicación me ayudaría a decidir si confiar en la decisión
     del sistema»). Se contrasta con la meta del 90% y el mínimo del 80%, como
     fijaba el plan («índices de claridad y confianza» obtenidos con formularios).
  2. El % de comprensión objetiva (preguntas C con respuesta correcta), como
     evidencia complementaria: muestra si lo que el evaluador dice entender
     coincide con lo que efectivamente lee en la explicación.
  3. El score SUS (0-100) y la subescala de Calidad de la Información del PSSUQ
     (InfoQual, Lewis 2002), como evidencia complementaria de usabilidad.

Formato de CSV esperado (una fila por evaluador; lo produce respuestas_a_csv.py)
-------------------------------------------------------------------------------
  evaluador_id                 : str, identificador anónimo (ej. "EVAL_01")
  valoracion_q1 ... q8         : int, 1-5. E1 a E8, dos por correo, en orden.
  comprension_correcta_q1 ...  : int, 0 o 1. Si la respuesta a C1, C2... fue la
                                 correcta según la clave de respuestas. El número
                                 de preguntas se toma de las columnas presentes.
  sus_q1 ... sus_q10           : int, 1-5. SUS adaptado (Sección A); los ítems
                                 impares son positivos y los pares negativos,
                                 como en Brooke (1996).
  pssuq_q7 ... pssuq_q12       : int, 1-5. Subescala InfoQual del PSSUQ, con la
                                 numeración original (en el formulario, B1 a B6).

Uso
---
  py script_scoring.py --input ruta/a/respuestas.csv
  py script_scoring.py --input ruta/a/respuestas.csv --output data/reports/xai_validation_results.json
  py script_scoring.py --input ruta/a/respuestas.csv --no-save

Sin argumentos corre sobre un panel sintético de 5 evaluadores, solo para
verificar que el script funciona antes de tener datos reales.
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

# Solo la subescala de Calidad de la Información (ítems 7 a 12 del PSSUQ).
PSSUQ_INFOQUAL_ITEMS = list(range(7, 13))  # q7-q12

VALORACION_N_ITEMS = 8   # dos por cada uno de los cuatro correos
VALORACION_FAVORABLE = 4  # una valoración cuenta como favorable desde 4 (de acuerdo)

UMBRAL_R3_3 = 80.0  # % mínimo comprometido en la Tabla 2 (IOV de R3.3)
# Meta objetivo sugerida por el Jurado 1 en la revisión del E3 ("se puede iniciar
# con 90 e ir bajando", pág. 12 del PDF).
# Se reporta contra AMBAS referencias: partir de una meta exigente sin renegociar
# el compromiso formal del IOV, y clasificar el resultado real en tres bandas.
META_OBJETIVO = 90.0

DEFAULT_OUTPUT_PATH = Path("data/reports/xai_validation_results.json")

LIKERT_MIN, LIKERT_MAX = 1, 5


def _columnas_comprension(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("comprension_correcta_q")]
    return sorted(cols, key=lambda c: int(c.rsplit("q", 1)[1]))


# ---------------------------------------------------------------------------
# Validación
# ---------------------------------------------------------------------------

def _expected_columns() -> list[str]:
    cols = ["evaluador_id"]
    cols += [f"valoracion_q{i}" for i in range(1, VALORACION_N_ITEMS + 1)]
    cols += [f"sus_q{i}" for i in range(1, SUS_N_ITEMS + 1)]
    cols += [f"pssuq_q{i}" for i in PSSUQ_INFOQUAL_ITEMS]
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
    if not _columnas_comprension(df):
        raise ValueError("No hay columnas comprension_correcta_q* en el CSV.")

    likert = ([f"valoracion_q{i}" for i in range(1, VALORACION_N_ITEMS + 1)]
              + [f"sus_q{i}" for i in range(1, SUS_N_ITEMS + 1)]
              + [f"pssuq_q{i}" for i in PSSUQ_INFOQUAL_ITEMS])
    for col in likert:
        out_of_range = ~df[col].between(LIKERT_MIN, LIKERT_MAX)
        if out_of_range.any():
            bad_rows = df.loc[out_of_range, "evaluador_id"].tolist()
            raise ValueError(
                f"Valores fuera de rango [{LIKERT_MIN}-{LIKERT_MAX}] en '{col}' "
                f"para evaluador(es): {bad_rows}"
            )

    for col in _columnas_comprension(df):
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

def compute_valoracion(row: pd.Series) -> dict[str, float]:
    """% de valoraciones favorables (4 o 5): total, claridad (E impares) y utilidad (E pares)."""
    def pct(items: list[int]) -> float:
        favorables = sum(row[f"valoracion_q{i}"] >= VALORACION_FAVORABLE for i in items)
        return favorables / len(items) * 100.0

    todos = list(range(1, VALORACION_N_ITEMS + 1))
    return {"indice": pct(todos),
            "claridad": pct([i for i in todos if i % 2 == 1]),
            "utilidad": pct([i for i in todos if i % 2 == 0])}


def compute_sus_score(row: pd.Series) -> float:
    """SUS estándar (Brooke, 1996), adaptado a escala 1-5. Rango resultado: 0-100."""
    total = 0.0
    for i in SUS_ODD_ITEMS:
        total += row[f"sus_q{i}"] - 1
    for i in SUS_EVEN_ITEMS:
        total += 5 - row[f"sus_q{i}"]
    return total * 2.5


def compute_pssuq_subscales(row: pd.Series) -> dict[str, float]:
    """Subescala InfoQual del PSSUQ (Lewis, 2002), adaptada. Rango: 1-5, mayor = mejor."""
    vals = [row[f"pssuq_q{i}"] for i in PSSUQ_INFOQUAL_ITEMS]
    return {"infoqual": sum(vals) / len(vals)}


def compute_comprehension_pct(row: pd.Series, columnas: list[str]) -> float:
    """% de respuestas correctas en la Sección C (comprensión objetiva), 0-100."""
    return sum(int(row[c]) for c in columnas) / len(columnas) * 100.0


# ---------------------------------------------------------------------------
# Scoring del panel completo
# ---------------------------------------------------------------------------

def _clasificar_banda(indice_promedio: float) -> dict:
    """
    Clasifica el resultado del panel en tres bandas, contra la meta objetivo
    (90%, sugerida por el Jurado 1) y el mínimo comprometido (80%, IOV de R3.3).

    Reportar ambas referencias permite ser honesto con el resultado real sin
    renegociar el compromiso formal: un panel al 85% cumple el IOV aunque no
    alcance la meta más exigente, y eso debe poder decirse con precisión en
    vez de presentarlo como éxito o fracaso absoluto.
    """
    if indice_promedio >= META_OBJETIVO:
        return {
            "banda": "meta_alcanzada",
            "interpretacion": (
                f"El {indice_promedio:.1f}% de las valoraciones del panel considera las "
                f"explicaciones claras y útiles, por encima de la meta objetivo de "
                f"{META_OBJETIVO:.0f}% y, por tanto, del mínimo de {UMBRAL_R3_3:.0f}% "
                "comprometido en el IOV de R3.3."
            ),
        }
    if indice_promedio >= UMBRAL_R3_3:
        return {
            "banda": "minimo_cumplido",
            "interpretacion": (
                f"El {indice_promedio:.1f}% de las valoraciones del panel considera las "
                f"explicaciones claras y útiles: cumple el mínimo de {UMBRAL_R3_3:.0f}% "
                f"comprometido en el IOV de R3.3, sin llegar a la meta objetivo de "
                f"{META_OBJETIVO:.0f}%. El resultado es válido para acreditar R3.3 y debe "
                "reportarse con esta precisión."
            ),
        }
    return {
        "banda": "no_cumple",
        "interpretacion": (
            f"Solo el {indice_promedio:.1f}% de las valoraciones del panel considera las "
            f"explicaciones claras y útiles, por debajo del mínimo de {UMBRAL_R3_3:.0f}% "
            "exigido por el IOV de R3.3. Debe reportarse como no cumplido y analizarse qué "
            "elementos de la explicación no resultaron comprensibles."
        ),
    }


def score_panel(df: pd.DataFrame) -> dict:
    validate_dataframe(df)
    comprension_cols = _columnas_comprension(df)

    per_evaluator = []
    for _, row in df.iterrows():
        val = compute_valoracion(row)
        per_evaluator.append(
            {
                "evaluador_id": row["evaluador_id"],
                "indice_claridad_utilidad_pct": round(val["indice"], 2),
                "claridad_pct": round(val["claridad"], 2),
                "utilidad_pct": round(val["utilidad"], 2),
                "comprension_objetiva_pct": round(
                    compute_comprehension_pct(row, comprension_cols), 2),
                "sus_score": round(compute_sus_score(row), 2),
                "pssuq_infoqual": round(compute_pssuq_subscales(row)["infoqual"], 2),
            }
        )

    per_df = pd.DataFrame(per_evaluator)
    indice = per_df["indice_claridad_utilidad_pct"].mean()

    panel_summary = {
        "n_evaluadores": int(len(per_df)),
        "indice_claridad_utilidad_pct_promedio": round(indice, 2),
        "claridad_pct_promedio": round(per_df["claridad_pct"].mean(), 2),
        "utilidad_pct_promedio": round(per_df["utilidad_pct"].mean(), 2),
        "umbral_r3_3": UMBRAL_R3_3,
        "umbral_r3_3_alcanzado": bool(indice >= UMBRAL_R3_3),
        "meta_objetivo": META_OBJETIVO,
        "meta_objetivo_alcanzada": bool(indice >= META_OBJETIVO),
        "banda_resultado": _clasificar_banda(indice),
        "comprension_objetiva_pct_promedio": round(per_df["comprension_objetiva_pct"].mean(), 2),
        "n_preguntas_de_comprension": len(comprension_cols),
        "sus_score_promedio": round(per_df["sus_score"].mean(), 2),
        "sus_score_referencia_promedio_literatura": 68.0,
        "sus_por_encima_del_promedio": bool(per_df["sus_score"].mean() > 68.0),
        "pssuq_infoqual_promedio": round(per_df["pssuq_infoqual"].mean(), 2),
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
    print("VALIDACIÓN DE LA COMPRENSIBILIDAD DE LAS EXPLICACIONES — R3.3")
    print("=" * 72)

    print(f"\nEvaluadores procesados: {panel['n_evaluadores']}\n")

    header = (f"{'evaluador_id':<14}{'Índice%':>9}{'Clar.%':>8}{'Útil%':>8}"
              f"{'Compr.%':>9}{'SUS':>7}{'InfoQ':>7}")
    print(header)
    print("-" * len(header))
    for row in per_evaluator:
        print(
            f"{row['evaluador_id']:<14}"
            f"{row['indice_claridad_utilidad_pct']:>9.1f}"
            f"{row['claridad_pct']:>8.1f}"
            f"{row['utilidad_pct']:>8.1f}"
            f"{row['comprension_objetiva_pct']:>9.1f}"
            f"{row['sus_score']:>7.1f}"
            f"{row['pssuq_infoqual']:>7.2f}"
        )

    print("\n" + "-" * 72)
    print("CRITERIO PRIMARIO R3.3 — ÍNDICE DE CLARIDAD Y UTILIDAD")
    print("-" * 72)
    print(f"Valoraciones favorables (4 o 5) : {panel['indice_claridad_utilidad_pct_promedio']:.2f}%"
          f"  (claridad {panel['claridad_pct_promedio']:.1f}%,"
          f" utilidad {panel['utilidad_pct_promedio']:.1f}%)")
    print(f"Meta objetivo (Jurado 1)        : {panel['meta_objetivo']:.1f}%"
          f"  [{'ALCANZADA' if panel['meta_objetivo_alcanzada'] else 'no alcanzada'}]")
    print(f"Mínimo comprometido (IOV R3.3)  : {panel['umbral_r3_3']:.1f}%"
          f"  [{'CUMPLE' if panel['umbral_r3_3_alcanzado'] else 'NO CUMPLE'}]")
    print()
    print(f">>> {panel['banda_resultado']['interpretacion']} <<<")

    print("\n" + "-" * 72)
    print("EVIDENCIA COMPLEMENTARIA")
    print("-" * 72)
    print(f"Comprensión objetiva promedio   : {panel['comprension_objetiva_pct_promedio']:.2f}%"
          f"  ({panel['n_preguntas_de_comprension']} preguntas)")
    print(f"SUS promedio                    : {panel['sus_score_promedio']:.2f} / 100"
          f"  ({'por encima' if panel['sus_por_encima_del_promedio'] else 'por debajo'}"
          f" del punto de referencia habitual de 68)")
    print(f"PSSUQ InfoQual promedio         : {panel['pssuq_infoqual_promedio']:.2f} / 5")
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
    # Patrones variados a propósito, para que el promedio del panel sintético no
    # sea trivialmente 100%.
    sus_patterns = [
        [5, 1, 5, 1, 5, 1, 5, 1, 5, 1],  # evaluador muy positivo -> SUS alto
        [4, 2, 4, 2, 4, 2, 4, 2, 4, 2],  # evaluador moderadamente positivo
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],  # evaluador neutro
        [4, 1, 5, 2, 4, 1, 4, 2, 5, 1],  # evaluador positivo con matices
        [2, 4, 2, 4, 3, 3, 2, 4, 2, 4],  # evaluador crítico -> SUS bajo
    ]
    pssuq_base = [4, 3, 4, 4, 3, 4]
    valoracion_patterns = [
        [5, 5, 4, 5, 5, 4, 5, 5],  # todo favorable
        [4, 4, 4, 3, 5, 4, 4, 4],  # una no favorable
        [3, 3, 4, 2, 3, 3, 4, 3],  # mayoría no favorable
        [5, 4, 4, 4, 4, 4, 5, 3],
        [4, 5, 5, 5, 4, 4, 4, 4],
    ]
    comprension_correcta_patterns = [
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 0, 1, 1],
        [1, 0, 1, 0, 0, 1],
        [1, 1, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 0],
    ]

    for idx in range(n):
        row = {"evaluador_id": f"EVAL_SINTETICO_{idx + 1:02d}"}
        sus_vals = sus_patterns[idx % len(sus_patterns)]
        # pequeña variación por evaluador para no repetir filas idénticas
        pssuq_vals = [max(1, min(5, v - (idx % 2))) for v in pssuq_base]
        val = valoracion_patterns[idx % len(valoracion_patterns)]
        comp = comprension_correcta_patterns[idx % len(comprension_correcta_patterns)]
        row.update({f"valoracion_q{i + 1}": v for i, v in enumerate(val)})
        row.update({f"comprension_correcta_q{i + 1}": c for i, c in enumerate(comp)})
        row.update({f"sus_q{i + 1}": sus_vals[i] for i in range(SUS_N_ITEMS)})
        row.update({f"pssuq_q{q}": v for q, v in zip(PSSUQ_INFOQUAL_ITEMS, pssuq_vals)})
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scoring del panel de validación de las explicaciones (R3.3)."
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
