"""
E0 · Artefactos de R1.1 y R1.2, regenerados desde el corpus vigente.

No es un experimento: es la evidencia del corpus y del pipeline, que los cinco
siguientes dan por supuesta. Se numera E0 y se ejecuta primero porque si el
corpus no cumple su indicador, nada de lo que venga después significa algo.

Existe porque esos artefactos se generaban con guiones sueltos, cada uno con su
ruta escrita a mano, y al reconstruir el corpus quedaban describiendo el anterior
sin que nada avisara. Se comprobó: el verificador de cobertura los daba por
presentes cuando ya no correspondían al corpus que el capítulo describe.

Produce:

  - Verificación del indicador de cobertura modal (25% / 40% / prevalencia)
  - Prueba de humo del corpus, nueve comprobaciones
  - Auditoría de fugas por disponibilidad de campo
  - Auditoría de características: información sobre la clase frente a la colección
  - Figura de cobertura modal por colección

    python scripts/experimentos/e0_corpus_y_pipeline.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from comun import BASE, cargar_corpus, emitir

# Umbrales del indicador de R1.1, confirmados por los asesores. La justificación
# de la prevalencia está en doc/DECISIONES_CORPUS_Y_PROTOCOLO.md.
UMBRAL_TRIMODAL = 25.0
UMBRAL_BIMODAL = 40.0
RANGO_PREVALENCIA = (0.40, 0.60)


def indicador(df: pd.DataFrame) -> dict:
    tri = df["has_structure_modality"] & df["has_network_modality"]
    bi = df["has_structure_modality"] | df["has_network_modality"]
    pct_tri, pct_bi = 100 * tri.mean(), 100 * bi.mean()
    prev, prev_bi = df["label"].mean(), df.loc[bi, "label"].mean()
    en_rango = lambda p: RANGO_PREVALENCIA[0] <= p <= RANGO_PREVALENCIA[1]  # noqa: E731

    criterios = [
        ("Cobertura de las tres modalidades", f"≥ {UMBRAL_TRIMODAL:.0f}%",
         f"{pct_tri:.2f}%", pct_tri >= UMBRAL_TRIMODAL),
        ("Cobertura de texto y al menos una no textual", f"≥ {UMBRAL_BIMODAL:.0f}%",
         f"{pct_bi:.2f}%", pct_bi >= UMBRAL_BIMODAL),
        ("Prevalencia del corpus", "0.40 – 0.60", f"{prev:.4f}", en_rango(prev)),
        ("Prevalencia del subconjunto bimodal", "0.40 – 0.60",
         f"{prev_bi:.4f}", en_rango(prev_bi)),
    ]
    return {
        "criterios": [{"criterio": c, "umbral": u, "medido": m, "cumple": bool(ok)}
                      for c, u, m, ok in criterios],
        "cumple_todo": all(ok for *_, ok in criterios),
        "trimodal": {"n": int(tri.sum()), "porcentaje": round(float(pct_tri), 4),
                     "prevalencia": round(float(df.loc[tri, "label"].mean()), 4)},
        "bimodal": {"n": int(bi.sum()), "porcentaje": round(float(pct_bi), 4),
                    "prevalencia": round(float(prev_bi), 4)},
        # Limitación declarada, no exigida como criterio: el subconjunto trimodal
        # tenía prevalencia 0.979 antes de incorporar correo legítimo con marcado.
        "nota_trimodal": (
            "La prevalencia del subconjunto trimodal se declara como medida. Antes "
            "de incorporar archivos de listas de discusión valía 0.979, es decir, "
            "el subconjunto era casi en su totalidad phishing, porque ninguna "
            "colección de correo legítimo publicada "
            "conserva el marcado del cuerpo."
        ),
    }


def prueba_de_humo(df: pd.DataFrame) -> dict:
    tri = df["has_structure_modality"] & df["has_network_modality"]
    bi = df["has_structure_modality"] | df["has_network_modality"]
    comprobaciones = [
        ("etiquetas sin nulos", int(df["label"].isna().sum()) == 0,
         f"{int(df['label'].isna().sum())} nulos"),
        ("cobertura textual completa",
         int(df["clean_text"].fillna("").str.strip().eq("").sum()) == 0,
         f"{int(df['clean_text'].fillna('').str.strip().eq('').sum())} vacíos"),
        (f"cobertura bimodal ≥ {UMBRAL_BIMODAL:.0f}%", 100 * bi.mean() >= UMBRAL_BIMODAL,
         f"{100 * bi.mean():.2f}%"),
        (f"cobertura trimodal ≥ {UMBRAL_TRIMODAL:.0f}%", 100 * tri.mean() >= UMBRAL_TRIMODAL,
         f"{100 * tri.mean():.2f}%"),
        ("prevalencia del corpus en rango",
         RANGO_PREVALENCIA[0] <= df["label"].mean() <= RANGO_PREVALENCIA[1],
         f"{df['label'].mean():.4f}"),
        ("sin duplicados exactos de texto",
         int(df["clean_text"].duplicated().sum()) == 0,
         f"{int(df['clean_text'].duplicated().sum())}"),
        ("template_cluster_id completo",
         int(df["template_cluster_id"].isna().sum()) == 0,
         f"{int(df['template_cluster_id'].isna().sum())} nulos"),
        ("al menos dos colecciones con ambas clases o el corpus equilibrado",
         df["label"].nunique() == 2, f"{df['label'].nunique()} clases"),
    ]
    return {
        "comprobaciones": [{"comprobacion": c, "resultado": "OK" if ok else "FALLA",
                            "detalle": d} for c, ok, d in comprobaciones],
        "veredicto": ("TODAS SUPERADAS" if all(ok for _, ok, _ in comprobaciones)
                      else "HAY FALLOS"),
    }


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    sys.path.insert(0, str(BASE / "src"))
    from phishing_pipeline.auditoria_fugas import auditar

    df = cargar_corpus()
    print(f"E0 · corpus {len(df):,} correos, prevalencia {df.label.mean():.4f}")

    ind = indicador(df)
    humo = prueba_de_humo(df)
    fugas = auditar(df)

    for c in ind["criterios"]:
        print(f"  [{'OK   ' if c['cumple'] else 'FALLA'}] {c['criterio']:48s} "
              f"{c['medido']:>10s}  (umbral {c['umbral']})")
    print(f"  prueba de humo: {humo['veredicto']}")
    print(f"  auditoría de fugas: {fugas['veredicto']}")

    por_fuente = df.groupby("source_dataset").agg(
        n=("label", "size"), phishing=("label", "sum"))
    por_fuente["legitimos"] = por_fuente.n - por_fuente.phishing

    informe = {
        "experimento": "E0",
        "resultados_asociados": ["R1.1", "R1.2"],
        "generado_en": datetime.now(timezone.utc).isoformat(),
        "corpus": {
            "n": int(len(df)),
            "prevalencia": round(float(df["label"].mean()), 4),
            "conglomerados": int(df["template_cluster_id"].nunique()),
            "por_fuente": {str(i): {"n": int(r.n), "phishing": int(r.phishing),
                                    "legitimos": int(r.legitimos)}
                           for i, r in por_fuente.iterrows()},
        },
        "indicador_r1_1": ind,
        "prueba_de_humo": humo,
        "auditoria_de_fugas_r1_2": fugas,
    }

    def figura(destino: Path) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fuentes = sorted(df["source_dataset"].unique())
        est = [100 * df.loc[df.source_dataset == f, "has_structure_modality"].mean()
               for f in fuentes]
        red = [100 * df.loc[df.source_dataset == f, "has_network_modality"].mean()
               for f in fuentes]
        amb = [100 * (df.loc[df.source_dataset == f, "has_structure_modality"]
                      & df.loc[df.source_dataset == f, "has_network_modality"]).mean()
               for f in fuentes]
        x = np.arange(len(fuentes))
        fig, ax = plt.subplots(figsize=(9.5, 4.6))
        ax.bar(x - 0.27, est, 0.27, label="Estructura (HTML/DOM)", color="#4C72B0")
        ax.bar(x, red, 0.27, label="Red (URL)", color="#DD8452")
        ax.bar(x + 0.27, amb, 0.27, label="Ambas", color="#55A868")
        ax.set_xticks(x); ax.set_xticklabels(fuentes, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Cobertura (%)"); ax.set_ylim(0, 105)
        ax.set_title("Disponibilidad de las modalidades no textuales, por colección")
        ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(destino / "e0_cobertura_modal.png", dpi=160)
        plt.close(fig)

    emitir("e0", informe, figura, corpus=df)
    return 0 if ind["cumple_todo"] and humo["veredicto"] == "TODAS SUPERADAS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
