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
  - Prueba de humo del corpus, ocho comprobaciones
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

from comun import BASE, CORPUS, cargar_corpus, emitir

# Umbrales del indicador de R1.1, confirmados por los asesores. La prevalencia
# objetivo de 0.40 es un parámetro de diseño de phishing_pipeline.corpus_real.
UMBRAL_TRIMODAL = 25.0
UMBRAL_BIMODAL = 40.0
RANGO_PREVALENCIA = (0.40, 0.60)

FORMATO_DE_PUBLICACION = {
    "phishing_pot": "mensaje original",
    "Nazario": "mensaje original",
    "SpamAssassin": "mensaje original",
    "Fedora": "mensaje original",
    "kernel_lists": "mensaje original",
    "CEAS_08": "tabular",
    "datacon2023": "tabular",
    "Kaggle": "tabular",
}


def cobertura_por_coleccion(df: pd.DataFrame) -> dict:
    """Cobertura modal medida en cada coleccion, junto a su formato declarado.

    Existe porque el capitulo sostenia sobre el formato de publicacion una
    afirmacion que la propia figura de cobertura desmentia: que las colecciones
    tabulares no aportan modalidades no textuales. Se mide y se emite, de modo que
    el texto pueda escribirse con la cifra y no con la expectativa.
    """
    tri = df["has_structure_modality"] & df["has_network_modality"]
    por_coleccion, agregado = {}, {}
    for nombre, sub in df.groupby("source_dataset"):
        formato = FORMATO_DE_PUBLICACION.get(str(nombre), "no declarado")
        m = tri.loc[sub.index]
        por_coleccion[str(nombre)] = {
            "formato_de_publicacion": formato,
            "n": int(len(sub)),
            "con_estructura": int(sub["has_structure_modality"].sum()),
            "con_red": int(sub["has_network_modality"].sum()),
            "trimodales": int(m.sum()),
            "pct_estructura": round(100 * float(sub["has_structure_modality"].mean()), 2),
            "pct_red": round(100 * float(sub["has_network_modality"].mean()), 2),
            "pct_trimodal": round(100 * float(m.mean()), 2),
        }
        agr = agregado.setdefault(formato, {"colecciones": 0, "n": 0, "trimodales": 0})
        agr["colecciones"] += 1
        agr["n"] += int(len(sub))
        agr["trimodales"] += int(m.sum())
    total_tri = int(tri.sum())
    for formato, agr in agregado.items():
        agr["pct_de_los_trimodales"] = (round(100 * agr["trimodales"] / total_tri, 2)
                                        if total_tri else 0.0)
    return {"por_coleccion": por_coleccion, "por_formato": agregado,
            "trimodales_del_corpus": total_tri}


def composicion_de_la_prueba(df: pd.DataFrame) -> dict:
    """Como se reparte el conjunto de prueba por tamano de conglomerado de campana.

    E4 mide el desempeno en las categorias que contienen ambas clases, que son las
    unicas evaluables, y por eso su informe no menciona las demas. El capitulo
    llego a afirmar que no existian conglomerados grandes en prueba, cuando lo que
    ocurre es que los que hay pertenecen todos a una sola clase: es la limitacion
    central del trabajo asomando otra vez, y conviene medirla y no deducirla.
    """
    from comun import particion_agrupada

    particion = particion_agrupada(df)
    prueba = df.loc[particion.prueba] if hasattr(particion.prueba, "dtype") else particion.prueba
    tam = df["template_cluster_id"].value_counts()
    tamanos = prueba["template_cluster_id"].map(tam)
    tramos = [(1, 1, "campañas únicas"), (2, 5, "conglomerados de 2 a 5"),
              (6, 10 ** 9, "conglomerados de más de 5")]
    salida = {}
    for lo, hi, nombre in tramos:
        sub = prueba[((tamanos >= lo) & (tamanos <= hi)).values]
        if not len(sub):
            continue
        clases = sorted(int(c) for c in sub["label"].unique())
        salida[nombre] = {
            "n": int(len(sub)),
            "prevalencia": round(float(sub["label"].mean()), 4),
            "clases_presentes": clases,
            "evaluable": len(clases) == 2,
        }
    return {"n_prueba": int(len(prueba)), "por_tamano": salida}


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
        ("Formato estándar listo para ingesta", ".parquet, .csv o .json",
         CORPUS.suffix, CORPUS.suffix in {".parquet", ".csv", ".json"}),
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
    por_coleccion = df.groupby("source_dataset")["label"].nunique()
    mixtas = int((por_coleccion == 2).sum())
    prevalencia = float(df["label"].mean())
    equilibrado = RANGO_PREVALENCIA[0] <= prevalencia <= RANGO_PREVALENCIA[1]
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
        ("identificador de conglomerado de plantilla sin valores ausentes",
         int(df["template_cluster_id"].isna().sum()) == 0,
         f"{int(df['template_cluster_id'].isna().sum())} nulos"),
        # Se comprueba lo que el enunciado declara, y no otra cosa. La version
        # anterior evaluaba `df["label"].nunique() == 2`, esto es, que el corpus
        # contuviera dos clases: cierto por construccion e incapaz de fallar. El
        # detalle que emitia, "2 clases", se leia ademas como si hubiera dos
        # colecciones con ambas clases, cuando solo hay una. El enunciado es una
        # disyuncion, de modo que se miden sus dos terminos y se declara cual lo
        # sostiene.
        ("al menos dos colecciones con ambas clases o el corpus equilibrado",
         mixtas >= 2 or equilibrado,
         f"{mixtas} de {len(por_coleccion)} colecciones con ambas clases; "
         f"prevalencia del corpus {prevalencia:.4f}"),
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
        "cobertura_modal_por_coleccion": cobertura_por_coleccion(df),
        "composicion_de_la_prueba": composicion_de_la_prueba(df),
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
