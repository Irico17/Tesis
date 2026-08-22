"""
Detección de casi-duplicados (plantillas) vía MinHashLSH (Fase A1).

Una "plantilla" es un artefacto intra-fuente: el clustering se corre por
separado dentro de cada `source_family` (Kaggle_Phishing_Email,
Spam_Genuine_Mail, PhishMMF), nunca cruzando fuentes. Esto alimenta
`splits.create_group_aware_splits()`, que agrupa por `template_cluster_id`
con StratifiedGroupKFold para que plantillas casi-idénticas no queden
repartidas entre train y test.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from datasketch import MinHash, MinHashLSH

from phishing_pipeline.config import PROCESSED_DIR, REPORTS_DIR, get_source_family
from phishing_pipeline.logging_utils import get_logger

logger = get_logger(__name__)

SINGLETON_SANITY_THRESHOLD_PCT = 50.0
MIN_SUBJECT_LENGTH_FOR_GROUPING = 8


class _UnionFind:
    """Union-find simple (compresión de camino) sobre claves arbitrarias hashables."""

    def __init__(self) -> None:
        self._parent: dict[Any, Any] = {}

    def find(self, x: Any) -> Any:
        self._parent.setdefault(x, x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        # Compresión de camino
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, x: Any, y: Any) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self._parent[rx] = ry


def _build_shingles(text: str, shingle_size: int) -> set[str]:
    tokens = str(text).strip().lower().split()
    if len(tokens) < shingle_size:
        return set()
    return {
        " ".join(tokens[i : i + shingle_size])
        for i in range(len(tokens) - shingle_size + 1)
    }


def _cluster_one_family(
    text_series: pd.Series,
    family_name: str,
    num_perm: int,
    shingle_size: int,
    jaccard_threshold: float,
) -> pd.Series:
    """Clustering de casi-duplicados dentro de UNA source_family. Devuelve Serie de template_cluster_id."""
    uf = _UnionFind()
    lsh = MinHashLSH(threshold=jaccard_threshold, num_perm=num_perm)
    minhashes: dict[Any, MinHash] = {}
    key_to_idx: dict[str, Any] = {}

    for idx, text in text_series.items():
        uf.find(idx)  # registra la fila aunque termine sin unión (singleton)
        shingles = _build_shingles(text, shingle_size)
        if not shingles:
            # Texto vacío o demasiado corto para shinglear -> cluster propio (singleton).
            continue
        m = MinHash(num_perm=num_perm)
        for shingle in shingles:
            m.update(shingle.encode("utf8"))
        minhashes[idx] = m
        key_to_idx[str(idx)] = idx

    # Inserción completa antes de consultar (evita orden-dependencia en las consultas).
    for idx, m in minhashes.items():
        lsh.insert(str(idx), m)

    for idx, m in minhashes.items():
        candidates = lsh.query(m)
        for cand_key in candidates:
            cand_idx = key_to_idx[cand_key]
            if cand_idx != idx:
                uf.union(idx, cand_idx)

    root_to_cluster_idx: dict[Any, int] = {}
    next_cluster_idx = 0
    result: dict[Any, str] = {}
    for idx in text_series.index:
        root = uf.find(idx)
        if root not in root_to_cluster_idx:
            root_to_cluster_idx[root] = next_cluster_idx
            next_cluster_idx += 1
        result[idx] = f"{family_name}_{root_to_cluster_idx[root]}"

    logger.info(
        "%s: %d filas -> %d clusters (%d con MinHash, %d singletons por texto corto/vacío)",
        family_name,
        len(text_series),
        next_cluster_idx,
        len(minhashes),
        len(text_series) - len(minhashes),
    )
    return pd.Series(result, name="template_cluster_id")


def find_near_duplicate_clusters(
    df: pd.DataFrame,
    text_col: str = "clean_text",
    source_col: str = "source_dataset",
    num_perm: int = 128,
    shingle_size: int = 5,
    jaccard_threshold: float = 0.8,
) -> pd.Series:
    """
    Asigna `template_cluster_id` por fila vía MinHashLSH, clustering SEPARADO por source_family.

    Devuelve una Serie indexada igual que `df`, valores `f"{source_family}_{cluster_idx}"`
    con `cluster_idx` entero incremental por source_family, empezando en 0.
    """
    families = df[source_col].apply(get_source_family)
    cluster_ids = pd.Series(index=df.index, dtype=object, name="template_cluster_id")

    for family in sorted(families.unique()):
        family_index = families[families == family].index
        family_clusters = _cluster_one_family(
            df.loc[family_index, text_col],
            family,
            num_perm=num_perm,
            shingle_size=shingle_size,
            jaccard_threshold=jaccard_threshold,
        )
        cluster_ids.loc[family_index] = family_clusters

    return cluster_ids


def _normalize_subject(subject: Any) -> str | None:
    """Normaliza un asunto para agrupación exacta; None si es nulo o demasiado corto/genérico."""
    if subject is None or (isinstance(subject, float) and pd.isna(subject)):
        return None
    s = re.sub(r"\s+", " ", str(subject).strip().lower())
    if len(s) < MIN_SUBJECT_LENGTH_FOR_GROUPING:
        return None
    return s


def merge_clusters_by_subject(
    df: pd.DataFrame,
    cluster_ids: pd.Series,
    source_col: str = "source_dataset",
    subject_col: str = "subject",
) -> pd.Series:
    """
    Segunda pasada: para filas con asunto informativo, reemplaza su
    `template_cluster_id` (de la pasada por cuerpo) por un grupo basado en
    (source_family, asunto normalizado) exacto. Filas sin asunto informativo
    conservan su cluster de cuerpo tal cual.

    Justificación empírica (verificada sobre el corpus real, no supuesta): en
    Spam_Genuine_Mail, 220 asuntos con >=10 repeticiones cubren el 77% de las
    filas de esa fuente, pero el clustering por similitud de CUERPO (Jaccard de
    shingles) solo agrupa como casi-duplicados ~13% de esas copias -- el resto
    varía demasiado en el cuerpo (probablemente personalización de
    nombres/cuentas/enlaces) para superar el umbral de Jaccard, aunque comparten
    la misma plantilla de campaña. El asunto exacto es una señal de plantilla más
    robusta a esa variación y captura directamente el riesgo de fuga que señaló
    el asesor de tesis (correos "casi idénticos" de la misma campaña repartidos
    entre train y test).

    Diseño DELIBERADAMENTE NO transitivo a través de los clusters de cuerpo: una
    primera versión unía clusters de cuerpo completos vía union-find cada vez que
    compartían una fila con el mismo asunto, lo que en la práctica encadenó
    decenas de asuntos distintos en un único súper-cluster de ~39,675 filas
    (48% de Spam_Genuine_Mail) por "efecto puente" -- un cluster de cuerpo que por
    casualidad contenía filas de dos asuntos distintos fusionaba de golpe TODOS
    los clusters de ambos asuntos entre sí, y así sucesivamente. Es un modo de
    fallo conocido del cierre transitivo de casi-duplicados. Esta versión evita
    el problema por construcción: el grupo final de una fila con asunto
    informativo es únicamente `(source_family, asunto)`, nunca se fusiona con
    otro asunto distinto aunque ambos compartan un cluster de cuerpo -- el tamaño
    máximo de un grupo queda acotado por la repetición real de un único asunto
    (a lo sumo unos pocos miles de filas), no por el tamaño del componente
    conexo transitivo.
    """
    families = df[source_col].apply(get_source_family)
    subjects = df[subject_col].apply(_normalize_subject) if subject_col in df.columns else pd.Series(
        None, index=df.index
    )

    new_cluster_ids: dict[Any, str] = {}
    for idx in df.index:
        subj = subjects.loc[idx]
        # OJO: con algunas versiones/dtypes de pandas, Series.apply() coacciona los
        # `None` devueltos por _normalize_subject a NaN (float) al inferir dtype
        # "str" para la columna resultante -- `subj is None` NO detecta ese caso y
        # dejaría pasar filas sin asunto real como si tuvieran un asunto válido
        # (ej. Kaggle, donde el 100% de las filas tiene subject=""). Se usa
        # pd.isna() explícito, no solo `is None`.
        has_subject = not (subj is None or (isinstance(subj, float) and pd.isna(subj)))
        if has_subject:
            fam = families.loc[idx]
            new_cluster_ids[idx] = f"{fam}::subj::{subj[:80]}"
        else:
            new_cluster_ids[idx] = cluster_ids.loc[idx]

    result = pd.Series(new_cluster_ids, name="template_cluster_id")
    logger.info(
        "Fusión por asunto: %d clusters (cuerpo) -> %d clusters (cuerpo + asunto)",
        cluster_ids.nunique(),
        result.nunique(),
    )
    return result


def save_cluster_report(
    cluster_ids: pd.Series,
    source_family: pd.Series,
    path: Path,
    duration_seconds: float | None = None,
) -> dict[str, Any]:
    """Calcula estadísticas de clustering y las guarda como JSON en `path`."""
    total_rows = int(len(cluster_ids))
    cluster_sizes = cluster_ids.value_counts()
    total_clusters = int(cluster_sizes.shape[0])
    singleton_clusters = int((cluster_sizes == 1).sum())
    singleton_pct = round(singleton_clusters / total_clusters * 100, 2) if total_clusters else 0.0

    def _bucket(size: int) -> str:
        if size == 1:
            return "1"
        if size == 2:
            return "2"
        if size <= 5:
            return "3-5"
        if size <= 10:
            return "6-10"
        return "11+"

    size_histogram: dict[str, int] = {"1": 0, "2": 0, "3-5": 0, "6-10": 0, "11+": 0}
    for size in cluster_sizes:
        size_histogram[_bucket(int(size))] += 1

    combined = pd.DataFrame({"cluster_id": cluster_ids, "source_family": source_family})
    by_family: dict[str, Any] = {}
    for family, group in combined.groupby("source_family"):
        fam_sizes = group["cluster_id"].value_counts()
        fam_total_clusters = int(fam_sizes.shape[0])
        fam_singletons = int((fam_sizes == 1).sum())
        by_family[str(family)] = {
            "rows": int(len(group)),
            "clusters": fam_total_clusters,
            "singleton_clusters": fam_singletons,
            "singleton_pct": round(fam_singletons / fam_total_clusters * 100, 2) if fam_total_clusters else 0.0,
            "largest_cluster_size": int(fam_sizes.max()) if fam_total_clusters else 0,
        }

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(duration_seconds, 2) if duration_seconds is not None else None,
        "total_rows": total_rows,
        "total_clusters": total_clusters,
        "singleton_clusters": singleton_clusters,
        "singleton_pct": singleton_pct,
        "largest_cluster_size": int(cluster_sizes.max()) if total_clusters else 0,
        "cluster_size_histogram": size_histogram,
        "by_source_family": by_family,
        "sanity_check": {
            "threshold_pct": SINGLETON_SANITY_THRESHOLD_PCT,
            "passed": singleton_pct >= SINGLETON_SANITY_THRESHOLD_PCT,
            "note": (
                "La mayoria de clusters deberian ser singletons (chequeo de que el metodo no "
                "sobre-fusiona). Si singleton_pct cae por debajo del umbral, se documenta como "
                "hallazgo real en vez de ocultarlo."
            ),
        },
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    logger.info("Reporte de clusters guardado en %s", path)
    return report


def main() -> None:
    """
    Ejecuta la detección de casi-duplicados y guarda la tabla de agrupación.

    Existe como función y no solo como bloque de ejecución directa para que el
    orquestador de reconstrucción (`phishing_pipeline.rebuild_corpus`) pueda
    invocarla como un paso más de la secuencia, en lugar de depender de que
    alguien recuerde ejecutar este módulo por separado.
    """
    from phishing_pipeline.splits import load_unified_for_splits

    t0 = time.time()
    logger.info("Cargando dataset unificado...")
    df = load_unified_for_splits()
    logger.info("Dataset cargado: %d filas", len(df))

    cluster_ids_body = find_near_duplicate_clusters(df)
    body_clusters = cluster_ids_body.nunique()

    cluster_ids = merge_clusters_by_subject(df, cluster_ids_body)
    elapsed = time.time() - t0

    source_family = df["source_dataset"].apply(get_source_family)
    report = save_cluster_report(
        cluster_ids,
        source_family,
        REPORTS_DIR / "near_duplicate_clusters.json",
        duration_seconds=elapsed,
    )
    report["clusters_before_subject_merge"] = int(body_clusters)
    (REPORTS_DIR / "near_duplicate_clusters.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    out = pd.DataFrame(
        {
            "email_id": df["email_id"].values,
            "template_cluster_id": cluster_ids.values,
        }
    )
    out_path = PROCESSED_DIR / "template_clusters.parquet"
    out.to_parquet(out_path, index=False)
    logger.info("Tabla lateral de clusters guardada en %s (%d filas)", out_path, len(out))

    print("=" * 70)
    print("Near-duplicate clustering (MinHashLSH) - resumen")
    print("=" * 70)
    print(f"Tiempo total: {elapsed:.2f}s")
    print(f"Filas totales: {report['total_rows']}")
    print(f"Clusters totales: {report['total_clusters']}")
    print(f"Singleton clusters: {report['singleton_clusters']} ({report['singleton_pct']}%)")
    print(f"Cluster mas grande: {report['largest_cluster_size']} filas")
    print(f"Sanity check (>= {SINGLETON_SANITY_THRESHOLD_PCT}% singletons): {report['sanity_check']['passed']}")
    print("Por source_family:")
    for family, stats in report["by_source_family"].items():
        print(
            f"  {family}: {stats['rows']} filas -> {stats['clusters']} clusters "
            f"({stats['singleton_pct']}% singletons, max={stats['largest_cluster_size']})"
        )
    print(f"Reporte JSON: {REPORTS_DIR / 'near_duplicate_clusters.json'}")
    print(f"Tabla lateral: {out_path}")



if __name__ == "__main__":
    main()
