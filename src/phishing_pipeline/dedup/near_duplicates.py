"""
Detección de casi-duplicados (plantillas) vía MinHashLSH.

Una "plantilla" es un artefacto intra-fuente: el clustering se corre por
separado dentro de cada familia de fuente, nunca cruzando fuentes. El
constructor del corpus (`corpus_real`) usa el identificador resultante para
agrupar por campaña al particionar, de modo que plantillas casi idénticas no
queden repartidas entre entrenamiento y prueba.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from datasketch import MinHash, MinHashLSH

from phishing_pipeline.config import get_source_family
from phishing_pipeline.logging_utils import get_logger

logger = get_logger(__name__)

SINGLETON_SANITY_THRESHOLD_PCT = 50.0


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
