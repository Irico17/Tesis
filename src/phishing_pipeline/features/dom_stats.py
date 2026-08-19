"""
Estadísticos de complejidad estructural del DOM: número de nodos y profundidad
de anidamiento.

Motivación (ver `doc/REVISION_CODIGO_MODELO.md`, hallazgos 1 y 5). La rama
estructural del modelo dependía de características que resultaron constantes en
todo el corpus: `has_form`, `has_iframe` y `has_javascript` valen cero en las
110,152 filas, porque las fuentes publicaron el HTML ya sanitizado, sin
elementos interactivos ni ejecutables. En cambio, el número de nodos y la
profundidad de anidamiento **sí presentan variabilidad** sobre este mismo
corpus (mediana de 7 nodos, máximo de 1,649; profundidad mediana de 4 y máxima
de 109), y capturan la noción de complejidad estructural que la literatura
asocia a las técnicas de ofuscación.

`features/dom_parser.py` ya contenía una función equivalente
(`extract_html_dom_stats`), pero su resultado se descartaba sin incorporarse al
esquema canónico. Este módulo la sustituye con una implementación basada
directamente en lxml —del orden de 0.1 ms por correo, frente al costo mayor de
BeautifulSoup— y con cálculo de profundidad iterativo en lugar de recursivo,
para no depender del límite de recursión en documentos muy anidados.
"""

from __future__ import annotations

import pandas as pd
from lxml import html as lxml_html

DOM_STATS_COLUMNS = ["total_nodos_dom", "profundidad_dom"]


def compute_dom_stats(body_raw: str | None) -> dict[str, int]:
    """
    Calcula número de nodos y profundidad máxima de anidamiento del DOM.

    Devuelve ceros para cuerpos vacíos, no textuales o que lxml no logre
    interpretar: un correo de texto plano tiene, por definición, complejidad
    estructural nula, y esa es la codificación correcta -- no un valor ausente.
    """
    if not isinstance(body_raw, str) or not body_raw.strip():
        return {"total_nodos_dom": 0, "profundidad_dom": 0}

    try:
        root = lxml_html.fromstring(body_raw)
    except Exception:
        # lxml falla ante cuerpos sin ningún elemento interpretable como marcado;
        # es el caso esperado para texto plano, no una condición de error.
        return {"total_nodos_dom": 0, "profundidad_dom": 0}

    total_nodes = 0
    max_depth = 0
    for element in root.iter():
        total_nodes += 1
        depth = 0
        parent = element
        while parent is not None:
            depth += 1
            parent = parent.getparent()
        if depth > max_depth:
            max_depth = depth

    return {"total_nodos_dom": total_nodes, "profundidad_dom": max_depth}


def compute_dom_stats_batch(bodies: pd.Series) -> pd.DataFrame:
    """Aplica `compute_dom_stats` sobre una serie de cuerpos y devuelve un DataFrame."""
    records = [compute_dom_stats(b) for b in bodies]
    return pd.DataFrame(records, index=bodies.index)
