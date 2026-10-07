"""
Estadísticos de complejidad estructural del DOM: número de nodos y profundidad
de anidamiento.

Motivación. La rama
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

from phishing_pipeline.features.dom_parser import ETIQUETA_HTML

DOM_STATS_COLUMNS = ["total_nodos_dom", "profundidad_dom"]


def compute_dom_stats(body_raw: str | None) -> dict[str, int]:
    """
    Calcula número de nodos y profundidad máxima de anidamiento del DOM.

    Devuelve ceros para cuerpos vacíos o sin marcado: un correo de texto plano
    tiene, por definición, complejidad estructural nula, y esa es la codificación
    correcta -- no un valor ausente.

    **La comprobación de marcado es imprescindible y no la suplía lxml.** El
    docstring anterior confiaba en que `lxml_html.fromstring` fallara ante texto
    plano, y no falla: envuelve el texto en `<span><p>…</p></span>`, de modo que
    todo correo de texto plano recibía `total_nodos_dom = 1` y
    `profundidad_dom = 3`. Se verificó sobre el corpus: las dos fuentes legítimas
    de PhishMMF tienen exactamente esos valores en el 100% de sus filas, y esa
    constante se estaba interpretando como una medida de complejidad. El contrato
    documentado era, por tanto, falso, y la característica no distinguía entre
    «sin estructura» y «una estructura mínima».

    Se exige ahora que exista al menos una etiqueta HTML reconocida, con el mismo
    patrón que gobierna `has_html`, de modo que ambas características sean
    coherentes entre sí por construcción.
    """
    if not isinstance(body_raw, str) or not body_raw.strip():
        return {"total_nodos_dom": 0, "profundidad_dom": 0}

    if not ETIQUETA_HTML.search(body_raw):
        return {"total_nodos_dom": 0, "profundidad_dom": 0}

    try:
        root = lxml_html.fromstring(body_raw)
    except Exception:
        # Cuerpo con algo que parece marcado pero que lxml no logra interpretar.
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


def compute_dom_stats_enrichment(unified_path=None):
    """
    Calcula los estadísticos de DOM sobre TODAS las filas del corpus unificado.

    Sigue el mismo patrón que `compute_network_lexical_enrichment`: se calcula
    fila por fila sobre `body_raw`, la misma columna que emplean los tres
    limpiadores, y se devuelve un DataFrame indexado por `email_id` listo para
    fusionar con el corpus.
    """
    from phishing_pipeline.config import PROCESSED_DIR
    from phishing_pipeline.splits import load_unified_for_splits

    unified_path = unified_path or (PROCESSED_DIR / "Dataset_Unificado.parquet")
    unified = load_unified_for_splits(unified_path)
    stats = compute_dom_stats_batch(unified["body_raw"].fillna("").astype(str))
    stats.insert(0, "email_id", unified["email_id"].values)
    return stats


def main() -> None:
    """Genera el parquet de enriquecimiento de complejidad del DOM."""
    import json
    from datetime import datetime, timezone

    from phishing_pipeline.config import PROCESSED_DIR, REPORTS_DIR
    from phishing_pipeline.logging_utils import get_logger

    logger = get_logger(__name__)
    output_path = PROCESSED_DIR / "dom_stats_enrichment.parquet"

    logger.info("=== Enriquecimiento de complejidad del DOM sobre el corpus completo ===")
    stats = compute_dom_stats_enrichment()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    stats.to_parquet(output_path, index=False)
    logger.info("Guardado: %s (%d filas)", output_path, len(stats))

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(len(stats)),
        "total_nodos_dom": {
            "media": round(float(stats["total_nodos_dom"].mean()), 4),
            "mediana": int(stats["total_nodos_dom"].median()),
            "maximo": int(stats["total_nodos_dom"].max()),
        },
        "profundidad_dom": {
            "media": round(float(stats["profundidad_dom"].mean()), 4),
            "mediana": int(stats["profundidad_dom"].median()),
            "maximo": int(stats["profundidad_dom"].max()),
        },
    }
    report_path = REPORTS_DIR / "dom_stats_enrichment_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Reporte: %s", report_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
