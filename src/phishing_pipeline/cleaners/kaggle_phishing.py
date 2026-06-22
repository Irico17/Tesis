"""Cleaner para Kaggle phishing-email (celdas 3 + 16)."""

from __future__ import annotations

import uuid

import pandas as pd

from phishing_pipeline.config import SOURCE_KAGGLE
from phishing_pipeline.features.network import extract_technical_features
from phishing_pipeline.schema import empty_canonical_row, normalize_label


def clean_kaggle_phishing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y mapea Kaggle phishing-email al esquema canónico.

    - dropna Email Text + Email Type
    - eliminar body == 'empty'
    - drop_duplicates Email Text
    - word_count > 0
    """
    work = df.copy()
    work = work.dropna(subset=["Email Text", "Email Type"])
    work = work[work["Email Text"].astype(str).str.strip().str.lower() != "empty"]
    work = work.drop_duplicates(subset=["Email Text"])
    work["Word_Count"] = work["Email Text"].apply(lambda x: len(str(x).split()))
    work = work[work["Word_Count"] > 0]

    rows = []
    for _, row in work.iterrows():
        body = str(row["Email Text"])
        try:
            label_int, label_text = normalize_label(row["Email Type"], SOURCE_KAGGLE)
        except ValueError:
            continue
        has_html, network = extract_technical_features(body)
        parsed = parse_row(body, label_int, label_text, has_html, network)
        rows.append(parsed)

    return pd.DataFrame(rows)


def parse_row(
    body: str,
    label_int: int,
    label_text: str,
    has_html: int,
    network: str,
) -> dict:
    """Construye fila canónica."""
    base = empty_canonical_row()
    base.update(
        {
            "email_id": str(uuid.uuid4()),
            "source_dataset": SOURCE_KAGGLE,
            "label": label_int,
            "label_text": label_text,
            "body_raw": body,
            "body_plain": body,
            "clean_text": body,
            "has_html": has_html,
            "network_indicators": network,
            "word_count": len(body.split()),
            "processing_status": "ok",
        }
    )
    return base
