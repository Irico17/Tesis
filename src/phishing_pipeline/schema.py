"""Esquema canónico unificado y validadores."""

from __future__ import annotations

from typing import Any

import pandas as pd

from phishing_pipeline.config import (
    LABEL_PHISHING,
    LABEL_SAFE,
    LABEL_TEXT_PHISHING,
    LABEL_TEXT_SAFE,
)

CANONICAL_COLUMNS: list[str] = [
    "email_id",
    "source_dataset",
    "label",
    "label_text",
    "subject",
    "sender",
    "body_raw",
    "body_plain",
    "body_html",
    "clean_text",
    "has_html",
    "network_indicators",
    "num_links",
    "num_images",
    "has_form",
    "has_iframe",
    "has_javascript",
    "has_ip_link",
    "word_count",
    "received_origin_ip",
    "spf_result",
    "dkim_result",
    "dmarc_result",
    "num_urls_metadata",
    "language",
    "processing_status",
    "processing_error",
    # R1.3 — dedup/splits (Fase A1)
    "template_cluster_id",
    # R1.3 — features léxicas de URL, agregadas a nivel de mensaje (Fase A2)
    "url_max_length",
    "url_max_digit_ratio",
    "url_max_subdomain_count",
    "url_has_at_symbol",
    "url_has_shortener",
    "url_has_suspicious_tld",
    "url_domain_max_entropy",
]


def empty_canonical_row() -> dict[str, Any]:
    """Fila vacía con valores por defecto del esquema canónico."""
    return {
        "email_id": None,
        "source_dataset": None,
        "label": None,
        "label_text": None,
        "subject": "",
        "sender": "",
        "body_raw": "",
        "body_plain": "",
        "body_html": "",
        "clean_text": "",
        "has_html": 0,
        "network_indicators": "None",
        "num_links": 0,
        "num_images": 0,
        "has_form": 0,
        "has_iframe": 0,
        "has_javascript": 0,
        "has_ip_link": 0,
        "word_count": 0,
        "received_origin_ip": None,
        "spf_result": None,
        "dkim_result": None,
        "dmarc_result": None,
        "num_urls_metadata": None,
        "language": None,
        "processing_status": "ok",
        "processing_error": None,
        "template_cluster_id": None,
        "url_max_length": 0,
        "url_max_digit_ratio": 0.0,
        "url_max_subdomain_count": 0,
        "url_has_at_symbol": 0,
        "url_has_shortener": 0,
        "url_has_suspicious_tld": 0,
        "url_domain_max_entropy": 0.0,
    }


def normalize_label(value: Any, source: str) -> tuple[int, str]:
    """
    Normaliza etiqueta a (label_int, label_text).

    Soporta: Email Type (Kaggle), label binario (spam-genuine), inferencia PhishMMF.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        raise ValueError(f"Etiqueta nula para fuente {source}")

    # Binario numérico (spam-genuine)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        iv = int(value)
        if iv == 0:
            return LABEL_SAFE, LABEL_TEXT_SAFE
        if iv == 1:
            return LABEL_PHISHING, LABEL_TEXT_PHISHING
        raise ValueError(f"Label binario inválido: {value}")

    text = str(value).strip().lower()

    phishing_tokens = {"phishing email", "phishing", "spam", "1", "malicious"}
    safe_tokens = {"safe email", "safe", "ham", "genuine", "legitimate", "0", "benign"}

    if text in phishing_tokens or "phish" in text:
        return LABEL_PHISHING, LABEL_TEXT_PHISHING
    if text in safe_tokens or "safe" in text:
        return LABEL_SAFE, LABEL_TEXT_SAFE

    raise ValueError(f"Etiqueta no reconocida '{value}' para fuente {source}")


def validate_schema(df: pd.DataFrame, strict: bool = True) -> bool:
    """Valida que el DataFrame contenga columnas canónicas obligatorias."""
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        if strict:
            raise ValueError(f"Columnas canónicas faltantes: {missing}")
        return False
    return True


def ensure_canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Añade columnas canónicas faltantes con valores por defecto."""
    defaults = empty_canonical_row()
    out = df.copy()
    for col in CANONICAL_COLUMNS:
        if col not in out.columns:
            out[col] = defaults.get(col)
    return out[CANONICAL_COLUMNS]
