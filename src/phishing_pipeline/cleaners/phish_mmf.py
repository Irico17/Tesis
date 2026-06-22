"""Cleaner para PhishMMF JSONL (celda 19 — versión correcta)."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pandas as pd

from phishing_pipeline.config import LABEL_PHISHING, LABEL_SAFE, LABEL_TEXT_PHISHING, LABEL_TEXT_SAFE
from phishing_pipeline.features.network import extract_technical_features
from phishing_pipeline.schema import empty_canonical_row


def _infer_label(filename: str) -> tuple[int, str] | None:
    fn = filename.lower()
    if "phishing_pot" in fn or "datacon2023" in fn:
        return LABEL_PHISHING, LABEL_TEXT_PHISHING
    if "spamassasin" in fn or "ceas_08" in fn:
        return LABEL_SAFE, LABEL_TEXT_SAFE
    return None


def _extract_body(item: dict) -> str:
    body = (
        item.get("Body")
        or item.get("body")
        or (item.get("content") or {}).get("body")
        or item.get("text")
        or ""
    )
    return str(body).strip()


def _extract_meta(item: dict) -> tuple[str, str]:
    meta = item.get("Metadata") or {}
    sender = item.get("sender") or meta.get("From") or meta.get("from") or ""
    subject = item.get("subject") or meta.get("Subject") or meta.get("subject") or ""
    return str(sender), str(subject)


def parse_phish_mmf_jsonl(extract_path: Path) -> pd.DataFrame:
    """
    Parsea archivos JSONL línea a línea (NO leer archivo entero).

    Args:
        extract_path: carpeta con JSONL extraídos de PhishMMF
    """
    rows = []
    invalid_lines = 0
    total_lines = 0

    for file_path in sorted(extract_path.rglob("*")):
        if not file_path.is_file():
            continue
        label_info = _infer_label(file_path.name)
        if label_info is None:
            continue

        label_int, label_text = label_info
        source_name = f"PhishMMF_{file_path.name}"

        try:
            with file_path.open(encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    total_lines += 1
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        invalid_lines += 1
                        continue

                    body = _extract_body(item)
                    if len(body) <= 5:
                        continue

                    sender, subject = _extract_meta(item)
                    has_html, network = extract_technical_features(body)

                    base = empty_canonical_row()
                    base.update(
                        {
                            "email_id": str(uuid.uuid4()),
                            "source_dataset": source_name,
                            "label": label_int,
                            "label_text": label_text,
                            "subject": subject,
                            "sender": sender,
                            "body_raw": body,
                            "body_plain": body,
                            "clean_text": body,
                            "has_html": has_html,
                            "network_indicators": network,
                            "word_count": len(body.split()),
                            "processing_status": "ok",
                        }
                    )
                    rows.append(base)
        except OSError:
            continue

    df = pd.DataFrame(rows)
    df.attrs["phish_mmf_stats"] = {
        "total_lines": total_lines,
        "invalid_lines": invalid_lines,
        "invalid_pct": round(invalid_lines / total_lines * 100, 2) if total_lines else 0,
    }
    return df


def clean_phish_mmf(extract_path: Path) -> pd.DataFrame:
    """Wrapper de limpieza PhishMMF."""
    return parse_phish_mmf_jsonl(extract_path)
