"""Cleaner para PhishMMF JSONL (celda 19 — versión correcta)."""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

import pandas as pd

from phishing_pipeline.config import LABEL_PHISHING, LABEL_SAFE, LABEL_TEXT_PHISHING, LABEL_TEXT_SAFE
from phishing_pipeline.features.network import extract_technical_features
from phishing_pipeline.schema import empty_canonical_row

# Sub-fuentes de PhishMMF cuyo JSON crudo trae un dict `Metadata` con cabeceras
# de autenticación (Authentication-Results / Received-SPF / X-Sender-IP).
# CEAS_08_0.jsonl y SpamAssasin_0.jsonl usan un esquema distinto (sender/receiver/
# date/subject/body) sin este dict, así que quedan fuera intencionalmente.
_AUTH_HEADER_SOURCES = {"phishing_pot.jsonl", "datacon2023_1.jsonl", "datacon2023_2.jsonl"}

_SPF_RE = re.compile(r"spf\s*=\s*(\w+)", re.IGNORECASE)
_DKIM_RE = re.compile(r"dkim\s*=\s*(\w+)", re.IGNORECASE)
_DMARC_RE = re.compile(r"dmarc\s*=\s*(\w+)", re.IGNORECASE)


def _parse_authentication_results(auth_str: str | None) -> dict:
    """
    Extrae spf/dkim/dmarc del string crudo de `Metadata.Authentication-Results`.

    Formato típico: "spf=pass (sender IP is ...) smtp.mailfrom=...;
    dkim=pass (signature was verified) header.d=...;dmarc=pass action=none ...".
    Valores devueltos tal cual aparecen en minúsculas (pass/fail/none/softfail/
    neutral/permerror/...), sin forzar un mapeo a un set fijo de categorías.
    """
    result = {"spf_result": None, "dkim_result": None, "dmarc_result": None}
    if not auth_str:
        return result

    spf_match = _SPF_RE.search(auth_str)
    if spf_match:
        result["spf_result"] = spf_match.group(1).lower()

    dkim_match = _DKIM_RE.search(auth_str)
    if dkim_match:
        result["dkim_result"] = dkim_match.group(1).lower()

    dmarc_match = _DMARC_RE.search(auth_str)
    if dmarc_match:
        result["dmarc_result"] = dmarc_match.group(1).lower()

    return result


def _parse_received_spf(spf_str: str | None) -> str | None:
    """
    Fallback: extrae el resultado SPF de `Metadata.Received-SPF`.

    Formato típico: "Pass (protection.outlook.com: domain of ... )" -> "pass".
    """
    if not spf_str:
        return None
    stripped = str(spf_str).strip()
    if not stripped:
        return None
    first_word = stripped.split()[0]
    return first_word.lower() or None


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
        has_auth_headers = file_path.name in _AUTH_HEADER_SOURCES

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

                    if has_auth_headers:
                        meta = item.get("Metadata") or {}
                        auth_results = meta.get("Authentication-Results")
                        received_spf = meta.get("Received-SPF")
                        sender_ip = meta.get("X-Sender-IP")

                        parsed_auth = _parse_authentication_results(auth_results)
                        spf_result = parsed_auth["spf_result"]
                        if spf_result is None:
                            # Fallback: Received-SPF solo confiable en phishing_pot.jsonl,
                            # pero se intenta igual en datacon2023 por si acaso existe.
                            spf_result = _parse_received_spf(received_spf)

                        base.update(
                            {
                                "spf_result": spf_result,
                                "dkim_result": parsed_auth["dkim_result"],
                                "dmarc_result": parsed_auth["dmarc_result"],
                                "received_origin_ip": str(sender_ip).strip() if sender_ip else None,
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
