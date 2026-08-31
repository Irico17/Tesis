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
# date/subject/body) sin este dict.
_AUTH_HEADER_SOURCES = {"phishing_pot.jsonl", "datacon2023_1.jsonl", "datacon2023_2.jsonl"}

# Extraer o no las cabeceras de autenticación de PhishMMF.
#
# POR QUÉ ESTÁ DESACTIVADO POR DEFECTO. Los tres ficheros que traen esas cabeceras
# son exactamente los tres ficheros de phishing de PhishMMF: `phishing_pot` y los
# dos de `datacon2023`. Los dos que no las traen —CEAS_08 y SpamAssassin— son
# exactamente los dos de la clase legítima. La disponibilidad del campo coincide
# así, punto por punto, con la etiqueta.
#
# La consecuencia se midió sobre el corpus y es una fuga dura: dentro de PhishMMF,
# `p(phishing | spf_result presente) = 1.0000` sobre 4,478 correos, y la
# información mutua entre la mera PRESENCIA del campo y la etiqueta alcanza 0.6016
# nats —el 87% de la entropía de una etiqueta binaria equilibrada—. El modelo no
# necesita leer el valor del campo: le basta con notar que existe.
#
# El daño no se detiene en la rama de red. `compute_modality_availability` define
# la disponibilidad de esa modalidad como «hay URL O hay SPF», de modo que la
# bandera `has_network` que el modelo recibe como entrada explícita heredaba la
# fuga. Es el origen real del «atajo de disponibilidad» de 0.0790 nats que el
# trabajo había atribuido a una propiedad del dominio: no lo era, era un defecto
# de este limpiador.
#
# La única corrección válida es simétrica. O se extrae el campo de las cinco
# sub-fuentes, o no se extrae de ninguna. Como CEAS_08 y SpamAssassin no publican
# esas cabeceras, la primera opción no está disponible y se toma la segunda:
# perder una característica es preferible a conservar una que revela la etiqueta.
# El valor de la característica era además ilusorio, porque procedía íntegramente
# de la asimetría.
#
# Se deja gobernable para poder reproducir las corridas anteriores y para que la
# decisión quede explícita en el código en lugar de implícita en una omisión.
EXTRAER_CABECERAS_PHISHMMF = False

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


def parse_phish_mmf_jsonl(
    extract_path: Path, extraer_cabeceras: bool = EXTRAER_CABECERAS_PHISHMMF
) -> pd.DataFrame:
    """
    Parsea archivos JSONL línea a línea (NO leer archivo entero).

    Args:
        extract_path: carpeta con JSONL extraídos de PhishMMF
        extraer_cabeceras: si se pueblan spf/dkim/dmarc/IP de origen. Por defecto
            NO se pueblan, porque solo están disponibles en los ficheros de una de
            las dos clases y su presencia revela la etiqueta. Ver la justificación
            completa en `EXTRAER_CABECERAS_PHISHMMF`.
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
        has_auth_headers = (
            extraer_cabeceras and file_path.name in _AUTH_HEADER_SOURCES
        )

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


def clean_phish_mmf(
    extract_path: Path, extraer_cabeceras: bool = EXTRAER_CABECERAS_PHISHMMF
) -> pd.DataFrame:
    """Wrapper de limpieza PhishMMF."""
    return parse_phish_mmf_jsonl(extract_path, extraer_cabeceras=extraer_cabeceras)
