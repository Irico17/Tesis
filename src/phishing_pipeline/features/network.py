"""Extracción de URLs, IPs y features de red."""

from __future__ import annotations

import re

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
IP_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")


def extract_technical_features(text: str) -> tuple[int, str]:
    """
    Detecta HTML y extrae URLs/IPs (celda 16 del notebook).

    Returns:
        (has_html, network_indicators)
    """
    text = str(text) if text is not None else ""
    has_html = 1 if HTML_TAG_PATTERN.search(text) else 0
    urls = URL_PATTERN.findall(text)
    ips = IP_PATTERN.findall(text)
    network_data = ", ".join(urls + ips) if (urls or ips) else "None"
    return has_html, network_data


def count_urls(text: str) -> int:
    """Cuenta URLs en texto."""
    return len(URL_PATTERN.findall(str(text)))
