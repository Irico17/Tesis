"""Extracción de URLs, IPs y features de red."""

from __future__ import annotations

import math
import re
from urllib.parse import urlparse

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
IP_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")

# Acortadores de URL conocidos (lista pequeña, versionada en código).
_KNOWN_SHORTENERS = {
    "bit.ly",
    "tinyurl.com",
    "goo.gl",
    "t.co",
    "ow.ly",
    "is.gd",
    "buff.ly",
    "rebrand.ly",
    "cutt.ly",
    "shorturl.at",
}

# TLDs frecuentemente reportados como abusados en campañas de phishing
# (lista pequeña, versionada en código).
_SUSPICIOUS_TLDS = {
    "xyz",
    "top",
    "club",
    "work",
    "support",
    "click",
    "link",
    "zip",
    "review",
    "country",
    "kim",
    "science",
    "gq",
    "tk",
    "ml",
}

_LEXICAL_FEATURE_DEFAULTS: dict[str, float] = {
    "url_max_length": 0,
    "url_max_digit_ratio": 0.0,
    "url_max_subdomain_count": 0,
    "url_has_at_symbol": 0,
    "url_has_shortener": 0,
    "url_has_suspicious_tld": 0,
    "url_domain_max_entropy": 0.0,
}


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


def _clean_url(url: str) -> str:
    """
    Recorta puntuación final pegada a la URL (bug de URL_PATTERN: usa \\S+
    sin límite, así que captura puntos/paréntesis/comillas de cierre que en
    realidad pertenecen a la puntuación del texto, no a la URL).
    """
    return url.rstrip(".,;:)]}'\"")


def _is_ip_literal(host: str) -> bool:
    return bool(IP_PATTERN.fullmatch(host))


def _shannon_entropy(s: str) -> float:
    """Entropía de Shannon estándar de un string, en bits."""
    if not s:
        return 0.0
    freq: dict[str, int] = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    length = len(s)
    entropy = 0.0
    for count in freq.values():
        p = count / length
        entropy -= p * math.log2(p)
    return entropy


def extract_lexical_url_features(text: str) -> dict:
    """
    Extrae features léxicas deterministas de las URLs presentes en `text`.

    Agrega tomando el MÁXIMO entre todas las URLs del mensaje (un phishing
    suele mezclar un link benigno con uno malicioso, así que el máximo
    captura la señal más sospechosa). Los booleanos son 1 si CUALQUIER URL
    cumple la condición.

    Returns:
        dict con las 7 claves: url_max_length, url_max_digit_ratio,
        url_max_subdomain_count, url_has_at_symbol, url_has_shortener,
        url_has_suspicious_tld, url_domain_max_entropy.
    """
    text = str(text) if text is not None else ""
    raw_urls = URL_PATTERN.findall(text)
    urls = [_clean_url(u) for u in raw_urls if u]

    if not urls:
        return dict(_LEXICAL_FEATURE_DEFAULTS)

    max_length = 0
    max_digit_ratio = 0.0
    max_subdomain_count = 0
    has_at_symbol = 0
    has_shortener = 0
    has_suspicious_tld = 0
    max_entropy = 0.0

    for url in urls:
        if not url:
            continue

        length = len(url)
        if length > max_length:
            max_length = length

        digit_count = sum(1 for ch in url if ch.isdigit())
        digit_ratio = digit_count / length if length else 0.0
        if digit_ratio > max_digit_ratio:
            max_digit_ratio = digit_ratio

        if "@" in url:
            has_at_symbol = 1

        parse_target = url
        if parse_target.lower().startswith("www."):
            parse_target = "http://" + parse_target

        try:
            parsed = urlparse(parse_target)
            host = (parsed.netloc or parsed.hostname or "").lower()
            # Quitar puerto/credenciales embebidas del netloc para aislar el dominio.
            host = host.split("@")[-1]
            host = host.split(":")[0]
        except Exception:
            continue

        if not host:
            continue

        if _is_ip_literal(host):
            subdomain_count = 0
        else:
            subdomain_count = max(host.count(".") - 1, 0)
        if subdomain_count > max_subdomain_count:
            max_subdomain_count = subdomain_count

        if host in _KNOWN_SHORTENERS:
            has_shortener = 1

        if not _is_ip_literal(host) and "." in host:
            tld = host.rsplit(".", 1)[-1]
            if tld in _SUSPICIOUS_TLDS:
                has_suspicious_tld = 1

        domain_entropy = _shannon_entropy(host)
        if domain_entropy > max_entropy:
            max_entropy = domain_entropy

    return {
        "url_max_length": max_length,
        "url_max_digit_ratio": max_digit_ratio,
        "url_max_subdomain_count": max_subdomain_count,
        "url_has_at_symbol": has_at_symbol,
        "url_has_shortener": has_shortener,
        "url_has_suspicious_tld": has_suspicious_tld,
        "url_domain_max_entropy": max_entropy,
    }
