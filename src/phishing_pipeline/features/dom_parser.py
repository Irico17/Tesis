"""Parser DOM multimodal (refactor celda 22 del notebook)."""

from __future__ import annotations

import re
import warnings
from typing import Any

from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

# Filtrar warning de BeautifulSoup
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

IP_PATTERN = re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b")


def parse_email_multimodal(body_raw: str) -> dict[str, Any]:
    """
    Extrae features DOM y clean_text desde body_raw.

    Returns dict con num_links, num_images, has_form, has_iframe,
    has_javascript, has_ip_link, clean_text, has_html.
    """
    if not isinstance(body_raw, str):
        body_raw = str(body_raw) if body_raw is not None else ""

    result: dict[str, Any] = {
        "num_links": 0,
        "num_images": 0,
        "has_form": 0,
        "has_iframe": 0,
        "has_javascript": 0,
        "has_ip_link": 0,
        "clean_text": "",
        "has_html": 0,
        "processing_status": "ok",
        "processing_error": None,
    }

    if not body_raw.strip():
        result["processing_status"] = "skipped"
        return result

    try:
        soup = BeautifulSoup(body_raw, "lxml")

        result["num_links"] = len(soup.find_all("a"))
        result["num_images"] = len(soup.find_all("img"))
        result["has_form"] = 1 if soup.find_all("form") else 0
        result["has_iframe"] = 1 if soup.find_all("iframe") else 0
        result["has_javascript"] = 1 if soup.find_all("script") else 0

        for link in soup.find_all("a", href=True):
            if IP_PATTERN.search(link["href"]):
                result["has_ip_link"] = 1
                break

        clean_text = soup.get_text(separator=" ", strip=True)
        clean_text = re.sub(r"\s+", " ", clean_text)
        result["clean_text"] = clean_text
        result["has_html"] = 1 if re.search(r"<[^>]+>", body_raw) else 0

        # Si no hay HTML, usar texto plano
        if not result["clean_text"] and body_raw:
            result["clean_text"] = re.sub(r"\s+", " ", body_raw.strip())

    except Exception as exc:
        result["processing_status"] = "error"
        result["processing_error"] = str(exc)
        result["clean_text"] = re.sub(r"\s+", " ", body_raw.strip())

    return result


def extract_html_dom_stats(body_html: str) -> dict[str, float]:
    """Stats DOM adicionales para spam-genuine (celda 6)."""
    if not body_html or not isinstance(body_html, str):
        return {
            "total_nodos_dom": 0,
            "profundidad_dom": 0,
            "num_scripts_js": 0,
            "longitud_js": 0,
            "densidad_js_porcentaje": 0.0,
        }

    soup = BeautifulSoup(body_html, "lxml")
    scripts = soup.find_all("script")
    longitud_js = sum(len(s.string or "") for s in scripts)
    longitud_html = len(body_html)

    def _depth(tag, d=1):
        children = tag.find_all(recursive=False) if hasattr(tag, "find_all") else []
        if not children:
            return d
        return max(_depth(c, d + 1) for c in children)

    return {
        "total_nodos_dom": len(soup.find_all()),
        "profundidad_dom": _depth(soup),
        "num_scripts_js": len(scripts),
        "longitud_js": longitud_js,
        "densidad_js_porcentaje": (longitud_js / longitud_html * 100) if longitud_html else 0.0,
    }
