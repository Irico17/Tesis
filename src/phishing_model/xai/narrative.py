"""
Módulo de narrativa (R3.2): traduce atribuciones XAI (de cualquiera de los 3
candidatos de R3.1 -- atención, SHAP, LIME) en texto de justificación en
lenguaje natural, siguiendo el ejemplo ya escrito en la tesis (Cap. 2.2.3.2):
"el dominio recién creado pesó 45% en la decisión de bloqueo y la 'táctica de
urgencia en el asunto' un 30%".

Diseñado para ser AGNÓSTICO a qué candidato XAI ganó la matriz de decisión de
R3.1: los tres producen una lista de (nombre_factor, peso) que se traduce con
la misma plantilla Jinja2 -- ver `ExplanationInput` y los adaptadores
`from_attention_summary`/`from_shap_result`/`from_lime_result` más abajo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from jinja2 import Template

# Nombres legibles para las features técnicas de estructura/red (candidato de
# atención, nivel de modalidad). Los candidatos SHAP/LIME operan a nivel de
# PALABRA -- para esos, el nombre "técnico" ya es la palabra literal del texto,
# no necesita traducción (ver `humanize_factor_name`).
FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "has_html": "la presencia de código HTML",
    "num_links": "la cantidad de enlaces",
    "num_images": "la cantidad de imágenes",
    "has_form": "la presencia de un formulario",
    "has_iframe": "la presencia de un iframe oculto",
    "has_javascript": "la presencia de código JavaScript",
    "word_count": "la longitud del mensaje",
    "has_ip_link": "un enlace con dirección IP en vez de dominio",
    "num_urls_metadata": "la cantidad de URLs reportadas",
    "url_max_length": "la longitud de las URLs",
    "url_max_digit_ratio": "la proporción de dígitos en las URLs",
    "url_max_subdomain_count": "la cantidad de subdominios en las URLs",
    "url_has_at_symbol": "el uso del símbolo '@' en una URL (técnica de ofuscación)",
    "url_has_shortener": "el uso de un acortador de enlaces",
    "url_has_suspicious_tld": "un dominio de nivel superior (TLD) frecuentemente asociado a phishing",
    "url_domain_max_entropy": "un dominio con apariencia generada aleatoriamente",
    "spf_result": "el resultado de verificación SPF del remitente",
    "dkim_result": "el resultado de verificación DKIM del remitente",
    "dmarc_result": "el resultado de verificación DMARC del remitente",
}

XAI_METHOD_DISPLAY: dict[str, str] = {
    "attention": "los pesos de atención intrínseca del modelo",
    "shap": "atribuciones GradientSHAP",
    "lime": "aproximaciones locales LIME",
}

NARRATIVE_TEMPLATE = Template(
    "Este correo fue clasificado como {{ label_text }} con una confianza del "
    "{{ confidence_pct }}%. "
    "{% if top_factors %}Los factores que más influyeron en la decisión fueron: "
    "{% for name, pct in top_factors %}{{ name }} (peso {{ pct }}%)"
    "{% if not loop.last %}, {% endif %}{% endfor %}. "
    "{% else %}No se identificó un factor individual dominante. {% endif %}"
    "{% if xai_method_display %}Explicación generada mediante {{ xai_method_display }}.{% endif %}"
)


@dataclass
class ExplanationInput:
    """Formato unificado que consume la plantilla, sin importar qué candidato XAI lo produjo."""

    label: int  # 0=legítimo, 1=phishing
    confidence: float  # 0-1, probabilidad de la clase predicha
    top_factors: list[tuple[str, float]]  # (nombre técnico o palabra, peso; puede ser negativo)
    xai_method: str  # "attention" | "shap" | "lime"
    max_factors: int = 3


def humanize_factor_name(name: str) -> str:
    """Traduce un nombre técnico de feature a frase legible; si es una palabra del texto
    (candidatos SHAP/LIME), la cita directamente entre comillas."""
    return FEATURE_DISPLAY_NAMES.get(name, f'la palabra "{name}"')


def generate_narrative(explanation: ExplanationInput) -> str:
    """Genera el párrafo de justificación en lenguaje natural."""
    label_text = "phishing" if explanation.label == 1 else "legítimo"
    confidence_pct = round(explanation.confidence * 100, 1)

    top = sorted(explanation.top_factors, key=lambda item: -abs(item[1]))[: explanation.max_factors]
    top_display = [(humanize_factor_name(name), round(abs(weight) * 100, 1)) for name, weight in top]

    xai_method_display = XAI_METHOD_DISPLAY.get(explanation.xai_method, explanation.xai_method)

    return NARRATIVE_TEMPLATE.render(
        label_text=label_text,
        confidence_pct=confidence_pct,
        top_factors=top_display,
        xai_method_display=xai_method_display,
    )


# --- Adaptadores: convierten la salida cruda de cada candidato de R3.1 a ExplanationInput ---


def from_attention_summary(
    modality_summary: dict[str, float], label: int, confidence: float, max_factors: int = 3
) -> ExplanationInput:
    """Adaptador para `xai.attention_explainer.extract_modality_level_summary()` (una fila)."""
    return ExplanationInput(
        label=label,
        confidence=confidence,
        top_factors=list(modality_summary.items()),
        xai_method="attention",
        max_factors=max_factors,
    )


def from_shap_result(shap_result: dict, label: int, confidence: float, max_factors: int = 3) -> ExplanationInput:
    """Adaptador para `xai.shap_explainer.explain_shap()` (claves 'tokens'/'attributions')."""
    pairs = list(zip(shap_result["tokens"], shap_result["attributions"]))
    return ExplanationInput(
        label=label, confidence=confidence, top_factors=pairs, xai_method="shap", max_factors=max_factors
    )


def from_lime_result(lime_result: dict, label: int, confidence: float, max_factors: int = 3) -> ExplanationInput:
    """Adaptador para `xai.lime_explainer.explain_lime()` (claves 'words'/'attributions')."""
    pairs = list(zip(lime_result["words"], lime_result["attributions"]))
    return ExplanationInput(
        label=label, confidence=confidence, top_factors=pairs, xai_method="lime", max_factors=max_factors
    )
