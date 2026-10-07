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

from dataclasses import dataclass

from jinja2 import Template

from phishing_model.config import NETWORK_CATEGORICAL_COLS
from phishing_pipeline.features.vectorizer import NETWORK_FEATURE_COLS, STRUCTURAL_FEATURE_COLS
from phishing_pipeline.logging_utils import get_logger

logger = get_logger(__name__)

# Nombres legibles para las features técnicas de estructura/red (candidato de
# atención, nivel de modalidad). Los candidatos SHAP/LIME operan a nivel de
# PALABRA -- para esos, el nombre "técnico" ya es la palabra literal del texto,
# no necesita traducción (ver `humanize_factor_name`).
FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "has_html": "la presencia de código HTML",
    "num_links": "la cantidad de enlaces",
    "num_images": "la cantidad de imágenes",
    "word_count": "la longitud del mensaje",
    "total_nodos_dom": "la complejidad estructural del mensaje (cantidad de elementos HTML)",
    "profundidad_dom": "el grado de anidamiento de la estructura HTML",
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

# Modalidades del reporte estructurado (IOV de R3.3: "Reporte estructurado POR
# MODALIDAD que traduzca el peso matemático en una justificación técnica
# comprensible"). Las listas de features se importan de la fuente canónica
# (phishing_pipeline.features.vectorizer / phishing_model.config) en vez de
# duplicarse aquí, para que no diverjan si el esquema cambia.
MODALITY_TEXT = "texto"
MODALITY_STRUCTURE = "estructura"
MODALITY_NETWORK = "red"

MODALITY_DISPLAY_NAMES: dict[str, str] = {
    MODALITY_TEXT: "Contenido textual (semántica del mensaje)",
    MODALITY_STRUCTURE: "Estructura HTML/DOM",
    MODALITY_NETWORK: "Metadatos de red y autenticación",
}

_STRUCTURAL_FEATURES = set(STRUCTURAL_FEATURE_COLS)
_NETWORK_FEATURES = set(NETWORK_FEATURE_COLS) | set(NETWORK_CATEGORICAL_COLS)


def missing_display_names() -> list[str]:
    """
    Devuelve las características canónicas que carecen de una etiqueta legible.

    Existe como guarda contra un fallo silencioso ya observado: al modificarse la
    composición de la rama estructural, las características entrantes quedaron sin
    entrada en `FEATURE_DISPLAY_NAMES`, de modo que la narrativa habría mostrado
    el identificador técnico crudo (`total_nodos_dom`) en lugar de una frase
    comprensible -- precisamente lo contrario de lo que exige R3.2. La
    comprobación se ejecuta dentro de `build_modality_report` y en las pruebas
    del módulo, para que un cambio futuro del esquema falle de forma visible.
    """
    canonical = list(STRUCTURAL_FEATURE_COLS) + list(NETWORK_FEATURE_COLS) + list(
        NETWORK_CATEGORICAL_COLS
    )
    return [c for c in canonical if c not in FEATURE_DISPLAY_NAMES]


def classify_factor_modality(name: str) -> str:
    """
    Asigna un factor de atribución a su modalidad.

    Los candidatos SHAP/LIME atribuyen a PALABRAS del texto (no a features
    tabulares), así que cualquier nombre que no esté en las listas canónicas
    de estructura/red se considera modalidad de texto -- es el comportamiento
    correcto, no un fallback perezoso: una palabra literal del correo es, por
    definición, evidencia de la modalidad textual.
    """
    if name in _STRUCTURAL_FEATURES:
        return MODALITY_STRUCTURE
    if name in _NETWORK_FEATURES:
        return MODALITY_NETWORK
    return MODALITY_TEXT

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


# --- Reporte estructurado POR MODALIDAD (IOV de R3.3) ---

MODALITY_REPORT_TEMPLATE = Template(
    "VEREDICTO: {{ label_text|upper }} (confianza {{ confidence_pct }}%)\n"
    "Técnica de explicabilidad: {{ xai_method_display }}\n"
    "\n"
    "{% for block in modality_blocks -%}"
    "[{{ block.display_name }}] — contribución agregada: {{ block.total_pct }}%\n"
    "{% if block.factors %}"
    "{% for f in block.factors %}  · {{ f.human_name }}: {{ f.pct }}%\n{% endfor %}"
    "{% else %}  · Sin evidencia atribuida a esta modalidad en esta predicción.\n"
    "{% endif %}"
    "\n"
    "{% endfor -%}"
    "SÍNTESIS: {{ synthesis }}\n"
)


def build_modality_report(explanation: ExplanationInput, max_factors_per_modality: int = 5) -> dict:
    """
    Agrupa las atribuciones por modalidad y devuelve la estructura de datos
    del reporte (sin renderizar). Satisface el IOV de R3.3, que exige un
    "reporte estructurado POR MODALIDAD" y no una lista plana de factores.

    Devuelve un dict con `modality_blocks` (uno por modalidad, en orden fijo
    texto → estructura → red, incluyendo las modalidades sin evidencia para
    que el analista vea explícitamente que se evaluaron y no aportaron) y los
    metadatos de la predicción.
    """
    faltantes = missing_display_names()
    if faltantes:
        logger.warning(
            "Características canónicas sin etiqueta legible en FEATURE_DISPLAY_NAMES: %s. "
            "La narrativa mostrará su identificador técnico crudo, lo que incumple el "
            "criterio de comprensibilidad de R3.2.",
            faltantes,
        )

    total_abs = sum(abs(w) for _, w in explanation.top_factors) or 1.0

    grouped: dict[str, list[tuple[str, float]]] = {
        MODALITY_TEXT: [],
        MODALITY_STRUCTURE: [],
        MODALITY_NETWORK: [],
    }
    for name, weight in explanation.top_factors:
        grouped[classify_factor_modality(name)].append((name, weight))

    modality_blocks = []
    for modality in (MODALITY_TEXT, MODALITY_STRUCTURE, MODALITY_NETWORK):
        factors = sorted(grouped[modality], key=lambda item: -abs(item[1]))
        total_pct = round(sum(abs(w) for _, w in factors) / total_abs * 100, 1)
        modality_blocks.append(
            {
                "modality": modality,
                "display_name": MODALITY_DISPLAY_NAMES[modality],
                "total_pct": total_pct,
                "factors": [
                    {
                        "name": name,
                        "human_name": humanize_factor_name(name),
                        "pct": round(abs(w) / total_abs * 100, 1),
                        "raw_weight": w,
                    }
                    for name, w in factors[:max_factors_per_modality]
                ],
                "n_factors_total": len(factors),
            }
        )

    dominant = max(modality_blocks, key=lambda b: b["total_pct"])
    label_text = "phishing" if explanation.label == 1 else "legítimo"
    if dominant["total_pct"] == 0:
        synthesis = "No se identificó evidencia atribuible a ninguna modalidad."
    else:
        synthesis = (
            f"La clasificación como {label_text} se apoya principalmente en la modalidad "
            f"'{dominant['display_name']}' ({dominant['total_pct']}% del peso total de la decisión)."
        )

    return {
        "label": explanation.label,
        "label_text": label_text,
        "confidence_pct": round(explanation.confidence * 100, 1),
        "xai_method": explanation.xai_method,
        "xai_method_display": XAI_METHOD_DISPLAY.get(explanation.xai_method, explanation.xai_method),
        "modality_blocks": modality_blocks,
        "dominant_modality": dominant["modality"],
        "synthesis": synthesis,
    }


def generate_modality_report(explanation: ExplanationInput, max_factors_per_modality: int = 5) -> str:
    """Renderiza el reporte estructurado por modalidad como texto legible para el analista SOC."""
    data = build_modality_report(explanation, max_factors_per_modality=max_factors_per_modality)
    return MODALITY_REPORT_TEMPLATE.render(**data)


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
