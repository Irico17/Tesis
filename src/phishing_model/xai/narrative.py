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


def es_palabra_legible(cadena: str) -> bool:
    """Si el factor citado es una palabra que un analista puede leer.

    Los correos arrastran fronteras MIME, identificadores codificados en base 64 y
    signos sueltos, y el codificador los convierte en tokens como cualquier otro.
    El discriminante es la longitud junto con la proporción de letras y la
    presencia de vocales: una cadena de cuarenta y ocho caracteres cae por larga,
    una secuencia de consonantes por impronunciable, y un término alfanumérico
    corto, como el nombre de un protocolo, se conserva. Mide legibilidad y no
    pertinencia.
    """
    if not (2 <= len(cadena) <= 20):
        return False
    limpia = cadena.replace("-", "").replace("'", "")
    letras = sum(1 for c in limpia if c.isalpha())
    if not limpia or letras / len(limpia) < 0.6:
        return False
    return any(v in cadena.lower() for v in "aeiouáéíóú")


def _tipo_de_token(cadena: str) -> str:
    """Qué es un token que no es una palabra legible, dicho para el analista.

    La explicación es fiel a lo que el modelo atiende, y eso incluye elementos que
    no son lenguaje. Citarlos por su cadena («cm9kcmlnby1m…») no le dice nada al
    analista; citarlos por su tipo le dice que el modelo se apoyó en el formato
    del mensaje. El peso no cambia, y la cadena se conserva, abreviada, para que
    el reporte siga siendo auditable.
    """
    if not any(c.isalnum() for c in cadena):
        return "un signo de puntuación"
    if all(c.isdigit() or c in ".,:/-" for c in cadena):
        return "una cifra"
    if any(c.isalpha() and not c.isascii() for c in cadena) and \
            not any(c.isascii() and c.isalpha() for c in cadena):
        return "un texto en otro alfabeto"
    if cadena.isascii() and cadena.isalpha() and len(cadena) <= 5:
        return "una abreviatura"
    return "un fragmento codificado"


def humanize_factor_name(name: str) -> str:
    """Traduce un nombre técnico de feature a frase legible; si es una palabra del texto
    (candidatos SHAP/LIME), la cita directamente entre comillas, y si es un token que
    no es una palabra legible, lo nombra por su tipo."""
    if name in FEATURE_DISPLAY_NAMES:
        return FEATURE_DISPLAY_NAMES[name]
    if es_palabra_legible(name):
        return f'la palabra "{name}"'
    visible = name if len(name) <= 12 else name[:10] + "…"
    return f'{_tipo_de_token(name)} "{visible}"'


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
    "{% if block.total_pct > 0 %}"
    "[{{ block.display_name }}] — contribución agregada: {{ block.total_pct }}%\n"
    "{% for f in block.factors if f.pct > 0 %}  · {{ f.human_name }}: {{ f.pct }}%\n{% endfor %}"
    "{% else %}"
    "[{{ block.display_name }}] — sin peso en esta decisión\n"
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


# --- Reporte en lenguaje sencillo, para quien lee la alerta ---
#
# El reporte técnico de arriba nombra la técnica, usa términos como «contribución
# agregada» y convive con una narrativa que expresa los pesos en otra escala (el
# peso crudo de atención, 1.4%, frente a su parte dentro del bloque, 29.9%). Las
# primeras pruebas del panel mostraron que eso confunde. Esta versión presenta lo
# mismo, los tres bloques y sus factores, con una sola escala y sin jerga. No
# cambia ninguna cifra: se construye a partir del mismo `build_modality_report`.

NOMBRES_SENCILLOS: dict[str, str] = {
    MODALITY_TEXT: "Texto del correo",
    MODALITY_STRUCTURE: "Diseño del correo (HTML)",
    MODALITY_NETWORK: "Enlaces y verificación del remitente",
}


def _factor_sencillo(human_name: str) -> str:
    """'la palabra "inbox"' pasa a «inbox»; 'una cifra "0514"' a 'la cifra «0514»'."""
    import re as _re
    m = _re.match(r'^la palabra "(.*)"$', human_name)
    if m:
        return f"«{m.group(1)}»"
    texto = _re.sub(r'"(.*)"', r"«\1»", human_name)
    return _re.sub(r"^un[a]? ", lambda x: "el " if x.group(0) == "un " else "la ", texto)


def reporte_sencillo(estructurado: dict) -> str:
    """El reporte por modalidad en lenguaje sencillo, a partir de su estructura."""
    clase = ("PHISHING (intento de fraude)" if estructurado["label"] == 1
             else "LEGÍTIMO (correo normal)")
    lineas = [f"Decisión del sistema: {clase}, con {estructurado['confidence_pct']}% de confianza.",
              "", "¿En qué se basó?"]
    for b in estructurado["modality_blocks"]:
        nombre = NOMBRES_SENCILLOS[b["modality"]]
        factores = [f for f in b["factors"] if f["pct"] > 0]
        if b["total_pct"] <= 0 or not factores:
            lineas.append(f"• {nombre}: no influyó en esta decisión.")
            continue
        lineas.append(f"• {nombre}: {b['total_pct']}% del peso. Lo que más pesó:")
        lineas += [f"    {i}. {_factor_sencillo(f['human_name'])}: {f['pct']}%"
                   for i, f in enumerate(factores, 1)]
    return "\n".join(lineas)


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
