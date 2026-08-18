"""
R3.3 — chequeo secundario de fidelidad narrativa, ACOTADO a verificación
NUMÉRICA (no un juicio abierto de "¿es buena esta explicación?"), para evitar
la circularidad que señala la tesis: un sistema de IA opaco no debe validar la
CALIDAD de la explicación de otro; solo verifica que las afirmaciones
numéricas/de ranking del texto generado (R3.2) correspondan fielmente a los
valores del vector de atribución que lo originó (R3.1).

El criterio PRIMARIO de R3.3 sigue siendo el panel humano SUS/PSSUQ (ver Fase
E0, `doc/validacion_xai/`) -- este módulo es un complemento automatizado y
escalable, no un sustituto (tal como especifica la tesis).

Este módulo NO se acopla a un proveedor de LLM específico ni requiere
credenciales para poder desarrollarse/probarse: recibe `llm_call_fn` inyectado
por quien lo use (una función `str -> str` que envuelva la llamada real a la
API del LLM que el tesista tenga disponible).

CONTRATO DE UNIDADES (verificado con pruebas reales, no asumido -- ver sesión
de desarrollo): `narrative.generate_narrative()` recibe pesos como FRACCIÓN
0-1 y los multiplica por 100 solo para mostrarlos en el texto ("peso 45.0%").
El `attribution` que se pasa a `run_fidelity_check()` debe estar en el MISMO
espacio que el texto muestra -- es decir, YA multiplicado por 100 (45.0, no
0.45) -- porque este módulo compara directamente contra los números que
aparecen literalmente en la narrativa. Si se pasa el vector en fracción 0-1
por error, todas las comparaciones fallan sistemáticamente (falso negativo
generalizado, no un error aleatorio).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from typing import Callable

VERIFICATION_PROMPT_TEMPLATE = """Tu única tarea es verificar afirmaciones NUMÉRICAS. No evalúes si la explicación es "buena", "clara" o "útil" -- eso lo hace un panel humano por separado, no tú.

Se te da:
1. Un vector de atribución (factor -> peso en %) calculado por una técnica XAI.
2. Un texto en lenguaje natural generado a partir de ese vector.

IMPORTANTE: el texto puede mencionar un porcentaje de "confianza" de la
predicción del modelo (ej. "clasificado con una confianza del 92%") -- ESE
número NO es parte del vector de atribución y NO debe verificarse contra él;
es la probabilidad de la clase predicha, un valor distinto. Verifica
ÚNICAMENTE los porcentajes que el texto atribuye explícitamente a un FACTOR
o palabra específica (ej. "el dominio sospechoso (peso 45%)").

Verifica ÚNICAMENTE:
- Si el texto menciona un porcentaje para un factor específico, ¿corresponde (tolerancia ±2 puntos porcentuales) al peso real de ese factor en el vector?
- Si el texto afirma que un factor pesó MÁS o MENOS que otro, ¿es cierto según el vector?
- Si el texto menciona un factor que NO está en el vector de atribución, señálalo como error.

No comentes nada más. No opines sobre la calidad, claridad o utilidad del texto.

Vector de atribución (factor -> peso %):
{attribution_json}

Texto generado a verificar:
"{narrative_text}"

Responde ÚNICAMENTE con JSON válido, sin texto adicional antes o después, con este formato exacto:
{{"all_claims_correct": true o false, "errors": ["descripción breve de cada error numérico encontrado"]}}
Si no hay errores, "errors" debe ser una lista vacía [].
"""


@dataclass
class FidelityCheckResult:
    all_claims_correct: bool
    errors: list[str]
    raw_llm_response: str


def build_verification_prompt(attribution: dict[str, float], narrative_text: str) -> str:
    """Construye el prompt de verificación acotada -- no expone el texto a un juicio abierto."""
    return VERIFICATION_PROMPT_TEMPLATE.format(
        attribution_json=json.dumps(attribution, indent=2, ensure_ascii=False),
        narrative_text=narrative_text,
    )


def _extract_json_block(text: str) -> str:
    """Algunos LLMs envuelven el JSON en texto/markdown pese a la instrucción -- se
    recorta al primer bloque {...} balanceado como fallback antes de fallar."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text


def run_fidelity_check(
    attribution: dict[str, float], narrative_text: str, llm_call_fn: Callable[[str], str]
) -> FidelityCheckResult:
    """
    Corre la verificación numérica sobre UN ejemplo.

    `llm_call_fn`: función `prompt (str) -> respuesta cruda (str)` que envuelve
    la llamada real al LLM disponible (Anthropic, OpenAI, etc.) -- inyectada,
    no hardcodeada, para no acoplar el repo a un proveedor ni requerir
    credenciales en tiempo de desarrollo/pruebas. Ver `mock_rule_based_llm_call`
    para pruebas deterministas sin LLM real.
    """
    prompt = build_verification_prompt(attribution, narrative_text)
    raw_response = llm_call_fn(prompt)
    try:
        parsed = json.loads(_extract_json_block(raw_response))
        return FidelityCheckResult(
            all_claims_correct=bool(parsed.get("all_claims_correct", False)),
            errors=list(parsed.get("errors", [])),
            raw_llm_response=raw_response,
        )
    except (json.JSONDecodeError, AttributeError) as exc:
        return FidelityCheckResult(
            all_claims_correct=False,
            errors=[f"Respuesta del LLM no parseable como JSON: {exc}"],
            raw_llm_response=raw_response,
        )


def run_fidelity_check_batch(
    examples: list[tuple[dict[str, float], str]], llm_call_fn: Callable[[str], str]
) -> dict:
    """Aplica `run_fidelity_check` sobre varios (atribución, narrativa) y agrega el % correcto."""
    results = [run_fidelity_check(attr, text, llm_call_fn) for attr, text in examples]
    n_correct = sum(1 for r in results if r.all_claims_correct)
    return {
        "n_examples": len(results),
        "n_correct": n_correct,
        "pct_correct": round(n_correct / len(results) * 100, 2) if results else None,
        "results": [asdict(r) for r in results],
    }


def mock_rule_based_llm_call(prompt: str) -> str:
    """
    Mock DETERMINISTA (sin LLM real) para probar el harness durante desarrollo:
    extrae los porcentajes mencionados en el texto generado vía regex y los
    compara directamente contra el vector de atribución del propio prompt, sin
    usar ningún modelo de lenguaje. Útil para CI/pruebas locales; el chequeo
    real de la tesis debe usar un LLM real vía `llm_call_fn`.

    Solo extrae porcentajes en el patrón "(peso X%)" (el formato exacto que usa
    `narrative.NARRATIVE_TEMPLATE` para atribuciones de factor) -- IGNORA
    deliberadamente el porcentaje de "confianza" de la predicción, que no es
    parte del vector de atribución y no debe verificarse contra él (ver
    instrucción explícita en `VERIFICATION_PROMPT_TEMPLATE`). Un LLM real
    infiere esta distinción del lenguaje natural; este mock, al ser puramente
    basado en reglas/regex, necesita el patrón explícito.
    """
    attribution_match = re.search(r"\{[\s\S]*?\}(?=\s*Texto)", prompt)
    text_match = re.search(r'Texto generado a verificar:\s*"(.*)"', prompt, re.DOTALL)
    if not attribution_match or not text_match:
        return json.dumps({"all_claims_correct": False, "errors": ["mock: no se pudo parsear el prompt"]})

    attribution = json.loads(attribution_match.group(0))
    narrative = text_match.group(1)

    mentioned_pcts = {float(m) for m in re.findall(r"peso\s+(\d+(?:\.\d+)?)%", narrative, re.IGNORECASE)}
    real_pcts = {round(abs(v), 1) for v in attribution.values()}

    errors = []
    for pct in mentioned_pcts:
        if not any(abs(pct - real) <= 2.0 for real in real_pcts):
            errors.append(f"Porcentaje {pct}% mencionado en el texto no corresponde a ningún peso real (±2pp)")

    return json.dumps({"all_claims_correct": len(errors) == 0, "errors": errors})
