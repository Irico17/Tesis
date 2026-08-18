"""
Candidato XAI "LIME" (R3.1): `lime.lime_text.LimeTextExplainer` sobre la rama
de texto del modelo multimodal.

LIME perturba el texto ORIGINAL (palabras, no sub-tokens de WordPiece/BPE) --
por eso primero se decodifica `input_ids` de vuelta a un string legible con
`tokenizer.decode(...)`, y luego cada perturbación que genera LIME internamente
(quitando palabras al azar) se re-tokeniza con el mismo tokenizer antes de
pasarla al modelo. Las ramas no-textuales (estructura/red) se mantienen FIJAS
en los valores de la fila original durante toda la explicación -- LIME en este
módulo solo ataca la modalidad de texto, así que perturbar también estructura/
red mezclaría dos fuentes de variación en una sola atribución.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from lime.lime_text import LimeTextExplainer

CLASS_NAMES = ["legítimo", "phishing"]


def _decode_original_text(tokenizer: Any, row_batch: dict[str, torch.Tensor]) -> str:
    input_ids = row_batch["input_ids"]
    attention_mask = row_batch["attention_mask"]
    if input_ids.dim() == 2:
        input_ids = input_ids.squeeze(0)
    if attention_mask.dim() == 2:
        attention_mask = attention_mask.squeeze(0)
    real_ids = input_ids[attention_mask.bool()]
    return tokenizer.decode(real_ids, skip_special_tokens=True)


def _build_predict_proba(
    model: torch.nn.Module,
    tokenizer: Any,
    row_batch: dict[str, torch.Tensor],
    max_token_length: int = 512,
):
    """
    Cierra sobre el modelo/tokenizer/fila original y devuelve una función
    `predict_proba(list[str]) -> np.ndarray [n, 2]` compatible con la API de
    LIME (`LimeTextExplainer.explain_instance` la llama repetidamente con
    listas de tamaño variable -- normalmente varias decenas de textos
    perturbados a la vez).

    Las ramas no-textuales se fijan en los valores de `row_batch` (fila
    original) y se repiten (`.expand`) para igualar el tamaño del batch de
    textos perturbados en cada llamada.
    """
    device = row_batch["input_ids"].device

    def _row(key: str) -> torch.Tensor:
        t = row_batch[key]
        if t.dim() == 1 and key not in ("has_structure", "has_network"):
            t = t.unsqueeze(0)
        elif t.dim() == 0:
            t = t.unsqueeze(0)
        return t.to(device)

    structural_row = _row("structural_continuous")  # [1, 7]
    network_continuous_row = _row("network_continuous")  # [1, 9]
    network_categorical_row = _row("network_categorical")  # [1, 3]
    has_structure_row = _row("has_structure")  # [1]
    has_network_row = _row("has_network")  # [1]

    def predict_proba(texts: list[str]) -> np.ndarray:
        n = len(texts)
        encoded = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_token_length,
            return_tensors="pt",
        )
        batch = {
            "input_ids": encoded["input_ids"].to(device),
            "attention_mask": encoded["attention_mask"].to(device),
            "structural_continuous": structural_row.expand(n, -1),
            "network_continuous": network_continuous_row.expand(n, -1),
            "network_categorical": network_categorical_row.expand(n, -1),
            "has_structure": has_structure_row.expand(n),
            "has_network": has_network_row.expand(n),
        }
        model.eval()
        with torch.no_grad():
            logits = model(batch)
            probs = torch.softmax(logits, dim=-1)
        return probs.cpu().numpy()

    return predict_proba


def explain_lime(
    model: torch.nn.Module,
    tokenizer: Any,
    row_batch: dict[str, torch.Tensor],
    target_class: int = 1,
    num_features: int = 15,
    num_samples: int = 200,
) -> dict[str, Any]:
    """
    Explica la predicción para `target_class` con LIME a nivel de palabra
    sobre el texto decodificado de `row_batch["input_ids"]`.

    `num_samples` controla cuántas perturbaciones evalúa LIME (cada una es un
    forward pass del modelo completo) -- 200 es el valor por defecto usado
    para los resultados reportados; para pruebas de humo rápidas en CPU se
    puede bajar (p.ej. 50) pasando el argumento explícitamente.

    Returns:
        {
            "words": [str, ...],
            "attributions": [float, ...],   # peso LIME, mismo orden que "words"
            "wall_time_seconds": float,
        }
    """
    t0 = time.time()

    original_text = _decode_original_text(tokenizer, row_batch)
    predict_proba = _build_predict_proba(model, tokenizer, row_batch)

    explainer = LimeTextExplainer(class_names=CLASS_NAMES)
    explanation = explainer.explain_instance(
        original_text,
        predict_proba,
        labels=(target_class,),
        num_features=num_features,
        num_samples=num_samples,
    )

    # as_list(label=...) devuelve [(palabra, peso), ...] ya ordenado por LIME
    # por importancia absoluta descendente -- se conserva ese orden tal cual.
    pairs = explanation.as_list(label=target_class)
    # `str(...)` explícito: LIME devuelve `numpy.str_` (subclase de `str`, pero
    # se normaliza aquí para no sorprender a consumidores que hagan `type() is str`).
    words = [str(w) for w, _ in pairs]
    attributions = [float(score) for _, score in pairs]

    wall_time_seconds = time.time() - t0

    return {
        "words": words,
        "attributions": attributions,
        "wall_time_seconds": wall_time_seconds,
    }
