"""
Candidato XAI "SHAP" (R3.1) -- en la práctica, Integrated Gradients aplicado a
nivel de capa de embeddings del backbone de texto (`LayerIntegratedGradients`
de Captum), el patrón estándar para modelos HuggingFace tipo BERT ("Captum +
BERT interpretability"). No es SHAP exacto (KernelSHAP/DeepSHAP), pero
Integrated Gradients converge al mismo objeto matemático (valores de Shapley
bajo un juego cooperativo continuo, Sundararajan et al. 2017) y es el enfoque
que escala a un backbone de 512 tokens sin explosión combinatoria -- de ahí
que las guías oficiales de Captum lo usen como sustituto de SHAP para
transformers.

No modifica la firma de `forward()` del modelo ya validado (Fase B): se
construye un `forward_func` externo que reconstruye el batch dict y llama a
`model(batch)`, y Captum ataca la capa de embeddings del backbone de
DistilBERT (`model.text_encoder.backbone.embeddings`) en vez de los
`input_ids` discretos directamente (Captum no puede diferenciar respecto a
índices enteros).
"""

from __future__ import annotations

import time
from typing import Any

import torch
from captum.attr import LayerIntegratedGradients


def _build_forward_func(model: torch.nn.Module):
    """
    Cierra sobre `model` y devuelve una función que Captum puede llamar
    repetidamente con `input_ids` (el input atribuido) más el resto de campos
    del batch como `additional_forward_args`, en el orden fijo documentado en
    el contrato: attention_mask, structural_continuous, network_continuous,
    network_categorical, has_structure, has_network.

    Devuelve probabilidades (softmax de los logits) -- Captum necesita una
    salida diferenciable respecto a la capa de embeddings interceptada por
    `LayerIntegratedGradients`, no respecto a `input_ids` en sí (los ids
    discretos nunca participan del grafo de autograd; lo que Captum perturba
    internamente es la salida de la capa de embeddings a la que apunta).
    """

    def forward_func(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        structural_continuous: torch.Tensor,
        network_continuous: torch.Tensor,
        network_categorical: torch.Tensor,
        has_structure: torch.Tensor,
        has_network: torch.Tensor,
    ) -> torch.Tensor:
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "structural_continuous": structural_continuous,
            "network_continuous": network_continuous,
            "network_categorical": network_categorical,
            "has_structure": has_structure,
            "has_network": has_network,
        }
        logits = model(batch)
        return torch.softmax(logits, dim=-1)

    return forward_func


def explain_shap(
    model: torch.nn.Module,
    tokenizer: Any,
    row_batch: dict[str, torch.Tensor],
    target_class: int = 1,
    n_steps: int = 20,
) -> dict[str, Any]:
    """
    Atribución por token vía Integrated Gradients a nivel de capa de
    embeddings (`LayerIntegratedGradients`), sustituto estándar de SHAP para
    backbones tipo BERT.

    `row_batch`: batch de una sola fila (batch_size=1) con las claves
    input_ids/attention_mask/structural_continuous/network_continuous/
    network_categorical/has_structure/has_network -- mismo formato que
    produce `MultimodalPhishingDataset.__getitem__` pero con dimensión de
    batch al frente (usar `.unsqueeze(0)` si hace falta antes de llamar).

    Returns:
        {
            "tokens": [str, ...],               # tokens de texto no-padding
            "attributions": [float, ...],        # misma longitud que "tokens"
            "wall_time_seconds": float,
            "convergence_delta": float | None,
        }
    """
    t0 = time.time()

    model.eval()

    input_ids = row_batch["input_ids"]
    attention_mask = row_batch["attention_mask"]
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)

    device = input_ids.device
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        # Fallback defensivo: algunos tokenizers no definen pad_token_id explícito.
        pad_token_id = tokenizer.eos_token_id or 0

    baseline_ids = torch.full_like(input_ids, fill_value=pad_token_id, device=device)

    additional_forward_args = (
        attention_mask,
        row_batch["structural_continuous"].to(device),
        row_batch["network_continuous"].to(device),
        row_batch["network_categorical"].to(device),
        row_batch["has_structure"].to(device),
        row_batch["has_network"].to(device),
    )

    forward_func = _build_forward_func(model)
    lig = LayerIntegratedGradients(forward_func, model.text_encoder.backbone.embeddings)

    attributions, convergence_delta = lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        additional_forward_args=additional_forward_args,
        target=target_class,
        n_steps=n_steps,
        return_convergence_delta=True,
    )

    # attributions: [1, seq_len, embedding_dim] -> score escalar por token
    token_scores = attributions.sum(dim=-1).squeeze(0)  # [seq_len]
    norm = torch.norm(token_scores)
    if norm.item() > 0:
        token_scores = token_scores / norm

    mask = attention_mask.squeeze(0).bool()
    kept_scores = token_scores[mask].tolist()

    all_tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())
    kept_tokens = [tok for tok, keep in zip(all_tokens, mask.tolist()) if keep]

    delta_value: float | None
    if convergence_delta is not None:
        delta_value = float(convergence_delta.detach().abs().mean().item())
    else:
        delta_value = None

    wall_time_seconds = time.time() - t0

    return {
        "tokens": kept_tokens,
        "attributions": kept_scores,
        "wall_time_seconds": wall_time_seconds,
        "convergence_delta": delta_value,
    }
