"""
Candidato XAI "atención intrínseca" (R3.1): dos niveles complementarios.

1. Pesos de cross-attention de Stage 2 (`fusion/cross_attention.py`) a nivel de
   FEATURE de modalidad -- cuánto atendió el texto de un email a `num_links`,
   `spf_result=fail`, etc. Gracias a la fusión a nivel de token de Fase B, esto
   ya es una señal genuina por-feature, no un gate agregado de 2 vías.
2. Attention-rollout (Abnar & Zuidema, 2020) sobre las self-attention de
   DistilBERT, a nivel de PALABRA -- complementa el nivel de modalidad con
   saliencia léxica.

`nn.TransformerDecoderLayer` de PyTorch NO expone los pesos de cross-attention
en su forward de alto nivel (internamente llama a `need_weights=False` por la
ruta rápida). Se capturan sustituyendo temporalmente el `multihead_attn`
interno por un wrapper "espía" que fuerza `need_weights=True` -- verificado
empíricamente antes de escribir este módulo (ver sesión de desarrollo), sin
modificar el modelo ya validado en Fase B (Checkpoint B1): el wrapper se
retira al salir del context manager, sin efectos secundarios permanentes.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn

from phishing_pipeline.features.vectorizer import NETWORK_FEATURE_COLS, STRUCTURAL_FEATURE_COLS
from phishing_model.config import NETWORK_CATEGORICAL_COLS

# Orden exacto de los 19 tokens de modalidad tal como los concatena
# `model.py` (`torch.cat([structural_tokens, network_tokens], dim=1)`, y dentro
# de network_tokens: continuos primero, categóricos después -- ver
# `encoders/network_tokenizer.py`).
MODALITY_TOKEN_NAMES: list[str] = STRUCTURAL_FEATURE_COLS + NETWORK_FEATURE_COLS + NETWORK_CATEGORICAL_COLS


class _AttentionCapturingMHA(nn.Module):
    """Envuelve un `nn.MultiheadAttention` real, forzando `need_weights=True` para
    capturar los pesos que `TransformerDecoderLayer` normalmente descarta."""

    def __init__(self, mha: nn.MultiheadAttention) -> None:
        super().__init__()
        self.mha = mha
        self.last_weights: torch.Tensor | None = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = True,  # ignorado deliberadamente -- ver docstring del módulo
        attn_mask: torch.Tensor | None = None,
        average_attn_weights: bool = True,  # ignorado deliberadamente -- necesitamos por-head
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out, weights = self.mha(
            query,
            key,
            value,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            attn_mask=attn_mask,
            average_attn_weights=False,
            is_causal=is_causal,
        )
        self.last_weights = weights.detach()
        return out, weights


@contextmanager
def capture_cross_attention_weights(cross_attention_fusion: nn.Module):
    """
    Sustituye temporalmente los `multihead_attn` internos de
    `CrossAttentionFusion.decoder.layers` por wrappers que capturan pesos, y
    los restaura al salir (try/finally) -- sin efectos secundarios permanentes
    en el modelo, ni siquiera si el forward dentro del `with` lanza una excepción.
    """
    layers = cross_attention_fusion.decoder.layers
    originals = [layer.multihead_attn for layer in layers]
    wrappers = [_AttentionCapturingMHA(orig) for orig in originals]
    for layer, wrapper in zip(layers, wrappers):
        layer.multihead_attn = wrapper
    try:
        yield wrappers
    finally:
        for layer, original in zip(layers, originals):
            layer.multihead_attn = original


def extract_modality_attention(model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Corre un forward pass y devuelve los pesos de cross-attention de la ÚLTIMA
    capa de fusión (Stage 2), promediados sobre heads.

    Returns:
        [batch, seq_len_texto, n_tokens_modalidad] -- solo válido para
        `fusion_type` con cross-attention (lanza `ValueError` para
        `text_only`/`concat_late_fusion`, que no tienen este mecanismo).
    """
    if not hasattr(model, "cross_attention"):
        raise ValueError(
            f"El modelo con fusion_type={model.config.fusion_type} no tiene mecanismo de "
            "cross-attention del cual extraer pesos (solo válido para las variantes "
            "cross_attention_modality_level / cross_attention_token_level)."
        )
    model.eval()
    with capture_cross_attention_weights(model.cross_attention) as wrappers:
        with torch.no_grad():
            model(batch)
    last_layer_weights = wrappers[-1].last_weights
    if last_layer_weights is None:
        raise RuntimeError(
            "No se capturaron pesos de atención -- el forward no pasó por CrossAttentionFusion "
            "(¿fusion_type incorrecto o modelo modificado?)."
        )
    return last_layer_weights.mean(dim=1)  # promedio sobre heads -> [batch, tgt_len, src_len]


def extract_modality_level_summary(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    feature_names: list[str] | None = None,
) -> list[dict[str, float]]:
    """
    Resume los pesos de atención a nivel de FEATURE de modalidad (no por token
    de texto individual) -- cuánto "atendió" en promedio el texto de cada email
    a cada feature de estructura/red. Es exactamente el tipo de resumen que
    necesita el módulo de narrativa (R3.2): "el dominio recién creado pesó
    45%..." es una instancia de esto.

    Solo aplica a la variante `cross_attention_token_level` (19 tokens de
    modalidad, uno por feature) -- para `cross_attention_modality_level` (2
    tokens agregados) el resumen ya es trivial y no necesita esta función.
    """
    feature_names = feature_names or MODALITY_TOKEN_NAMES
    attn = extract_modality_attention(model, batch)  # [batch, tgt_len, n_modality_tokens]
    if attn.shape[-1] != len(feature_names):
        raise ValueError(
            f"extract_modality_level_summary espera {len(feature_names)} tokens de modalidad "
            f"(variante cross_attention_token_level), pero el modelo produjo {attn.shape[-1]} -- "
            "¿se pasó un modelo con fusion_type=cross_attention_modality_level? Ese caso no "
            "necesita este resumen (ya son solo 2 tokens: estructura y red)."
        )

    attention_mask = batch["attention_mask"].float()  # [batch, tgt_len]
    weighted = (attn * attention_mask.unsqueeze(-1)).sum(dim=1)
    weights_per_email = weighted / attention_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    # Normaliza a que sume 1 por fila -- interpretación directa como "% de peso".
    weights_per_email = weights_per_email / weights_per_email.sum(dim=1, keepdim=True).clamp(min=1e-8)

    results = []
    for row in weights_per_email:
        pairs = sorted(zip(feature_names, row.tolist()), key=lambda item: -item[1])
        results.append({name: round(val, 4) for name, val in pairs})
    return results


def extract_text_saliency_rollout(
    text_encoder: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """
    Attention-rollout (Abnar & Zuidema, 2020) sobre las self-attention layers de
    DistilBERT -- saliencia a nivel de PALABRA, complemento opcional del resumen
    a nivel de modalidad (más grueso). No requiere modificar `TextEncoder`: HF
    expone `output_attentions=True` directamente en el backbone.

    NOTA (bug real encontrado y corregido, no hipotético): en `transformers`
    5.x, `AutoModel.from_pretrained` carga DistilBERT con `attn_implementation
    ="sdpa"` por defecto (kernels fusionados de PyTorch) -- con esa
    implementación, `output_attentions=True` NO lanza error pero devuelve una
    tupla VACÍA en vez de los pesos reales (el kernel fusionado no los
    materializa), fallando en silencio. Se fuerza `attn_implementation="eager"`
    en el backbone antes de pedir attentions -- matemáticamente equivalente a
    SDPA (mismo mecanismo de atención, sin cambio de resultados), el efecto
    lateral es que las llamadas subsecuentes al backbone quedan en modo eager
    (ligeramente más lento, no incorrecto) -- aceptable porque esta función
    solo se usa en inferencia post-entrenamiento (XAI), nunca durante el
    entrenamiento en sí.
    """
    if getattr(text_encoder.backbone.config, "_attn_implementation", None) != "eager":
        text_encoder.backbone.set_attn_implementation("eager")

    text_encoder.eval()
    with torch.no_grad():
        outputs = text_encoder.backbone(
            input_ids=input_ids, attention_mask=attention_mask, output_attentions=True
        )
    attentions = outputs.attentions  # tupla de [batch, n_heads, seq, seq], una por capa
    if not attentions:
        raise RuntimeError(
            "El backbone no devolvió attentions incluso tras forzar attn_implementation="
            "'eager' -- reabrir esta investigación si vuelve a ocurrir, no es el fallo ya "
            "diagnosticado (sdpa)."
        )

    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    rollout = torch.eye(seq_len, device=device).unsqueeze(0).expand(batch_size, -1, -1).clone()
    identity = torch.eye(seq_len, device=device).unsqueeze(0)
    for layer_attn in attentions:
        avg_heads = layer_attn.mean(dim=1)  # promedio sobre heads -> [batch, seq, seq]
        avg_heads = avg_heads + identity  # conexión residual (Abnar & Zuidema)
        avg_heads = avg_heads / avg_heads.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        rollout = torch.bmm(avg_heads, rollout)

    return rollout[:, 0, :]
