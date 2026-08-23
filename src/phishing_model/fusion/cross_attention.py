"""Stage 2: Cross-Attention real a nivel de token (texto = Query, modalidad = Key/Value).

Implementa el mecanismo central de R1.3 usando `nn.TransformerDecoderLayer`
estándar de PyTorch -- literalmente la arquitectura decoder de Vaswani et al.
(2017), ya citada en el Cap. 1.3 de la tesis, aplicada como fusión cross-modal
en vez de generación de secuencia. `tgt` = secuencia completa de hidden states
de texto (no un vector pooled), `memory` = tokens de modalidad (estructura+red,
ya auto-atendidos en Stage 1 vía `modality_encoder.ModalityEncoder`, o
colapsados a 1 token por rama para la variante `cross_attention_modality_level`
-- ver `model.py`, esta clase no distingue entre ambas variantes, solo consume
lo que se le pase como `memory`).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """Envuelve `nn.TransformerDecoder` (self-attn + cross-attn + FFN por capa)."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        n_layers: int = 1,
        norm_first: bool = True,
        activation: str = "gelu",
    ) -> None:
        """
        `norm_first=True` aplica la normalización por capa ANTES de cada
        subcomponente (pre-normalización) en lugar de después. La formulación
        original de 2017 normaliza después, y por eso exige un calentamiento
        cuidadoso de la tasa de aprendizaje para no divergir: con
        post-normalización, la magnitud del gradiente que llega a las capas
        inferiores crece con la profundidad. La pre-normalización deja una
        trayectoria residual sin normalizar de extremo a extremo y es la
        formulación estándar desde entonces. Aquí importa además por una razón
        propia de esta arquitectura: las capas de fusión se inicializan al azar y
        se conectan a un codificador preentrenado, de modo que la estabilidad de
        los primeros pasos condiciona cuánto se degradan los pesos aprendidos.

        `activation="gelu"` coincide con la activación de DistilBERT, el
        codificador que alimenta esta capa; el valor por defecto de PyTorch es
        ReLU, que introducía una discontinuidad de criterio dentro del modelo.
        """
        super().__init__()
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first,
            activation=activation,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers)

    def forward(
        self,
        text_tokens: torch.Tensor,
        text_key_padding_mask: torch.Tensor,
        memory_tokens: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            text_tokens: [batch, seq_len, d_model] -- secuencia de DistilBERT proyectada.
            text_key_padding_mask: [batch, seq_len] booleano, True = token de padding (a excluir).
            memory_tokens: [batch, n_memory_tokens, d_model] -- tokens de modalidad (Stage 1
                o pooled a 1 token/rama según la variante), con el token centinela de
                "sin modalidad" en la posición 0 (ver `model.py`).
            memory_key_padding_mask: [batch, n_memory_tokens] booleano, True = modalidad AUSENTE
                (natural o por enmascaramiento de entrenamiento) -- se excluye de la atención.

        Returns:
            [batch, seq_len, d_model] -- secuencia de texto ya fusionada con la modalidad.
        """
        # Salvaguarda numérica: sin ella, un softmax sobre un conjunto vacío de
        # Key/Value produce NaN. `model.py` antepone un token centinela que nunca
        # se enmascara, de modo que la condición ya no puede darse por esa vía;
        # esta comprobación se conserva para quien invoque la capa directamente,
        # y ahora es inocua porque la posición 0 es precisamente el centinela.
        #
        # Implementación branchless (sin `if tensor.any():`) -- ver justificación
        # completa en `modality_encoder.ModalityEncoder.forward`: un `if` sobre el
        # CONTENIDO de un tensor rompe la exportación a ONNX (`torch.export`),
        # error real encontrado durante el desarrollo de R2.3.
        fully_masked_memory = memory_key_padding_mask.all(dim=1, keepdim=True)  # [batch, 1]
        safe_col0 = memory_key_padding_mask[:, 0:1] & (~fully_masked_memory)
        memory_key_padding_mask = torch.cat([safe_col0, memory_key_padding_mask[:, 1:]], dim=1)

        return self.decoder(
            tgt=text_tokens,
            memory=memory_tokens,
            tgt_key_padding_mask=text_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )


def masked_mean_pool(tokens: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean-pool sobre tokens válidos (donde key_padding_mask es False).

    Args:
        tokens: [batch, n_tokens, d_model]
        key_padding_mask: [batch, n_tokens] booleano, True = posición a EXCLUIR del pooling.

    Returns:
        [batch, d_model]
    """
    valid_mask = (~key_padding_mask).unsqueeze(-1).to(tokens.dtype)  # [batch, n_tokens, 1]
    summed = (tokens * valid_mask).sum(dim=1)
    counts = valid_mask.sum(dim=1).clamp(min=1.0)  # evita división por cero si (tras la
    # salvaguarda de forward) alguna fila quedara sin tokens válidos
    return summed / counts
