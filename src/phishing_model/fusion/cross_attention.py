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
    ) -> None:
        super().__init__()
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
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
                o pooled a 1 token/rama según la variante).
            memory_key_padding_mask: [batch, n_memory_tokens] booleano, True = modalidad AUSENTE
                (natural o por dropout de entrenamiento) -- se excluye de la atención cruzada.

        Returns:
            [batch, seq_len, d_model] -- secuencia de texto ya fusionada con la modalidad.
        """
        # Salvaguarda: sin esto, softmax sobre un conjunto de Key/Value vacío
        # produce NaN. Ocurre para las filas sin NINGUNA modalidad no-textual
        # disponible (~13% del corpus) -- se relaja la máscara dejando un token
        # visible; su contenido (vector de padding/cero) no aporta información
        # real, pero evita que el forward completo colapse a NaN para esas filas.
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
    valid_mask = (~key_padding_mask).unsqueeze(-1).float()  # [batch, n_tokens, 1]
    summed = (tokens * valid_mask).sum(dim=1)
    counts = valid_mask.sum(dim=1).clamp(min=1.0)  # evita división por cero si (tras la
    # salvaguarda de forward) alguna fila quedara sin tokens válidos
    return summed / counts
