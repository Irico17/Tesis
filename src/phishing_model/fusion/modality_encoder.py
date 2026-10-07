"""Stage 1 (jerárquico): auto-atención entre los tokens de estructura + red.

Antes de que el texto los atienda (Stage 2, `cross_attention.py`), los tokens
de estructura y red se dejan "verse entre sí" vía un `nn.TransformerEncoderLayer`
estándar -- esto es lo que hace la fusión genuinamente jerárquica (Tabla 2 de la
tesis: "Fusión Jerárquica o Atención Cruzada") y no solo una concatenación.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ModalityEncoder(nn.Module):
    """Envuelve `nn.TransformerEncoderLayer` con máscara de padding por disponibilidad."""

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
        """`norm_first` y `activation`: ver la justificación en
        `cross_attention.CrossAttentionFusion.__init__`. Ambas capas de fusión
        comparten criterio para que la comparación entre variantes no confunda el
        efecto de la arquitectura con el de su formulación interna."""
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first,
            activation=activation,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(
        self,
        modality_tokens: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            modality_tokens: [batch, n_modality_tokens, d_model] -- estructura + red concatenados.
            key_padding_mask: [batch, n_modality_tokens] booleano, True = posición AUSENTE
                (se excluye de la atención). Construida en `model.py` a partir de las
                máscaras de disponibilidad NATURAL (`has_structure`/`has_network`) combinadas
                con el dropout de modalidad de entrenamiento -- ver `model.py`.

        Nota sobre filas totalmente enmascaradas: si TODOS los tokens de una fila están
        marcados como ausentes (key_padding_mask todo True para esa fila), `nn.MultiheadAttention`
        internamente produciría NaN (softmax sobre un conjunto vacío). `model.py` antepone a la
        secuencia un token centinela de "sin modalidad" que nunca se enmascara, de modo que la
        condición no puede darse por esa vía; la salvaguarda de abajo se conserva para quien
        invoque la capa directamente y resulta inocua, porque la posición 0 que relaja es
        precisamente la del centinela.

        Implementación SIN bifurcación condicional sobre valores de tensor (`if
        tensor.any():`): el exportador ONNX de PyTorch (`torch.export`, usado por
        `torch.onnx.export` desde torch 2.x) traza el grafo simbólicamente y no
        puede manejar ramas Python que dependen del CONTENIDO de un tensor (solo
        de su forma) -- un `if` así rompe la exportación (error real encontrado y
        corregido durante el desarrollo de R2.3). La forma branchless de abajo logra el
        mismo efecto con álgebra de tensores pura, exportable a ONNX.
        """
        fully_masked = key_padding_mask.all(dim=1, keepdim=True)  # [batch, 1]
        # Fuerza la posición 0 a "visible" (False) exactamente cuando la fila está
        # totalmente enmascarada; para el resto de filas, la posición 0 queda
        # exactamente como estaba (sin efecto).
        safe_col0 = key_padding_mask[:, 0:1] & (~fully_masked)
        key_padding_mask = torch.cat([safe_col0, key_padding_mask[:, 1:]], dim=1)

        return self.encoder(modality_tokens, src_key_padding_mask=key_padding_mask)
