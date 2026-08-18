"""Encoder de texto: DistilBERT, secuencia completa de hidden states (no pooled).

Expone `[batch, seq_len, d_model]` -- toda la secuencia de tokens, no un solo
vector agregado -- para que Stage 2 (fusion/cross_attention.py) pueda hacer
cross-attention real a nivel de token entre cada palabra del correo y cada
token de modalidad (estructura/red).
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from transformers import AutoModel

from phishing_model.config import TEXT_HIDDEN_SIZE

logger = logging.getLogger(__name__)


class TextEncoder(nn.Module):
    """DistilBERT (u otro backbone HF compatible con AutoModel) -> secuencia proyectada a d_model."""

    def __init__(
        self,
        model_name: str,
        d_model: int,
        freeze: bool = False,
        grad_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # AutoModel (no AutoModelForSequenceClassification): necesitamos los
        # hidden states crudos de la secuencia completa, no logits pooled.
        self.backbone = AutoModel.from_pretrained(model_name)

        if d_model != TEXT_HIDDEN_SIZE:
            self.projection: nn.Module = nn.Linear(TEXT_HIDDEN_SIZE, d_model)
        else:
            self.projection = nn.Identity()

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

        if grad_checkpointing:
            try:
                self.backbone.gradient_checkpointing_enable()
            except Exception as exc:  # pragma: no cover - depende de la versión de transformers
                logger.warning(
                    "No se pudo habilitar gradient checkpointing en el backbone de texto (%s): %s",
                    model_name,
                    exc,
                )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Devuelve [batch, seq_len, d_model] -- secuencia completa de hidden states proyectada."""
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self.projection(outputs.last_hidden_state)

    def get_param_groups(self, backbone_lr: float, head_lr: float) -> list[dict]:
        """Grupos de parámetros para AdamW: backbone a `backbone_lr`, proyección a `head_lr`.

        Si `d_model == TEXT_HIDDEN_SIZE` la proyección es `nn.Identity()` (sin
        parámetros) -- ese grupo se omite del resultado en vez de pasarle una
        lista vacía a AdamW (que no lo acepta).
        """
        groups = [
            {
                "params": [p for p in self.backbone.parameters() if p.requires_grad],
                "lr": backbone_lr,
            }
        ]
        projection_params = list(self.projection.parameters())
        if projection_params:
            groups.append({"params": projection_params, "lr": head_lr})
        return groups
