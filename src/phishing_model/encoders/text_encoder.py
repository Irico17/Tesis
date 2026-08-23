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

        # La dimensión oculta se lee del modelo cargado y no de la constante del
        # módulo. Fijarla a 768 obligaba a que cualquier codificador alternativo
        # tuviera exactamente esa anchura: RoBERTa base o BERT multilingüe la
        # cumplen por casualidad, pero un modelo grande (1024) construía una
        # proyección de la forma equivocada y el error aparecía más tarde, ya
        # dentro del primer paso hacia adelante. Leerla del modelo permite
        # sustituir el codificador sin tocar el código.
        hidden = getattr(self.backbone.config, "hidden_size", None) or TEXT_HIDDEN_SIZE
        self.hidden_size = int(hidden)

        if d_model != self.hidden_size:
            self.projection: nn.Module = nn.Linear(self.hidden_size, d_model)
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
        from phishing_model.model import sin_decaimiento  # import diferido: evita ciclo

        cuerpo = [(n, p) for n, p in self.backbone.named_parameters() if p.requires_grad]
        groups = []
        con = [p for n, p in cuerpo if not sin_decaimiento(n)]
        sin = [p for n, p in cuerpo if sin_decaimiento(n)]
        if con:
            groups.append({"params": con, "lr": backbone_lr})
        # Sesgos y normalización por capa quedan exentos del decaimiento de peso
        # (ver `phishing_model.model.sin_decaimiento`).
        if sin:
            groups.append({"params": sin, "lr": backbone_lr, "weight_decay": 0.0})
        proyeccion = [(n, p) for n, p in self.projection.named_parameters() if p.requires_grad]
        con_proy = [p for n, p in proyeccion if not sin_decaimiento(n)]
        sin_proy = [p for n, p in proyeccion if sin_decaimiento(n)]
        if con_proy:
            groups.append({"params": con_proy, "lr": head_lr})
        if sin_proy:
            groups.append({"params": sin_proy, "lr": head_lr, "weight_decay": 0.0})
        return groups
