"""MultimodalPhishingClassifier — ensambla encoders + tokenizers + fusión (Stage 1+2) + cabeza.

Orquesta las 4 variantes de `FusionType` (switch de ablación de R2.2) bajo una
única interfaz `forward(batch) -> logits [batch, 2]`, y aplica el dropout de
modalidad durante entrenamiento (mecanismo central de manejo de modalidades
ausentes, ver plan Fase B / docstring de `_apply_modality_dropout`).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from phishing_model.config import ModelConfig, FusionType, NETWORK_CATEGORICAL_VOCABS
from phishing_model.encoders.network_tokenizer import NetworkTokenizer
from phishing_model.encoders.structural_tokenizer import StructuralTokenizer
from phishing_model.encoders.text_encoder import TextEncoder
from phishing_model.fusion.cross_attention import CrossAttentionFusion, masked_mean_pool
from phishing_model.fusion.modality_encoder import ModalityEncoder


class MultimodalPhishingClassifier(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model

        self.text_encoder = TextEncoder(
            config.text_model_name,
            d,
            freeze=config.freeze_text_encoder,
            grad_checkpointing=config.grad_checkpointing,
        )

        needs_modality_branches = config.fusion_type != FusionType.TEXT_ONLY
        if needs_modality_branches:
            self.structural_tokenizer = StructuralTokenizer(
                config.n_structural_features, d, dropout=config.dropout
            )
            self.network_tokenizer = NetworkTokenizer(
                config.n_network_continuous, NETWORK_CATEGORICAL_VOCABS, d
            )

        self.uses_stage1 = config.fusion_type == FusionType.CROSS_ATTENTION_TOKEN_LEVEL
        self.uses_cross_attention = config.fusion_type in (
            FusionType.CROSS_ATTENTION_TOKEN_LEVEL,
            FusionType.CROSS_ATTENTION_MODALITY_LEVEL,
        )
        if self.uses_stage1:
            self.modality_encoder = ModalityEncoder(
                d, config.n_heads, config.dim_feedforward, config.dropout, n_layers=config.n_fusion_layers
            )
        if self.uses_cross_attention:
            self.cross_attention = CrossAttentionFusion(
                d, config.n_heads, config.dim_feedforward, config.dropout, n_layers=config.n_fusion_layers
            )

        if config.fusion_type == FusionType.TEXT_ONLY:
            head_in = d
        elif config.fusion_type == FusionType.CONCAT_LATE_FUSION:
            head_in = d * 3  # texto + estructura + red, cada uno pooled independientemente
        else:  # ambas variantes de cross-attention devuelven texto fusionado pooled
            head_in = d

        self.classifier = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Dropout(config.dropout),
            nn.Linear(head_in, 2),
        )

        self.modality_dropout_prob = config.modality_dropout_prob

    def _apply_modality_dropout(
        self, has_structure: torch.Tensor, has_network: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Dropout de modalidad -- SOLO durante entrenamiento (`self.training`), nunca en
        eval()/inferencia real. Con probabilidad `modality_dropout_prob`, por fila y por
        rama, enmascara artificialmente una rama que SÍ está disponible (no toca las que
        ya están ausentes de forma natural -- nada que enmascarar ahí). Esto evita que el
        modelo aprenda a usar la mera DISPONIBILIDAD de una rama como atajo hacia la
        identidad de la fuente (la preocupación explícita del asesor de tesis: "el modelo
        aprenda patrones de disponibilidad de datos... de origen del dataset"), forzándolo
        a depender del CONTENIDO de cada rama cuando está presente.

        Devuelve las máscaras de disponibilidad EFECTIVAS (tras dropout); las máscaras
        NATURALES (`has_structure`/`has_network` originales) nunca se modifican in-place.
        """
        if not self.training or self.modality_dropout_prob <= 0:
            return has_structure, has_network

        drop_structure = (torch.rand_like(has_structure) < self.modality_dropout_prob) & (has_structure > 0.5)
        drop_network = (torch.rand_like(has_network) < self.modality_dropout_prob) & (has_network > 0.5)

        effective_structure = has_structure.clone()
        effective_network = has_network.clone()
        effective_structure[drop_structure] = 0.0
        effective_network[drop_network] = 0.0
        return effective_structure, effective_network

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Devuelve logits [batch, 2] (sin softmax -- CrossEntropyLoss/FocalLoss lo aplican internamente)."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        text_key_padding_mask = attention_mask == 0  # True = token de padding a excluir

        text_tokens = self.text_encoder(input_ids, attention_mask)  # [batch, seq_len, d_model]

        if self.config.fusion_type == FusionType.TEXT_ONLY:
            pooled_text = masked_mean_pool(text_tokens, text_key_padding_mask)
            return self.classifier(pooled_text)

        has_structure, has_network = self._apply_modality_dropout(
            batch["has_structure"], batch["has_network"]
        )

        structural_tokens = self.structural_tokenizer(batch["structural_continuous"])  # [batch, 7, d]
        network_tokens = self.network_tokenizer(
            batch["network_continuous"], batch["network_categorical"]
        )  # [batch, 12, d]

        # Máscara de padding por rama: True = excluir. Se replica el flag de la fila
        # (escalar) a lo largo de todos los tokens de esa rama -- si la rama está
        # ausente, TODOS sus tokens se excluyen en bloque, no individualmente.
        struct_pad = (has_structure < 0.5).unsqueeze(1).expand(-1, structural_tokens.shape[1])
        net_pad = (has_network < 0.5).unsqueeze(1).expand(-1, network_tokens.shape[1])

        if self.config.fusion_type == FusionType.CONCAT_LATE_FUSION:
            pooled_text = masked_mean_pool(text_tokens, text_key_padding_mask)
            pooled_struct = masked_mean_pool(structural_tokens, struct_pad)
            pooled_net = masked_mean_pool(network_tokens, net_pad)
            combined = torch.cat([pooled_text, pooled_struct, pooled_net], dim=-1)
            return self.classifier(combined)

        if self.config.fusion_type == FusionType.CROSS_ATTENTION_MODALITY_LEVEL:
            # Variante ligera: cada rama se colapsa a 1 token (mean-pool) antes de la
            # cross-attention -- comparación directa contra la fusión a nivel de token
            # (ablación de R2.2: ¿la riqueza de tokens individuales realmente aporta?).
            struct_token = masked_mean_pool(structural_tokens, struct_pad).unsqueeze(1)
            net_token = masked_mean_pool(network_tokens, net_pad).unsqueeze(1)
            memory = torch.cat([struct_token, net_token], dim=1)  # [batch, 2, d]
            memory_pad = torch.stack([has_structure < 0.5, has_network < 0.5], dim=1)  # [batch, 2]
            fused_text = self.cross_attention(text_tokens, text_key_padding_mask, memory, memory_pad)
            pooled = masked_mean_pool(fused_text, text_key_padding_mask)
            return self.classifier(pooled)

        # CROSS_ATTENTION_TOKEN_LEVEL (variante principal, ver plan Fase B)
        modality_tokens = torch.cat([structural_tokens, network_tokens], dim=1)  # [batch, 19, d]
        modality_pad = torch.cat([struct_pad, net_pad], dim=1)  # [batch, 19]
        modality_tokens = self.modality_encoder(modality_tokens, modality_pad)  # Stage 1
        fused_text = self.cross_attention(
            text_tokens, text_key_padding_mask, modality_tokens, modality_pad
        )  # Stage 2
        pooled = masked_mean_pool(fused_text, text_key_padding_mask)
        return self.classifier(pooled)

    def get_optimizer_param_groups(self, backbone_lr: float, head_lr: float) -> list[dict]:
        """
        Grupos de parámetros para AdamW con LR discriminativo: backbone de DistilBERT
        a `backbone_lr` (más bajo), resto de la arquitectura (tokenizers, fusión,
        cabeza de clasificación) a `head_lr` (más alto) -- ver justificación en el plan,
        Fase B, sección "Rama de texto".
        """
        groups = self.text_encoder.get_param_groups(backbone_lr, head_lr)
        other_params = [
            p
            for name, module in self.named_children()
            if name != "text_encoder"
            for p in module.parameters()
            if p.requires_grad
        ]
        if other_params:
            groups.append({"params": other_params, "lr": head_lr})
        return groups
