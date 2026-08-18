"""Tokenización FT-Transformer-style para features tabulares continuas.

Cada feature escalar (ya escalada 0-1 por `dataset.py`) se proyecta a `d_model`
con SU PROPIO peso lineal (una proyección independiente escalar->vector por
columna, no una proyección compartida `n_features -> d_model`) y se le suma un
embedding de "identidad de feature", siguiendo la tokenización de datos
tabulares de FT-Transformer (Gorishniy et al., 2021, *Revisiting Deep Learning
Models for Tabular Data*). Así el modelo puede distinguir "este token es
`num_links`" de "este token es `word_count`" incluso si comparten el mismo
valor escalar.

Reutilizado también por `NetworkTokenizer` para su parte continua (ver
`network_tokenizer.py`).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class StructuralTokenizer(nn.Module):
    """Tokeniza `[batch, n_features]` (continuo, 0-1) -> `[batch, n_features, d_model]`."""

    def __init__(self, n_features: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model

        # Proyección por-feature: weight[i] y bias[i] son la proyección escalar->vector
        # dedicada a la columna i. Vectorizado en forward() vía broadcasting.
        self.weight = nn.Parameter(torch.empty(n_features, d_model))
        self.bias = nn.Parameter(torch.zeros(n_features, d_model))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Embedding de "identidad de feature" -- una fila por columna, sumada a
        # cada token para que el modelo sepa qué feature está viendo.
        self.feature_embedding = nn.Embedding(n_features, d_model)
        self.register_buffer("feature_ids", torch.arange(n_features), persistent=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, n_features] (continuo, escalado 0-1) -> [batch, n_features, d_model]."""
        # [batch, n_features, 1] * [1, n_features, d_model] + [1, n_features, d_model]
        tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        tokens = tokens + self.feature_embedding(self.feature_ids).unsqueeze(0)
        return self.dropout(tokens)
