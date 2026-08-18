"""Tokenización de la rama de red: features léxicas de URL (continuas) +
SPF/DKIM/DMARC (categóricas, con "missing" explícito).

La parte continua reutiliza exactamente la misma técnica FT-Transformer-style
que `StructuralTokenizer` (proyección por-feature + embedding de identidad),
delegada como submódulo en vez de duplicar la lógica. La parte categórica usa
un `nn.Embedding` dedicado por columna categórica (spf/dkim/dmarc), indexado
por el vocabulario ya mapeado en `dataset.py` (`config.NETWORK_CATEGORICAL_VOCABS`).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from phishing_model.encoders.structural_tokenizer import StructuralTokenizer


class NetworkTokenizer(nn.Module):
    """Tokeniza la rama de red -> [batch, n_continuous + n_categorical, d_model]."""

    def __init__(
        self,
        n_continuous: int,
        categorical_vocabs: dict[str, dict[str, int]],
        d_model: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_continuous = n_continuous
        # Orden fijo: el orden de iteración de `categorical_vocabs` (dict de Python,
        # preserva orden de inserción) determina el orden esperado de las columnas
        # en el tensor `categorical` del forward -- en la práctica siempre
        # NETWORK_CATEGORICAL_COLS de config.py, pero no se hardcodea aquí.
        self.categorical_cols = list(categorical_vocabs.keys())

        self.continuous_tokenizer = StructuralTokenizer(n_continuous, d_model, dropout=dropout)
        self.categorical_embeddings = nn.ModuleDict(
            {col: nn.Embedding(len(vocab), d_model) for col, vocab in categorical_vocabs.items()}
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, continuous: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        """
        continuous: [batch, n_continuous] (escalado 0-1)
        categorical: [batch, len(categorical_vocabs)] (índices long, mismo orden que
            categorical_vocabs.keys() / NETWORK_CATEGORICAL_COLS)
        -> [batch, n_continuous + len(categorical_vocabs), d_model]
        """
        continuous_tokens = self.continuous_tokenizer(continuous)  # [batch, n_continuous, d_model]

        categorical_tokens = [
            self.categorical_embeddings[col](categorical[:, j])  # [batch, d_model]
            for j, col in enumerate(self.categorical_cols)
        ]
        categorical_tokens = torch.stack(categorical_tokens, dim=1)  # [batch, n_categorical, d_model]
        categorical_tokens = self.dropout(categorical_tokens)

        return torch.cat([continuous_tokens, categorical_tokens], dim=1)
