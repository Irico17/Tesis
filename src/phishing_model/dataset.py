"""Dataset PyTorch multimodal: texto + estructura + red, con máscaras de disponibilidad.

Contrato de batch (una fila por email) que consumen `encoders/` y `model.py`:

    input_ids            LongTensor [max_token_length]
    attention_mask        LongTensor [max_token_length]
    structural_continuous FloatTensor [n_structural_features]   (escalado 0-1)
    network_continuous    FloatTensor [n_network_continuous]    (escalado 0-1)
    network_categorical   LongTensor [n_network_categorical]    (índices de vocabulario, ver config.py)
    has_structure         FloatTensor []  (escalar 1.0/0.0 -- disponibilidad NATURAL, no dropout)
    has_network           FloatTensor []  (escalar 1.0/0.0 -- disponibilidad NATURAL, no dropout)
    label                 LongTensor []   (escalar 0/1)
    email_id              str             (trazabilidad; el DataLoader por defecto lo agrupa en una lista)

El dropout de modalidad (enmascarar artificialmente una rama disponible durante
el entrenamiento) NO ocurre aquí -- se aplica dentro de `model.py` a partir de
`has_structure`/`has_network`, para poder desactivarlo limpiamente en eval().
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

from phishing_model.config import (
    MAX_TOKEN_LENGTH,
    MODEL_DIR,
    NETWORK_CATEGORICAL_COLS,
    NETWORK_CATEGORICAL_VOCABS,
    ModelConfig,
)
from phishing_pipeline.features.vectorizer import (
    NETWORK_FEATURE_COLS,
    STRUCTURAL_FEATURE_COLS,
    compute_modality_availability,
)

REQUIRED_COLS = (
    ["clean_text", "label", "email_id"]
    + STRUCTURAL_FEATURE_COLS
    + NETWORK_FEATURE_COLS
    + NETWORK_CATEGORICAL_COLS
    + ["has_html", "network_indicators"]  # usados por compute_modality_availability
)

SCALER_PATH = MODEL_DIR / "feature_scalers.pkl"


def _encode_categorical_column(df: pd.DataFrame, col: str) -> np.ndarray:
    """Mapea una columna categórica (spf/dkim/dmarc) a índices de vocabulario, 'missing' por defecto."""
    vocab = NETWORK_CATEGORICAL_VOCABS[col]
    missing_idx = vocab["missing"]
    out = np.full(len(df), missing_idx, dtype="int64")
    values = df[col].tolist()
    for i, v in enumerate(values):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        key = str(v).strip().lower()
        out[i] = vocab.get(key, missing_idx)  # valor no reconocido (no debería pasar) -> missing, no crashea
    return out


class MultimodalPhishingDataset(Dataset):
    """Dataset PyTorch sobre un split (train/val/test) ya cargado como DataFrame."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: Any,
        structural_scaler: MinMaxScaler | None = None,
        network_scaler: MinMaxScaler | None = None,
        max_token_length: int = MAX_TOKEN_LENGTH,
        fit_scalers: bool = False,
    ) -> None:
        missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Faltan columnas requeridas en el DataFrame: {missing_cols}")

        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_token_length = max_token_length

        structural_raw = self.df[STRUCTURAL_FEATURE_COLS].fillna(0).astype(float).values
        network_raw = self.df[NETWORK_FEATURE_COLS].fillna(0).astype(float).values

        if fit_scalers:
            self.structural_scaler = MinMaxScaler().fit(structural_raw)
            self.network_scaler = MinMaxScaler().fit(network_raw)
        else:
            if structural_scaler is None or network_scaler is None:
                raise ValueError(
                    "fit_scalers=False requiere structural_scaler y network_scaler ya ajustados "
                    "(sobre el split de TRAIN únicamente, para no filtrar información de val/test)."
                )
            self.structural_scaler = structural_scaler
            self.network_scaler = network_scaler

        self.structural_scaled = self.structural_scaler.transform(structural_raw).astype("float32")
        self.network_scaled = self.network_scaler.transform(network_raw).astype("float32")

        self.network_categorical = np.stack(
            [_encode_categorical_column(self.df, col) for col in NETWORK_CATEGORICAL_COLS],
            axis=1,
        )

        availability = compute_modality_availability(self.df)
        self.has_structure = availability["has_structure_modality"].astype(bool).to_numpy()
        self.has_network = availability["has_network_modality"].astype(bool).to_numpy()

        self.texts = self.df["clean_text"].fillna("").astype(str).tolist()
        self.labels = self.df["label"].astype(int).to_numpy()
        self.email_ids = self.df["email_id"].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_token_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "structural_continuous": torch.from_numpy(self.structural_scaled[idx]),
            "network_continuous": torch.from_numpy(self.network_scaled[idx]),
            "network_categorical": torch.from_numpy(self.network_categorical[idx]),
            "has_structure": torch.tensor(float(self.has_structure[idx])),
            "has_network": torch.tensor(float(self.has_network[idx])),
            "label": torch.tensor(int(self.labels[idx]), dtype=torch.long),
            "email_id": self.email_ids[idx],
        }


def save_scalers(structural_scaler: MinMaxScaler, network_scaler: MinMaxScaler, path: Path | None = None) -> Path:
    """Persiste los scalers ajustados sobre train (para reutilizar en val/test/inferencia)."""
    path = path or SCALER_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"structural_scaler": structural_scaler, "network_scaler": network_scaler}, path)
    return path


def load_scalers(path: Path | None = None) -> tuple[MinMaxScaler, MinMaxScaler]:
    """Carga scalers previamente guardados con `save_scalers`."""
    path = path or SCALER_PATH
    artifact = joblib.load(path)
    return artifact["structural_scaler"], artifact["network_scaler"]


def make_synthetic_batch(batch_size: int, config: ModelConfig, device: str = "cpu") -> dict[str, torch.Tensor]:
    """
    Batch 100% sintético (sin tocar el dataset real) para el chequeo de forma
    de tensores requerido por R1.3 ("verificación estática de tensores").
    Usa un vocabulario de tokens mínimo válido para distilbert-base-uncased
    (ids en [0, 30521], vocab_size real del tokenizer) para no requerir carga
    del tokenizer HF en este chequeo puramente estructural.
    """
    vocab_size_placeholder = 30522  # tamaño real de distilbert-base-uncased-vocab
    g = torch.Generator().manual_seed(config_seed_for_synthetic())
    return {
        "input_ids": torch.randint(
            0, vocab_size_placeholder, (batch_size, config.max_token_length), generator=g
        ).to(device),
        "attention_mask": torch.ones(batch_size, config.max_token_length, dtype=torch.long).to(device),
        "structural_continuous": torch.rand(batch_size, config.n_structural_features, generator=g).to(device),
        "network_continuous": torch.rand(batch_size, config.n_network_continuous, generator=g).to(device),
        "network_categorical": torch.zeros(batch_size, config.n_network_categorical, dtype=torch.long).to(device),
        "has_structure": torch.randint(0, 2, (batch_size,), generator=g).float().to(device),
        "has_network": torch.randint(0, 2, (batch_size,), generator=g).float().to(device),
        "label": torch.randint(0, 2, (batch_size,), generator=g).to(device),
    }


def config_seed_for_synthetic() -> int:
    """Semilla fija y determinista para datos sintéticos (no usa Date.now/random real)."""
    return 42
