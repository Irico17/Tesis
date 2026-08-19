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
from torch.utils.data import Dataset, Sampler

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


# Columnas de CONTEO con distribución de cola larga: se les aplica log1p antes de
# escalar. Sin esta transformación, un único valor atípico arruina la columna --
# verificado empíricamente sobre el corpus: `word_count` tiene máximo 3,525,778 y
# percentil 99 de 1,423, de modo que MinMaxScaler comprimía el 99% de las
# observaciones por debajo de 0.0004, dejando la característica indistinguible de
# cero. log1p es monótona (preserva el orden), está definida en 0 y acota el
# efecto de los extremos. Ver doc/REVISION_CODIGO_MODELO.md, hallazgo 2.
LONG_TAIL_COUNT_COLS = frozenset(
    {
        "word_count",
        "num_links",
        "num_images",
        "num_urls_metadata",
        "url_max_length",
        "total_nodos_dom",
    }
)


def _log_transform_counts(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Aplica log1p a las columnas de conteo de cola larga y devuelve la matriz."""
    out = df.copy()
    for col in cols:
        if col in LONG_TAIL_COUNT_COLS:
            out[col] = np.log1p(out[col].clip(lower=0))
    return out.values


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
        pad_to_max_length: bool = False,
    ) -> None:
        missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Faltan columnas requeridas en el DataFrame: {missing_cols}")

        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_token_length = max_token_length
        # pad_to_max_length=False (por defecto): cada elemento se tokeniza SIN
        # relleno y el relleno se aplica por lote en `collate_multimodal`, hasta
        # la longitud del elemento más largo del lote. Verificado sobre el
        # corpus: la mediana de longitud es de 19 palabras y el percentil 75 de
        # 28, frente a un máximo de 512 -- rellenar siempre a 512 desperdiciaba
        # más de un orden de magnitud de cómputo en atención, que escala de
        # forma cuadrática con la longitud. Ver doc/REVISION_CODIGO_MODELO.md,
        # hallazgo 3.
        #
        # pad_to_max_length=True se usa donde la longitud debe ser fija: la
        # exportación a ONNX traza el grafo con la dimensión de secuencia fija
        # (solo el lote es dinámico), de modo que la inferencia sobre el modelo
        # exportado requiere secuencias de longitud `max_token_length`.
        self.pad_to_max_length = pad_to_max_length

        structural_raw = _log_transform_counts(
            self.df[STRUCTURAL_FEATURE_COLS].fillna(0).astype(float), STRUCTURAL_FEATURE_COLS
        )
        network_raw = _log_transform_counts(
            self.df[NETWORK_FEATURE_COLS].fillna(0).astype(float), NETWORK_FEATURE_COLS
        )

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
        # Proxy de longitud para LengthGroupedSampler: contar palabras es
        # ~1000x más barato que tokenizar el corpus completo por adelantado y
        # correlaciona muy fuerte con el número de sub-palabras. Se acota a
        # max_token_length porque más allá el texto se trunca igual.
        self.approx_lengths = [
            min(len(t.split()), self.max_token_length) for t in self.texts
        ]
        self.labels = self.df["label"].astype(int).to_numpy()
        self.email_ids = self.df["email_id"].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length" if self.pad_to_max_length else False,
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


def collate_multimodal(batch: list[dict[str, Any]], pad_token_id: int = 0) -> dict[str, Any]:
    """
    Agrupa elementos en un lote aplicando relleno dinámico a la longitud del
    elemento más largo del propio lote (en lugar de a `max_token_length`).

    OBLIGATORIA como `collate_fn` del DataLoader cuando el Dataset se construye
    con `pad_to_max_length=False` (el valor por defecto): la función de
    agrupación estándar de PyTorch no puede apilar secuencias de longitud
    distinta y fallaría. Las columnas no textuales tienen longitud fija y se
    apilan directamente.

    El relleno de `input_ids` usa el identificador del token de relleno y el de
    `attention_mask` usa ceros, de modo que el modelo excluye esas posiciones
    del cálculo de atención exactamente igual que con el relleno fijo.
    """
    max_len = max(item["input_ids"].shape[0] for item in batch)

    input_ids, attention_mask = [], []
    for item in batch:
        ids = item["input_ids"]
        mask = item["attention_mask"]
        pad_len = max_len - ids.shape[0]
        if pad_len > 0:
            ids = torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=ids.dtype)])
            mask = torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)])
        input_ids.append(ids)
        attention_mask.append(mask)

    collated: dict[str, Any] = {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
    }
    for key in (
        "structural_continuous",
        "network_continuous",
        "network_categorical",
        "has_structure",
        "has_network",
        "label",
    ):
        collated[key] = torch.stack([item[key] for item in batch])
    collated["email_id"] = [item["email_id"] for item in batch]
    return collated


class LengthGroupedSampler(Sampler[list[int]]):
    """
    Muestreador por lotes que agrupa ejemplos de longitud similar, conservando
    aleatoriedad entre épocas.

    Motivación (medida sobre el corpus, no estimada): la longitud de los correos
    es de cola muy larga -- percentil 50 de 48 unidades de sub-palabra, percentil
    90 de 399 y un 8.3% truncado a 512. Con lotes formados al azar, basta un
    correo largo para que TODO el lote se rellene hasta esa longitud, de modo que
    el relleno dinámico por sí solo apenas reduce el costo (factor 1.18x medido).
    Agrupando por longitud, el costo de atención -- cuadrático en la longitud de
    secuencia -- se reduce en un factor de **7.63x** sobre el mismo corpus.

    Para no introducir sesgo por correlación entre la composición del lote y la
    longitud, se aplica el esquema estándar de "megalotes": se permutan los
    índices al azar, se agrupan en bloques de `batch_size * megabatch_mult`, se
    ordena por longitud DENTRO de cada bloque, y finalmente se permuta el orden
    de los lotes resultantes. Así cada época produce lotes distintos y el orden
    de presentación sigue siendo aleatorio, pero cada lote es internamente
    homogéneo en longitud.
    """

    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        shuffle: bool = True,
        megabatch_mult: int = 50,
        seed: int = 0,
    ) -> None:
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.megabatch_size = batch_size * megabatch_mult
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Cambia la semilla efectiva para que cada época produzca lotes distintos."""
        self.epoch = epoch

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.lengths)
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(n, generator=g).tolist()
        else:
            indices = list(range(n))

        batches: list[list[int]] = []
        for start in range(0, n, self.megabatch_size):
            megabatch = indices[start : start + self.megabatch_size]
            megabatch.sort(key=lambda i: self.lengths[i], reverse=True)
            for b_start in range(0, len(megabatch), self.batch_size):
                batches.append(megabatch[b_start : b_start + self.batch_size])

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch + 1)
            order = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in order]

        yield from batches


def make_collate_fn(tokenizer: Any):
    """Devuelve una `collate_fn` ligada al token de relleno del tokenizador dado."""
    pad_token_id = getattr(tokenizer, "pad_token_id", 0) or 0

    def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        return collate_multimodal(batch, pad_token_id=pad_token_id)

    return _collate


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
