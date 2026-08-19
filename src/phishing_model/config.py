"""Configuración de la arquitectura multimodal R1.3/R1.4.

Fuente única de verdad para: dimensiones del modelo, vocabularios categóricos
de red (SPF/DKIM/DMARC), hiperparámetros de entrenamiento y rutas de
checkpoint/artefactos. `dataset.py` y `encoders/network_tokenizer.py` importan
los vocabularios de aquí para no divergir.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from phishing_pipeline.config import BASE_DIR, RANDOM_STATE

MODEL_DIR = BASE_DIR / "data" / "model"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
SANITY_CHECK_REPORT_PATH = BASE_DIR / "data" / "reports" / "model_sanity_check.json"

TEXT_MODEL_NAME = "distilbert-base-uncased"
TEXT_HIDDEN_SIZE = 768  # dimensión nativa de salida de distilbert-base-uncased
MAX_TOKEN_LENGTH = 512


class FusionType(str, Enum):
    """Variantes de fusión — ablación de R2.2 (texto solo / metadatos concatenados / atención)."""

    TEXT_ONLY = "text_only"
    CONCAT_LATE_FUSION = "concat_late_fusion"
    CROSS_ATTENTION_MODALITY_LEVEL = "cross_attention_modality_level"
    CROSS_ATTENTION_TOKEN_LEVEL = "cross_attention_token_level"


# Vocabularios categóricos de red — extraídos de los valores REALES presentes en
# el corpus unificado tras el enriquecimiento de Fase A2 (Spam_Genuine_Mail +
# PhishMMF/phishing_pot+datacon2023). Índice 0 reservado para "missing"
# (ausencia natural del campo, p.ej. Kaggle_Phishing_Email no lo aporta).
# Verificado con:
#   df[col].dropna().unique() sobre Dataset_Unificado.parquet, 2026-08-18.
SPF_VOCAB: dict[str, int] = {
    "missing": 0,
    "pass": 1,
    "fail": 2,
    "neutral": 3,
    "none": 4,
    "softfail": 5,
    "permerror": 6,
    "temperror": 7,
}
DKIM_VOCAB: dict[str, int] = {
    "missing": 0,
    "pass": 1,
    "fail": 2,
    "neutral": 3,
    "none": 4,
    "ignore": 5,
    "test": 6,
    "timeout": 7,
}
DMARC_VOCAB: dict[str, int] = {
    "missing": 0,
    "pass": 1,
    "fail": 2,
    "none": 3,
    "bestguesspass": 4,
    "permerror": 5,
    "temperror": 6,
}

# Orden fijo de las columnas categóricas de red dentro del tensor `network_categorical`.
NETWORK_CATEGORICAL_COLS: list[str] = ["spf_result", "dkim_result", "dmarc_result"]
NETWORK_CATEGORICAL_VOCABS: dict[str, dict[str, int]] = {
    "spf_result": SPF_VOCAB,
    "dkim_result": DKIM_VOCAB,
    "dmarc_result": DMARC_VOCAB,
}

# Recordatorio de las columnas continuas/estructurales — las listas canónicas
# viven en phishing_pipeline.features.vectorizer (STRUCTURAL_FEATURE_COLS,
# NETWORK_FEATURE_COLS); no se duplican aquí para no divergir, dataset.py las
# importa directamente de ahí.


@dataclass
class ModelConfig:
    """Hiperparámetros de arquitectura (Fase B)."""

    d_model: int = 256
    n_heads: int = 8
    n_fusion_layers: int = 1
    dim_feedforward: int = 512
    dropout: float = 0.1
    fusion_type: FusionType = FusionType.CROSS_ATTENTION_TOKEN_LEVEL
    text_model_name: str = TEXT_MODEL_NAME
    max_token_length: int = MAX_TOKEN_LENGTH
    freeze_text_encoder: bool = False  # fine-tuning completo por defecto (ver plan, Fase B)
    grad_checkpointing: bool = True

    # Dropout de modalidad durante entrenamiento (mecanismo central de manejo
    # de modalidades ausentes, ver plan Fase B) — probabilidad de enmascarar
    # artificialmente una rama disponible, por fila y por rama, SOLO en train().
    modality_dropout_prob: float = 0.15

    n_structural_features: int = 6  # STRUCTURAL_FEATURE_COLS (ver nota allí sobre las 3 excluidas)
    n_network_continuous: int = 9  # NETWORK_FEATURE_COLS
    n_network_categorical: int = 3  # spf/dkim/dmarc

    @property
    def n_network_tokens(self) -> int:
        """Tokens de la rama de red: 9 continuos + 3 categóricos = 12."""
        return self.n_network_continuous + self.n_network_categorical

    @property
    def n_modality_tokens(self) -> int:
        """Tokens no-textuales totales que entran a Stage 1 (fusion/modality_encoder.py)."""
        return self.n_structural_features + self.n_network_tokens


@dataclass
class TrainConfig:
    """Hiperparámetros de entrenamiento (Fase C)."""

    backbone_lr: float = 2e-5
    head_lr: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 16
    eval_batch_size: int = 32
    epochs: int = 3
    max_grad_norm: float = 1.0
    mixed_precision: bool = True
    checkpoint_every_n_steps: int = 500
    use_class_weights: bool = False  # decisión empírica (Fase C), no a priori
    use_focal_loss: bool = False
    focal_loss_gamma: float = 2.0
    seed: int = RANDOM_STATE
    # Planificador de tasa de aprendizaje: calentamiento lineal seguido de
    # decaimiento lineal, práctica estándar en el ajuste fino de modelos de
    # lenguaje preentrenados desde BERT. Sin él, las capas de fusión
    # inicializadas al azar propagan gradientes grandes hacia el codificador
    # preentrenado en los primeros pasos (riesgo de degradar los pesos
    # aprendidos), y no hay refinamiento fino al final del entrenamiento.
    # `warmup_ratio` se expresa como fracción del total de pasos previstos.
    use_lr_scheduler: bool = True
    warmup_ratio: float = 0.1
    # Parada temprana por épocas sin mejora de la pérdida de validación.
    # 0 la desactiva. Con el presupuesto de GPU acotado y compartido del
    # laboratorio, interrumpir una ejecución que ya dejó de mejorar libera
    # tiempo para las variantes de ablación restantes.
    early_stopping_patience: int = 2
    num_workers: int = 0  # 0 por defecto: seguro en Windows/CPU; subir en GPU lab si hace falta


def get_checkpoint_path(run_name: str) -> Path:
    """Ruta de checkpoint para una corrida nombrada (p.ej. por fusion_type)."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    return CHECKPOINT_DIR / f"{run_name}.pt"
