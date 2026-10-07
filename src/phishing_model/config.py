"""Configuración de la arquitectura multimodal R1.3/R1.4.

Fuente única de verdad para: dimensiones del modelo, vocabularios categóricos
de red (SPF/DKIM/DMARC), hiperparámetros de entrenamiento y rutas de
checkpoint/artefactos. `dataset.py` y `encoders/network_tokenizer.py` importan
los vocabularios de aquí para no divergir.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
    # Cuarta manera de tratar la ausencia de ramas: enrutamiento por
    # patrón de disponibilidad. Se decide por medición en E3.
    MIXTURE_OF_EXPERTS = "mixture_of_experts"


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
    # "independent": cada rama disponible se suprime con probabilidad fija.
    # "randomize": se sortea el patrón de disponibilidad de la distribución marginal
    # del corpus, con independencia de la etiqueta.
    #
    # Medido sobre el corpus real, la información mutua entre el patrón de
    # disponibilidad y la etiqueta —el atajo verificado en los datos— pasa de
    # 0.0790 nats a 0.0235 con aleatorización, frente a 0.0602 con descarte
    # independiente al 15%. La aleatorización REDUCE el atajo en un 70%, pero no
    # lo elimina: el patrón sorteado se intersecta con la disponibilidad natural,
    # porque no puede añadirse una modalidad que la fila no posee, y esa
    # intersección conserva la dependencia que proviene de la disponibilidad real.
    # Elevar la probabilidad de descarte no sustituye a la aleatorización: se
    # comprobó con tasas de 0.15 a 0.60 y no mejora el pliegue afectado.
    modality_dropout_mode: str = "independent"
    # Distribución marginal medida sobre el corpus consolidado, en el orden
    # (ninguna, solo red, solo estructura, ambas).
    modality_pattern_probs: tuple[float, float, float, float] = (0.1318, 0.4697, 0.0035, 0.3951)

    # Valor inicial de la compuerta de contribución modal, expresado ya como
    # tanh(g) y por tanto en [0, 1). Ver la justificación de por qué no se
    # inicializa en cero en `model.MultimodalPhishingClassifier.__init__`.
    modality_gate_init: float = 0.5
    # Forma de la compuerta modal. Ver la justificación en `model.py`.
    # "global"      : un escalar único para todo el corpus (formulación original).
    # "conditional" : un valor por correo, calculado a partir de la representación
    #                 textual y de las banderas de disponibilidad. Permite que el
    #                 modelo apague la fusión donde no aporta, y habilita reportar
    #                 la distribución de la compuerta por fuente y por clase.
    modality_gate_mode: str = "global"

    # Formulación interna de las capas de fusión. Ver la justificación en
    # `fusion/cross_attention.py`: pre-normalización por estabilidad frente al
    # codificador preentrenado, y GELU por coherencia con DistilBERT.
    norm_first: bool = True
    fusion_activation: str = "gelu"

    # Escalado de las características tabulares. Ver `dataset.py`.
    # "quantile"     : QuantileTransformer, acotado por construcción (recomendado).
    # "minmax_clip"  : MinMaxScaler con recorte a [0, 1].
    # "minmax"       : MinMaxScaler sin recorte — formulación original, NO acotada.
    scaler_kind: str = "quantile"

    # Número de fuentes que debe distinguir la cabeza adversaria. 0 la desactiva.
    # Lo fija `train()` a partir de los grupos presentes en el entrenamiento, de
    # modo que quede guardado en el punto de control y la arquitectura se
    # reconstruya idéntica al evaluar. Ver `losses.CabezaAdversariaDeFuente`.
    # Anchura de la capa oculta de la cabeza de clasificación. Con None la
    # cabeza es lineal --LayerNorm, Dropout, Linear-- que es lo que usaban
    # todas las variantes. Un valor entero la convierte en un perceptrón
    # multicapa, que es el fusor que el asesor propuso como línea base
    # frente a la atención cruzada.
    cabeza_oculta: int | None = None
    n_expertos_moe: int = 4
    n_fuentes_adversario: int = 0
    # Peso de la pérdida adversaria frente a la de clasificación.
    peso_adversario: float = 1.0

    n_structural_features: int = 6  # STRUCTURAL_FEATURE_COLS (ver nota allí sobre las 3 excluidas)
    n_network_continuous: int = 9  # NETWORK_FEATURE_COLS
    n_network_categorical: int = 3  # spf/dkim/dmarc

    @property
    def n_network_tokens(self) -> int:
        """Tokens de la rama de red: 9 continuos + 3 categóricos = 12."""
        return self.n_network_continuous + self.n_network_categorical

    @property
    def n_modality_tokens(self) -> int:
        """Tokens no-textuales totales que entran a Stage 1 (fusion/modality_encoder.py).

        Son 6 estructurales + 12 de red = 18. A la memoria de atención se le
        antepone además el token centinela de "sin modalidad" (ver `model.py`),
        de modo que la memoria efectiva tiene 19 posiciones; el centinela no es
        una característica del correo y por eso no se cuenta aquí.
        """
        return self.n_structural_features + self.n_network_tokens

    def to_dict(self) -> dict:
        """Representación serializable, para persistir junto al punto de control."""
        datos = asdict(self)
        datos["fusion_type"] = self.fusion_type.value
        datos["modality_pattern_probs"] = list(self.modality_pattern_probs)
        return datos

    @classmethod
    def from_dict(cls, datos: dict) -> "ModelConfig":
        """
        Reconstruye la configuración desde un punto de control.

        Se ignoran las claves que no correspondan a ningún campo actual, de modo
        que un punto de control generado por una versión anterior siga cargándose
        en lugar de fallar; los campos que falten toman su valor por defecto.
        """
        campos = {f for f in cls.__dataclass_fields__}
        limpio = {k: v for k, v in datos.items() if k in campos}
        if "fusion_type" in limpio:
            limpio["fusion_type"] = FusionType(limpio["fusion_type"])
        if "modality_pattern_probs" in limpio:
            limpio["modality_pattern_probs"] = tuple(limpio["modality_pattern_probs"])
        return cls(**limpio)


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

    # Minimización del riesgo del peor grupo, con la FUENTE como grupo. Responde a
    # un desbalance distinto del que atacan la ponderación de clases y la pérdida
    # focal: el de fuente, que en este corpus es el dominante —en el pliegue que
    # retiene Kaggle, el 88.5% del entrenamiento procede de una sola fuente—.
    # Ver la justificación completa en `losses.GroupDROLoss`.
    use_group_dro: bool = False
    group_dro_eta: float = 0.01
    # Muestreo con reposición que iguala la presencia esperada de cada fuente en
    # cada época. Alternativa más simple a la anterior y compatible con ella:
    # reponderar el muestreo ataca la frecuencia, reponderar la pérdida ataca el
    # riesgo. Se dejan separadas para poder atribuir el efecto observado.
    balancear_por_fuente: bool = False

    # Criterio con el que se elige el punto de control que después se evalúa.
    # Ver `train.CRITERIOS_DE_SELECCION` para la justificación de por qué el
    # criterio histórico está sesgado bajo cambio de dominio.
    criterio_seleccion: str = "val_loss"

    # Cabeza adversaria que borra la fuente de la representación (Ganin et al.,
    # 2016). Responde al hallazgo de que las fuentes de este corpus son
    # identificables con un 96.4% de exactitud desde el texto, de modo que el
    # modelo dispone de un atajo casi perfecto si no se le impide usarlo.
    usar_adversario_de_fuente: bool = False
    peso_adversario: float = 1.0
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
    # Determinismo completo. Fijar las semillas de random/numpy/torch no basta:
    # las rutinas de cuDNN eligen algoritmos por heurística de rendimiento y
    # algunas acumulan en orden no determinista, de modo que dos ejecuciones con
    # la misma semilla pueden diferir. La tesis declara la ejecución reproducible
    # como criterio metodológico, y eso exige fijarlo de forma explícita.
    # Tiene un costo de rendimiento, por lo que se deja gobernable.
    deterministic: bool = True


def get_checkpoint_path(run_name: str) -> Path:
    """Ruta de checkpoint para una corrida nombrada (p.ej. por fusion_type)."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    return CHECKPOINT_DIR / f"{run_name}.pt"
