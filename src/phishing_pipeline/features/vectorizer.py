"""Vectorizador HF tokenizer + MinMaxScaler (R1.2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from phishing_pipeline.config import FEATURES_DIR, MAX_TOKEN_LENGTH, TOKENIZER_MODEL
from phishing_pipeline.logging_utils import get_logger

logger = get_logger(__name__)

# Rama estructural del modelo. `has_form`, `has_iframe` y `has_javascript` se
# EXCLUYEN deliberadamente pese a formar parte del esquema canónico: valen cero
# en las 110,152 filas del corpus, porque las fuentes publicaron el HTML ya
# sanitizado. Como características constantes no pueden aportar capacidad
# predictiva, y sí provocan que la explicabilidad les asigne peso espurio (se
# observó `has_iframe` como segundo factor de una predicción). En su lugar se
# incorporan los estadísticos de complejidad del DOM, que sí discriminan sobre
# este corpus: media de 13.78 nodos en phishing frente a 1.87 en legítimos.
# Se conservan en el esquema para que el pipeline siga extrayéndolas -- serían
# informativas sobre un corpus no sanitizado. Ver doc/REVISION_CODIGO_MODELO.md.
STRUCTURAL_FEATURE_COLS = [
    "has_html",
    "num_links",
    "num_images",
    "word_count",
    "total_nodos_dom",
    "profundidad_dom",
]

# Nota: spf_result/dkim_result/dmarc_result son categóricas (pass/fail/none/...,
# con "missing" explícito), NO numéricas, así que NO van en esta lista pese a
# vivir conceptualmente en la rama de red — MultimodalVectorizer/MinMaxScaler
# solo maneja columnas numéricas. Su codificación categórica (embedding por
# categoría incl. "missing") se implementa en
# src/phishing_model/encoders/network_tokenizer.py (Fase B del plan,
# construido por otro proceso/agente) — no se inventa ni se adelanta aquí.
NETWORK_FEATURE_COLS = [
    "has_ip_link",
    "num_urls_metadata",
    "url_max_length",
    "url_max_digit_ratio",
    "url_max_subdomain_count",
    "url_has_at_symbol",
    "url_has_shortener",
    "url_has_suspicious_tld",
    "url_domain_max_entropy",
]

# Alias de compatibilidad hacia atrás: código existente que importa
# NUMERIC_FEATURE_COLS (p.ej. baselines estructurales) sigue funcionando igual.
NUMERIC_FEATURE_COLS = STRUCTURAL_FEATURE_COLS + NETWORK_FEATURE_COLS


def compute_modality_availability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula disponibilidad de modalidad por fila (mismo criterio que
    unifier.run_smoke_test para consistencia con el smoke test R1.1/R1.2).

    La disponibilidad de red se define ÚNICAMENTE por la presencia de una URL o
    una dirección IP en el mensaje. La formulación anterior admitía además que
    `spf_result` no fuese nulo, y se comprobó que eso convertía la bandera en un
    revelador de la etiqueta: las cabeceras de autenticación solo se extraían de
    los tres ficheros de phishing de PhishMMF, de modo que dentro de esa fuente
    `p(phishing | spf presente)` valía 1.0000 sobre 4,478 correos. La bandera
    `has_network` entra al modelo como entrada explícita y gobierna el
    enmascaramiento de toda la rama, así que la fuga llegaba intacta a la
    arquitectura. Es el origen del «atajo de disponibilidad» de 0.0790 nats.

    La presencia de una URL es una propiedad del correo y sobrevive a cualquier
    cambio en el pipeline de ingesta; la de una cabecera extraída depende de qué
    fichero la traía. Solo la primera puede sostener la definición.

    Ver `cleaners/phish_mmf.EXTRAER_CABECERAS_PHISHMMF` y
    `phishing_pipeline.auditoria_fugas`, que comprueba el invariante en cada
    ejecución para que la corrección no se pierda con el tiempo.

    Returns:
        DataFrame con 3 columnas booleanas: has_text_modality (siempre True),
        has_structure_modality (has_html == 1), has_network_modality
        (URL o IP presente en network_indicators).
    """
    has_text_modality = pd.Series(True, index=df.index)
    has_structure_modality = df["has_html"] == 1
    has_network_modality = df["network_indicators"].fillna("None") != "None"

    return pd.DataFrame(
        {
            "has_text_modality": has_text_modality,
            "has_structure_modality": has_structure_modality,
            "has_network_modality": has_network_modality,
        }
    )


class MultimodalVectorizer:
    """Tokenización HuggingFace + escalado MinMax para features tabulares."""

    def __init__(
        self,
        model_name: str = TOKENIZER_MODEL,
        max_length: int = MAX_TOKEN_LENGTH,
        features_dir: Path | None = None,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.features_dir = features_dir or FEATURES_DIR
        self.tokenizer = None
        self.scaler: MinMaxScaler | None = None
        self.numeric_cols: list[str] = []

    def _load_tokenizer(self):
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self.tokenizer

    def fit_transform_text(
        self,
        texts: list[str],
        save: bool = True,
        batch_size: int = 256,
        max_encode: int | None = 5000,
    ) -> dict[str, Any]:
        """Tokeniza textos con HF tokenizer (batched, memory-safe)."""
        tokenizer = self._load_tokenizer()
        if save:
            tok_dir = self.features_dir / "tokenizer"
            tok_dir.mkdir(parents=True, exist_ok=True)
            tokenizer.save_pretrained(tok_dir)
            logger.info("Tokenizer guardado en %s", tok_dir)

        encode_texts = texts[: max_encode or len(texts)]
        all_ids: list[np.ndarray] = []
        all_masks: list[np.ndarray] = []
        for i in range(0, len(encode_texts), batch_size):
            batch = encode_texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="np",
            )
            all_ids.append(enc["input_ids"])
            all_masks.append(enc["attention_mask"])

        if not all_ids:
            return {"input_ids": np.array([]), "attention_mask": np.array([]), "encoded_rows": 0}

        return {
            "input_ids": np.concatenate(all_ids, axis=0),
            "attention_mask": np.concatenate(all_masks, axis=0),
            "encoded_rows": len(encode_texts),
            "total_rows": len(texts),
        }

    def fit_transform_metadata(
        self, df: pd.DataFrame, numeric_cols: list[str] | None = None
    ) -> np.ndarray:
        """Ajusta MinMaxScaler sobre columnas numéricas."""
        self.numeric_cols = numeric_cols or [
            c for c in NUMERIC_FEATURE_COLS if c in df.columns
        ]
        if not self.numeric_cols:
            raise ValueError("No hay columnas numéricas para escalar")

        data = df[self.numeric_cols].fillna(0).astype(float).values
        self.scaler = MinMaxScaler()
        scaled = self.scaler.fit_transform(data)

        scaler_path = self.features_dir / "scaler.pkl"
        self.features_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"scaler": self.scaler, "columns": self.numeric_cols}, scaler_path)
        logger.info("Scaler guardado en %s", scaler_path)
        return scaled

    def transform(self, df: pd.DataFrame, texts: list[str] | None = None) -> dict[str, Any]:
        """Transforma dataset completo."""
        if texts is None:
            texts = df["clean_text"].fillna("").astype(str).tolist()

        text_features = self.fit_transform_text(texts, save=False)
        meta_features = None
        if self.scaler is not None and self.numeric_cols:
            data = df[self.numeric_cols].fillna(0).astype(float).values
            meta_features = self.scaler.transform(data)

        return {
            "input_ids": text_features["input_ids"],
            "attention_mask": text_features["attention_mask"],
            "metadata_scaled": meta_features,
            "numeric_columns": self.numeric_cols,
        }

    def load_artifacts(self) -> None:
        """Carga tokenizer y scaler desde disco."""
        from transformers import AutoTokenizer

        tok_dir = self.features_dir / "tokenizer"
        if tok_dir.exists():
            self.tokenizer = AutoTokenizer.from_pretrained(tok_dir)
        scaler_path = self.features_dir / "scaler.pkl"
        if scaler_path.exists():
            artifact = joblib.load(scaler_path)
            self.scaler = artifact["scaler"]
            self.numeric_cols = artifact["columns"]
