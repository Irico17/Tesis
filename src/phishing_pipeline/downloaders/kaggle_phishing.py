"""Descarga dataset Kaggle mohammadaoalhija/phishing-email."""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from phishing_pipeline.config import (
    KAGGLE_PHISHING_DIR,
    KAGGLE_PHISHING_FILE,
    KAGGLE_PHISHING_PANDAS_KWARGS,
    KAGGLE_PHISHING_SLUG,
    MIN_KAGGLE_PHISHING_ROWS,
)
from phishing_pipeline.downloaders.validation import is_valid_csv, remove_path
from phishing_pipeline.logging_utils import get_logger

logger = get_logger(__name__)


def _write_metadata(dest: Path, df: pd.DataFrame) -> None:
    md5 = hashlib.md5(dest.read_bytes()).hexdigest()
    meta = {
        "dataset": KAGGLE_PHISHING_SLUG,
        "file": KAGGLE_PHISHING_FILE,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(df),
        "columns": list(df.columns),
        "md5": md5,
    }
    (dest.parent / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _resolve_csv_source(cache_path: Path) -> Path:
    """Localiza el CSV real; en Kaggle a veces viene empaquetado como ZIP."""
    for candidate in cache_path.rglob(KAGGLE_PHISHING_FILE):
        if candidate.suffix.lower() == ".csv" and not candidate.name.endswith(".zip"):
            if candidate.read_bytes()[:2] == b"PK":
                extract_dir = candidate.parent / "_extracted"
                extract_dir.mkdir(exist_ok=True)
                with zipfile.ZipFile(candidate, "r") as zf:
                    zf.extractall(extract_dir)
                inner = extract_dir / KAGGLE_PHISHING_FILE
                if inner.exists():
                    return inner
            return candidate
    csvs = [p for p in cache_path.rglob("*.csv") if p.read_bytes()[:2] != b"PK"]
    if not csvs:
        raise FileNotFoundError(f"No se encontró {KAGGLE_PHISHING_FILE} en {cache_path}")
    return max(csvs, key=lambda x: x.stat().st_size)


def _read_phishing_csv(source: Path) -> pd.DataFrame:
    read_kwargs = {k: v for k, v in KAGGLE_PHISHING_PANDAS_KWARGS.items() if k != "index_col"}
    try:
        return pd.read_csv(source, **read_kwargs, index_col=0)
    except UnicodeDecodeError:
        return pd.read_csv(source, **read_kwargs, index_col=0, encoding="latin-1")


def download_kaggle_phishing(raw_dir: Path | None = None) -> Path:
    """
    Descarga Phishing_Email.csv vía kagglehub (mismo flujo que el notebook Colab).

    Returns:
        Path al CSV en Datasets_Originales/Kaggle_Phishing_Email/
    """
    out_dir = (raw_dir or KAGGLE_PHISHING_DIR.parent) / "Kaggle_Phishing_Email"
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / KAGGLE_PHISHING_FILE

    if dest.exists() and is_valid_csv(dest, MIN_KAGGLE_PHISHING_ROWS):
        logger.info("Kaggle phishing ya existe: %s", dest)
        return dest

    if dest.exists():
        logger.warning("Eliminando CSV placeholder/incompleto: %s", dest)
        remove_path(dest)
        metadata = out_dir / "metadata.json"
        if metadata.exists():
            remove_path(metadata)

    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError(
            "kagglehub no está instalado. Ejecute: pip install kagglehub"
        ) from exc

    logger.info("Descargando %s desde Kaggle (kagglehub)...", KAGGLE_PHISHING_SLUG)
    try:
        cache_path = Path(kagglehub.dataset_download(KAGGLE_PHISHING_SLUG))
        source = _resolve_csv_source(cache_path)
        df = _read_phishing_csv(source)
    except Exception as exc:
        raise RuntimeError(
            "No se pudo descargar mohammadaoalhija/phishing-email con kagglehub. "
            "Si Kaggle pide autenticación, configure credenciales (ver README_pipeline.md)."
        ) from exc

    if len(df) < MIN_KAGGLE_PHISHING_ROWS:
        raise RuntimeError(
            f"Descarga Kaggle incompleta: {len(df)} filas (mínimo {MIN_KAGGLE_PHISHING_ROWS})."
        )

    valid_types = {"Safe Email", "Phishing Email"}
    before = len(df)
    df = df[df["Email Type"].isin(valid_types)].copy()
    if len(df) < before:
        logger.warning(
            "Filtradas %d filas Kaggle con Email Type inválido (CSV corrupto)",
            before - len(df),
        )

    df.to_csv(dest, index=False)
    _write_metadata(dest, df)
    logger.info("Guardado %s (%d filas)", dest, len(df))
    return dest
