"""Validación de archivos descargados (rechaza placeholders)."""

from __future__ import annotations

from pathlib import Path


from phishing_pipeline.logging_utils import get_logger

logger = get_logger(__name__)


def count_csv_rows(path: Path) -> int:
    """Cuenta filas de datos en un CSV (excluye cabecera)."""
    with path.open(encoding="utf-8", errors="ignore") as handle:
        total_lines = sum(1 for _ in handle)
    return max(total_lines - 1, 0)


def count_jsonl_lines(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def is_valid_csv(path: Path, min_rows: int) -> bool:
    if not path.exists():
        return False
    rows = count_csv_rows(path)
    if rows < min_rows:
        logger.warning(
            "CSV insuficiente: %s (%d filas, mínimo %d)",
            path,
            rows,
            min_rows,
        )
        return False
    return True


def is_valid_phish_mmf_extracted(extract_dir: Path, min_total_lines: int) -> bool:
    if not extract_dir.exists():
        return False
    jsonl_files = list(extract_dir.rglob("*.jsonl"))
    if not jsonl_files:
        logger.warning("PhishMMF extracted sin archivos .jsonl: %s", extract_dir)
        return False
    total_lines = sum(count_jsonl_lines(p) for p in jsonl_files)
    if total_lines < min_total_lines:
        logger.warning(
            "PhishMMF extracted insuficiente: %d líneas en %d archivos (mínimo %d)",
            total_lines,
            len(jsonl_files),
            min_total_lines,
        )
        return False
    return True


def remove_path(path: Path) -> None:
    if path.is_file():
        path.unlink()
    elif path.is_dir():
        import shutil

        shutil.rmtree(path)
