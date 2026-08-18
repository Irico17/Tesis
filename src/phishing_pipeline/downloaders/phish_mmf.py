"""Descarga y extracción de PhishMMF desde GitHub."""

from __future__ import annotations

import json
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlretrieve

from phishing_pipeline.config import (
    MIN_PHISH_MMF_JSONL_LINES,
    PHISH_MMF_DIR,
    PHISH_MMF_EXTRACTED,
    PHISH_MMF_GITHUB_URL,
    PHISH_MMF_PINNED_COMMIT,
    PHISH_MMF_ZIP_FILES,
)
from phishing_pipeline.downloaders.validation import (
    count_jsonl_lines,
    is_valid_phish_mmf_extracted,
    remove_path,
)
from phishing_pipeline.logging_utils import get_logger

logger = get_logger(__name__)


def _count_jsonl_lines(path: Path) -> int:
    return count_jsonl_lines(path)


def _extract_zips(repo_dir: Path, extract_dir: Path) -> list[Path]:
    extract_dir.mkdir(parents=True, exist_ok=True)
    extracted_files: list[Path] = []
    for zip_name in PHISH_MMF_ZIP_FILES:
        zip_path = repo_dir / zip_name
        if not zip_path.exists():
            for candidate in repo_dir.rglob(zip_name):
                zip_path = candidate
                break
        if not zip_path.exists():
            logger.warning("ZIP no encontrado: %s", zip_name)
            continue
        logger.info("Extrayendo %s...", zip_path.name)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        extracted_files.append(zip_path)
    return extracted_files


def _write_metadata(extract_dir: Path) -> None:
    files_info = []
    for p in sorted(extract_dir.rglob("*")):
        if p.is_file():
            files_info.append(
                {
                    "path": str(p.relative_to(extract_dir)),
                    "size_bytes": p.stat().st_size,
                    "lines": _count_jsonl_lines(p) if p.suffix.lower() in {".jsonl", ".txt", ""} else None,
                }
            )
    meta = {
        "source": PHISH_MMF_GITHUB_URL,
        "pinned_commit": PHISH_MMF_PINNED_COMMIT,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "extracted_dir": str(extract_dir),
        "files": files_info,
        "file_count": len(files_info),
    }
    (PHISH_MMF_DIR / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _current_commit(repo_dir: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _clone_or_download(raw_dir: Path) -> Path:
    repo_dir = raw_dir / "PhishMMF" / "repo"
    if repo_dir.exists() and any(repo_dir.iterdir()):
        current = _current_commit(repo_dir)
        if current == PHISH_MMF_PINNED_COMMIT:
            logger.info("Repositorio PhishMMF ya presente en %s, commit correcto (%s)", repo_dir, current[:12])
            return repo_dir
        logger.warning(
            "Repositorio PhishMMF presente pero en commit %s (esperado %s) -- re-clonando "
            "para garantizar reproducibilidad.",
            current,
            PHISH_MMF_PINNED_COMMIT,
        )
        remove_path(repo_dir)

    if repo_dir.exists():
        remove_path(repo_dir)
    repo_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Clon completo (no --depth 1): un shallow clone solo trae el HEAD
        # actual del repo remoto, que puede no incluir PHISH_MMF_PINNED_COMMIT
        # si el repositorio avanzó desde entonces. El repo es modesto (~47MB
        # de historial), el costo de un clon completo es aceptable frente a la
        # garantía de reproducibilidad exacta.
        logger.info("Clonando PhishMMF desde GitHub (historial completo, para poder fijar commit)...")
        subprocess.run(
            ["git", "clone", PHISH_MMF_GITHUB_URL, str(repo_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "-C", str(repo_dir), "checkout", PHISH_MMF_PINNED_COMMIT],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("PhishMMF fijado a commit %s", PHISH_MMF_PINNED_COMMIT[:12])
        return repo_dir
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.warning(
            "git clone/checkout al commit fijado falló (%s). Intentando descarga ZIP de ese "
            "commit exacto (fallback sin git)...",
            exc,
        )
        zip_url = PHISH_MMF_GITHUB_URL.replace(".git", f"/archive/{PHISH_MMF_PINNED_COMMIT}.zip")
        zip_dest = raw_dir / "PhishMMF" / "main.zip"
        urlretrieve(zip_url, zip_dest)
        with zipfile.ZipFile(zip_dest, "r") as zf:
            zf.extractall(raw_dir / "PhishMMF")
        for p in (raw_dir / "PhishMMF").iterdir():
            if p.is_dir() and p.name.startswith("PhishMMF"):
                return p
        raise FileNotFoundError("No se pudo obtener PhishMMF en el commit fijado") from exc


def download_phish_mmf(raw_dir: Path | None = None) -> Path:
    """
    Clona PhishMMF, extrae ZIPs y conserva JSONL originales.

    Returns:
        Path a la carpeta extracted/
    """
    base = raw_dir or PHISH_MMF_DIR.parent
    extract_dir = base / "PhishMMF" / "extracted"

    if extract_dir.exists() and is_valid_phish_mmf_extracted(
        extract_dir, MIN_PHISH_MMF_JSONL_LINES
    ):
        logger.info("PhishMMF extracted ya existe: %s", extract_dir)
        _write_metadata(extract_dir)
        return extract_dir

    if extract_dir.exists():
        logger.warning("Eliminando extracted placeholder/incompleto: %s", extract_dir)
        remove_path(extract_dir)

    repo_dir = _clone_or_download(base)
    _extract_zips(repo_dir, extract_dir)

    if not is_valid_phish_mmf_extracted(extract_dir, MIN_PHISH_MMF_JSONL_LINES):
        raise RuntimeError(
            "PhishMMF extraído no contiene suficientes líneas JSONL. "
            "Verifique git clone y los ZIPs del repositorio."
        )

    _write_metadata(extract_dir)
    logger.info("PhishMMF listo en %s", extract_dir)
    return extract_dir
