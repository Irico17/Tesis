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
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "extracted_dir": str(extract_dir),
        "files": files_info,
        "file_count": len(files_info),
    }
    (PHISH_MMF_DIR / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _clone_or_download(raw_dir: Path) -> Path:
    repo_dir = raw_dir / "PhishMMF" / "repo"
    if repo_dir.exists() and any(repo_dir.iterdir()):
        logger.info("Repositorio PhishMMF ya presente en %s", repo_dir)
        return repo_dir

    if repo_dir.exists():
        remove_path(repo_dir)
    repo_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("Clonando PhishMMF desde GitHub...")
        subprocess.run(
            ["git", "clone", "--depth", "1", PHISH_MMF_GITHUB_URL, str(repo_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        return repo_dir
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.warning("git clone falló (%s). Intentando descarga ZIP del repo...", exc)
        zip_url = PHISH_MMF_GITHUB_URL.replace(".git", "/archive/refs/heads/main.zip")
        zip_dest = raw_dir / "PhishMMF" / "main.zip"
        urlretrieve(zip_url, zip_dest)
        with zipfile.ZipFile(zip_dest, "r") as zf:
            zf.extractall(raw_dir / "PhishMMF")
        for p in (raw_dir / "PhishMMF").iterdir():
            if p.is_dir() and p.name.startswith("PhishMMF"):
                return p
        raise FileNotFoundError("No se pudo obtener PhishMMF") from exc


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
