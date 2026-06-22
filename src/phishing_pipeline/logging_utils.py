"""Logging estructurado con conteos y tasa de éxito (indicador R1.2)."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import Any


def get_logger(name: str = "phishing_pipeline") -> logging.Logger:
    """Retorna un logger configurado para consola."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


@dataclass
class ProcessingStats:
    """Contadores de procesamiento para validación R1.2."""

    processed: int = 0
    errors: int = 0
    skipped: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return self.processed + self.errors + self.skipped

    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.processed / self.total

    def log_summary(self, logger: logging.Logger, label: str = "Pipeline") -> None:
        logger.info(
            "%s | processed=%d errors=%d skipped=%d total=%d success_rate=%.2f%%",
            label,
            self.processed,
            self.errors,
            self.skipped,
            self.total,
            self.success_rate * 100,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "processed": self.processed,
            "errors": self.errors,
            "skipped": self.skipped,
            "total": self.total,
            "success_rate": round(self.success_rate, 4),
            **self.details,
        }


def log_counts(logger: logging.Logger, prefix: str, counts: dict[str, int]) -> None:
    """Registra conteos clave-valor."""
    parts = ", ".join(f"{k}={v}" for k, v in counts.items())
    logger.info("%s: %s", prefix, parts)
