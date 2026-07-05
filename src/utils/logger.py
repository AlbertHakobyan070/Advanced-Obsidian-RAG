"""
logger.py — Centralized structured logging.

Usage:
    from src.utils.logger import get_logger
    log = get_logger(__name__)
    log.info("indexed %d chunks", n)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED = False
_DEFAULT_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FMT = "%H:%M:%S"


def configure_logging(
    level: str = "INFO",
    log_file: str | Path | None = None,
    console: bool = True,
) -> None:
    """Configure root logging once. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear any pre-existing handlers (e.g. from libraries)
    root.handlers.clear()

    formatter = logging.Formatter(_DEFAULT_FMT, datefmt=_DATE_FMT)

    if console:
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(formatter)
        root.addHandler(sh)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(formatter)
        root.addHandler(fh)

    # Quiet down noisy third-party loggers
    for noisy in ("httpx", "urllib3", "chromadb", "sentence_transformers", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Get a named logger. Auto-configures with defaults if not yet set up."""
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
