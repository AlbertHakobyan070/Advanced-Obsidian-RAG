"""
logger.py — Centralized structured logging.

Usage:
    from src.utils.logger import get_logger
    log = get_logger(__name__)
    log.info("indexed %d chunks", n)

TWO KINDS OF SETUP. Modules call get_logger() at import time, which sets the
process up with the defaults (INFO, stderr only, no file) when nothing has
configured logging yet — long before main.py's _bootstrap_logging() or
RAGPipeline.from_config() pass config.yaml's logging.level / logging.file. That
implicit setup is provisional: the first EXPLICIT configure_logging() call
replaces its handlers and applies its own arguments. After that, explicit
calls are idempotent and the first one wins.

The log file is appended to and NOT rotated, deliberately. :8051, the console
and every job subprocess share one logging.file, and Windows refuses to rename
a file another process has open — so a RotatingFileHandler would fail every
rollover once the file passed its limit and print a traceback for each record
after that. Known limit: the file grows until someone deletes it. The fix is a
per-process file or a multi-process-safe handler, not rotation.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED = False   # root logging is set up, by either kind of call
_EXPLICIT = False     # ...by a configure_logging() call, not get_logger()'s defaults
_DEFAULT_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FMT = "%H:%M:%S"


def configure_logging(
    level: str = "INFO",
    log_file: str | Path | None = None,
    console: bool = True,
) -> None:
    """Configure root logging. Idempotent: once a call has configured it, later
    calls return immediately, whatever arguments they are given. The one
    exception is the implicit setup get_logger() makes with the defaults — the
    first explicit call replaces that and applies its arguments."""
    global _CONFIGURED, _EXPLICIT
    if _EXPLICIT:
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
    _EXPLICIT = True


def get_logger(name: str) -> logging.Logger:
    """Get a named logger. Auto-configures with defaults if not yet set up; a
    later configure_logging() call still replaces that provisional setup."""
    global _EXPLICIT
    if not _CONFIGURED:
        configure_logging()
        _EXPLICIT = False
    return logging.getLogger(name)
