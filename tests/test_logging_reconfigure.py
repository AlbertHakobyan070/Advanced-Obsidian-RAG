"""An explicit configure_logging() must beat get_logger()'s implicit one.

THE BUG THIS PINS. Every module calls get_logger() at import time, which set
the root logger up with the defaults and marked logging as done. The
configure_logging(level=..., log_file=...) that main.py and
RAGPipeline.from_config() then make from config.yaml returned without doing
anything: the configured level never applied and the configured logs/rag.log
was never created.

The fix is two-sided, and so are these tests: the explicit call replaces the
implicit setup, and explicit calls stay idempotent among themselves (so a
CLI command that bootstraps logging and then builds the pipeline, which
bootstraps it again, does not stack handlers).
"""
import logging

import pytest

from src.utils import logger as L


@pytest.fixture
def fresh_logging(monkeypatch):
    """Start from 'nothing has configured logging yet' and put the real root
    logger back afterwards — pytest's own capture handlers live on it."""
    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level
    monkeypatch.setattr(L, "_CONFIGURED", False)
    monkeypatch.setattr(L, "_EXPLICIT", False)
    yield root
    for h in list(root.handlers):
        if h not in saved_handlers:
            h.close()                      # releases the log file on Windows
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)


def test_explicit_configure_after_implicit_get_logger_applies_its_arguments(
        fresh_logging, tmp_path):
    root = fresh_logging
    L.get_logger("tests.implicit")         # what every module's import-time call does
    assert root.level == logging.INFO

    log_file = tmp_path / "logs" / "rag.log"        # its folder does not exist yet
    L.configure_logging(level="DEBUG", log_file=log_file)

    assert root.level == logging.DEBUG
    # DEBUG is below the implicit INFO, so this record only reaches the file if
    # both the level AND the file handler from the explicit call took effect.
    logging.getLogger("tests.after").debug("record-for-the-file")
    for h in root.handlers:
        h.flush()
    assert "record-for-the-file" in log_file.read_text(encoding="utf-8")


def test_explicit_configure_replaces_the_implicit_handlers_instead_of_stacking(
        fresh_logging, tmp_path):
    root = fresh_logging
    L.get_logger("tests.implicit")
    (implicit,) = root.handlers            # exactly one: the stderr console handler

    L.configure_logging(log_file=tmp_path / "rag.log")

    assert implicit not in root.handlers
    # FileHandler subclasses StreamHandler, so naming the types pins exactly
    # one of each: one console handler, one file handler.
    assert sorted(type(h).__name__ for h in root.handlers) == [
        "FileHandler", "StreamHandler"]


def test_a_second_explicit_configure_is_a_no_op(fresh_logging, tmp_path):
    root = fresh_logging
    L.get_logger("tests.implicit")
    L.configure_logging(level="DEBUG", log_file=tmp_path / "first.log")
    handlers = list(root.handlers)

    L.configure_logging(level="ERROR", log_file=tmp_path / "second.log", console=False)

    assert root.handlers == handlers       # same objects: nothing added, nothing dropped
    assert root.level == logging.DEBUG     # the first explicit call wins
    assert not (tmp_path / "second.log").exists()


def test_get_logger_never_undoes_an_explicit_configure(fresh_logging, tmp_path):
    root = fresh_logging
    L.configure_logging(level="DEBUG", log_file=tmp_path / "rag.log")
    handlers = list(root.handlers)

    L.get_logger("tests.late")             # a module imported after startup

    assert root.handlers == handlers and root.level == logging.DEBUG
    # ...and the explicit call stays final: a further one is still ignored.
    L.configure_logging(level="ERROR")
    assert root.level == logging.DEBUG


def test_get_logger_alone_still_configures_the_defaults_once(fresh_logging):
    root = fresh_logging
    L.get_logger("tests.a")
    L.get_logger("tests.b")

    assert len(root.handlers) == 1 and root.level == logging.INFO


def test_the_log_file_is_appended_to_not_rotated(fresh_logging, tmp_path):
    """Several processes share logging.file and Windows refuses to rename a file
    another process holds open, so a rotating handler would fail every rollover
    and print a traceback per record. Appending is the safe default; see the
    module docstring for the known limit (the file grows)."""
    root = fresh_logging
    L.configure_logging(log_file=tmp_path / "rag.log")

    (fh,) = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
    assert type(fh) is logging.FileHandler and fh.mode == "a"


def test_main_bootstrap_applies_the_level_and_file_from_config(fresh_logging, tmp_path):
    """The config-driven caller itself, not just configure_logging: main.py
    imports (and so calls get_logger) before any command reads config.yaml."""
    import main
    from src.utils.config_loader import Config

    cfg = Config({"logging": {"level": "DEBUG", "file": "logs/rag.log",
                              "console": True}}, tmp_path)
    L.get_logger("tests.implicit")

    main._bootstrap_logging(cfg)

    assert fresh_logging.level == logging.DEBUG
    assert (tmp_path / "logs" / "rag.log").exists()
