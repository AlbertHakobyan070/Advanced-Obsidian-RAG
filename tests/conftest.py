"""Shared test fixtures.

The one thing worth centralising: several tests assert that *the config this
install ships* is well-formed. On a working machine that is `config.yaml`; in a
fresh clone `config.yaml` does not exist at all (it is gitignored — it holds
machine paths), and the shipped config is `config.example.yaml`. Asking for
`load_config()` unconditionally made those tests fail for every new clone,
which reads as "the project is broken" rather than "you haven't made your
config yet".
"""
import importlib
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

ROOT = Path(__file__).resolve().parents[1]


def shipped_config():
    """The most authoritative config present: the live one, else the example."""
    from src.utils.config_loader import load_config
    for name in ("config.yaml", "config.example.yaml"):
        path = ROOT / name
        if path.exists():
            return load_config(path)
    pytest.skip("neither config.yaml nor config.example.yaml is present")


@pytest.fixture
def shipped_cfg():
    return shipped_config()


@pytest.fixture(autouse=True)
def _chroma_telemetry_off(monkeypatch):
    """Chroma raises ValueError when a second client opens a store path with
    settings that differ from the first client's. The repo opens every store with
    Settings(anonymized_telemetry=False) (src/utils/chroma_client.py), so a test
    that builds its own bare chromadb.PersistentClient(path) on a path the code
    under test also opens would clash with it. Default Settings reads this
    variable, so the two agree. (tests/test_chroma_telemetry.py checks what the
    code passes to chromadb itself, which this variable cannot mask.)"""
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "False")


@pytest.fixture
def management_module(tmp_path, monkeypatch):
    """Import the console against the config a fresh clone actually ships.

    ``manage_api`` intentionally requires ``config.yaml`` at import time, while
    the repository ships only ``config.example.yaml``. A temporary working
    directory keeps that runtime contract intact without writing a machine
    config into the checkout.
    """
    shutil.copyfile(ROOT / "config.example.yaml", tmp_path / "config.yaml")
    monkeypatch.chdir(tmp_path)
    return importlib.import_module("manage_api")
