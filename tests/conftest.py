"""Shared test fixtures.

The one thing worth centralising: several tests assert that *the config this
install ships* is well-formed. On a working machine that is `config.yaml`; in a
fresh clone `config.yaml` does not exist at all (it is gitignored — it holds
machine paths), and the shipped config is `config.example.yaml`. Asking for
`load_config()` unconditionally made those tests fail for every new clone,
which reads as "the project is broken" rather than "you haven't made your
config yet".
"""
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
