"""
loader.py — Load versioned prompt templates from YAML.

Prompts live in src/prompts/<name>.yaml, each with at least a 'system' key and
usually a 'user' template with {placeholders}. Keeping them in files (not
hardcoded) means you can tune wording without touching Python, and diff prompt
changes in git.

    from src.prompts.loader import load_prompt
    p = load_prompt("generation")
    p["system"], p["user"]
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

_PROMPT_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=None)
def load_prompt(name: str) -> dict:
    path = _PROMPT_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "system" not in data:
        raise ValueError(f"Prompt '{name}' missing required 'system' key")
    return data
