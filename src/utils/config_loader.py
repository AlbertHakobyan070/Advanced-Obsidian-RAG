"""
config_loader.py — Load config.yaml + .env into a single accessor.

Usage:
    from src.utils.config_loader import load_config
    cfg = load_config()
    cfg.get("generation.provider")          # -> "anthropic"
    cfg.get("retrieval.dense_top_k", 20)     # -> 20
    cfg.secret("ANTHROPIC_API_KEY")          # -> from environment
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


class Config:
    """Thin wrapper over the parsed YAML dict with dotted-key access."""

    def __init__(self, data: dict[str, Any], project_root: Path):
        self._data = data
        self.project_root = project_root

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Fetch a nested value via 'a.b.c' notation."""
        node: Any = self._data
        for part in dotted_key.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def path(self, dotted_key: str, default: str | None = None) -> Path:
        """Like get() but resolves the value relative to project root."""
        raw = self.get(dotted_key, default)
        if raw is None:
            raise KeyError(f"No path configured at '{dotted_key}'")
        p = Path(raw)
        return p if p.is_absolute() else (self.project_root / p)

    @staticmethod
    def secret(env_key: str, default: str | None = None) -> str | None:
        """Read a secret from the environment (.env already loaded)."""
        return os.environ.get(env_key, default)

    def require_secret(self, env_key: str) -> str:
        val = self.secret(env_key)
        if not val:
            raise RuntimeError(
                f"Missing required secret '{env_key}'. "
                f"Add it to your .env file (see .env.example)."
            )
        return val

    def as_dict(self) -> dict[str, Any]:
        return self._data


def persist_config_values(cfg_path: str | Path, changes: dict[str, Any]) -> list[str]:
    """
    Rewrite scalar values in config.yaml IN PLACE, preserving comments and
    layout (no yaml round-trip, so nothing else in the file moves).

    `changes` maps LEAF key names (e.g. "rerank_top_k") to new values. Each key
    must appear exactly once at the start of a line (inline preset maps like
    `code: {rerank_top_k: 10}` don't count) — ambiguous or missing keys raise
    instead of guessing. None values are skipped. Returns the keys rewritten.
    """
    path = Path(cfg_path)
    text = path.read_text(encoding="utf-8")
    written: list[str] = []

    for key, value in changes.items():
        if value is None:
            continue
        if isinstance(value, bool):
            sval = "true" if value else "false"
        else:
            sval = str(value)
        # Match "  key: value   # optional comment" — keep indent and comment.
        pattern = re.compile(
            rf"(?m)^(?P<pre>[ \t]*{re.escape(key)}[ \t]*:[ \t]*)(?P<val>[^#\r\n]*?)(?P<post>[ \t]*(?:#[^\r\n]*)?)$"
        )
        matches = list(pattern.finditer(text))
        if len(matches) != 1:
            raise ValueError(
                f"Key '{key}' appears {len(matches)} times in {path.name}; "
                "refusing to rewrite ambiguously."
            )
        m = matches[0]
        text = text[: m.start()] + m.group("pre") + sval + m.group("post") + text[m.end():]
        written.append(key)

    if written:
        path.write_text(text, encoding="utf-8")
    return written


_ENV_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env(node: Any) -> Any:
    """Expand ${VAR} / ${VAR:-default} inside string values, recursively.

    Exists for the containerised builds: the vault lives at a path only the
    person running Docker knows, and it has to appear in TWO places that must
    agree — the compose bind-mount and parser.vault_path. Pointing both at
    ${RAG_HOST_HOME} means one line in .env instead of two edits that silently
    drift apart.

    An unset variable with no default is left as the literal ${VAR}: blanking
    it would turn a vault path into "" and make the parser scan the filesystem
    root. Left intact, it shows up in the console as an obviously-unset path.
    """
    if isinstance(node, str):
        return _ENV_REF_RE.sub(
            lambda m: os.environ.get(m.group(1),
                                     m.group(2) if m.group(2) is not None
                                     else m.group(0)),
            node)
    if isinstance(node, dict):
        return {k: expand_env(v) for k, v in node.items()}
    if isinstance(node, list):
        return [expand_env(v) for v in node]
    return node


def load_config(config_path: str | Path | None = None) -> Config:
    """
    Locate and load config.yaml, plus the sibling .env.

    Search order for config.yaml:
      1. explicit `config_path` argument
      2. ./config.yaml (current working dir)
      3. <project_root>/config.yaml  (two levels up from this file)
    """
    if config_path is not None:
        cfg_file = Path(config_path)
        project_root = cfg_file.resolve().parent
    else:
        candidates = [
            Path.cwd() / "config.yaml",
            Path(__file__).resolve().parents[2] / "config.yaml",
        ]
        cfg_file = next((c for c in candidates if c.exists()), None)
        if cfg_file is None:
            raise FileNotFoundError(
                "config.yaml not found in cwd or project root. "
                "Pass an explicit path to load_config()."
            )
        project_root = cfg_file.resolve().parent

    # Load .env from the same directory as config.yaml (if present)
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        load_dotenv()  # fall back to any .env on the default search path

    with open(cfg_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # After load_dotenv above, so .env values are visible to the expansion.
    cfg = Config(expand_env(data), project_root)

    # Apply `taxonomy:` over the parser's built-in folder/course maps. This
    # lives here, not in each of main.py's eleven load_config() call sites,
    # because the maps are process-global: configuring them anywhere but the
    # single point where config enters the process means one forgotten call
    # site silently ingests with the wrong taxonomy. obsidian_parser imports
    # nothing but stdlib + yaml, so there is no cycle and no real import cost.
    from src.ingestion.obsidian_parser import configure_taxonomy
    configure_taxonomy(cfg)

    return cfg
