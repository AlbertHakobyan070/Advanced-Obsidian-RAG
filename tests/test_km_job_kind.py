"""The `map` console job kind: argv, refusals (400), and the mutating tier (also over MCP)."""
import json
import sys

import pytest

from src.mcp_bridge.catalog import console_tiers


def test_build_argv_map(management_module):
    m = management_module
    assert m._build_argv("map", {"pdf": "b.pdf", "chapter": "Storage"}) == \
        [sys.executable, "main.py", "map", "b.pdf", "--chapter", "Storage"]
    assert m._build_argv("map", {"pdf": "b.pdf", "pages": "140-240", "template": "concept", "dry_run": True}) == \
        [sys.executable, "main.py", "map", "b.pdf", "--pages", "140-240", "--template", "concept", "--dry-run"]


def test_build_argv_map_title(management_module):
    """M3: a queued job can name its book (otherwise PDF metadata names the folder and canvas)."""
    assert management_module._build_argv("map", {"pdf": "Books/b.pdf", "pages": "1-2", "title": "DDIA"}) ==         [sys.executable, "main.py", "map", "Books/b.pdf", "--pages", "1-2", "--title", "DDIA"]


def test_map_params_refused(management_module):
    m = management_module
    for prm in ({}, {"pdf": ""}, {"pdf": 3, "chapter": "x"}):
        with pytest.raises(ValueError, match="pdf is required"):
            m._build_argv("map", prm)
    for prm in ({"pdf": "b.pdf"}, {"pdf": "b.pdf", "chapter": "x", "pages": "1-2"}):
        with pytest.raises(ValueError, match="exactly one of chapter or pages"):
            m._build_argv("map", prm)


def test_bad_map_params_are_a_400(management_module, monkeypatch):
    m = management_module
    monkeypatch.setattr(m, "_JOBS", {})
    monkeypatch.setattr(m, "_ORDER", [])
    r = m.jobs_create(m.JobIn(kind="map", params={"pdf": "b.pdf"}))
    assert r.status_code == 400 and "exactly one of chapter or pages" in json.loads(r.body)["error"]


def test_map_kind_has_mutating_tier(management_module):
    schema = management_module.api_schema()
    entry = schema["job_kinds"]["map"]
    assert entry["permission"] == "mutating"
    assert "staging" in entry["note"] and "never writes the vault" in entry["note"]
    assert "relative to parser.vault_path" in entry["params"]["pdf"] and "title" in entry["params"]
    _, _, kinds = console_tiers(schema)           # what the MCP catalog reads
    assert kinds["map"] == "mutating"
