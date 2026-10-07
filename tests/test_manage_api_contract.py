"""Management-console capability-map contract tests.

The agent skills discover this surface through ``GET /api/schema``.  These
tests keep callable management routes from silently disappearing from that map.
"""

import re
from pathlib import Path

from src.utils.config_loader import Config

ROOT = Path(__file__).resolve().parents[1]


def test_management_schema_lists_agent_relevant_routes(management_module):
    manage_api = management_module
    contract = manage_api.api_schema()
    endpoints = contract["endpoints"]

    expected = {
        "GET /api/schema",
        "GET /api/import/ocr_scan",
        "GET /api/ocr/status",
        "POST /api/ocr/warm",
        "POST /api/providers/key",
        "POST /api/service/restart",
    }

    assert expected <= set(endpoints)
    assert endpoints["POST /api/providers/key"]["permission"] == "mutating"
    assert endpoints["POST /api/ocr/warm"]["permission"] == "mutating"
    assert "bill" in endpoints["POST /api/ocr/warm"]["purpose"]
    assert endpoints["POST /api/providers/key"]["purpose"].endswith(
        "credential-prefix mismatches."
    )
    assert "Windows lane only" not in endpoints[
        "POST /api/service/restart"
    ]["purpose"]


def test_provider_and_model_update_still_warns_about_wrong_key_type(
        monkeypatch, tmp_path, management_module):
    manage_api = management_module
    cfg = Config(
        {
            "providers": {
                "minimax": {
                    "kind": "anthropic",
                    "model": "MiniMax-M3",
                    "api_key_env": "TEST_MINIMAX_SETTINGS_KEY",
                    "api_key_prefix": "sk-cp-",
                }
            },
            "generation": {"provider": "openai", "model": "auto"},
        },
        tmp_path,
    )
    monkeypatch.setattr(
        "src.utils.config_loader.load_config", lambda *args, **kwargs: cfg
    )
    monkeypatch.setattr(
        manage_api, "_persist_section_keys", lambda path, changes: list(changes)
    )
    monkeypatch.setattr(
        manage_api, "_torch_devices", lambda: ({}, ["auto", "cpu"])
    )
    monkeypatch.setenv("TEST_MINIMAX_SETTINGS_KEY", "sk-api-paygo")

    result = manage_api.settings_update(manage_api.SettingsIn(changes={
        "generation.provider": "minimax",
        "generation.model": "MiniMax-M3",
    }))

    assert result["ok"] is True
    assert "wrong credential type" in result["note"]
    assert "sk-cp-" in result["note"]


def test_full_index_rebuild_is_tiered_destructive(management_module):
    """`main.py index` deletes the dense collection and re-embeds chunks.jsonl
    alone, so every appended lane drops out of both indexes. The tier is what a
    gating agent reads before running it, and "mutating" told it the wrong
    thing."""
    kinds = management_module.api_schema()["job_kinds"]
    assert kinds["index_rebuild"]["permission"] == "destructive"
    assert "appended lane" in kinds["index_rebuild"]["note"]
    # Only the rebuild drops content: an append is an idempotent upsert and the
    # BM25 rebuild re-derives the sparse half from every JSONL.
    assert kinds["index_append"]["permission"] == "mutating"
    assert kinds["rebuild_bm25"]["permission"] == "mutating"


def test_ingest_tab_hint_warns_that_the_rebuild_drops_appended_lanes():
    """The tier and the hint must tell the same story. The hint used to say
    "re-embeds everything", the opposite of what happens to every lane except
    chunks.jsonl."""
    html = (ROOT / "webui" / "index.html").read_text(encoding="utf-8")
    hint = re.search(r'index_rebuild:"([^"]+)"', html).group(1)
    assert "DESTRUCTIVE" in hint and "appended lane" in hint
    assert "re-embeds everything" not in hint
