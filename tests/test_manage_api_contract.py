"""Management-console capability-map contract tests.

The agent skills discover this surface through ``GET /api/schema``.  These
tests keep callable management routes from silently disappearing from that map.
"""

from src.utils.config_loader import Config


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
