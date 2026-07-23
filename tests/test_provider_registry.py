"""Provider-registry resolution tests (session 16).

Run:  python -m pytest tests/ -q

Guards the swap-any-backend feature in src/llm/llm_client.py. The rule that
must never break: adding a `providers:` registry cannot change how an EXISTING
config resolves. Pre-registry configs say `provider: openai` with a base_url
pointing at a local proxy, and they have to keep hitting the legacy path even
though "openai" is a plausible registry name — hence RESERVED_PROVIDERS.

No network: constructing an OpenAI/Anthropic SDK client does not call out, so
these run offline.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.llm.llm_client import LLMClient
from src.utils.config_loader import Config

ROOT = Path(__file__).resolve().parents[1]


def cfg_of(data: dict) -> Config:
    return Config(data, ROOT)


REGISTRY = {
    "minimax": {
        "kind": "anthropic",
        "base_url": "https://api.minimax.io/anthropic",
        "model": "MiniMax-M3",
        "api_key_env": "TEST_MINIMAX_KEY",
        "api_key_prefix": "sk-cp-",
    },
    "kobold": {
        "kind": "openai",
        "base_url": "http://localhost:5001/v1",
        "model": "local-model",
        "api_key_env": None,
    },
    "proxy": {
        "kind": "openai",
        "base_url": "http://localhost:3001/v1",
        "model": "auto",
        "api_key_env": "TEST_OPTIONAL_KEY",
        "api_key_optional": True,
    },
}


# ------------------------------------------------------------ registry path

def test_registry_entry_resolves_endpoint_and_model(monkeypatch):
    monkeypatch.setenv("TEST_MINIMAX_KEY", "sk-cp-test")
    llm = LLMClient.from_config(cfg_of({
        "providers": REGISTRY,
        "generation": {"provider": "minimax"},
    }))
    assert llm.provider == "anthropic"       # wire protocol, not vendor name
    assert llm.backend == "minimax"          # configured registry alias
    assert llm.model == "MiniMax-M3"
    assert str(llm._client.base_url).startswith("https://api.minimax.io")


def test_key_comes_from_the_named_env_var(monkeypatch):
    monkeypatch.setenv("TEST_MINIMAX_KEY", "sk-cp-from-env")
    llm = LLMClient.from_config(cfg_of({
        "providers": REGISTRY, "generation": {"provider": "minimax"},
    }))
    assert llm._client.api_key == "sk-cp-from-env"


def test_missing_required_key_raises_naming_the_env_var(monkeypatch):
    monkeypatch.delenv("TEST_MINIMAX_KEY", raising=False)
    with pytest.raises(RuntimeError, match="TEST_MINIMAX_KEY"):
        LLMClient.from_config(cfg_of({
            "providers": REGISTRY, "generation": {"provider": "minimax"},
        }))


def test_declared_key_prefix_rejects_wrong_credential_type(monkeypatch):
    monkeypatch.setenv("TEST_MINIMAX_KEY", "sk-api-paygo")
    with pytest.raises(
        RuntimeError,
        match=r"TEST_MINIMAX_KEY.*wrong credential type.*sk-cp-",
    ):
        LLMClient.from_config(cfg_of({
            "providers": REGISTRY, "generation": {"provider": "minimax"},
        }))


def test_optional_and_absent_keys_do_not_raise(monkeypatch):
    monkeypatch.delenv("TEST_OPTIONAL_KEY", raising=False)
    LLMClient.from_config(cfg_of({
        "providers": REGISTRY, "generation": {"provider": "proxy"},
    }))
    # api_key_env: null -> keyless local endpoint
    LLMClient.from_config(cfg_of({
        "providers": REGISTRY, "generation": {"provider": "kobold"},
    }))


def test_role_model_overrides_provider_model(monkeypatch):
    monkeypatch.setenv("TEST_MINIMAX_KEY", "sk-cp-test")
    llm = LLMClient.from_config(cfg_of({
        "providers": REGISTRY,
        "generation": {"provider": "minimax", "model": "MiniMax-Text-01"},
    }))
    assert llm.model == "MiniMax-Text-01"


def test_per_role_sections_resolve_independently(monkeypatch):
    """generation and eval.judge can sit on different backends."""
    monkeypatch.setenv("TEST_MINIMAX_KEY", "sk-cp-test")
    data = {
        "providers": REGISTRY,
        "generation": {"provider": "proxy"},
        "eval": {"judge": {"provider": "minimax", "max_tokens": 700}},
    }
    gen = LLMClient.from_config(cfg_of(data), role="generation")
    judge = LLMClient.from_config(cfg_of(data), role="eval.judge")
    assert gen.model == "auto"
    assert judge.model == "MiniMax-M3"
    assert judge.default_max_tokens == 700


def test_bad_kind_is_rejected(monkeypatch):
    with pytest.raises(ValueError, match="kind"):
        LLMClient.from_config(cfg_of({
            "providers": {"weird": {"kind": "grpc", "model": "m"}},
            "generation": {"provider": "weird"},
        }))


def test_unknown_provider_lists_the_known_ones():
    with pytest.raises(ValueError, match="not in the `providers:` registry"):
        LLMClient.from_config(cfg_of({
            "providers": REGISTRY, "generation": {"provider": "typo"},
        }))


# ------------------------------------------ per-request provider override

def test_provider_override_uses_backend_default_without_mutating_config(
        monkeypatch):
    monkeypatch.setenv("TEST_MINIMAX_KEY", "sk-cp-test")
    data = {
        "providers": REGISTRY,
        "generation": {
            "provider": "minimax",
            # This process-wide role model must not leak into another backend.
            "model": "MiniMax-Text-01",
            "temperature": 0.25,
            "max_tokens": 700,
        },
    }
    cfg = cfg_of(data)

    llm = LLMClient.from_provider_override(cfg, "kobold")

    assert llm.backend == "kobold"
    assert llm.provider == "openai"
    assert llm.model == "local-model"
    assert llm.default_temperature == 0.25
    assert llm.default_max_tokens == 700
    assert cfg.get("generation.provider") == "minimax"
    assert cfg.get("generation.model") == "MiniMax-Text-01"


def test_provider_override_accepts_explicit_model():
    llm = LLMClient.from_provider_override(
        cfg_of({
            "providers": REGISTRY,
            "generation": {"provider": "minimax", "model": "stale-role-model"},
        }),
        "kobold",
        model="request-model",
    )
    assert llm.model == "request-model"


def test_provider_override_rejects_unknown_backend():
    with pytest.raises(ValueError, match="not in the `providers:` registry"):
        LLMClient.from_provider_override(
            cfg_of({
                "providers": REGISTRY,
                "generation": {"provider": "minimax"},
            }),
            "typo",
        )


# ------------------------------------------- legacy path must not regress

def test_reserved_name_ignores_a_registry_entry_of_the_same_name(monkeypatch):
    """The whole point of RESERVED_PROVIDERS.

    A pre-registry config says provider: openai + a local base_url. If someone
    later adds a registry entry called "openai" pointing at the real API, this
    config must STILL resolve the legacy way — otherwise every existing
    install silently starts calling a different endpoint.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-legacy")
    llm = LLMClient.from_config(cfg_of({
        "providers": {"openai": {"kind": "openai",
                                 "base_url": "https://api.openai.com/v1",
                                 "model": "gpt-4o-mini"}},
        "generation": {"provider": "openai",
                       "base_url": "http://localhost:3001/v1",
                       "model": "auto"},
    }))
    assert llm.model == "auto"                                   # legacy model
    assert "localhost:3001" in str(llm._client.base_url)          # legacy url


def test_legacy_config_without_any_registry_still_works(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-legacy")
    llm = LLMClient.from_config(cfg_of({
        "generation": {"provider": "openai",
                       "base_url": "http://localhost:3001/v1",
                       "model": "auto"},
    }))
    assert llm.provider == "openai" and llm.backend == "openai"
    assert llm.model == "auto"


def test_legacy_local_provider_still_works():
    llm = LLMClient.from_config(cfg_of({
        "providers": REGISTRY,
        "generation": {"provider": "local",
                       "local": {"model": "gemma", "base_url": "http://x/v1"}},
    }))
    assert llm.model == "gemma"


def test_shipped_config_yaml_registry_is_valid():
    """The shipped config must parse and obey the reserved-name rule."""
    from conftest import shipped_config
    cfg = shipped_config()
    registry = cfg.get("providers", {}) or {}
    assert registry, "the shipped config lost its providers: block"
    for name, spec in registry.items():
        assert name not in LLMClient.RESERVED_PROVIDERS, \
            f"providers.{name} shadows a reserved legacy name"
        assert str(spec.get("kind", "openai")).lower() in ("openai", "anthropic")
        assert spec.get("model"), f"providers.{name} has no model"


def test_shipped_legacy_local_block_is_under_generation():
    """The reserved local provider reads generation.local.*, never retrieval."""
    from conftest import shipped_config

    cfg = shipped_config()
    assert cfg.get("generation.local.base_url")
    assert cfg.get("generation.local.model")
    assert cfg.get("retrieval.local") is None


def test_management_key_writer_rejects_wrong_declared_key_type(
        management_module):
    management = management_module

    response = management.provider_key(management.ProviderKeyIn(
        env="MINIMAX_API_KEY",
        value="sk-api-paygo-key",
    ))

    assert response.status_code == 400
    assert b"wrong key type" in response.body
    assert b"sk-cp-" in response.body
