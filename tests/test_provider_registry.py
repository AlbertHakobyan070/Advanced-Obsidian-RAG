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
        "kind": "openai",
        "base_url": "https://api.minimax.io/v1",
        "model": "MiniMax-M2",
        "api_key_env": "TEST_MINIMAX_KEY",
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
    monkeypatch.setenv("TEST_MINIMAX_KEY", "sk-test")
    llm = LLMClient.from_config(cfg_of({
        "providers": REGISTRY,
        "generation": {"provider": "minimax"},
    }))
    assert llm.provider == "openai"          # wire protocol, not vendor name
    assert llm.model == "MiniMax-M2"
    assert str(llm._client.base_url).startswith("https://api.minimax.io")


def test_key_comes_from_the_named_env_var(monkeypatch):
    monkeypatch.setenv("TEST_MINIMAX_KEY", "sk-from-env")
    llm = LLMClient.from_config(cfg_of({
        "providers": REGISTRY, "generation": {"provider": "minimax"},
    }))
    assert llm._client.api_key == "sk-from-env"


def test_missing_required_key_raises_naming_the_env_var(monkeypatch):
    monkeypatch.delenv("TEST_MINIMAX_KEY", raising=False)
    with pytest.raises(RuntimeError, match="TEST_MINIMAX_KEY"):
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
    monkeypatch.setenv("TEST_MINIMAX_KEY", "sk-test")
    llm = LLMClient.from_config(cfg_of({
        "providers": REGISTRY,
        "generation": {"provider": "minimax", "model": "MiniMax-Text-01"},
    }))
    assert llm.model == "MiniMax-Text-01"


def test_per_role_sections_resolve_independently(monkeypatch):
    """generation and eval.judge can sit on different backends."""
    monkeypatch.setenv("TEST_MINIMAX_KEY", "sk-test")
    data = {
        "providers": REGISTRY,
        "generation": {"provider": "proxy"},
        "eval": {"judge": {"provider": "minimax", "max_tokens": 700}},
    }
    gen = LLMClient.from_config(cfg_of(data), role="generation")
    judge = LLMClient.from_config(cfg_of(data), role="eval.judge")
    assert gen.model == "auto"
    assert judge.model == "MiniMax-M2"
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
    assert llm.provider == "openai" and llm.model == "auto"


def test_legacy_local_provider_still_works():
    llm = LLMClient.from_config(cfg_of({
        "providers": REGISTRY,
        "generation": {"provider": "local",
                       "local": {"model": "gemma", "base_url": "http://x/v1"}},
    }))
    assert llm.model == "gemma"


def test_shipped_config_yaml_registry_is_valid():
    """The real config.yaml must parse and obey the reserved-name rule."""
    from src.utils.config_loader import load_config
    cfg = load_config()
    registry = cfg.get("providers", {}) or {}
    assert registry, "config.yaml lost its providers: block"
    for name, spec in registry.items():
        assert name not in LLMClient.RESERVED_PROVIDERS, \
            f"providers.{name} shadows a reserved legacy name"
        assert str(spec.get("kind", "openai")).lower() in ("openai", "anthropic")
        assert spec.get("model"), f"providers.{name} has no model"
