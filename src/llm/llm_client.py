"""
llm_client.py — One interface, any OpenAI- or Anthropic-compatible backend.

The rest of the pipeline NEVER imports a vendor SDK directly, so the whole
generation layer swaps by editing one line in config.yaml.

    from src.llm.llm_client import LLMClient
    llm = LLMClient.from_config(cfg)                  # role="generation"
    judge = LLMClient.from_config(cfg, role="eval.judge")

Two ways to name a backend, and they coexist:

1. PROVIDER REGISTRY (preferred). `providers:` in config.yaml holds named
   backends; a role points at one by name:

       providers:
         minimax:
           kind: openai                       # wire protocol, not vendor
           base_url: "https://api.minimax.io/v1"
           model: "MiniMax-M2"
           api_key_env: MINIMAX_API_KEY       # the NAME of an env var
       generation:
         provider: minimax

   Keys are NEVER stored in config.yaml — a provider names the environment
   variable holding its key, and .env (gitignored) is loaded at startup. Add a
   provider for any service exposing an OpenAI-compatible /v1/chat/completions
   or an Anthropic-compatible endpoint: MiniMax, Moonshot/Kimi, xAI, OpenAI,
   a local KoboldCPP, an office proxy. `kind` selects the wire protocol only.

2. LEGACY (unchanged, still fully supported). The three names `openai`,
   `anthropic` and `local` are RESERVED and always take the original code
   path with their original env vars (OPENAI_API_KEY / ANTHROPIC_API_KEY).
   Never add a registry entry under those names — existing configs, including
   the Docker bundle, depend on them resolving the old way.

Per-role: any config section may carry `provider:` / `temperature:` /
`max_tokens:`, so generation, HyDE and the eval judge can each run on a
different backend. `<role>.model` overrides the provider's default model.

Local note (KoboldCPP serving Gemma-4-E4B-it-Q8):
    KoboldCPP exposes an OpenAI-compatible API at /v1, so we reuse the OpenAI
    SDK and just point base_url at it with a dummy key. Context length must be
    set when LAUNCHING KoboldCPP (--contextsize N); it cannot be set per-request.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.utils.config_loader import Config
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    usage: dict[str, Any] | None = None


class LLMClient:
    """Provider-agnostic chat completion wrapper."""

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        default_temperature: float = 0.1,
        default_max_tokens: int = 1500,
    ):
        self.provider = provider
        self.model = model
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self._client = self._build_client(provider, api_key, base_url)
        log.info("LLMClient ready: provider=%s model=%s", provider, model)

    # ---- construction -------------------------------------------------

    # Names that always take the legacy path below. A registry entry under one
    # of these would silently redirect every pre-registry config.
    RESERVED_PROVIDERS = ("openai", "anthropic", "local")

    @classmethod
    def from_config(cls, cfg: Config, role: str = "generation") -> "LLMClient":
        """
        Build a client for a config section: "generation", "eval.judge", ...

        Resolution order: a `providers:` registry entry whose name matches
        `<role>.provider` wins; otherwise the reserved legacy names apply.
        """
        provider = cfg.get(f"{role}.provider", "anthropic")
        temperature = cfg.get(f"{role}.temperature", 0.1)
        max_tokens = cfg.get(f"{role}.max_tokens", 1500)

        registry = cfg.get("providers", {}) or {}
        if provider in registry and provider not in cls.RESERVED_PROVIDERS:
            return cls._from_registry(
                cfg, role, provider, registry[provider], temperature, max_tokens
            )
        if provider not in cls.RESERVED_PROVIDERS and registry:
            raise ValueError(
                f"{role}.provider = {provider!r} is not in the `providers:` "
                f"registry and is not one of {cls.RESERVED_PROVIDERS}. "
                f"Known providers: {sorted(registry)}"
            )

        if provider == "anthropic":
            return cls(
                provider="anthropic",
                model=cfg.get(f"{role}.model", "claude-sonnet-4-20250514"),
                api_key=cfg.require_secret("ANTHROPIC_API_KEY"),
                default_temperature=temperature,
                default_max_tokens=max_tokens,
            )
        if provider == "openai":
            return cls(
                provider="openai",
                model=cfg.get(f"{role}.model", "gpt-4o-mini"),
                api_key=cfg.require_secret("OPENAI_API_KEY"),
                base_url=cfg.get(f"{role}.base_url"),
                default_temperature=temperature,
                default_max_tokens=max_tokens,
            )
        if provider == "local":
            return cls(
                provider="local",
                model=cfg.get(f"{role}.local.model", "local-model"),
                api_key=cfg.get(f"{role}.local.api_key", "local"),
                base_url=cfg.get(f"{role}.local.base_url", "http://localhost:5001/v1"),
                default_temperature=temperature,
                default_max_tokens=max_tokens,
            )
        raise ValueError(f"Unknown LLM provider: {provider!r}")

    @classmethod
    def _from_registry(cls, cfg: Config, role: str, name: str, spec: dict,
                       temperature: float, max_tokens: int) -> "LLMClient":
        """Build a client from a `providers:` entry.

        The provider owns the backend facts (endpoint, model, which env var
        holds its key); the role may override `model` to run a different model
        on the same backend. That override is logged, because pointing a role
        at a new provider while leaving a stale `<role>.model` behind is the
        one way to misconfigure this quietly — the model name simply gets
        rejected by the new endpoint.
        """
        spec = spec or {}
        kind = str(spec.get("kind", "openai")).lower()
        if kind not in ("openai", "anthropic"):
            raise ValueError(
                f"providers.{name}.kind = {kind!r}; expected 'openai' or "
                f"'anthropic' (the wire protocol, not the vendor)"
            )

        spec_model = spec.get("model")
        role_model = cfg.get(f"{role}.model")
        model = role_model or spec_model
        if not model:
            raise ValueError(
                f"No model for role {role!r}: set providers.{name}.model "
                f"or {role}.model"
            )

        # Keys come from the environment, never from config.yaml. A provider
        # with no api_key_env is a keyless local endpoint (KoboldCPP, Ollama,
        # a localhost proxy) — the OpenAI SDK still wants a non-empty string.
        key_env = spec.get("api_key_env")
        if key_env:
            api_key = (cfg.secret(key_env) if spec.get("api_key_optional")
                       else cfg.require_secret(key_env))
            api_key = api_key or "not-needed"
        else:
            api_key = "not-needed"

        if role_model and spec_model and role_model != spec_model:
            log.warning(
                "%s.model=%r overrides providers.%s.model=%r — make sure %s "
                "actually serves that model",
                role, role_model, name, spec_model, name,
            )
        log.info("LLMClient[%s]: provider=%s kind=%s model=%s", role, name, kind, model)

        return cls(
            provider=kind,
            model=model,
            api_key=api_key,
            base_url=spec.get("base_url"),
            default_temperature=temperature,
            default_max_tokens=max_tokens,
        )

    def _build_client(self, provider: str, api_key, base_url):
        if provider == "anthropic":
            try:
                import anthropic
            except ImportError as e:
                raise ImportError("pip install anthropic") from e
            # base_url matters here too: several providers (MiniMax among them)
            # expose an Anthropic-compatible endpoint alongside their /v1 one.
            kwargs: dict[str, Any] = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            return anthropic.Anthropic(**kwargs)
        # openai + local both use the OpenAI SDK
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("pip install openai") from e
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAI(**kwargs)

    # ---- completion ---------------------------------------------------

    def complete(
        self,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Single-turn system+user completion. Synchronous."""
        temp = self.default_temperature if temperature is None else temperature
        maxtok = self.default_max_tokens if max_tokens is None else max_tokens

        if self.provider == "anthropic":
            return self._complete_anthropic(system, user, temp, maxtok)
        return self._complete_openai_compatible(system, user, temp, maxtok)

    def _complete_anthropic(self, system, user, temp, maxtok) -> LLMResponse:
        resp = self._client.messages.create(
            model=self.model,
            system=system,
            max_tokens=maxtok,
            temperature=temp,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        )
        usage = None
        if getattr(resp, "usage", None):
            usage = {
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
            }
        return LLMResponse(text=text, model=self.model, provider=self.provider, usage=usage)

    def _complete_openai_compatible(self, system, user, temp, maxtok) -> LLMResponse:
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=temp,
            max_tokens=maxtok,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = resp.choices[0].message.content or ""
        usage = None
        if getattr(resp, "usage", None):
            usage = {
                "input_tokens": resp.usage.prompt_tokens,
                "output_tokens": resp.usage.completion_tokens,
            }
        return LLMResponse(text=text, model=self.model, provider=self.provider, usage=usage)
