"""
llm_client.py — One interface, three backends.

    anthropic -> Anthropic Claude via the Anthropic SDK
    openai    -> OpenAI GPT via the OpenAI SDK
    local     -> KoboldCPP / Ollama / vLLM via OpenAI-compatible /v1 endpoint

The point of this module: the rest of the pipeline NEVER imports a vendor SDK
directly. Swap the whole generation layer by editing one line in config.yaml.

    from src.llm.llm_client import LLMClient
    llm = LLMClient.from_config(cfg)
    text = llm.complete(system="...", user="...", temperature=0.1)

Local note (e.g. KoboldCPP serving a small open model):
    KoboldCPP exposes an OpenAI-compatible API at /v1, so we reuse the OpenAI
    SDK and just point base_url at it with a dummy key. Context length must
    be set when LAUNCHING KoboldCPP (--contextsize N); it cannot be set
    per-request.
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

    @classmethod
    def from_config(cls, cfg: Config, role: str = "generation") -> "LLMClient":
        """
        Build a client from the config section named by `role`
        (currently only 'generation', but kept generic for HyDE/verify reuse).
        """
        provider = cfg.get(f"{role}.provider", "openai")
        temperature = cfg.get(f"{role}.temperature", 0.1)
        max_tokens = cfg.get(f"{role}.max_tokens", 1500)

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

    def _build_client(self, provider: str, api_key, base_url):
        if provider == "anthropic":
            try:
                import anthropic
            except ImportError as e:
                raise ImportError("pip install anthropic") from e
            return anthropic.Anthropic(api_key=api_key)
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
