"""Images through LLMClient.complete (src/llm/llm_client.py).

What must never happen: a text-only request changing shape, or an image going to
a backend that was not marked `vision: true` (some OpenAI-compatible endpoints
drop image parts without a word). No network: the SDK clients are fakes.
"""
import base64
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from src.llm.llm_client import IMAGE_MIMES, LLMClient
from src.utils.config_loader import Config

ROOT = Path(__file__).resolve().parents[1]
PNG = b"\x89PNG\r\n\x1a\nfake"


def _client(provider: str, vision: bool = False):
    """A client whose SDK records every create() call and answers "ok"."""
    calls = []

    def create(**kw):
        calls.append(kw)
        if provider == "anthropic":
            return NS(content=[NS(type="text", text="ok")], usage=None)
        return NS(choices=[NS(message=NS(content="ok"))], usage=None, model="m")
    llm = LLMClient.__new__(LLMClient)
    llm.provider, llm.backend, llm.model = provider, "kobold", "m"
    llm.default_temperature, llm.default_max_tokens = 0.1, 100
    if vision:
        llm.vision = True
    llm._client = NS(messages=NS(create=create), chat=NS(completions=NS(create=create)))
    return llm, calls


def test_text_only_requests_are_unchanged():
    llm, calls = _client("openai")
    assert llm.complete("sys", "usr", temperature=0.0, max_tokens=5).text == "ok"
    assert calls == [{"model": "m", "temperature": 0.0, "max_tokens": 5,
                      "messages": [{"role": "system", "content": "sys"},
                                   {"role": "user", "content": "usr"}]}]
    llm, calls = _client("anthropic")
    assert llm.complete("sys", "usr", temperature=0.0, max_tokens=5).text == "ok"
    assert calls == [{"model": "m", "system": "sys", "max_tokens": 5, "temperature": 0.0,
                      "messages": [{"role": "user", "content": "usr"}]}]


def test_images_go_to_openai_as_image_url_parts():
    llm, calls = _client("openai", vision=True)
    llm.complete("sys", "usr", images=[(PNG, "image/png")])
    b64 = base64.b64encode(PNG).decode("ascii")
    assert calls[0]["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": [
            {"type": "text", "text": "usr"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}]


def test_images_go_to_anthropic_as_base64_blocks():
    llm, calls = _client("anthropic", vision=True)
    llm.complete("sys", "usr", images=[(PNG, "image/png"), (b"jpg", "image/jpeg")])
    assert calls[0]["system"] == "sys"
    assert calls[0]["messages"] == [{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                     "data": base64.b64encode(PNG).decode("ascii")}},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                     "data": base64.b64encode(b"jpg").decode("ascii")}},
        {"type": "text", "text": "usr"}]}]


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_images_to_a_client_without_vision_are_refused_before_any_call(provider):
    llm, calls = _client(provider)
    with pytest.raises(ValueError, match=r"providers\.kobold is not marked vision: true"):
        llm.complete("sys", "usr", images=[(PNG, "image/png")])
    assert calls == []


def test_an_unsupported_image_type_is_refused():
    llm, calls = _client("openai", vision=True)
    with pytest.raises(ValueError, match="image/bmp"):
        llm.complete("sys", "usr", images=[(PNG, "image/png"), (b"BM", "image/bmp")])
    assert calls == [] and "image/bmp" not in IMAGE_MIMES


def _registry_client(vision):
    entry = {"kind": "openai", "base_url": "http://localhost:1/v1", "model": "m"}
    if vision is not ...:
        entry["vision"] = vision
    return LLMClient.from_config(Config({"providers": {"v": entry},
                                         "generation": {"provider": "v"}}, ROOT))


def test_the_registry_vision_flag_reaches_the_client(monkeypatch):
    assert _registry_client(True).vision is True
    assert _registry_client(False).vision is False
    assert _registry_client(...).vision is False                      # absent
    for bad in ("true", 1, "yes"):
        with pytest.raises(ValueError, match=r"providers\.v\.vision = .* must be true or false"):
            _registry_client(bad)
    # The reserved names never read a registry entry, so they stay text-only even
    # when an entry under the same name says vision: true.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    shadow = {"providers": {"openai": {"kind": "openai", "model": "x", "vision": True}},
              "generation": {"provider": "openai", "model": "auto"}}
    assert LLMClient.from_config(Config(shadow, ROOT)).vision is False
    assert LLMClient.from_provider_override(Config(shadow, ROOT), "local").vision is False
