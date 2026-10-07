"""
registry.py — Which embedder a config means, resolved once and validated.

`embedding.provider` is one of:
  local   -> a sentence-transformers model on this machine
             (embedding.local_model, embedding.device)
  openai  -> LEGACY: OpenAI's own API (OPENAI_API_KEY, embedding.model,
             embedding.dimensions). Kept so every pre-registry config resolves
             exactly as it always did.
  <name>  -> an entry of `embedding.providers:`: anything serving an
             OpenAI-compatible /v1/embeddings (Ollama, llama.cpp --embedding,
             a gateway, OpenAI itself).

Mirrors the LLM registry in src/llm/llm_client.py, with one deliberate
difference. That registry silently ignores a reserved name used as an entry and
passes unknown keys through. Here both are errors: a prefix or `dimensions` that
a typo turned into a no-op yields vectors built differently from what the file
says, and nothing downstream would notice.

Stdlib only (no torch, no openai) so the console, the bench and the tests can
resolve a spec without loading a model. A spec never holds a secret: it carries
the NAME of the environment variable, like the LLM registry's `api_key_env`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.utils.config_loader import Config

# Always the built-in paths. A registry entry under one of these names would
# silently redirect every pre-registry config.
RESERVED_EMBEDDING_PROVIDERS = ("local", "openai")
# The wire protocols a `providers:` entry may use (the protocol, not the vendor).
ENTRY_KINDS = ("openai",)
DEVICE_PATTERN = re.compile(r"^(auto|cpu|mps|cuda(:\d+)?)$")
ENTRY_KEYS = frozenset({
    "kind", "base_url", "model", "api_key_env", "api_key_optional",
    "dimensions", "query_prefix", "doc_prefix", "label", "description",
})


@dataclass(frozen=True)
class EmbeddingSpec:
    """Everything that decides which vectors an embedder produces, plus the
    non-secret facts needed to build it. Frozen: a spec is a value that is
    compared with a collection's recorded fingerprint, never edited in place."""

    provider: str                  # label: "local", legacy "openai", or the providers: entry name
    kind: str                      # "local" | "openai": the runtime / wire protocol
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    api_key_optional: bool = False
    dimensions: int | None = None
    device: str = "auto"
    query_prefix: str = ""
    doc_prefix: str = ""

    @property
    def normalize(self) -> bool:
        """Local vectors are L2-normalised by the embedder; hosted ones are
        passed through as the endpoint returns them. Recorded with a
        collection, not a knob."""
        return self.kind == "local"

    def identity(self) -> dict:
        """What must match between the configured embedder and a collection's
        vectors. Not in it: the provider NAME, base_url, device and batch size,
        none of which changes the vector space."""
        return {
            "kind": self.kind,
            "model": self.model,
            "normalize": self.normalize,
            "query_prefix": self.query_prefix,
            "doc_prefix": self.doc_prefix,
        }

    def label(self) -> str:
        return f"{self.provider} · {self.model}"


def check_prefix(value: Any, where: str) -> str:
    """A query/doc prefix: None means none. One line, no '#' and no '\\', since
    the console's in-place YAML writer cannot round-trip them. Returned
    untouched: a trailing space ("query: ") is part of the prefix."""
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{where} = {value!r} must be a string")
    if any(ch in value for ch in "\r\n#\\"):
        raise ValueError(
            f"{where} = {value!r} must be a single line with no '#' or '\\' "
            f"(the console's config writer cannot round-trip them)"
        )
    return value


def check_device(value: Any, where: str) -> str:
    """`auto` (let torch pick) | cpu | mps | cuda | cuda:N, lower-cased."""
    if isinstance(value, str) and DEVICE_PATTERN.fullmatch(value.lower()):
        return value.lower()
    raise ValueError(
        f"{where} = {value!r} is not a device; use auto, cpu, mps, cuda or cuda:N"
    )


def resolve_embedding_spec(cfg: "Config", provider: str | None = None) -> EmbeddingSpec:
    """The embedder `cfg` configures, or the one named by `provider` (a
    re-embed target) resolved against the same registry. An unset
    `embedding.provider` still means legacy `openai`, as it always has. Only
    the selected entry is validated, so a broken sibling cannot take down a
    service that is not using it."""
    name = provider if provider is not None else cfg.get("embedding.provider", "openai")

    registry = cfg.get("embedding.providers", {})
    if registry is None:                       # `providers:` left empty in YAML
        registry = {}
    if not isinstance(registry, dict):
        raise ValueError(
            f"embedding.providers must be a mapping of name -> entry, "
            f"got {type(registry).__name__}"
        )
    for reserved in RESERVED_EMBEDDING_PROVIDERS:
        if reserved in registry:
            raise ValueError(
                f"embedding.providers.{reserved} is reserved: "
                f"{RESERVED_EMBEDDING_PROVIDERS} always take the built-in paths, "
                f"so an entry under that name would silently redirect every "
                f"pre-registry config. Use another name."
            )

    if name == "local":
        return EmbeddingSpec(
            provider="local",
            kind="local",
            model=cfg.get("embedding.local_model", "BAAI/bge-small-en-v1.5"),
            device=check_device(cfg.get("embedding.device", "auto"), "embedding.device"),
            query_prefix=check_prefix(cfg.get("embedding.query_prefix", ""), "embedding.query_prefix"),
            doc_prefix=check_prefix(cfg.get("embedding.doc_prefix", ""), "embedding.doc_prefix"),
        )
    if name == "openai":
        return EmbeddingSpec(
            provider="openai",
            kind="openai",
            model=cfg.get("embedding.model", "text-embedding-3-small"),
            api_key_env="OPENAI_API_KEY",
            dimensions=cfg.get("embedding.dimensions"),
            query_prefix=check_prefix(cfg.get("embedding.query_prefix", ""), "embedding.query_prefix"),
            doc_prefix=check_prefix(cfg.get("embedding.doc_prefix", ""), "embedding.doc_prefix"),
        )
    if not isinstance(name, str) or name not in registry:
        raise ValueError(
            f"embedding.provider = {name!r} is not in embedding.providers and "
            f"is not one of {RESERVED_EMBEDDING_PROVIDERS}. "
            f"Known providers: {sorted(str(k) for k in registry)}"
        )
    return _spec_from_entry(name, registry[name])


def _spec_from_entry(name: str, entry: Any) -> EmbeddingSpec:
    """Validate one `embedding.providers:` entry and turn it into a spec. A
    prefix or `dimensions` that did not take effect would change the vectors
    without any error, so anything unrecognised is refused outright."""
    where = f"embedding.providers.{name}"
    if not isinstance(entry, dict):
        raise ValueError(f"{where} must be a mapping of settings, got {type(entry).__name__}")
    unknown = sorted(str(k) for k in set(entry) - ENTRY_KEYS)
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {unknown}; allowed: {sorted(ENTRY_KEYS)}")

    kind = str(entry.get("kind", "openai")).lower()
    if kind not in ENTRY_KINDS:
        raise ValueError(
            f"{where}.kind = {entry.get('kind')!r}; expected one of {ENTRY_KINDS} "
            f"(the wire protocol, not the vendor)"
        )
    model = entry.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"{where}.model is required: the model id the endpoint serves")
    dimensions = entry.get("dimensions")
    if dimensions is not None and (
            isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions < 1):
        raise ValueError(f"{where}.dimensions = {dimensions!r} must be a positive integer or null")
    base_url = entry.get("base_url")
    if base_url is not None and not isinstance(base_url, str):
        raise ValueError(f"{where}.base_url = {base_url!r} must be a string or null")
    api_key_env = entry.get("api_key_env")
    if api_key_env is not None and not isinstance(api_key_env, str):
        raise ValueError(
            f"{where}.api_key_env = {api_key_env!r} must be the NAME of an "
            f"environment variable (a string) or null"
        )
    api_key_optional = entry.get("api_key_optional", False)
    if not isinstance(api_key_optional, bool):
        raise ValueError(f"{where}.api_key_optional = {api_key_optional!r} must be true or false")

    return EmbeddingSpec(
        provider=name,
        kind=kind,
        model=model,
        base_url=base_url,
        api_key_env=api_key_env,
        api_key_optional=api_key_optional,
        dimensions=dimensions,
        query_prefix=check_prefix(entry.get("query_prefix"), f"{where}.query_prefix"),
        doc_prefix=check_prefix(entry.get("doc_prefix"), f"{where}.doc_prefix"),
    )


# The console's model picker: models whose facts are known, NOT a whitelist (any
# sentence-transformers id or hosted model is accepted). `dimension` is the
# vector size a collection built with the model will have. `prefixes_required`
# means retrieval degrades without them; bge v1.5 only suggests a query
# instruction. Sources: the FlagEmbedding docs (bge sizes, instruction, dims)
# and the e5 authors' repo (microsoft/unilm, e5/: dims, and the eval code that
# embeds queries as "query: ..." and passages as "passage: ...").
KNOWN_EMBEDDERS: dict[str, dict] = {
    "BAAI/bge-small-en-v1.5": {
        "label": "bge-small-en-v1.5 (384-d, 33M, English)",
        "dimension": 384,
        "multilingual": False,
        "query_prefix": "",
        "doc_prefix": "",
        "prefixes_required": False,
        "note": "The shipped default. Optional query instruction (queries only): "
                "\"Represent this sentence for searching relevant passages: \".",
    },
    "BAAI/bge-base-en-v1.5": {
        "label": "bge-base-en-v1.5 (768-d, 109M, English)",
        "dimension": 768,
        "multilingual": False,
        "query_prefix": "",
        "doc_prefix": "",
        "prefixes_required": False,
        "note": "About 3x bge-small on CPU. Same optional query instruction as bge-small.",
    },
    "BAAI/bge-m3": {
        "label": "bge-m3 (1024-d, ~570M, multilingual)",
        "dimension": 1024,
        "multilingual": True,
        "query_prefix": "",
        "doc_prefix": "",
        "prefixes_required": False,
        "note": "An order of magnitude over bge-small on CPU; GPU advised. Needs no instruction.",
    },
    "intfloat/multilingual-e5-base": {
        "label": "multilingual-e5-base (768-d, multilingual)",
        "dimension": 768,
        "multilingual": True,
        "query_prefix": "query: ",
        "doc_prefix": "passage: ",
        "prefixes_required": True,
        "note": "Needs both prefixes: the e5 authors' own eval embeds queries as "
                "\"query: ...\" and passages as \"passage: ...\".",
    },
}
