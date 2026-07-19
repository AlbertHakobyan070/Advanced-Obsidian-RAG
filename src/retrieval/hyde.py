"""
hyde.py — Hypothetical Document Embeddings.

Problem: a short question ("What is ARIMA?") and the lecture passage that
answers it live in different parts of embedding space — questions don't look
like answers. HyDE fixes this: ask the LLM to *write a hypothetical answer*,
then embed THAT and search with it. The fake answer is lexically and
semantically closer to real answer passages than the bare question is.

    query -> LLM drafts a plausible answer -> embed answer -> dense search

We keep it cheap: a small max_tokens, low temperature, one call. On a local
model (Gemma-4-E4B) this is essentially free.

Code-intent bypass: HyDE writes *prose*, so for "show me my ggplot code" the
hypothetical lands near lecture notes, not near actual code chunks — it biases
dense retrieval against the very thing being asked for. If the query trips one
of `retrieval.hyde_skip_signals` (config), expand() returns the raw query and
skips the LLM call entirely.

Usage:
    from src.retrieval.hyde import HyDE
    hyde = HyDE.from_config(cfg, llm)
    expanded = hyde.expand("What is ARIMA?")    # -> str (hypothetical answer)
    hyde.code_intent_signal("my ggplot code")   # -> "ggplot" (or None)
"""
from __future__ import annotations

import re

from src.llm.llm_client import LLMClient
from src.prompts.loader import load_prompt
from src.utils.config_loader import Config
from src.utils.logger import get_logger

log = get_logger(__name__)


def _compile_signals(signals: list[str]) -> list[tuple[str, re.Pattern]]:
    """
    Compile skip signals into word-boundary regexes.

    Boundaries are only asserted next to alphanumeric edge characters, so a
    trailing '_' or '.' acts as a prefix wildcard: "geom_" matches "geom_point"
    but "code" does NOT match "encoder". A simple plural is tolerated
    ("code" also matches "codes" — the author's actual ggplot query said "my codes").
    """
    compiled = []
    for sig in signals:
        sig = str(sig).strip()
        if not sig:
            continue
        pat = re.escape(sig.lower())
        if sig[0].isalnum():
            pat = r"\b" + pat
        if sig[-1].isalnum():
            pat = pat + r"s?\b"
        compiled.append((sig, re.compile(pat)))
    return compiled


class HyDE:
    def __init__(
        self,
        llm: LLMClient,
        enabled: bool = True,
        max_tokens: int = 200,
        skip_signals: list[str] | None = None,
    ):
        self.llm = llm
        self.enabled = enabled
        self.max_tokens = max_tokens
        self._signals = _compile_signals(skip_signals or [])
        self._prompt = load_prompt("hyde")

    @classmethod
    def from_config(cls, cfg: Config, llm: LLMClient) -> "HyDE":
        return cls(
            llm=llm,
            enabled=cfg.get("retrieval.use_hyde", True),
            max_tokens=200,
            skip_signals=cfg.get("retrieval.hyde_skip_signals", []),
        )

    def code_intent_signal(self, query: str) -> str | None:
        """Return the first matching skip signal, or None if the query is prose."""
        q = query.lower()
        for sig, pat in self._signals:
            if pat.search(q):
                return sig
        return None

    def expand(self, query: str, enabled: bool | None = None) -> str:
        """
        Return a hypothetical answer to embed. Falls back to the raw query.
        `enabled` overrides the configured default for this call only
        (presets pass use_hyde=False for code queries).
        """
        on = self.enabled if enabled is None else enabled
        if not on:
            return query
        signal = self.code_intent_signal(query)
        if signal:
            log.info("HyDE bypass: code-intent signal %r — searching with raw query", signal)
            return query
        try:
            resp = self.llm.complete(
                system=self._prompt["system"],
                user=self._prompt["user"].format(query=query),
                temperature=0.3,
                max_tokens=self.max_tokens,
            )
            hypo = resp.text.strip()
            if not hypo:
                return query
            # Concatenate query + hypothetical so we keep the original signal too.
            log.info("HyDE expanded query (%d chars)", len(hypo))
            return f"{query}\n\n{hypo}"
        except Exception as e:
            log.warning("HyDE failed (%s); using raw query", e)
            return query
