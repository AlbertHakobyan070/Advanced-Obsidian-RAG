"""
hyde.py — Hypothetical Document Embeddings.

Problem: a short question ("What is ARIMA?") and the lecture passage that
answers it live in different parts of embedding space — questions don't look
like answers. HyDE fixes this: ask the LLM to *write a hypothetical answer*,
then embed THAT and search with it. The fake answer is lexically and
semantically closer to real answer passages than the bare question is.

    query -> LLM drafts a plausible answer -> "query\n\ndraft" -> retrieve()

What comes back is the query WITH the draft appended, and the pipeline hands
that text to retrieve() — so it drives every lane, BM25 and the live-vault
lane included, not only the dense one. Scope detection and reranking still
see the raw question.

We keep it cheap: max_tokens 200, one call. The price of its temperature
(0.3) is repeatability: two identical queries get different drafts and can
retrieve different chunks. A disk cache (retrieval.hyde_cache, see
hyde_cache.py) keeps the first draft per question and reuses it, which makes
repeated searches identical; without one, switch HyDE off for any A/B
comparison or eval.

Code-intent bypass: HyDE writes *prose*, so for "show me my ggplot code" the
hypothetical lands near lecture notes, not near actual code chunks — it biases
dense retrieval against the very thing being asked for. If the query trips one
of `retrieval.hyde_skip_signals` (config), expand() returns the raw query and
skips the LLM call entirely.

Usage:
    from src.retrieval.hyde import HyDE
    hyde = HyDE.from_config(cfg, llm)
    expanded = hyde.expand("What is ARIMA?")    # -> str (hypothetical answer)
    text, status = hyde.expand_with_info("What is ARIMA?")   # + how it was made
    hyde.code_intent_signal("my ggplot code")   # -> "ggplot" (or None)
"""
from __future__ import annotations

import hashlib
import re

from src.llm.llm_client import LLMClient
from src.prompts.loader import load_prompt
from src.retrieval.hyde_cache import HydeCache
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
        cache: HydeCache | None = None,
    ):
        self.llm = llm
        self.enabled = enabled
        self.max_tokens = max_tokens
        self._signals = _compile_signals(skip_signals or [])
        self._prompt = load_prompt("hyde")
        # One constant, so the temperature the LLM is called with and the one
        # in the cache key cannot drift apart.
        self._temperature = 0.3
        # The prompt text is part of the cache key: a reworded prompt must not
        # be served drafts written under the old one.
        self._prompt_fp = hashlib.sha256(
            (self._prompt["system"] + "\x00" + self._prompt["user"]).encode("utf-8")
        ).hexdigest()[:16]
        # Assignable: the eval runner attaches one so its runs are repeatable.
        self.cache = cache

    @classmethod
    def from_config(cls, cfg: Config, llm: LLMClient) -> "HyDE":
        hc = cfg.get("retrieval.hyde_cache", {}) or {}
        cache = (HydeCache(cfg.path("retrieval.hyde_cache.path", "data/cache/hyde.sqlite"))
                 if hc.get("enabled") else None)
        return cls(
            llm=llm,
            enabled=cfg.get("retrieval.use_hyde", True),
            max_tokens=200,
            skip_signals=cfg.get("retrieval.hyde_skip_signals", []),
            cache=cache,
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
        Return the text to retrieve with: the query, a blank line, then a
        hypothetical answer. Falls back to the raw query alone when HyDE is
        off, when a code-intent signal fires, when the draft comes back empty,
        and on ANY LLM error (logged as a warning, never raised — which is why
        /search keeps working with the generation backend down).
        `enabled` overrides the configured default for this call only
        (presets pass use_hyde=False for code queries).
        """
        return self.expand_with_info(query, enabled)[0]

    def expand_with_info(
        self, query: str, enabled: bool | None = None
    ) -> tuple[str, str]:
        """
        expand(), plus WHY the text is what it is: (text, status), status being
          off      HyDE is off for this call
          bypass   a code-intent signal fired; no LLM call, no cache access
          hit      the draft came from the cache; no LLM call
          miss     the LLM wrote a draft, now stored in the cache (a concurrent
                   miss that stored its draft first wins: that one is used)
          nocache  the LLM wrote a draft and no cache is attached
          error    the LLM failed or returned nothing; raw query, not cached
        The pipeline echoes this as `hyde_cache`, so a run can tell a replayed
        draft from a fresh one.
        """
        on = self.enabled if enabled is None else enabled
        if not on:
            return query, "off"
        signal = self.code_intent_signal(query)
        if signal:
            log.info("HyDE bypass: code-intent signal %r — searching with raw query", signal)
            return query, "bypass"
        key = None
        if self.cache is not None:
            key = HydeCache.make_key(
                query, self._prompt_fp, str(getattr(self.llm, "model", "")),
                self._temperature)
            cached = self.cache.get(key)
            if cached is not None:
                log.info("HyDE cache hit (%d chars)", len(cached))
                return f"{query}\n\n{cached}", "hit"
        try:
            resp = self.llm.complete(
                system=self._prompt["system"],
                user=self._prompt["user"].format(query=query),
                temperature=self._temperature,
                max_tokens=self.max_tokens,
            )
            hypo = resp.text.strip()
        except Exception as e:
            log.warning("HyDE failed (%s); using raw query", e)
            return query, "error"
        if not hypo:
            log.warning("HyDE draft came back empty; using raw query")
            return query, "error"
        # Concatenate query + hypothetical so we keep the original signal too.
        log.info("HyDE expanded query (%d chars)", len(hypo))
        text = f"{query}\n\n{hypo}"
        if self.cache is None:
            return text, "nocache"
        # Outside the try above on purpose: a cache that cannot be written is a
        # broken cache, not a failed LLM call, and must not pass for one. put()
        # returns the STORED draft: if a concurrent miss got there first, that
        # one wins and this call retrieves with it, so both searches agree.
        stored = self.cache.put(key, hypo)
        return f"{query}\n\n{stored}", "miss"
