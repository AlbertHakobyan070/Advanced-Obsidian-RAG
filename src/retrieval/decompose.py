"""
decompose.py — split a multi-part question into searches, one branch each.

Searching a multi-part question as one string reaches its dominant topic and
not the second one. Measured on the dev split (2026-10-13): about half of the
required gold sources the final list missed were not even in a five-times-
deeper candidate pool, mostly on multi-source (T3) and follow-up (T4)
questions. A wider pool cannot fix that; a second search can.

    question -> LLM: [sub-query, ...]   one call, temperature 0
    branches = the original question + each distinct sub-query
    each branch: retrieve, then rerank against ITS OWN text (whole pool)
    merge: reciprocal rank over the branch rankings

Why rerank per branch: a chunk that answers one part of the question scores
low against the whole question (it answers half of it), and the shared rerank
is exactly what pushed such chunks out. Why reciprocal rank to merge: rerank
scores from different queries are not comparable, ranks are. Rank 1 of every
branch scores the same, so each part gets a seat near the top (ties go to the
original branch), and a chunk several branches agree on rises.

A one-element reply means the question is atomic and the caller proceeds
exactly as without decomposition. An LLM failure or an unparseable reply
raises: there is no silent fallback to the undecomposed path.
"""
from __future__ import annotations

import json
import re

from src.prompts.loader import load_prompt
from src.retrieval.retriever import RetrievedDoc
from src.utils.config_loader import Config


def parse_subqueries(text: str, question: str, max_subqueries: int) -> list[str]:
    """The distinct sub-queries in an LLM reply; [] when the question is atomic
    (a one-element array, or every item repeats the question). Raises
    ValueError on anything that is not a JSON array of strings."""
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start, end = t.find("["), t.rfind("]")
    if start == -1 or end < start:
        raise ValueError(f"decompose: no JSON array in the reply {text[:200]!r}")
    try:
        items = json.loads(t[start:end + 1])
    except json.JSONDecodeError as e:
        raise ValueError(f"decompose: unparseable reply {text[:200]!r} ({e})") from None
    if not isinstance(items, list) or not all(isinstance(s, str) for s in items):
        raise ValueError(f"decompose: expected a JSON array of strings, got {text[:200]!r}")
    if len(items) <= 1:
        return []
    seen = {" ".join(question.split()).casefold()}
    out = []
    for s in items:
        s = " ".join(s.split())
        if s and s.casefold() not in seen:
            seen.add(s.casefold())
            out.append(s)
    return out[:max_subqueries]


def merge_branches(rankings: list[list[RetrievedDoc]], rrf_k: int) -> list[RetrievedDoc]:
    """One list from several best-first branch rankings, by summed reciprocal
    rank. The first ranking is the original question's: on a tie its chunks
    come first, because the sort is stable over first-seen order."""
    score: dict[str, float] = {}
    first: dict[str, RetrievedDoc] = {}
    for ranking in rankings:
        for rank, d in enumerate(ranking, 1):
            score[d.id] = score.get(d.id, 0.0) + 1.0 / (rrf_k + rank)
            first.setdefault(d.id, d)
    return sorted(first.values(), key=lambda d: -score[d.id])


class Decomposer:
    def __init__(self, llm, enabled: bool = False, max_subqueries: int = 3):
        if max_subqueries < 2:
            raise ValueError(f"retrieval.decompose.max_subqueries must be at least 2, got {max_subqueries}")
        self.llm = llm                  # assignable: the eval swaps in a replay store
        self.enabled = enabled
        self.max_subqueries = max_subqueries
        self._prompt = load_prompt("decompose")

    @classmethod
    def from_config(cls, cfg: Config, llm) -> "Decomposer":
        d = cfg.get("retrieval.decompose", {}) or {}
        return cls(llm, enabled=bool(d.get("enabled", False)),
                   max_subqueries=int(d.get("max_subqueries", 3)))

    def split(self, question: str) -> list[str]:
        """[] = atomic; otherwise the distinct sub-queries. Raises on an LLM
        error or an unparseable reply."""
        resp = self.llm.complete(
            system=self._prompt["system"].format(max=self.max_subqueries),
            user=self._prompt["user"].format(query=question),
            temperature=0.0, max_tokens=300)
        return parse_subqueries(resp.text, question, self.max_subqueries)
