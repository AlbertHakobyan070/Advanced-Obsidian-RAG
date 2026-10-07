"""
metrics.py — per-question retrieval metrics from relevance judgements.

The input is one ranking: a list with one frozenset per rank, holding the
indices of the gold sources that chunk satisfies (relevance.Judged.matches),
empty for an irrelevant chunk. Everything here is arithmetic on that — no
index, no LLM — so each metric is checked on hand-built rankings.

Gains are SOURCE-level, not chunk-level. A rank earns gain 1 only if it
satisfies a REQUIRED gold source that no higher rank already satisfied, so ten
chunks of one book add one gain, not ten, and nDCG cannot be inflated by a
retriever that fills the list with near-duplicates. The ideal ranking finds
every required source in the first min(k, n_required) ranks; that is the nDCG
denominator. recall@k and complete@k count sources found for the same reason.
hit@k, MRR@k and precision@k only ask whether a rank is relevant to ANY gold
source.

`required` (spec §3.2, multihop): a source flagged required=false is optional
context and does not count toward recall, complete or nDCG. If none is flagged
required, all are, so a question never ends up with nothing to find. The
document-level variants (doc_hit@5, doc_recall@10) are the same arithmetic on
Judged.doc_matches.
"""
from __future__ import annotations

import math

from eval.bench.questions import GoldSource
from eval.bench.relevance import Judged

Ranking = list[frozenset[int]]


def required(gold: list[GoldSource]) -> list[int]:
    req = [i for i, g in enumerate(gold) if g.required]
    return req or list(range(len(gold)))


def hit_at(m: Ranking, k: int) -> float:
    return 1.0 if any(m[:k]) else 0.0


def precision_at(m: Ranking, k: int) -> float:
    top = m[:k]
    return sum(1 for s in top if s) / k if k else 0.0


def _found(m: Ranking, gold: list[GoldSource], k: int) -> tuple[set[int], set[int]]:
    req = set(required(gold))
    got = set().union(*m[:k]) if m[:k] else set()
    return req, got & req


def recall_at(m: Ranking, gold: list[GoldSource], k: int) -> float:
    req, found = _found(m, gold, k)
    return len(found) / len(req) if req else 0.0


def complete_at(m: Ranking, gold: list[GoldSource], k: int) -> float:
    req, found = _found(m, gold, k)
    return 1.0 if req and found == req else 0.0


def mrr_at(m: Ranking, k: int) -> float:
    for r, s in enumerate(m[:k]):
        if s:
            return 1.0 / (r + 1)
    return 0.0


def ndcg_at(m: Ranking, gold: list[GoldSource], k: int) -> float:
    req = set(required(gold))
    seen, dcg = set(), 0.0
    for r, s in enumerate(m[:k]):
        new = (s & req) - seen
        if new:
            dcg += 1.0 / math.log2(r + 2)
            seen |= new
    ideal = sum(1.0 / math.log2(r + 2) for r in range(min(k, len(req))))
    return dcg / ideal if ideal else 0.0


def retrieval_metrics(j: Judged, gold: list[GoldSource]) -> dict[str, float]:
    m, d = j.matches, j.doc_matches
    return {
        "hit@1": hit_at(m, 1), "hit@5": hit_at(m, 5), "hit@10": hit_at(m, 10),
        "recall@10": recall_at(m, gold, 10), "complete@10": complete_at(m, gold, 10),
        "mrr@10": mrr_at(m, 10), "ndcg@10": ndcg_at(m, gold, 10), "p@5": precision_at(m, 5),
        "doc_hit@5": hit_at(d, 5), "doc_recall@10": recall_at(d, gold, 10),
    }
