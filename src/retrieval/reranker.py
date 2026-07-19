"""
reranker.py — Cross-encoder reranking.

Bi-encoder retrieval (dense vectors) is fast but approximate: query and doc are
embedded separately. A cross-encoder reads (query, doc) TOGETHER and scores
relevance directly — far more accurate, but too slow to run over the whole
corpus. So we use it to reorder the top-N hybrid candidates down to top-k.

    retrieve (20-40 candidates) -> rerank -> top 5 -> generation

Usage:
    from src.retrieval.reranker import Reranker
    rr = Reranker.from_config(cfg)
    top5 = rr.rerank("What is ARIMA?", candidates, top_k=5)
"""
from __future__ import annotations

import re

from src.retrieval.retriever import RetrievedDoc
from src.utils.config_loader import Config
from src.utils.logger import get_logger

log = get_logger(__name__)


RERANK_MODES = ("cross_encoder", "lexical", "none")

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _lexical_score(query_terms: dict[str, float], text: str) -> float:
    """Cheap query-term coverage score: sum of IDF-ish weights for each query
    term present, damped by repeat count, normalized by doc length. No model,
    no deps — a fast alternative ordering when the cross-encoder's semantic
    opinion is unwanted (exact-keyword hunts) or its load cost is."""
    toks = _TOKEN_RE.findall(text.lower())
    if not toks:
        return 0.0
    counts: dict[str, int] = {}
    for t in toks:
        counts[t] = counts.get(t, 0) + 1
    score = 0.0
    for term, w in query_terms.items():
        c = counts.get(term, 0)
        if c:
            score += w * (1.0 + 0.5 * min(c - 1, 3))
    return score / (1.0 + len(toks) / 500.0)


class Reranker:
    def __init__(self, model_name: str, top_k: int = 7,
                 mode: str = "cross_encoder"):
        self.model_name = model_name
        self.top_k = top_k
        if mode not in RERANK_MODES:
            raise ValueError(f"rerank mode must be one of {RERANK_MODES}, "
                             f"got {mode!r}")
        self.mode = mode
        self._model = None

    # Back-compat: some call sites check .enabled
    @property
    def enabled(self) -> bool:
        return self.mode == "cross_encoder"

    @classmethod
    def from_config(cls, cfg: Config) -> "Reranker":
        mode = str(cfg.get("retrieval.rerank_mode", "cross_encoder")).lower()
        if mode not in RERANK_MODES:            # historical configs used e.g. "off"
            mode = "none"
        return cls(
            model_name=cfg.get(
                "retrieval.cross_encoder_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            ),
            top_k=cfg.get("retrieval.rerank_top_k", 5),
            mode=mode,
        )

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            log.info("Loading cross-encoder: %s", self.model_name)
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self, query: str, docs: list[RetrievedDoc], top_k: int | None = None,
        mode: str | None = None,
    ) -> list[RetrievedDoc]:
        """Reorder + truncate the fused candidates. `mode` overrides the
        configured method for this call: cross_encoder | lexical | none."""
        k = top_k or self.top_k
        m = (mode or self.mode).lower()
        if m not in RERANK_MODES:
            raise ValueError(f"rerank mode must be one of {RERANK_MODES}, "
                             f"got {m!r}")

        if m == "none" or not docs:
            # No reranking — just truncate the fused ranking.
            return docs[:k]

        if m == "lexical":
            terms = _TOKEN_RE.findall(query.lower())
            # rarer-looking (longer) terms weigh more; stopword-ish shorties less
            qw = {t: min(len(t), 8) / 8.0 for t in terms if len(t) > 2}
            for doc in docs:
                doc.rerank_score = _lexical_score(qw, doc.text)
            reranked = sorted(docs, key=lambda d: d.rerank_score, reverse=True)
            log.info("lexical-reranked %d candidates -> top %d", len(docs), k)
            return reranked[:k]

        model = self._get_model()
        pairs = [(query, d.text) for d in docs]
        scores = model.predict(pairs)

        for doc, score in zip(docs, scores):
            doc.rerank_score = float(score)

        reranked = sorted(docs, key=lambda d: d.rerank_score, reverse=True)
        log.info("reranked %d candidates -> top %d", len(docs), k)
        return reranked[:k]
