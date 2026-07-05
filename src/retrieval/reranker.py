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

from src.retrieval.retriever import RetrievedDoc
from src.utils.config_loader import Config
from src.utils.logger import get_logger

log = get_logger(__name__)


class Reranker:
    def __init__(self, model_name: str, top_k: int = 7, enabled: bool = True):
        self.model_name = model_name
        self.top_k = top_k
        self.enabled = enabled
        self._model = None

    @classmethod
    def from_config(cls, cfg: Config) -> "Reranker":
        mode = cfg.get("retrieval.rerank_mode", "cross_encoder")
        return cls(
            model_name=cfg.get(
                "retrieval.cross_encoder_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            ),
            top_k=cfg.get("retrieval.rerank_top_k", 5),
            enabled=(mode == "cross_encoder"),
        )

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            log.info("Loading cross-encoder: %s", self.model_name)
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self, query: str, docs: list[RetrievedDoc], top_k: int | None = None
    ) -> list[RetrievedDoc]:
        k = top_k or self.top_k

        if not self.enabled or not docs:
            # No reranking configured — just truncate the fused ranking.
            return docs[:k]

        model = self._get_model()
        pairs = [(query, d.text) for d in docs]
        scores = model.predict(pairs)

        for doc, score in zip(docs, scores):
            doc.rerank_score = float(score)

        reranked = sorted(docs, key=lambda d: d.rerank_score, reverse=True)
        log.info("reranked %d candidates -> top %d", len(docs), k)
        return reranked[:k]
