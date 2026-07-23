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
import time

from src.retrieval.retriever import RetrievedDoc
from src.utils.config_loader import Config
from src.utils.logger import get_logger

log = get_logger(__name__)


RERANK_MODES = ("cross_encoder", "http", "lexical", "none")

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


class RerankerExecutionError(RuntimeError):
    """The configured cross-encoder loaded, but failed while scoring pairs."""


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


# Cross-encoders known to work as a drop-in here. Sizes are the models' own
# published parameter counts; the cost column is a RATIO relative to MiniLM,
# because absolute seconds/query depend entirely on the machine — quoting one
# box's numbers as if they were the model's property is how a reader on
# different hardware ends up with a wrong expectation.
# Any HF cross-encoder id works — this list only drives the console's picker
# and documents the tradeoff, it is not a whitelist.
KNOWN_RERANKERS = {
    "cross-encoder/ms-marco-MiniLM-L-6-v2": {
        "label": "MiniLM-L6 — 22M params, the baseline cost (1x)",
        # max_length is the console's recommended interactive setting;
        # context_length is the hard model limit validated at construction.
        "max_length": 512,
        "context_length": 512,
    },
    "BAAI/bge-reranker-base": {
        "label": "bge-reranker-base — 278M, roughly 10x MiniLM's cost",
        "max_length": 512,
        "context_length": 512,
    },
    "BAAI/bge-reranker-v2-m3": {
        # XLM-RoBERTa-large, multilingual, 8k context. Stronger on public
        # benchmarks; on a CPU-only box the measured cost here was ~22x MiniLM,
        # which makes it a GPU-or-offline choice rather than an interactive one.
        "label": "bge-reranker-v2-m3 — 568M, multilingual/8k, ~22x MiniLM (GPU advised)",
        "max_length": 512,
        "context_length": 8192,
    },
}


# Ready-made reranker setups. The console applies one as a group, because the
# four knobs are only correct together: a big model with the wrong device, or
# `http` mode with no endpoint, fails in ways that read like a broken install.
#
# `default` is deliberately first and deliberately boring — it is the setup that
# works on any machine, laptop or server, with or without a GPU. The rest are
# opt-in for people who know which way they want to trade.
RERANK_PROFILES = {
    "default": {
        "label": "Default — works everywhere",
        "detail": "MiniLM cross-encoder, device auto-detected. The right "
                  "starting point on any modern machine; a GPU is used if "
                  "torch can see one, otherwise it runs fine on CPU.",
        "settings": {
            "retrieval.rerank_mode": "cross_encoder",
            "retrieval.cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "retrieval.cross_encoder_max_length": "512",
            "retrieval.cross_encoder_device": "auto",
        },
    },
    "quality": {
        "label": "Higher quality — needs a GPU or patience",
        "detail": "bge-reranker-base: better ordering on public benchmarks at "
                  "roughly 10x the cost. Worth it only if you have measured "
                  "that it helps YOUR corpus (main.py eval --retrieval-only).",
        "settings": {
            "retrieval.rerank_mode": "cross_encoder",
            "retrieval.cross_encoder_model": "BAAI/bge-reranker-base",
            "retrieval.cross_encoder_max_length": "512",
            "retrieval.cross_encoder_device": "auto",
        },
    },
    "low_power": {
        "label": "Old or low-power machine — no model at all",
        "detail": "Model-free lexical reranking: query-term coverage instead of "
                  "a neural cross-encoder. Nothing to download, no torch load "
                  "time, answers in milliseconds. Ordering is weaker on "
                  "paraphrased questions, better on exact-keyword hunts.",
        "settings": {
            "retrieval.rerank_mode": "lexical",
            "retrieval.cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "retrieval.cross_encoder_max_length": "512",
            "retrieval.cross_encoder_device": "cpu",
        },
    },
    "external": {
        "label": "External rerank server (advanced)",
        "detail": "Score against a /v1/rerank endpoint you run yourself, so a "
                  "big reranker can use hardware this process cannot. Set "
                  "retrieval.rerank_http.base_url first — this mode does not "
                  "fail soft if the endpoint is down.",
        "settings": {
            "retrieval.rerank_mode": "http",
        },
    },
}


class Reranker:
    def __init__(self, model_name: str, top_k: int = 7,
                 mode: str = "cross_encoder", max_length: int = 512,
                 device: str | None = None,
                 http_url: str | None = None, http_model: str | None = None,
                 http_timeout: int = 120):
        self.model_name = model_name
        self.top_k = top_k
        if mode not in RERANK_MODES:
            raise ValueError(f"rerank mode must be one of {RERANK_MODES}, "
                             f"got {mode!r}")
        self.mode = mode
        self.max_length = int(max_length)
        known = KNOWN_RERANKERS.get(self.model_name)
        if known and self.max_length > known["context_length"]:
            raise ValueError(
                "retrieval.cross_encoder_max_length="
                f"{self.max_length} exceeds {self.model_name!r}'s "
                f"{known['context_length']}-token context limit"
            )
        # None => let sentence-transformers choose (cuda when torch sees a GPU).
        # An explicit "cuda:1" pins a specific card; "cpu" forces CPU even on a
        # GPU box, which is what you want when VRAM is busy with something else.
        self.device = (device or "").strip().lower() or None
        if self.device == "auto":
            self.device = None
        # mode="http": the model is served OUT OF PROCESS behind an
        # OpenAI-style /v1/rerank endpoint (llama-server --reranking
        # --pooling rank). Same shape as the VLM-OCR lane, and the reason it
        # exists: llama.cpp compiles its own sm_61 kernels, so a big reranker
        # runs on GPUs that modern PyTorch wheels no longer support.
        self.http_url = (http_url or "").rstrip("/")
        self.http_model = http_model or model_name
        self.http_timeout = int(http_timeout)
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
            max_length=cfg.get("retrieval.cross_encoder_max_length", 512),
            device=cfg.get("retrieval.cross_encoder_device", "auto"),
            http_url=cfg.get("retrieval.rerank_http.base_url"),
            http_model=cfg.get("retrieval.rerank_http.model"),
            http_timeout=cfg.get("retrieval.rerank_http.timeout", 120),
        )

    # ---- http lane -----------------------------------------------------

    def _rerank_http(self, query: str, docs: list[RetrievedDoc],
                     k: int) -> list[RetrievedDoc]:
        """Score via an external /v1/rerank endpoint (llama-server et al).

        Deliberately NOT fail-soft: if the endpoint is configured and down, a
        silent fall-through to fused order would quietly change what the model
        answers from while every log line still said "reranked". The caller
        can pick mode="lexical" per call if it wants a model-free path.
        """
        import httpx

        if not self.http_url:
            raise ValueError(
                "rerank_mode='http' needs retrieval.rerank_http.base_url "
                "(e.g. http://127.0.0.1:8101/v1)")
        payload = {"model": self.http_model, "query": query,
                   "documents": [d.text for d in docs], "top_n": len(docs)}
        try:
            r = httpx.post(f"{self.http_url}/rerank", json=payload,
                           timeout=self.http_timeout)
            r.raise_for_status()
            data = r.json()
            # llama.cpp returns
            # {"results":[{"index":i,"relevance_score":x},...]}
            results = data.get("results")
            if not isinstance(results, list):
                raise ValueError(
                    f"unexpected /rerank response keys: {sorted(data)}")
            for item in results:
                i = item.get("index")
                score = item.get("relevance_score", item.get("score"))
                if i is None or score is None or not (0 <= i < len(docs)):
                    raise ValueError(f"bad /rerank result entry: {item}")
                docs[i].rerank_score = float(score)
            if any(d.rerank_score is None for d in docs):
                raise ValueError("/rerank did not score every document")
        except Exception as e:
            raise RerankerExecutionError(
                f"HTTP reranker {self.http_url!r} failed while scoring "
                f"{len(docs)} candidate(s): {type(e).__name__}: {e}"
            ) from e
        reranked = sorted(docs, key=lambda d: d.rerank_score, reverse=True)
        log.info("http-reranked %d candidates -> top %d via %s",
                 len(docs), k, self.http_url)
        return reranked[:k]

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            log.info("Loading cross-encoder: %s (max_length=%d, device=%s)",
                     self.model_name, self.max_length, self.device or "auto")
            t0 = time.time()
            kwargs: dict = {"max_length": self.max_length}
            if self.device:
                kwargs["device"] = self.device
            self._model = CrossEncoder(self.model_name, **kwargs)
            # Big rerankers (bge-reranker-v2-m3 is 568M params / 2.2 GB) take
            # minutes to load on CPU the first time. Say so, or the first query
            # after a restart looks like a hang.
            log.info("cross-encoder ready in %.1fs", time.time() - t0)
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

        if m == "http":
            return self._rerank_http(query, docs, k)

        if m == "lexical":
            terms = _TOKEN_RE.findall(query.lower())
            # rarer-looking (longer) terms weigh more; stopword-ish shorties less
            qw = {t: min(len(t), 8) / 8.0 for t in terms if len(t) > 2}
            for doc in docs:
                doc.rerank_score = _lexical_score(qw, doc.text)
            reranked = sorted(docs, key=lambda d: d.rerank_score, reverse=True)
            log.info("lexical-reranked %d candidates -> top %d", len(docs), k)
            return reranked[:k]

        pairs = [(query, d.text) for d in docs]
        try:
            # Model resolution belongs inside the typed boundary too: missing
            # files, download failures, and invalid devices are reranker
            # failures just as much as predict-time tensor errors.
            model = self._get_model()
            scores = model.predict(pairs)
        except Exception as e:
            raise RerankerExecutionError(
                f"cross-encoder {self.model_name!r} failed while loading or scoring "
                f"{len(pairs)} candidate(s) at max_length={self.max_length}: "
                f"{type(e).__name__}: {e}"
            ) from e

        for doc, score in zip(docs, scores):
            doc.rerank_score = float(score)

        reranked = sorted(docs, key=lambda d: d.rerank_score, reverse=True)
        log.info("reranked %d candidates -> top %d", len(docs), k)
        return reranked[:k]
