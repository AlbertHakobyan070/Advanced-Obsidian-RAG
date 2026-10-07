"""
reranker.py — Reorder the fused candidate pool, then cut it to top-k.

Bi-encoder retrieval (dense vectors) is fast but approximate: query and doc are
embedded separately. A cross-encoder reads (query, doc) TOGETHER and scores
relevance directly — far more accurate, but too slow to run over the whole
corpus. So we use it to reorder the top-N hybrid candidates down to top-k.

    retrieve (20-40 candidates) -> rerank -> top k -> generation

Five modes (RERANK_MODES), selectable per call:
  cross_encoder  a sentence-transformers CrossEncoder, in process
  http           the same job behind an external /v1/rerank endpoint
  lexical        model-free query-term coverage (no torch, milliseconds)
  none           keep the fused RRF order and just truncate
  laya           EXPERIMENTAL: P(relevant) from a Laya checkpoint fine-tuned on
                 this vault (laya_reranker.py); refused unless
                 retrieval.laya.enabled, and it ignores the instruction below

An optional rerank INSTRUCTION states the ranking criterion ("prefer worked
procedures over definitions"). It changes only what the reranker scores
against, so it can reorder the pool but never shrink it.

The model modes fail loudly: they raise RerankerExecutionError rather than
falling back to fused order, so an answer is never built from a ranking
nobody asked for while the logs still say "reranked".

Usage:
    from src.retrieval.reranker import Reranker
    rr = Reranker.from_config(cfg)
    top5 = rr.rerank("What is ARIMA?", candidates, top_k=5)
"""
from __future__ import annotations

import re
import time

from src.retrieval.laya_reranker import LayaScorer
from src.retrieval.retriever import RetrievedDoc
from src.utils.config_loader import Config
from src.utils.logger import get_logger

log = get_logger(__name__)


RERANK_MODES = ("cross_encoder", "http", "lexical", "none", "laya")

_TOKEN_RE = re.compile(r"[a-z0-9_]+")

# How a rerank instruction is joined to the question before scoring.
#
#   prefix   — "<instruction>\n<question>". Query REFRAMING: it works with any
#              cross-encoder because it simply changes the text the model is
#              matching against. This is the default precisely because it makes
#              no claim about the model understanding instructions.
#   instruct — the labelled form instruction-tuned rerankers are trained on.
#              Only worth selecting for a model documented to expect it; on a
#              plain cross-encoder the literal tags are just noise in the query.
INSTRUCTION_FORMATS = {
    "prefix": "{instruction}\n{query}",
    "instruct": "<Instruct>: {instruction}\n<Query>: {query}",
}


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
                 http_timeout: int = 120,
                 instruction: str | None = None,
                 instruction_format: str = "prefix",
                 laya_scorer: LayaScorer | None = None):
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

        # ---- rerank instruction ------------------------------------------
        # WHAT IT IS FOR. A cross-encoder answers "how relevant is this passage
        # to this query" — but relevant *for what* is left implicit, and the
        # model's own idea of it comes from whatever it was trained on. Two
        # passages can be equally on-topic while only one is the kind of thing
        # you wanted: a worked procedure rather than a definition, a primary
        # source rather than a summary, code rather than prose about code.
        #
        # The instruction makes that criterion explicit and applies it at
        # ranking time — after retrieval has already found the on-topic pool, so
        # it costs nothing in recall. It is a ranking-criterion knob, not a
        # filter: it re-weights an existing candidate set and can never remove
        # material from it.
        self.instruction = (instruction or "").strip() or None
        if instruction_format not in INSTRUCTION_FORMATS:
            raise ValueError(
                f"retrieval.rerank_instruction_format must be one of "
                f"{sorted(INSTRUCTION_FORMATS)}, got {instruction_format!r}")
        self.instruction_format = instruction_format
        self._warned_lexical_instruction = False
        self._warned_laya_instruction = False

        # mode="laya" (EXPERIMENTAL): scores with a Laya checkpoint
        # fine-tuned on this vault. None = the feature is switched off
        # (retrieval.laya.enabled is false), and the mode is then REFUSED
        # rather than answered by some other reranker. The scorer loads
        # lazily, and the pipeline hands this same instance to the relevance
        # gate's `laya` scorer, so the checkpoint is only ever loaded once.
        self.laya_scorer = laya_scorer

    # Back-compat: some call sites check .enabled
    @property
    def enabled(self) -> bool:
        return self.mode == "cross_encoder"

    @classmethod
    def from_config(cls, cfg: Config) -> "Reranker":
        mode = str(cfg.get("retrieval.rerank_mode", "cross_encoder")).lower()
        # Any unrecognised CONFIG value maps to "none": historical configs used
        # e.g. "off" — but a typo such as "cross-encoder" lands here too, and
        # it switches reranking off without an error. (A per-call mode is
        # validated in rerank() and raises instead.)
        if mode not in RERANK_MODES:
            mode = "none"
        # Built only when the flag is on, and even then it imports and loads
        # nothing until its first score(): a switched-off feature costs
        # nothing.
        laya_scorer = None
        if bool(cfg.get("retrieval.laya.enabled", False)):
            laya_scorer = LayaScorer(
                model_dir=cfg.path("retrieval.laya.model_dir",
                                   "models/laya-noetrix"),
                device=cfg.get("retrieval.laya.device", "cpu"),
                batch_size=cfg.get("retrieval.laya.batch_size", 16),
            )
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
            instruction=cfg.get("retrieval.rerank_instruction"),
            instruction_format=str(
                cfg.get("retrieval.rerank_instruction_format", "prefix")),
            laya_scorer=laya_scorer,
        )

    # ---- instruction ---------------------------------------------------

    def scoring_query(self, query: str, mode: str,
                      instruction: str | None = None) -> str:
        """The text the reranker actually scores against.

        Routed PER MODE, because the modes can genuinely use it or genuinely
        cannot:

          cross_encoder — joined to the question. The model reads the pair
                          together, so a stated criterion shifts what counts as
                          a good match. This is the lane the feature is for.
          http          — same joined text goes to the external service, which
                          is the only channel the /v1/rerank shape gives us. A
                          service that is instruction-tuned benefits; one that
                          is not sees a longer query, exactly as above.
          lexical       — IGNORED, deliberately. Lexical scoring is query-term
                          coverage; folding a sentence of instruction into the
                          term set would dilute every real query term and
                          quietly make ranking worse. Warned once, not silent.
          none          — nothing is scored at all.
          laya          — IGNORED, deliberately. The query goes into Laya's own
                          question ("Does the passage answer: <query>?"), the
                          exact wording its checkpoint was fine-tuned on; a
                          prepended instruction would change the input it was
                          trained on. Warned once, not silent.
        """
        text = (instruction if instruction is not None else self.instruction)
        text = (text or "").strip()
        if not text:
            return query
        if mode in ("lexical", "none"):
            if not self._warned_lexical_instruction:
                self._warned_lexical_instruction = True
                log.warning(
                    "rerank instruction ignored: mode=%r scores query-term "
                    "coverage, and mixing instruction words into the term set "
                    "would dilute the query's own terms. Use "
                    "rerank_mode=cross_encoder or http to apply it.", mode)
            return query
        if mode == "laya":
            if not self._warned_laya_instruction:
                self._warned_laya_instruction = True
                log.warning(
                    "rerank instruction ignored: mode='laya' puts the query "
                    "into the model's own question, the wording its checkpoint "
                    "was trained on, and an instruction would change that "
                    "input. Use rerank_mode=cross_encoder or http to apply it.")
            return query
        return INSTRUCTION_FORMATS[self.instruction_format].format(
            instruction=text, query=query)

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

    # ---- laya lane (EXPERIMENTAL) --------------------------------------

    def _rerank_laya(self, query: str, docs: list[RetrievedDoc],
                     k: int) -> list[RetrievedDoc]:
        """Score with the Laya checkpoint: rerank_score = P(true), in [0, 1].

        Deliberately NOT fail-soft, and never a quiet switch to another
        reranker: with the feature off, or the package or checkpoint missing,
        this raises, because an answer built from a ranking nobody asked for
        would still be logged as "reranked". P(true) is rounded to 4 places, so
        ties are normal; sorted() is stable, so they keep the fused order.
        """
        if self.laya_scorer is None:
            raise RerankerExecutionError(
                "rerank mode 'laya' is EXPERIMENTAL and switched off "
                "(retrieval.laya.enabled is false). Turn it on in the "
                "management console under Settings > Experimental features, or "
                "set retrieval.laya.enabled: true in config.yaml, then restart "
                ":8051; it also needs the laya package and a checkpoint "
                "(docs/laya-finetune.md). It never falls back to another "
                "rerank mode: pick one explicitly")
        try:
            scores = self.laya_scorer.score(query, [d.text for d in docs])
        except Exception as e:
            raise RerankerExecutionError(
                f"laya reranker ({self.laya_scorer.model_dir}) failed while "
                f"loading or scoring {len(docs)} candidate(s): "
                f"{type(e).__name__}: {e}"
            ) from e
        if len(scores) != len(docs):
            # zip() would pin scores on the wrong docs and drop the rest.
            raise RerankerExecutionError(
                f"laya reranker returned {len(scores)} score(s) for "
                f"{len(docs)} candidate(s)")
        for doc, score in zip(docs, scores):
            doc.rerank_score = float(score)
        reranked = sorted(docs, key=lambda d: d.rerank_score, reverse=True)
        log.info("laya-reranked %d candidates -> top %d", len(docs), k)
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
        mode: str | None = None, instruction: str | None = None,
    ) -> list[RetrievedDoc]:
        """Reorder + truncate the fused candidates.

        `mode` overrides the configured method for this call
        (cross_encoder | http | lexical | none | laya). `instruction`
        overrides the configured ranking criterion; pass "" to switch it off
        for one call while the config keeps it on.
        """
        k = top_k or self.top_k
        m = (mode or self.mode).lower()
        if m not in RERANK_MODES:
            raise ValueError(f"rerank mode must be one of {RERANK_MODES}, "
                             f"got {m!r}")

        if m == "none" or not docs:
            # No reranking — just truncate the fused ranking.
            return docs[:k]

        # The instruction changes only what the RERANKER scores against. The
        # retrieval that produced `docs` already happened against the real
        # question, so a criterion can reweight the pool but never shrink it.
        scoring_query = self.scoring_query(query, m, instruction)

        if m == "http":
            return self._rerank_http(scoring_query, docs, k)

        if m == "laya":
            return self._rerank_laya(scoring_query, docs, k)

        if m == "lexical":
            terms = _TOKEN_RE.findall(scoring_query.lower())
            # rarer-looking (longer) terms weigh more; stopword-ish shorties less
            qw = {t: min(len(t), 8) / 8.0 for t in terms if len(t) > 2}
            for doc in docs:
                doc.rerank_score = _lexical_score(qw, doc.text)
            reranked = sorted(docs, key=lambda d: d.rerank_score, reverse=True)
            log.info("lexical-reranked %d candidates -> top %d", len(docs), k)
            return reranked[:k]

        pairs = [(scoring_query, d.text) for d in docs]
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
