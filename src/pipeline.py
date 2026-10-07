"""
pipeline.py — The full Advanced-RAG query path, wired end to end.

    query
      -> preset resolution   explicit preset, or 'code' auto-applied when the
                             question trips a code-intent signal
      -> scope detection     on the RAW question (domain / content hints)
      -> HyDE expansion      optional; its text feeds every retrieval lane
      -> hybrid retrieve     up to eight lanes, weighted RRF, metadata boost
      -> rerank -> top k     cross_encoder | http | lexical | none | laya
      -> relevance gate      optional; drops chunks below a threshold, and
                             leaving none makes generation abstain
      -> small-to-big        optional parent swap / adjacent PDF pages
    search() stops here: retrieval only, no generation backend needed.
      -> grounded generation with inline [n] citations + a CONFIDENCE line
      -> optional citation verification (a second LLM pass)
    -> Answer, whose .retrieval echoes every setting that actually ran

Graph RAG (self.graph) is a separate mode that is merely wired here: neither
search() nor query() calls it.

Lazy construction: heavy objects (embedding model, cross-encoder, chroma client)
are built once and reused. Build the pipeline once, call query() many times.

    from src.pipeline import RAGPipeline
    rag = RAGPipeline.from_config(cfg)
    answer = rag.query("Explain knowledge distillation in my capstone")
"""
from __future__ import annotations

import time
from contextlib import nullcontext

from src.embeddings.embedder import Embedder
from src.generation.generator import Answer, Generator
from src.llm.llm_client import LLMClient
from src.retrieval.context_expand import NeighborContext, ParentContext
from src.retrieval.graph_expand import GraphExpander
from src.retrieval.hyde import HyDE
from src.retrieval.relevance_gate import RelevanceGate
from src.retrieval.reranker import Reranker
from src.retrieval.retriever import LANES, HybridRetriever, _clean_lane_set
from src.retrieval.scope import ScopeRouter
from src.utils.config_loader import Config, load_config
from src.utils.logger import configure_logging, get_logger
from src.utils.timing import StageTrace

log = get_logger(__name__)


class RAGPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: Reranker,
        hyde: HyDE,
        generator: Generator,
        rerank_top_k: int = 7,
        presets: dict[str, dict] | None = None,
        scope_router: ScopeRouter | None = None,
        parent_ctx: ParentContext | None = None,
        neighbor_ctx: NeighborContext | None = None,
        graph: GraphExpander | None = None,
        gate: RelevanceGate | None = None,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.hyde = hyde
        self.generator = generator
        self.rerank_top_k = rerank_top_k
        # Named override bundles from retrieval.presets (code/concept/synthesis).
        self.presets = presets or {}
        # Domain/content-type routing (retrieval.domain_signals / content_signals).
        self.scope_router = scope_router or ScopeRouter(None, None)
        # E2 small-to-big context expansion (both default OFF; see
        # retrieval.parent_context / retrieval.neighbor_context).
        self.parent_ctx = parent_ctx
        self.neighbor_ctx = neighbor_ctx
        # Post-rerank relevance gate (retrieval.relevance_gate). None = a gate
        # that is off, so a per-call gate / gate_threshold still has something
        # to switch on (rerank_score, the default scorer) instead of being
        # silently ignored.
        self.gate = gate if gate is not None else RelevanceGate()
        # Graph RAG mode (POST /graph/expand). A SEPARATE path: nothing in
        # search() or query() below touches it, and it touches nothing here.
        self.graph = graph

    @classmethod
    def from_config(cls, cfg: Config | None = None) -> "RAGPipeline":
        cfg = cfg or load_config()
        configure_logging(
            level=cfg.get("logging.level", "INFO"),
            log_file=cfg.path("logging.file") if cfg.get("logging.file") else None,
            console=cfg.get("logging.console", True),
        )

        embedder = Embedder.from_config(cfg)
        llm = LLMClient.from_config(cfg, role="generation")

        retriever = HybridRetriever.from_config(cfg, embedder)
        reranker = Reranker.from_config(cfg)
        hyde = HyDE.from_config(cfg, llm)
        generator = Generator.from_config(cfg, llm)

        return cls(
            retriever=retriever,
            reranker=reranker,
            hyde=hyde,
            generator=generator,
            rerank_top_k=cfg.get("retrieval.rerank_top_k", 7),
            presets=cfg.get("retrieval.presets", {}) or {},
            scope_router=ScopeRouter.from_config(cfg),
            parent_ctx=ParentContext.from_config(cfg),
            neighbor_ctx=NeighborContext.from_config(cfg),
            graph=GraphExpander.from_config(cfg, retriever, reranker),
            # Built here so a bad gate block fails the build, not a request.
            gate=RelevanceGate.from_config(cfg, reranker),
        )

    def _resolve_overrides(
        self,
        question: str,
        preset: str | None,
        auto_preset: bool = True,
    ) -> tuple[dict, str | None]:
        """
        Pick the override bundle for this query.

        Explicit preset wins. With no preset, a code-intent query (one that
        trips retrieval.hyde_skip_signals) auto-applies the 'code' preset so
        the candidate pool widens enough for code chunks to survive fusion.
        Returns ({overrides}, preset_label_or_None). Never mutates defaults.
        """
        if preset:
            if preset not in self.presets:
                raise KeyError(
                    f"Unknown preset {preset!r}. Available: {sorted(self.presets)}"
                )
            return dict(self.presets[preset]), preset
        signal = self.hyde.code_intent_signal(question) if auto_preset else None
        if signal and "code" in self.presets:
            log.info("auto-applying 'code' preset (signal %r)", signal)
            return dict(self.presets["code"]), "code (auto)"
        return {}, None

    def _lazy_handles(self) -> tuple[bool, ...]:
        """Which lazily-loaded handles are resident right now: the Chroma
        collection, the BM25 payload, the cross-encoder and the Laya
        checkpoint. /health reports ready before any of them loads, so the
        first search pays for them — a spike the latency numbers must flag, not
        average in. getattr, so a stand-in without these attributes reads as
        never-loaded, never cold."""
        return tuple(
            getattr(obj, name, None) is not None
            for obj, name in ((self.retriever, "_collection"),
                              (self.retriever, "_bm25_payload"),
                              (self.reranker, "_model"),
                              (getattr(self.reranker, "laya_scorer", None),
                               "_agent"))
        )

    def search(
        self,
        question: str,
        preset: str | None = None,
        top_k: int | None = None,
        dense_top_k: int | None = None,
        sparse_top_k: int | None = None,
        hyde: bool | None = None,
        omnisearch: bool | None = None,
        parent_context: bool | None = None,
        neighbor_context: bool | None = None,
        hype: bool | None = None,
        rerank: str | None = None,
        rerank_instruction: str | None = None,
        lane_weights: dict[str, float] | None = None,
        auto_preset: bool = True,
        lanes: list[str] | None = None,
        metadata_boost: bool | None = None,
        gate: bool | None = None,
        gate_threshold: float | None = None,
        gate_scorer: str | None = None,
    ) -> tuple[list, dict]:
        """
        Retrieval only — everything query() does EXCEPT generation. Returns
        (reranked_docs, retrieval_info). This is what the /search endpoint and
        `retrieve_only` queries use: it needs no generation backend at all
        (HyDE degrades to the raw query if the LLM is down), so an agent can
        pull grounded chunks even when FreeLLMAPI isn't running and do its own
        reasoning over them.

        Per-call knobs (never mutate the warm defaults):
          preset      named bundle from retrieval.presets
          top_k       overrides rerank_top_k
          hyde        True/False forces HyDE on/off (beats the preset value)
          omnisearch  True/False forces the live-vault lane on/off
          parent_context / neighbor_context
                      True/False forces the E2 small-to-big lanes on/off
                      (beats preset, which beats the config default)
          dense_top_k / sparse_top_k
                      per-lane candidate pool sizes (beat the preset's)
          hype        True/False forces the hypothetical-question lane on/off
          rerank      rerank mode for this call only (cross_encoder | http |
                      lexical | none | laya)
          rerank_instruction
                      ranking criterion for this call; "" switches a
                      configured one off
          lane_weights
                      {lane: weight}, merged lane by lane over the preset's
                      map, which is itself merged over retrieval.lane_weights
          auto_preset False disables the implicit code preset, giving compare
                      calls an explicit config-only baseline
          lanes       restrict the call to these lanes (names from LANES): one
                      outside the list is not run at all. Restricts only — a
                      conditional lane still needs its own trigger. Empty or
                      unknown raises ValueError
          metadata_boost
                      True/False forces the course/domain/tag boost on or off
                      (None follows retrieval.metadata_boost)
          gate        True/False forces the relevance gate on or off (None
                      follows retrieval.relevance_gate.enabled). It runs after
                      the rerank and before the small-to-big expansion and
                      drops what scores below the threshold; if nothing
                      survives, query() abstains. The rerank_score scorer
                      needs a scoring rerank mode: `none` raises ValueError
          gate_threshold
                      cutoff in the scorer's own units; beats the configured
                      one, and on its own it turns the gate on for this call
          gate_scorer rerank_score | laya for this call (None follows
                      retrieval.relevance_gate.scorer); a scorer other than
                      the configured one needs its own gate_threshold

        The echo also reports where the time went: timings (ms per stage),
        lanes_run (candidates each lane returned) and cold (this call loaded
        the index or the cross-encoder, so its timings are not steady-state),
        and what the gate did: gate ({"enabled": False} when it did not run).
        """
        # First thing, before HyDE: retrieve() rejects a bad lane set too, but by
        # then the draft has already cost an LLM call. The cleaned value is not
        # used; retrieve() cleans the same input again.
        _clean_lane_set(lanes, "lanes")
        log.info("=== SEARCH: %s", question)
        trace = StageTrace()
        t_all = time.perf_counter()
        loaded_before = self._lazy_handles()

        with trace.span("preset"):
            overrides, preset_label = self._resolve_overrides(
                question, preset, auto_preset=auto_preset)
        k = top_k or overrides.get("rerank_top_k") or self.rerank_top_k
        # Per-lane pool sizes: explicit per-call beats preset beats config default.
        dk = dense_top_k if dense_top_k is not None else overrides.get("dense_top_k")
        sk = sparse_top_k if sparse_top_k is not None else overrides.get("sparse_top_k")
        use_hyde = hyde if hyde is not None else overrides.get("use_hyde")
        use_omni = omnisearch if omnisearch is not None else overrides.get("omnisearch")
        use_hype = hype if hype is not None else overrides.get("hype")
        # Lane weights compose rather than replace: the preset's map merges
        # over the configured one, and the per-call map merges over that. So
        # a preset can reweight sparse while a call reweights hype, and both
        # apply — replacing wholesale would silently drop the preset's intent.
        weights = dict(overrides.get("lane_weights") or {})
        weights.update(lane_weights or {})
        # Domain/content hints are detected on the RAW question (HyDE prose
        # would dilute the keywords) and routed as extra fusion lanes.
        with trace.span("scope"):
            scope = self.scope_router.detect(question)

        with trace.span("hyde"):
            search_text, hyde_status = self.hyde.expand_with_info(
                question, enabled=use_hyde)
        candidates = self.retriever.retrieve(
            search_text,
            dense_top_k=dk,
            sparse_top_k=sk,
            boost_code=overrides.get("boost_code", False),
            scope=scope if scope else None,
            omnisearch=use_omni,
            hype=use_hype,
            lane_weights=weights or None,
            lanes=lanes,
            metadata_boost=metadata_boost,
            trace=trace,
        )
        rerank_mode = rerank if rerank is not None else overrides.get("rerank_mode")
        # "" is meaningful here: it turns a configured criterion OFF for this
        # call, which `or` would silently discard.
        instruction = (rerank_instruction if rerank_instruction is not None
                       else overrides.get("rerank_instruction"))
        with trace.span("rerank"):
            top = self.reranker.rerank(question, candidates, top_k=k,
                                       mode=rerank_mode, instruction=instruction)

        # Relevance gate (post-rerank, pre-expansion). It judges the ORIGINAL
        # question — HyDE's text is a hypothetical answer, not what was asked.
        # The span exists only for a call that gates, so a gate-off call's
        # timings carry no stage that never ran.
        gate_on = self.gate.is_on(gate, gate_threshold, gate_scorer)
        with trace.span("gate") if gate_on else nullcontext():
            top, gate_info = self.gate.apply(
                question, top, enabled=gate, threshold=gate_threshold,
                scorer=gate_scorer)

        # E2 small-to-big (post-rerank): per-call beats preset beats config.
        parent_on = parent_context
        if parent_on is None:
            parent_on = overrides.get("parent_context")
        if parent_on is None:
            parent_on = bool(self.parent_ctx and self.parent_ctx.enabled)
        swaps = siblings = 0
        if parent_on and self.parent_ctx:
            with trace.span("parent"):
                top, swaps, siblings = self.parent_ctx.apply(top)
        neighbor_on = neighbor_context
        if neighbor_on is None:
            neighbor_on = overrides.get("neighbor_context")
        if neighbor_on is None:
            neighbor_on = bool(self.neighbor_ctx and self.neighbor_ctx.enabled)
        neighbors = 0
        if neighbor_on and self.neighbor_ctx:
            with trace.span("neighbor"):
                top, neighbors = self.neighbor_ctx.apply(
                    top, self.retriever._get_collection())

        info = {
            "preset": preset_label,
            "auto_preset": bool(auto_preset),
            "rerank_top_k": k,
            "dense_top_k": dk or self.retriever.dense_top_k,
            "sparse_top_k": sk or self.retriever.sparse_top_k,
            "hyde_used": search_text != question,
            # off | bypass | hit | miss | nocache | error — whether the draft
            # was replayed from the cache or written fresh (see HyDE).
            "hyde_cache": hyde_status,
            "boost_code": overrides.get("boost_code", False),
            "scope": scope.labels if scope else [],
            "candidates": len(candidates),
            "omnisearch": bool(
                self.retriever.omnisearch is not None
                and (self.retriever.omnisearch.enabled if use_omni is None else use_omni)
            ),
            "live_hits": sum(1 for d in top if d.metadata.get("live")),
            "parent_context": bool(parent_on),
            "parent_swaps": swaps,
            "parent_siblings_dropped": siblings,
            "neighbor_context": bool(neighbor_on),
            "neighbors_added": neighbors,
            "hype": bool(use_hype if use_hype is not None
                         else self.retriever.hype_enabled),
            "lanes_requested": sorted(lanes) if lanes is not None else None,
            "metadata_boost": bool(self.retriever.metadata_boost
                                   if metadata_boost is None else metadata_boost),
            # The weights actually used, every lane spelled out — an echo that
            # showed only the overrides would leave the reader guessing what
            # the other lanes were fused at.
            "lane_weights": {
                lane: (weights.get(lane)
                       if weights.get(lane) is not None
                       else self.retriever.lane_weights.get(lane, 1.0))
                for lane in LANES
            },
            "rerank_mode": (rerank_mode or self.reranker.mode),
            # Report the criterion that was APPLIED, not the one that was
            # asked for: `lexical`, `none` and `laya` ignore it by design, and
            # an echo that claimed otherwise would make a no-op look like a
            # setting.
            "rerank_instruction": (
                (instruction if instruction is not None
                 else self.reranker.instruction) or None
                if (rerank_mode or self.reranker.mode) in ("cross_encoder", "http")
                else None
            ),
            "rerank_instruction_applied": (
                self.reranker.scoring_query(
                    question, (rerank_mode or self.reranker.mode), instruction)
                != question
            ),
            # The cross-encoder's model and length. http and laya report
            # neither: a Laya checkpoint reads its own max_len from its config.
            "reranker_model": (
                self.reranker.model_name
                if (rerank_mode or self.reranker.mode) == "cross_encoder"
                else None
            ),
            "reranker_max_length": (
                self.reranker.max_length
                if (rerank_mode or self.reranker.mode) == "cross_encoder"
                else None
            ),
            # What the relevance gate did; {"enabled": False} when it did not
            # run. `abstained` is why a /query came back with the fixed
            # "nothing relevant" answer and no LLM call.
            "gate": gate_info,
        }
        trace.ms["total"] = (time.perf_counter() - t_all) * 1000.0
        info["timings"] = trace.timings()
        info["lanes_run"] = dict(trace.lanes)
        info["cold"] = self._lazy_handles() != loaded_before
        return top, info

    def query(
        self,
        question: str,
        preset: str | None = None,
        top_k: int | None = None,
        dense_top_k: int | None = None,
        sparse_top_k: int | None = None,
        hyde: bool | None = None,
        omnisearch: bool | None = None,
        parent_context: bool | None = None,
        neighbor_context: bool | None = None,
        hype: bool | None = None,
        rerank: str | None = None,
        rerank_instruction: str | None = None,
        lane_weights: dict[str, float] | None = None,
        max_tokens: int | None = None,
        auto_preset: bool = True,
        lanes: list[str] | None = None,
        metadata_boost: bool | None = None,
        gate: bool | None = None,
        gate_threshold: float | None = None,
        gate_scorer: str | None = None,
    ) -> Answer:
        """
        Run the full RAG path (search + grounded generation). All knobs are
        per-call overrides that leave the warm pipeline's defaults untouched;
        see search() for their meaning. max_tokens caps the ANSWER length
        (output tokens) for this call only — None keeps generation.max_tokens.
        When the relevance gate leaves no docs the generator gets an empty list
        and returns its fixed LOW "nothing relevant" answer without calling the
        LLM; the echo's gate.abstained says why.
        """
        top, info = self.search(
            question, preset=preset, top_k=top_k,
            dense_top_k=dense_top_k, sparse_top_k=sparse_top_k,
            hyde=hyde, omnisearch=omnisearch,
            parent_context=parent_context, neighbor_context=neighbor_context,
            hype=hype, rerank=rerank, rerank_instruction=rerank_instruction,
            lane_weights=lane_weights, auto_preset=auto_preset,
            lanes=lanes, metadata_boost=metadata_boost,
            gate=gate, gate_threshold=gate_threshold, gate_scorer=gate_scorer,
        )
        answer = self.generator.generate(question, top, max_tokens=max_tokens)
        answer.retrieval = info
        return answer
