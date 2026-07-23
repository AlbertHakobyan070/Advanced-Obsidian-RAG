"""
pipeline.py — The full Advanced-RAG query path, wired end to end.

    query
      -> HyDE expansion (optional)
      -> hybrid retrieve (dense + BM25 + RRF)
      -> cross-encoder rerank -> top k
      -> grounded generation with inline citations
      -> optional citation verification
    -> Answer

Lazy construction: heavy objects (embedding model, cross-encoder, chroma client)
are built once and reused. Build the pipeline once, call query() many times.

    from src.pipeline import RAGPipeline
    rag = RAGPipeline.from_config(cfg)
    answer = rag.query("Explain knowledge distillation in my capstone")
"""
from __future__ import annotations

from src.embeddings.embedder import Embedder
from src.generation.generator import Answer, Generator
from src.llm.llm_client import LLMClient
from src.retrieval.context_expand import NeighborContext, ParentContext
from src.retrieval.hyde import HyDE
from src.retrieval.reranker import Reranker
from src.retrieval.retriever import HybridRetriever
from src.retrieval.scope import ScopeRouter
from src.utils.config_loader import Config, load_config
from src.utils.logger import configure_logging, get_logger

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
        auto_preset: bool = True,
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
          auto_preset False disables the implicit code preset, giving compare
                      calls an explicit config-only baseline
        """
        log.info("=== SEARCH: %s", question)

        overrides, preset_label = self._resolve_overrides(
            question, preset, auto_preset=auto_preset)
        k = top_k or overrides.get("rerank_top_k") or self.rerank_top_k
        # Per-lane pool sizes: explicit per-call beats preset beats config default.
        dk = dense_top_k if dense_top_k is not None else overrides.get("dense_top_k")
        sk = sparse_top_k if sparse_top_k is not None else overrides.get("sparse_top_k")
        use_hyde = hyde if hyde is not None else overrides.get("use_hyde")
        use_omni = omnisearch if omnisearch is not None else overrides.get("omnisearch")
        use_hype = hype if hype is not None else overrides.get("hype")
        # Domain/content hints are detected on the RAW question (HyDE prose
        # would dilute the keywords) and routed as extra fusion lanes.
        scope = self.scope_router.detect(question)

        search_text = self.hyde.expand(question, enabled=use_hyde)
        candidates = self.retriever.retrieve(
            search_text,
            dense_top_k=dk,
            sparse_top_k=sk,
            boost_code=overrides.get("boost_code", False),
            scope=scope if scope else None,
            omnisearch=use_omni,
            hype=use_hype,
        )
        rerank_mode = rerank if rerank is not None else overrides.get("rerank_mode")
        top = self.reranker.rerank(question, candidates, top_k=k,
                                   mode=rerank_mode)

        # E2 small-to-big (post-rerank): per-call beats preset beats config.
        parent_on = parent_context
        if parent_on is None:
            parent_on = overrides.get("parent_context")
        if parent_on is None:
            parent_on = bool(self.parent_ctx and self.parent_ctx.enabled)
        swaps = siblings = 0
        if parent_on and self.parent_ctx:
            top, swaps, siblings = self.parent_ctx.apply(top)
        neighbor_on = neighbor_context
        if neighbor_on is None:
            neighbor_on = overrides.get("neighbor_context")
        if neighbor_on is None:
            neighbor_on = bool(self.neighbor_ctx and self.neighbor_ctx.enabled)
        neighbors = 0
        if neighbor_on and self.neighbor_ctx:
            top, neighbors = self.neighbor_ctx.apply(
                top, self.retriever._get_collection())

        info = {
            "preset": preset_label,
            "auto_preset": bool(auto_preset),
            "rerank_top_k": k,
            "dense_top_k": dk or self.retriever.dense_top_k,
            "sparse_top_k": sk or self.retriever.sparse_top_k,
            "hyde_used": search_text != question,
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
            "rerank_mode": (rerank_mode or self.reranker.mode),
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
        }
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
        max_tokens: int | None = None,
        auto_preset: bool = True,
    ) -> Answer:
        """
        Run the full RAG path (search + grounded generation). All knobs are
        per-call overrides that leave the warm pipeline's defaults untouched;
        see search() for their meaning. max_tokens caps the ANSWER length
        (output tokens) for this call only — None keeps generation.max_tokens.
        """
        top, info = self.search(
            question, preset=preset, top_k=top_k,
            dense_top_k=dense_top_k, sparse_top_k=sparse_top_k,
            hyde=hyde, omnisearch=omnisearch,
            parent_context=parent_context, neighbor_context=neighbor_context,
            hype=hype, rerank=rerank, auto_preset=auto_preset,
        )
        answer = self.generator.generate(question, top, max_tokens=max_tokens)
        answer.retrieval = info
        return answer
