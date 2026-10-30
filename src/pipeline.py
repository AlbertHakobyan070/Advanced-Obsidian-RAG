"""
pipeline.py — The full Advanced-RAG query path, wired end to end.

    query
      -> preset resolution   explicit preset, or 'code' auto-applied when the
                             question trips a code-intent signal
      -> scope detection     on the RAW question (domain / content hints)
      -> HyDE expansion      optional; its text feeds every retrieval lane
      -> decomposition       optional; a multi-part question becomes one
                             retrieve + rerank branch per part, merged by rank
      -> hybrid retrieve     the lanes the call runs, weighted RRF, metadata boost
      -> rerank -> top k     cross_encoder | http | lexical | none | laya
      -> per-file cap        optional; at most N chunks per source file
      -> relevance gate      optional; drops chunks below a threshold, and
                             leaving none makes generation abstain
      -> small-to-big        optional parent swap / adjacent PDF pages
    search() stops here: retrieval only, no generation backend needed.
      -> grounded generation with inline [n] citations + a CONFIDENCE line
      -> optional citation verification (a second LLM pass)
    -> Answer, whose .retrieval echoes every setting that actually ran

Graph RAG (self.graph) is a separate mode: neither search() nor query() calls
it. graph_search() (the bench and POST /evidence_pack) runs search(), then a
walk, then one union rerank.

Lazy construction: heavy objects (embedding model, cross-encoder, chroma client)
are built once and reused. Build the pipeline once, call query() many times.

    from src.pipeline import RAGPipeline
    rag = RAGPipeline.from_config(cfg)
    answer = rag.query("Explain knowledge distillation in my capstone")
"""
from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path

from src.embeddings.embedder import Embedder
from src.embeddings.sidecar import iter_ids
from src.generation.generator import Answer, Generator
from src.index_ops import PAGE
from src.llm.llm_client import LLMClient
from src.retrieval.context_expand import NeighborContext, ParentContext
from src.retrieval.decompose import Decomposer, merge_branches
from src.retrieval.diversity import cap_per_file
from src.retrieval.graph_expand import GraphExpander, _paths
from src.retrieval.hyde import HyDE
from src.retrieval.outline import build_outline
from src.retrieval.relevance_gate import RelevanceGate
from src.retrieval.reranker import Reranker
from src.retrieval.retriever import (FILTER_TANTIVY, LANES, ROUTE_TANTIVY, HybridRetriever,
                                     _clean_lane_set)
from src.retrieval.route import RoutePointers
from src.retrieval.scope import ScopeRouter
from src.utils.config_loader import Config, load_config
from src.utils.logger import configure_logging, get_logger
from src.utils.provenance import (
    RouteUnavailableError, UnstampedProvenanceError, check_mode, provenance_of, provenance_rule,
    unstamped_generated)
from src.utils.timing import StageTrace

log = get_logger(__name__)

# The walk options graph_search() takes: GraphExpander.expand's, plus seed_top_k.
WALK_KEYS = frozenset({"depth", "max_per_hop", "max_total", "seed_top_k", "bridges",
                       "cross_canvas", "wikilinks", "direction", "edge_types"})
# The label on an unstamped row the active collection holds and the bm25 payload does
# not. It may still be in a chunk JSONL (appended with no bm25 rebuild since), where
# stamp-provenance reaches it; only a row still reported after the stamp and the
# rebuild is in no JSONL.
CHROMA_ONLY = "in Chroma but not in the bm25 index"
# The Chroma write keeps no .bak, so the backup comes first.
STAMP_FIX = ("back up chroma_db and the bm25 index first, then "
             "run `python main.py stamp-provenance --dry-run`, then `--yes`, "
             "then `python rebuild_bm25.py`")


def _stored_provenance(retriever, rule) -> tuple[dict[str, int], int, list[tuple[str, str]]]:
    """(unstamped, generated_rows, chroma_only) over the stored rows (spec §5.2).
    unstamped maps a source_file to its rows the rule calls generated but not
    stamped generated: the payload's, plus the active collection's rows that are
    not in the payload, keyed "<file> [CHROMA_ONLY]". generated_rows counts the
    payload rows stamped generated. chroma_only is those collection rows as sorted
    (id, source_file). No bm25 file (a fresh install): nothing to check."""
    if not Path(retriever.bm25_index).exists():
        return {}, 0, []
    payload = retriever._get_bm25()
    unstamped = unstamped_generated(payload["metadatas"], rule)
    generated_rows = sum(m.get("provenance") == "generated" for m in payload["metadatas"])
    col = retriever._get_collection()
    have = set(payload["ids"])
    extra = [i for i in iter_ids(col, PAGE) if i not in have]
    # Chroma returns None for a row stored with no metadata: no key, no source_file.
    only = []
    for o in range(0, len(extra), PAGE):
        got = col.get(ids=extra[o:o + PAGE], include=["metadatas"])
        only += [(i, m or {}) for i, m in zip(got["ids"], got["metadatas"])]
    unstamped.update({f"{f} [{CHROMA_ONLY}]": n
                      for f, n in unstamped_generated([m for _, m in only], rule).items()})
    chroma_only = sorted((i, m.get("source_file") or "") for i, m in only if unstamped_generated([m], rule))
    return unstamped, generated_rows, chroma_only


def _unstamped_message(unstamped: dict[str, int], chroma_only: list[tuple[str, str]] = ()) -> str:
    """Up to 10 files with their counts, then the fix. The Chroma-only step appears
    only with Chroma-only rows, and only after the stamp: the JSONLs are primary data,
    and the stamp may fix those rows too. It names up to 10 row ids and removes only
    those: deleting their source would take its valid rows with it."""
    names = sorted(unstamped, key=lambda f: (-unstamped[f], f))
    shown = "; ".join(f"{f} ({unstamped[f]})" for f in names[:10])
    more = f"; and {len(names) - 10} more files" if len(names) > 10 else ""
    fix = STAMP_FIX
    if chroma_only:
        rows = "; ".join(f"{i} ({f})" for i, f in chroma_only[:10])
        rows += f"; and {len(chroma_only) - 10} more" if len(chroma_only) > 10 else ""
        fix += (f". The rows {CHROMA_ONLY}: {rows}. If any is still reported after the stamp "
                f"and the rebuild, it exists only in the Chroma chunk collections, with no JSONL "
                f"row behind it: back up chroma_db, then remove exactly those ids from the chunk "
                f"collections. Do not delete their source: that removes its valid rows too")
    return (f"{sum(unstamped.values())} stored rows the provenance rule calls generated are "
            f"not stamped provenance: generated, so generated_mode drop or route cannot keep "
            f"them out of the dense lanes: {shown}{more}. Fix: {fix}")


# route-pointers itself refuses without this block, so it is refused before the sidecar.
ROUTE_BLOCK = "generated_mode route needs the retrieval.route block in config.yaml"


def _no_sidecar_message(generated_rows: int) -> str:
    return (f"generated_mode route needs the route pointer sidecar (retrieval.route.pointers_file): "
            f"the store holds {generated_rows} generated rows and none is loaded. Build it with "
            f"`python main.py route-pointers` (or the route_pointers job), then restart :8051")


def _route_coverage(ids, metas, pointers: RoutePointers | None, sidecar: str | None) -> dict:
    """Generated ids in the store vs in the sidecar (spec §5.4): the payload rows stamped
    generated, those of them whose targets() is not None, and the loaded sidecar's path."""
    gen = [i for i, m in zip(ids, metas) if m.get("provenance") == "generated"]
    return {"generated_rows": len(gen),
            "with_pointers": 0 if pointers is None else sum(pointers.targets(i) is not None for i in gen),
            "sidecar": sidecar}


def _file_filter(source_file: str | None, pages) -> dict | None:
    """search()'s source_file and pages as the route target retrieve() filters by,
    {source_file, page_start, page_end}, or None. pages is [first, last] with
    1 <= first <= last, and restricts a file, so it needs source_file."""
    if pages is not None:
        if source_file is None:
            raise ValueError("pages needs source_file: a page range restricts one file, so name it "
                             "(GET /outline lists the sources)")
        if len(pages) != 2 or not 1 <= pages[0] <= pages[1]:
            raise ValueError(f"pages must be [first, last] with 1 <= first <= last, got {list(pages)!r}")
    if source_file is None:
        return None
    first, last = pages if pages is not None else (None, None)
    return {"source_file": source_file, "page_start": first, "page_end": last}


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
        scope_lanes: bool = True,
        max_per_file: int = 0,
        decomposer: Decomposer | None = None,
        provenance_map: dict[str, str] | None = None,
        generated_mode: str = "off",
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
        # retrieval.scope_lanes: whether a detected scope opens the dense_scope /
        # sparse_scope lanes. False still detects (and echoes) the scope.
        self.scope_lanes = scope_lanes
        # retrieval.max_per_file: at most this many chunks of one source file in
        # the final list (0 = no cap). See diversity.py.
        self.max_per_file = max_per_file
        # retrieval.decompose: None = no decomposer (a per-call decompose=True
        # then does nothing to switch on). See decompose.py.
        self.decomposer = decomposer
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
        # provenance_rule(cfg) (a test fake may set a plain by_file_type map), for
        # graph_search()'s per-doc provenance echo.
        self.provenance_map = provenance_map or {}
        # retrieval.generated_mode (spec §5.1): what generated chunks do as evidence
        # when a call does not say. from_config defaults a missing key to drop.
        self.generated_mode = check_mode(generated_mode, "generated_mode")
        # Filled by from_config (_stored_provenance): stored rows the rule calls
        # generated but not stamped so, by file; the payload rows stamped generated;
        # and the unstamped Chroma-only rows as (id, source_file).
        self.unstamped: dict[str, int] = {}
        self.generated_rows = 0
        self.chroma_only: list[tuple[str, str]] = []
        # _route_coverage's dict, computed once by from_config; None = not computed.
        self.route_coverage: dict | None = None
        # GET /outline's answer (outline.build_outline), built once by from_config; None = not built.
        self.outline: dict | None = None

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

        # The startup invariant (spec §5.2): the dense lanes filter on the stored
        # key, so under drop or route a row the rule calls generated but not stamped
        # generated would slip through them. Refuse to build; /health says why.
        mode = check_mode(cfg.get("retrieval.generated_mode", "drop"), "retrieval.generated_mode")
        has_route = cfg.get("retrieval.route") is not None
        if mode == "route" and not has_route:
            raise ValueError(ROUTE_BLOCK)
        # The pointer sidecar (spec §5.4), loaded here only. In every mode, so a drop
        # pipeline can serve a per-call route; a bad sidecar therefore fails any boot.
        # The retriever is only assigned one that loads: off and drop read nothing new.
        pointers, sidecar = None, None
        if has_route:
            if not cfg.get("retrieval.route.pointers_file"):
                raise ValueError("retrieval.route.pointers_file is not set: name the route pointer "
                                 "sidecar's path in config.yaml (e.g. data/route_pointers.json)")
            pointers_file = cfg.path("retrieval.route.pointers_file")
            if pointers_file.exists():
                pointers = retriever.route_pointers = RoutePointers.load(pointers_file)
                sidecar = str(pointers_file)
        rule = provenance_rule(cfg)
        unstamped, generated_rows, chroma_only = _stored_provenance(retriever, rule)
        if mode != "off" and unstamped:
            raise UnstampedProvenanceError(_unstamped_message(unstamped, chroma_only))
        # Zero generated rows route nothing, so a fresh install starts with no sidecar.
        if mode == "route" and generated_rows and pointers is None:
            raise RouteUnavailableError(_no_sidecar_message(generated_rows))
        has_bm25 = Path(retriever.bm25_index).exists()
        payload = (retriever._get_bm25() if has_bm25
                   else {"ids": [], "metadatas": []})    # no bm25 file: as _stored_provenance
        coverage = _route_coverage(payload["ids"], payload["metadatas"], pointers, sidecar)
        # Served from memory until the next restart (spec §5.7).
        outline = build_outline(payload, rule, pointers)
        if pointers is not None and has_bm25:
            # The routed lane's per-file and per-id maps, from the payload just loaded, so
            # the first routed or map-view request after a start does not pay for them.
            retriever._file_rows()
            retriever._id_rows()

        rag = cls(
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
            scope_lanes=bool(cfg.get("retrieval.scope_lanes", True)),
            max_per_file=int(cfg.get("retrieval.max_per_file", 0) or 0),
            decomposer=Decomposer.from_config(cfg, llm),
            provenance_map=rule,
            generated_mode=mode,
        )
        # A configured off keeps them, so a per-call drop or route can refuse.
        rag.unstamped, rag.generated_rows, rag.chroma_only = unstamped, generated_rows, chroma_only
        rag.route_coverage = coverage
        rag.outline = outline
        return rag

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
        """Which heavy handles are resident right now: the Chroma collection, the
        BM25 payload, the cross-encoder and the Laya checkpoint. When a bm25 file
        exists, from_config loads the first two at boot (the provenance check);
        the rest load on the first search that uses them, after /health reports
        ready — a spike the latency numbers must flag, not average in. getattr,
        so a stand-in without these attributes reads as never-loaded, never cold."""
        return tuple(
            getattr(obj, name, None) is not None
            for obj, name in ((self.retriever, "_collection"),
                              (self.retriever, "_bm25_payload"),
                              (self.reranker, "_model"),
                              (getattr(self.reranker, "laya_scorer", None),
                               "_agent"))
        )

    def _context_switches(self, overrides: dict, parent_context,
                          neighbor_context) -> tuple[bool, bool]:
        """E2 small-to-big switches: per-call beats preset beats config. One rule for
        search() and for graph_search(), which applies them after its union rerank."""
        parent_on = parent_context
        if parent_on is None:
            parent_on = overrides.get("parent_context")
        if parent_on is None:
            parent_on = bool(self.parent_ctx and self.parent_ctx.enabled)
        neighbor_on = neighbor_context
        if neighbor_on is None:
            neighbor_on = overrides.get("neighbor_context")
        if neighbor_on is None:
            neighbor_on = bool(self.neighbor_ctx and self.neighbor_ctx.enabled)
        return parent_on, neighbor_on

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
        scope_lanes: bool | None = None,
        gate: bool | None = None,
        gate_threshold: float | None = None,
        gate_scorer: str | None = None,
        max_per_file: int | None = None,
        decompose: bool | None = None,
        generated_mode: str | None = None,
        source_file: str | None = None,
        pages: list[int] | None = None,
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
          scope_lanes True/False forces the dense_scope / sparse_scope lanes on
                      or off (None follows retrieval.scope_lanes). Off still
                      detects the scope; it just never runs those lanes
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
          max_per_file
                      at most this many chunks of one source file in the final
                      list, best first, backfilled if that leaves fewer than
                      top_k (None follows retrieval.max_per_file; 0 = no cap)
          decompose   True/False forces query decomposition on or off (None
                      follows retrieval.decompose.enabled): one LLM call splits
                      a multi-part question into sub-queries, each retrieved and
                      reranked on its own, merged by reciprocal rank. An atomic
                      question runs unchanged. LLM errors raise; it cannot be
                      combined with the relevance gate (ValueError)
          generated_mode
                      off | drop | route for this call (None follows
                      retrieval.generated_mode), passed to every retrieve();
                      drop or route raises ValueError while the store holds
                      rows the rule calls generated that are not stamped so;
                      route raises ValueError with no retrieval.route block
                      or on tantivy (both before HyDE), then
                      RouteUnavailableError while the store held
                      generated rows at startup and no pointer sidecar is loaded
          source_file / pages
                      restrict every lane of every retrieve() to one file's
                      primary chunks (any stored spelling of its path), and
                      with pages = [first, last] to those overlapping them;
                      the neighbour pages stay inside them too, and omnisearch
                      does not run. pages without source_file, not
                      1 <= first <= last, or a tantivy sparse backend raises
                      ValueError (before HyDE). The echo's file_filter
                      repeats them, with matched false when the store holds no
                      primary chunk there (an unknown file, pages outside it,
                      or a map)

        Every returned doc carries debug["provenance"], by the one rule. Under
        route the echo's route is the routed lane's summary, summed over every
        retrieve() of the call.

        The echo also reports where the time went: timings (ms per stage),
        lanes_run (candidates each lane returned) and cold (this call loaded
        the index or the cross-encoder, so its timings are not steady-state),
        and what the gate did: gate ({"enabled": False} when it did not run).
        """
        # First thing, before HyDE: retrieve() rejects a bad lane set too, but by
        # then the draft has already cost an LLM call. The cleaned value is not
        # used; retrieve() cleans the same input again.
        _clean_lane_set(lanes, "lanes")
        file_filter = _file_filter(source_file, pages)
        mode = check_mode(self.generated_mode if generated_mode is None else generated_mode,
                          "generated_mode")
        if mode != "off" and self.unstamped:
            # Never filter on stored keys that are known to be wrong.
            raise ValueError(_unstamped_message(self.unstamped, self.chroma_only))
        if mode == "route" and self.retriever.route is None:
            raise ValueError(ROUTE_BLOCK)
        # Before the 409, whose fix (a sidecar) would not make tantivy route either.
        if mode == "route" and self.retriever.sparse_backend == "tantivy":
            raise ValueError(ROUTE_TANTIVY)
        if file_filter is not None and self.retriever.sparse_backend == "tantivy":
            raise ValueError(FILTER_TANTIVY)          # retrieve() would, after HyDE's draft
        # The count taken at startup: rows appended since then run unrouted.
        if mode == "route" and self.generated_rows > 0 and self.retriever.route_pointers is None:
            raise RouteUnavailableError(_no_sidecar_message(self.generated_rows))
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
        scope_lanes_on = self.scope_lanes if scope_lanes is None else bool(scope_lanes)

        gate_on = self.gate.is_on(gate, gate_threshold, gate_scorer)
        dec_on = self.decomposer is not None and (
            self.decomposer.enabled if decompose is None else bool(decompose))
        if dec_on and gate_on:
            # The gate's rerank_score scorer would read scores computed against
            # sub-queries, not the question it claims to judge.
            raise ValueError("decompose and the relevance gate cannot run in one call: "
                             "turn one of them off")

        with trace.span("hyde"):
            search_text, hyde_status = self.hyde.expand_with_info(
                question, enabled=use_hyde)
        subqueries: list[str] = []
        if dec_on:
            with trace.span("decompose"):
                subqueries = self.decomposer.split(question)
        # Every retrieve() below takes lane_kw, so every branch applies the one mode:
        # retrieve() defaults to off, which would let generated text back in.
        lane_kw = dict(
            generated_mode=mode,
            dense_top_k=dk,
            sparse_top_k=sk,
            boost_code=overrides.get("boost_code", False),
            scope=scope if (scope and scope_lanes_on) else None,
            omnisearch=use_omni,
            hype=use_hype,
            lane_weights=weights or None,
            lanes=lanes,
            metadata_boost=metadata_boost,
            trace=trace,
            file_filter=file_filter,
        )
        candidates = self.retriever.retrieve(search_text, **lane_kw)
        rerank_mode = rerank if rerank is not None else overrides.get("rerank_mode")
        # "" is meaningful here: it turns a configured criterion OFF for this
        # call, which `or` would silently discard.
        instruction = (rerank_instruction if rerank_instruction is not None
                       else overrides.get("rerank_instruction"))
        cap = self.max_per_file if max_per_file is None else int(max_per_file)
        with trace.span("rerank"):
            # A cap or a branch merge needs the whole scored pool, not the top k.
            top = self.reranker.rerank(question, candidates,
                                       top_k=len(candidates) if (cap > 0 or subqueries) else k,
                                       mode=rerank_mode, instruction=instruction)
        branch_pools = [len(candidates)]
        if subqueries:
            # Each sub-query is its own branch: retrieved with its own text (no
            # HyDE) and reranked against itself; see decompose.py.
            rankings = [top]
            for sq in subqueries:
                pool = self.retriever.retrieve(sq, **lane_kw)
                branch_pools.append(len(pool))
                with trace.span("rerank"):
                    rankings.append(self.reranker.rerank(sq, pool, top_k=len(pool),
                                                         mode=rerank_mode, instruction=instruction))
            top = merge_branches(rankings, self.retriever.rrf_k)
        top = cap_per_file(top, k, cap)

        # Relevance gate (post-rerank, pre-expansion). It judges the ORIGINAL
        # question — HyDE's text is a hypothetical answer, not what was asked.
        # The span exists only for a call that gates, so a gate-off call's
        # timings carry no stage that never ran.
        with trace.span("gate") if gate_on else nullcontext():
            top, gate_info = self.gate.apply(
                question, top, enabled=gate, threshold=gate_threshold,
                scorer=gate_scorer)

        # E2 small-to-big (post-rerank): per-call beats preset beats config.
        parent_on, neighbor_on = self._context_switches(
            overrides, parent_context, neighbor_context)
        swaps = siblings = 0
        if parent_on and self.parent_ctx:
            with trace.span("parent"):
                top, swaps, siblings = self.parent_ctx.apply(top)
        neighbors = 0
        if neighbor_on and self.neighbor_ctx:
            with trace.span("neighbor"):
                top, neighbors = self.neighbor_ctx.apply(
                    top, self.retriever._get_collection())
        span = None
        if file_filter is not None:
            # The neighbour pages are fetched by page, not through retrieve(), so they are held
            # to the same span here; a parent swap keeps its child's id and file.
            span = self.retriever.span_ids(file_filter)
            top = [d for d in top if d.id in span]
            neighbors = sum("neighbor_of" in d.debug for d in top)
        for d in top:     # the final list: after the parent swap and the neighbours
            d.debug["provenance"] = provenance_of(d.metadata, self.provenance_map)

        info = {
            "generated_mode": mode,
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
            "scope_lanes": scope_lanes_on,
            "max_per_file": cap,
            # off | atomic (the LLM kept the question whole) | split; one
            # candidate-pool size per branch, the original question's first.
            "decompose": {"status": "off" if not dec_on else ("split" if subqueries else "atomic"),
                          "subqueries": subqueries, "branch_pools": branch_pools},
            "candidates": len(candidates),
            "omnisearch": bool(
                self.retriever.omnisearch is not None and file_filter is None   # a filter skips it
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
        if mode == "route":
            info["route"] = dict(trace.route)
        if file_filter is not None:
            # matched: the store holds a primary row of that file in those pages.
            info["file_filter"] = {**file_filter, "matched": bool(span)}
        info["cold"] = self._lazy_handles() != loaded_before
        return top, info

    def graph_search(self, question, graph: dict | None = None, **search_kw) -> tuple[list, dict]:
        """search(), then (graph not None) a walk seeded by the search's own canvas/md hits
        plus graph.seed_from_query, then ONE rerank of search hits + walked chunks down to
        the search's own top_k - the same budget, so graph-on never wins by having more
        slots. The union rerank scores CHILD texts on both sides: search() runs with parent
        swap and neighbour expansion off, and they are applied to the final list afterwards,
        with the caller's settings and search()'s own helpers. A walk that adds nothing
        returns the search result unreranked, so G-legacy with no additions equals G-off.
        graph=None: search() alone, in the same echo shape (the bench's graph-off row).
        graph.enabled is NOT consulted: the HTTP routes gate on it; the bench measures
        before anything is switched on. max_per_file and the relevance gate are not
        re-applied after the union rerank; the eval runs both off.
        info["graph"]: added_ids = what the walk brought in (hop >= 1 and bridges);
        seed_ids = seed_from_query's hits the search lacked (depth 0: in the union
        rerank, but reached by no edge); search_metas = the search's own list before the
        union rerank; paths from the walk's tree in every mode. Under drop, generated walked
        nodes leave; under route, they give way to the primaries they cite (routed_for),
        less any already a search hit or walked (with no sidecar loaded, they just leave).
        Such a primary counts as walked when an edge reached one of its routers, and
        info["graph"]["route"] is routed_for's summary, its new the primaries the walk
        added, present only when it ran."""
        walk_route = None
        if graph is None:
            docs, info = self.search(question, **search_kw)
            seeds, added, walked, walked_ids, searched = {}, [], {"stats": None}, set(), docs
        else:
            # The walk seeds and steps across the whole vault, so it cannot keep to one file.
            if search_kw.get("source_file") is not None or search_kw.get("pages") is not None:
                raise ValueError("graph true cannot take source_file or pages: they keep the evidence "
                                 "inside one file, and a graph walk leaves it. Send graph false, or "
                                 "drop source_file and pages")
            if self.graph is None:
                raise ValueError("graph_search needs a graph expander, and this pipeline has "
                                 "none: build it with RAGPipeline.from_config")
            opts = {k: v for k, v in graph.items() if v is not None}
            unknown = sorted(set(opts) - WALK_KEYS)
            if unknown:
                raise ValueError(f"unknown graph option(s) {unknown}: a walk takes "
                                 f"{sorted(WALK_KEYS)}")
            seed_top_k = opts.pop("seed_top_k", None)
            # For the walk guard only: search_kw reaches search() unchanged, which
            # resolves the same None to self.generated_mode, so both apply one mode.
            mode = check_mode(search_kw.get("generated_mode") or self.generated_mode,
                              "generated_mode")
            overrides, _ = self._resolve_overrides(
                question, search_kw.get("preset"), auto_preset=search_kw.get("auto_preset", True))
            parent_on, neighbor_on = self._context_switches(
                overrides, search_kw.get("parent_context"), search_kw.get("neighbor_context"))
            docs, info = self.search(question, **{**search_kw, "parent_context": False,
                                                  "neighbor_context": False})
            searched = docs
            tr = StageTrace()
            with tr.span("graph_seed"):
                kinds = {self.graph.file_type}
                if opts.get("wikilinks", self.graph.wikilinks):    # per call beats graph.wikilinks
                    kinds |= {"note", "daily_note"}
                seeds = {d.id: d for d in docs if d.metadata.get("file_type") in kinds}
                for d in self.graph.seed_from_query(question, seed_top_k=seed_top_k,
                                                    rerank=info["rerank_mode"]):
                    seeds.setdefault(d.id, d)
            with tr.span("graph_expand"):
                walked = self.graph.expand(list(seeds.values()), **opts)
                have = {d.id for d in docs}
                new = [n for n in walked["nodes"] if n["id"] not in have]
                added = self.graph._fetch_ids([n["id"] for n in new])
                walked_ids = {n["id"] for n in new if n["depth"] > 0}    # depth 0 = a query seed
                if mode == "drop":    # spec §5.5: generated nodes leave; the seeds stay
                    added = [d for d in added
                             if provenance_of(d.metadata, self.provenance_map) != "generated"]
                elif mode == "route":    # spec §5.5: generated nodes route to what they cite
                    gen = [d for d in added if provenance_of(d.metadata, self.provenance_map) == "generated"]
                    added = [d for d in added
                             if provenance_of(d.metadata, self.provenance_map) != "generated"]
                    # No sidecar: every router is unrouted, so they just leave, as under drop.
                    if gen and self.retriever.route_pointers is not None:
                        # In walk order, not _fetch_ids': max_targets keeps the first routers'.
                        pos = {n["id"]: i for i, n in enumerate(new)}
                        routed, walk_route = self.retriever.routed_for(
                            question, sorted((g.id for g in gen), key=pos.__getitem__))
                        taken = have | {d.id for d in added}       # the rerank does not dedupe
                        kept = [r for r in routed if r.id not in taken]
                        walk_route["new"] = len(kept)              # what the walk added, as the lane counts
                        depth = {n["id"]: n["depth"] for n in new}
                        walked_ids |= {r.id for r in kept
                                       if any(depth.get(f["id"], 0) > 0 for f in r.debug["found_via"])}
                        added += kept
            if added:
                with tr.span("graph_rerank"):
                    # The mode and criterion search() applied, presets included; "" = none was.
                    docs = self.reranker.rerank(question, docs + added, top_k=info["rerank_top_k"],
                                                mode=info["rerank_mode"],
                                                instruction=info["rerank_instruction"] or "")
            swaps = siblings = neighbors = 0
            if parent_on and self.parent_ctx:
                with tr.span("parent"):
                    docs, swaps, siblings = self.parent_ctx.apply(docs)
            if neighbor_on and self.neighbor_ctx:
                with tr.span("neighbor"):
                    docs, neighbors = self.neighbor_ctx.apply(
                        docs, self.retriever._get_collection())
            info.update(parent_context=bool(parent_on), parent_swaps=swaps,
                        parent_siblings_dropped=siblings, neighbor_context=bool(neighbor_on),
                        neighbors_added=neighbors)
            info["timings"] = {**info["timings"], **tr.timings(),
                               "total": round(info["timings"]["total"] + sum(tr.ms.values()), 2)}
        for d in docs:
            d.debug["provenance"] = provenance_of(d.metadata, self.provenance_map)
        info["graph"] = {"walk": graph is not None, "seeds": len(seeds),
                         "added_ids": [d.id for d in added if d.id in walked_ids],
                         "seed_ids": [d.id for d in added if d.id not in walked_ids],
                         "stats": walked["stats"] or None,
                         "paths": _paths(walked.get("tree", [])),
                         "search_metas": [d.metadata for d in searched],
                         "provenance": [d.debug["provenance"] for d in docs]}
        if walk_route is not None:
            info["graph"]["route"] = walk_route
        return docs, info

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
        scope_lanes: bool | None = None,
        gate: bool | None = None,
        gate_threshold: float | None = None,
        gate_scorer: str | None = None,
        max_per_file: int | None = None,
        decompose: bool | None = None,
        generated_mode: str | None = None,
        source_file: str | None = None,
        pages: list[int] | None = None,
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
            lanes=lanes, metadata_boost=metadata_boost, scope_lanes=scope_lanes,
            gate=gate, gate_threshold=gate_threshold, gate_scorer=gate_scorer,
            max_per_file=max_per_file, decompose=decompose, generated_mode=generated_mode,
            source_file=source_file, pages=pages,
        )
        answer = self.generator.generate(question, top, max_tokens=max_tokens)
        answer.retrieval = info
        return answer
