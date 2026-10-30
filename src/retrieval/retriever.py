"""
retriever.py — Hybrid search with weighted Reciprocal Rank Fusion.

Pipeline (every lane is a ranked list of (id, text, metadata)):
    query -> dense        ChromaDB, always
          -> sparse       BM25, always
          -> dense_code   + sparse_code    when boost_code (the code preset)
          -> dense_scope  + sparse_scope   when a Scope was detected
          -> omnisearch   live vault over HTTP, per config or per call
          -> hype         hypothetical-question collection, per config or call
          -> routed       generated_mode route: the primary pages generated hits cite
       -> weighted RRF fuse -> metadata boost -> ranked candidate pool
A per-call `lanes` list restricts which of these run (it never forces a
conditional one on) — see retrieve().

RRF (Reciprocal Rank Fusion) merges ranked lists without needing their scores
to be on the same scale. For a doc at rank r in lane L (r counts from 0 here):
    contribution = weight_L / (rrf_k + r)
summed across every lane the doc appears in. Each weight defaults to 1.0,
which is plain RRF. The fused order is then nudged by metadata (a course,
domain or tag named in the query; has_code on code queries) before the
reranker sees the pool.

Usage:
    from src.retrieval.retriever import HybridRetriever
    r = HybridRetriever.from_config(cfg, embedder)
    results = r.retrieve("What is ARIMA?")          # -> list[RetrievedDoc]
"""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass, field
from itertools import zip_longest
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.embeddings.embedder import Embedder, _tokenize
from src.retrieval import sparse_tantivy
from src.embeddings.sidecar import (
    check_collection, sidecar_label, sidecar_path, unstamped_warning)
from src.ingestion.canvas_loader import norm_vault_path
from src.utils.chroma_client import persistent_client
from src.utils.config_loader import Config
from src.utils.logger import get_logger
from src.utils.provenance import ProvenanceRule, check_mode, provenance_of, provenance_rule
from src.utils.timing import ROUTE_KEYS, StageTrace

if TYPE_CHECKING:
    from src.retrieval.route import RoutePointers

log = get_logger(__name__)


# Every lane that can reach RRF fusion. ONE list, so config, the API schema
# and the tests cannot drift from what retrieve() actually builds.
#
# Not just two: the code and scope lanes are separate filtered passes,
# omnisearch/hype are their own retrievers, and routed brings in the pages
# generated hits cite. Weighting them individually is therefore a far finer
# instrument than "dense vs sparse".
LANES = ("dense", "sparse", "dense_code", "sparse_code",
         "dense_scope", "sparse_scope", "omnisearch", "hype", "routed")

# Route reuses bm25s's whole score vector, which tantivy never computes (spec §5.4).
ROUTE_TANTIVY = "generated_mode route needs retrieval.sparse_backend: bm25s in config.yaml"
# A file filter ranks the same spans as route, so it needs that vector too (spec §5.7).
FILTER_TANTIVY = "source_file needs retrieval.sparse_backend: bm25s in config.yaml"

def _clean_lane_weights(raw, where: str) -> dict[str, float]:
    """Validate a lane_weights mapping. Unknown lane names are an ERROR, not a
    no-op: a typo'd key would otherwise look like a setting that does nothing."""
    if not raw:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{where} must be a mapping of lane -> weight, "
                         f"got {type(raw).__name__}")
    out: dict[str, float] = {}
    for lane, weight in raw.items():
        if lane not in LANES:
            raise ValueError(
                f"{where}: unknown lane {lane!r}. Known lanes: {', '.join(LANES)}")
        try:
            value = float(weight)
        except (TypeError, ValueError):
            raise ValueError(
                f"{where}[{lane!r}] must be a number, got {weight!r}") from None
        if value < 0:
            raise ValueError(f"{where}[{lane!r}] must be >= 0, got {value}")
        out[lane] = value
    return out


def _clean_lane_set(raw, where: str) -> frozenset[str] | None:
    """Validate a per-call lane selection. None = no restriction. An unknown
    name is an ERROR, as with lane weights: a typo that silently ran every
    lane would make an ablation measure the full pipeline and call it a
    baseline."""
    if raw is None:
        return None
    names = [str(x).strip() for x in raw]
    if not names:
        raise ValueError(f"{where}: name at least one lane (one of {', '.join(LANES)})")
    unknown = sorted(set(names) - set(LANES))
    if unknown:
        raise ValueError(f"{where}: unknown lane(s) {', '.join(unknown)}; "
                         f"expected any of {', '.join(LANES)}")
    return frozenset(names)


def _in_pages(meta: dict, a: int, b: int) -> bool:
    """meta's pages overlap a..b the way a Chroma page where would read them
    (page_start <= b and page_end >= a): a row missing either key is outside."""
    start, end = meta.get("page_start"), meta.get("page_end")
    return start is not None and end is not None and start <= b and end >= a


def _within(target: dict, span: dict) -> dict | None:
    """A route target clipped to a file filter's span: the same file, the pages both cover
    (null pages are the whole file), or None when the two do not meet."""
    if norm_vault_path(target["source_file"]) != norm_vault_path(span["source_file"]):
        return None
    if span["page_start"] is None:
        return target
    if target["page_start"] is None:
        return dict(target, page_start=span["page_start"], page_end=span["page_end"])
    a, b = max(target["page_start"], span["page_start"]), min(target["page_end"], span["page_end"])
    return dict(target, page_start=a, page_end=b) if a <= b else None


def _build_omnisearch(cfg: Config):
    """Construct the live-vault client iff the config block exists.

    Returns the client even when retrieval.omnisearch.enabled is false, so it
    can be toggled per-query ({"omnisearch": true}) or live via POST /config
    without a restart. Returns None only when the block is absent entirely.
    """
    if not (cfg.get("retrieval.omnisearch") or {}):
        return None
    from src.retrieval.omnisearch_client import OmnisearchClient
    return OmnisearchClient.from_config(cfg)


@dataclass
class RetrievedDoc:
    """One candidate as it moves through fusion, reranking and generation.

    `score` starts as the fused RRF score (then any metadata boost);
    `rerank_score` is set by the reranker and is on that model's own scale.
    dense_rank / sparse_rank are 0-based; every other lane records its rank in
    debug["<lane>_rank"]."""
    id: str
    text: str
    metadata: dict[str, Any]
    score: float = 0.0                       # fused/rerank score
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rerank_score: float | None = None
    debug: dict[str, Any] = field(default_factory=dict)

    @property
    def source_label(self) -> str:
        """Short label: '<group> (date)', else the file stem, else the id.

        <group> is read from course / canonical_course / domain, but the
        loaders write course_name / course_code instead (see the note in
        _apply_metadata_boost), so in practice it is the DOMAIN. Answer
        citations use Generator._source_label, which reads the real fields;
        this one is what the Streamlit source list and graph-mode nodes show,
        and that function's last-resort fallback."""
        m = self.metadata
        course = m.get("course") or m.get("canonical_course") or m.get("domain")
        date = m.get("date") or m.get("note_date")
        fname = m.get("source_file") or m.get("file") or m.get("path")
        if course and date:
            return f"{course} ({date})"
        if course:
            return str(course)
        if fname:
            return Path(str(fname)).stem
        return self.id


class HybridRetriever:
    """Runs the lanes and fuses them. Opens the Chroma collection and
    unpickles the BM25 payload lazily on first use, then keeps both for the
    life of this object — a BM25 rebuild on disk is not seen by a warm
    instance until a new retriever is constructed.

    Also the store other components borrow: GraphExpander calls
    _dense_search / _sparse_search / _get_collection directly, and the
    pipeline hands _get_collection() to NeighborContext."""

    # Class-level defaults, so a HybridRetriever.__new__ test retriever has them.
    # provenance_rule is None only there and in direct constructions; from_config
    # always builds one. route_pointers / route belong to the route lane.
    provenance_rule = None
    route_pointers = None
    route = None
    sparse_backend = "bm25s"
    _generated_cache = None          # (payload, rows) behind _generated_rows()
    _file_cache = None               # (payload, files) behind _file_rows()
    _id_cache = None                 # (payload, ids) behind _id_rows()
    _vec_cache = None                # (payload, {row: unit vector}) behind _span_vectors()

    def __init__(
        self,
        embedder: Embedder,
        chroma_dir: Path,
        bm25_index: Path,
        collection_name: str,
        dense_top_k: int = 20,
        sparse_top_k: int = 20,
        rrf_k: int = 60,
        metadata_boost: bool = True,
        code_file_types: list[str] | None = None,
        omnisearch=None,
        hype_enabled: bool = False,
        hype_collection: str = "hype_questions",
        hype_top_k: int = 15,
        lane_weights: dict[str, float] | None = None,
        sparse_backend: str = "bm25s",
        tantivy_filter_pool: int = 2000,
        provenance_rule: ProvenanceRule | None = None,
        route_pointers: RoutePointers | None = None,
        route: dict | None = None,
    ):
        # Which engine ranks the BM25 lanes (see sparse_tantivy.py). The pickle
        # supplies every row's text and metadata either way.
        self.sparse_backend = sparse_tantivy.check_backend(sparse_backend)
        if isinstance(tantivy_filter_pool, bool) or not isinstance(tantivy_filter_pool, int) \
                or tantivy_filter_pool < 1:
            raise ValueError("retrieval.tantivy_filter_pool must be a positive integer, "
                             f"got {tantivy_filter_pool!r}")
        self.tantivy_filter_pool = tantivy_filter_pool
        self._tantivy = None
        self.embedder = embedder
        self.chroma_dir = chroma_dir
        self.bm25_index = bm25_index
        self.collection_name = collection_name
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.rrf_k = rrf_k
        self.metadata_boost = metadata_boost
        # file_type values that count as "own code" for the code lane
        self.code_file_types = [str(t).lower() for t in (code_file_types or [])]
        # Optional live-vault lane (OmnisearchClient); None = feature absent.
        self.omnisearch = omnisearch
        # HyPE lane (build_hype.py): query→hypothetical-question matching,
        # mapped back to parent chunks. Fail-soft when the collection is absent.
        self.hype_enabled = hype_enabled
        self.hype_collection = hype_collection
        self.hype_top_k = hype_top_k
        # Per-lane RRF weights. EVERY lane defaults to 1.0, so an unset config
        # fuses byte-identically to the unweighted formula this replaced.
        self.lane_weights = _clean_lane_weights(lane_weights,
                                                "retrieval.lane_weights")
        # The one provenance rule; generated_mode drop classifies by it.
        self.provenance_rule = provenance_rule
        # generated_mode route: the retrieval.route block, and the pointer sidecar
        # (pipeline.from_config loads and assigns it; None = every router hit unrouted).
        # The block's knobs are checked here, so a bad edit fails the build naming its key.
        if route is not None:
            if not isinstance(route, dict):
                raise ValueError(f"retrieval.route must be a mapping, got {route!r}: fix it in config.yaml")
            for key in ("router_k", "per_target", "max_targets", "max_new"):
                value = route.get(key)
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ValueError(f"retrieval.route.{key} must be a positive integer, "
                                     f"got {value!r}: fix it in config.yaml")
        self.route_pointers = route_pointers
        self.route = route

        self._collection = None
        self._bm25_payload = None
        self._hype_col = None
        self._hype_missing_logged = False

    @classmethod
    def from_config(cls, cfg: Config, embedder: Embedder) -> "HybridRetriever":
        retriever = cls(
            embedder=embedder,
            chroma_dir=cfg.path("paths.chroma_dir"),
            bm25_index=cfg.path("paths.bm25_index"),
            collection_name=cfg.get("paths.collection_name", "obsidian_vault"),
            dense_top_k=cfg.get("retrieval.dense_top_k", 20),
            sparse_top_k=cfg.get("retrieval.sparse_top_k", 20),
            rrf_k=cfg.get("retrieval.rrf_k", 60),
            metadata_boost=cfg.get("retrieval.metadata_boost", True),
            code_file_types=cfg.get("retrieval.code_file_types",
                                    ["ipynb", "py", "r", "rmd"]),
            omnisearch=_build_omnisearch(cfg),
            hype_enabled=bool(cfg.get("retrieval.hype.enabled", False)),
            hype_collection=cfg.get("retrieval.hype.collection", "hype_questions"),
            hype_top_k=int(cfg.get("retrieval.hype.top_k", 15)),
            lane_weights=cfg.get("retrieval.lane_weights", None),
            sparse_backend=cfg.get("retrieval.sparse_backend", "bm25s"),
            tantivy_filter_pool=cfg.get("retrieval.tantivy_filter_pool", 2000),
            provenance_rule=provenance_rule(cfg),
            route=cfg.get("retrieval.route"),     # the sidecar is pipeline.from_config's to load
        )
        # Route scores bm25s's whole corpus once and reuses it (spec §5.4); tantivy never
        # computes that vector. Before the index check, whose RuntimeError would hide this.
        if retriever.sparse_backend == "tantivy" and cfg.get("retrieval.generated_mode") == "route":
            raise ValueError(ROUTE_TANTIVY)
        # A tantivy backend with no index fails the build here, so /health says
        # why, instead of the first query failing. The rows are checked against
        # the pickle on first use (that needs the pickle loaded).
        if (retriever.sparse_backend == "tantivy" and not
                (sparse_tantivy.index_dir(retriever.bm25_index) / "noetrix_meta.json").is_file()):
            raise sparse_tantivy.TantivyIndexError(
                f"retrieval.sparse_backend is tantivy but there is no index at "
                f"{sparse_tantivy.index_dir(retriever.bm25_index)}: run the "
                "build_tantivy job (python build_tantivy.py) or set it back to bm25s")
        # The fingerprint guard. Vectors from another embedder are searched
        # without complaint and rank nonsense, so a mismatch raises HERE: the
        # service comes up failed (and says why) instead of answering wrongly.
        # No sidecar is the live store's state until `rag stamp` records it:
        # warn and serve, never refuse.
        sidecar = check_collection(embedder, retriever.chroma_dir,
                                   retriever.collection_name, role="chunks")
        if sidecar is None:
            log.warning(unstamped_warning(
                retriever.collection_name,
                sidecar_path(retriever.chroma_dir, retriever.collection_name), embedder.spec))
        else:
            log.info("Embedding fingerprint OK: %r was built by %s",
                     retriever.collection_name, sidecar_label(sidecar))
        # An enabled HyPE lane has a collection of its own to check; whether the
        # collection exists at all is _get_hype_collection's business.
        if retriever.hype_enabled:
            check_collection(embedder, retriever.chroma_dir,
                             retriever.hype_collection, role="hype")
        return retriever

    # ---- lazy loaders ----

    def _get_collection(self):
        if self._collection is None:
            client = persistent_client(self.chroma_dir)
            self._collection = client.get_collection(self.collection_name)
        return self._collection

    def _get_bm25(self):
        if self._bm25_payload is None:
            with open(self.bm25_index, "rb") as f:
                self._bm25_payload = pickle.load(f)
        return self._bm25_payload

    def _get_hype_collection(self):
        """The HyPE question collection, or None when it has not been built (the
        lane is optional and build_hype.py may simply not have been run yet).

        A collection that EXISTS but was embedded by another embedder raises
        EmbeddingMismatchError: failing soft there would be the lane quietly
        matching queries against another model's questions."""
        if self._hype_col is None:
            client = persistent_client(self.chroma_dir)
            try:
                col = client.get_collection(self.hype_collection)
            except Exception:
                if not self._hype_missing_logged:
                    log.info("HyPE lane requested but collection %r doesn't "
                             "exist — run build_hype.py first (lane skipped).",
                             self.hype_collection)
                    self._hype_missing_logged = True
                self._hype_col = False           # sentinel: checked, absent
            else:
                # Outside the try: a mismatch must surface, not read as "not built yet".
                if check_collection(self.embedder, self.chroma_dir,
                                    self.hype_collection, role="hype") is None:
                    log.warning(unstamped_warning(
                        self.hype_collection,
                        sidecar_path(self.chroma_dir, self.hype_collection),
                        self.embedder.spec,
                        command=f"rag stamp --collection {self.hype_collection}"))
                self._hype_col = col
        return self._hype_col or None

    def _drop_rule(self) -> ProvenanceRule:
        if self.provenance_rule is None:
            raise ValueError("generated_mode drop or route needs the provenance rule: "
                             "build the retriever with from_config")
        return self.provenance_rule

    def _generated_rows(self) -> frozenset[int]:
        """Payload indices the FULL rule calls generated (stored key and path rule),
        so the sparse lanes agree with the rule even on rows no stamp has reached.
        Computed once per loaded payload."""
        payload = self._get_bm25()
        if self._generated_cache is None or self._generated_cache[0] is not payload:
            rule = self._drop_rule()
            self._generated_cache = (payload, frozenset(
                i for i, m in enumerate(payload["metadatas"])
                if provenance_of(m, rule) == "generated"))
        return self._generated_cache[1]

    def _file_rows(self) -> dict[str, tuple[list[int], frozenset[str]]]:
        """norm_vault_path(source_file) -> (its payload indices, every stored spelling of
        it). A route target names its file in its own spelling, while case, slash and
        Unicode form all vary, so a target finds its rows by the normalised path.
        Computed once per loaded payload."""
        payload = self._get_bm25()
        if self._file_cache is None or self._file_cache[0] is not payload:
            files: dict[str, tuple[list[int], set[str]]] = {}
            for i, m in enumerate(payload["metadatas"]):
                if m.get("source_file"):
                    rows, spellings = files.setdefault(norm_vault_path(m["source_file"]), ([], set()))
                    rows.append(i)
                    spellings.add(m["source_file"])
            self._file_cache = (payload, {k: (rows, frozenset(s)) for k, (rows, s) in files.items()})
        return self._file_cache[1]

    def _id_rows(self) -> dict[str, int]:
        """doc_id -> payload index. Computed once per loaded payload."""
        payload = self._get_bm25()
        if self._id_cache is None or self._id_cache[0] is not payload:
            self._id_cache = (payload, {d: i for i, d in enumerate(payload["ids"])})
        return self._id_cache[1]

    def _payload_rows(self, indices) -> list[tuple[str, str, dict]]:
        payload = self._get_bm25()
        return [(payload["ids"][i], payload["documents"][i], payload["metadatas"][i]) for i in indices]

    def _bm25_scores(self, query: str):
        """Every payload row's bm25s score for query. bm25s scores the whole corpus per
        call, so route computes this once per retrieve() and every sparse pass reuses it."""
        return self._get_bm25()["bm25"].get_scores(_tokenize(query))

    # ---- individual searches ----

    def _dense_search(
        self, qvec: list[float], top_k: int, where: dict | None = None,
        drop_generated: bool = False,
    ) -> list[tuple[str, str, dict]]:
        """drop_generated keeps what a Chroma {"provenance": {"$ne": "generated"}}
        filter keeps (rows without the key too), without asking Chroma for it: on
        the live store that filter cost ~60x this over-fetch. So the lane ranks with
        `where` alone and drops stored-generated rows here, doubling the fetch until
        top_k survive or the collection runs out. Ranks can differ slightly from
        Chroma's filtered search (both are approximate). The startup invariant
        (pipeline._stored_provenance) is what makes the stored key enough."""
        n = 2 * top_k if drop_generated else top_k
        while True:
            kwargs: dict[str, Any] = {
                "query_embeddings": [qvec],
                "n_results": n,
                "include": ["documents", "metadatas"],
            }
            if where:
                kwargs["where"] = where
            res = self._get_collection().query(**kwargs)
            rows = list(zip(res["ids"][0], res["documents"][0], res["metadatas"][0]))
            if not drop_generated:
                return rows
            kept = [r for r in rows if (r[2] or {}).get("provenance") != "generated"]
            if len(kept) >= top_k or len(rows) < n:
                return kept[:top_k]
            n *= 2

    def _sparse_search(
        self, query: str, top_k: int, predicate=None, drop_generated=False, scores=None
    ) -> list[tuple[str, str, dict]]:
        """predicate(meta) -> bool restricts the ranked list (code/scope lanes).
        drop_generated removes _generated_rows() before top_k, so the lane refills.
        scores (bm25s; route passes it) is this query's _bm25_scores, computed once."""
        payload = self._get_bm25()
        gen = self._generated_rows() if drop_generated else frozenset()
        if self.sparse_backend == "tantivy":
            if self._tantivy is None:
                self._tantivy = sparse_tantivy.TantivySparse(self.bm25_index, payload["ids"])
            # ponytail: a filtered lane (scope/code, or a generated drop) ranks the
            # best `tantivy_filter_pool` hits and filters those (bm25s filters the
            # whole ranking); raise the pool, or push the filter into the query, if
            # a filtered lane comes up short.
            pool = top_k if predicate is None and not drop_generated else max(top_k, self.tantivy_filter_pool)
            rows = [r for r, s in self._tantivy.search(_tokenize(query), pool) if s > 0]
            if predicate is not None:
                rows = [r for r in rows if predicate(payload["metadatas"][r])]
            if drop_generated:
                rows = [r for r in rows if r not in gen]
            return [(payload["ids"][i], payload["documents"][i], payload["metadatas"][i])
                    for i in rows[:top_k]]
        if scores is None:
            scores = self._bm25_scores(query)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        if predicate is not None:
            ranked = [i for i in ranked if predicate(payload["metadatas"][i])]
        if drop_generated:
            ranked = [i for i in ranked if i not in gen]
        top = ranked[:top_k]
        return [
            (payload["ids"][i], payload["documents"][i], payload["metadatas"][i])
            for i in top
            if scores[i] > 0
        ]

    def _dense_scope_search(
        self, qvec: list[float], top_k: int, scope, drop_generated=False
    ) -> list[tuple[str, str, dict]]:
        """
        Dense lane restricted to a Scope. Domain / file_type constraints are
        pushed into ChromaDB; path substrings can't be (no $contains on
        metadata), so those are filtered client-side from an oversampled fetch.
        drop_generated (generated_mode drop) is passed to _dense_search.
        """
        clauses: list[dict] = []
        if scope.domains:
            clauses.append({"domain": {"$in": list(scope.domains)}})
        # file_type is only pushed down when there's no path filter — the two
        # are OR'd inside a scope, and Chroma can only AND clauses.
        if scope.file_types and not scope.path_contains:
            clauses.append({"file_type": {"$in": list(scope.file_types)}})
        where = clauses[0] if len(clauses) == 1 else ({"$and": clauses} if clauses else None)
        # Under off this is empty, so _dense_search gets exactly today's arguments.
        drop_kw = {"drop_generated": True} if drop_generated else {}

        lane_k = max(top_k // 2, 10)
        if scope.path_contains:
            rows = self._dense_search(qvec, min(200, max(top_k * 4, 80)), where=where, **drop_kw)
            return [r for r in rows if scope.matches(r[2])][:lane_k]
        return self._dense_search(qvec, lane_k, where=where, **drop_kw)

    def _hype_search(self, qvec: list[float], top_k: int) -> list[tuple[str, str, dict]]:
        """HyPE lane: match the query against hypothetical QUESTIONS, then map
        hits back to their parent chunks (deduped, first-hit order). Two cheap
        Chroma calls; empty list on any failure."""
        col = self._get_hype_collection()
        if col is None:
            return []
        try:
            res = col.query(query_embeddings=[qvec], n_results=top_k,
                            include=["metadatas"])
            parent_ids: list[str] = []
            for m in res["metadatas"][0]:
                did = str((m or {}).get("doc_id", ""))
                if did and did not in parent_ids:
                    parent_ids.append(did)
            if not parent_ids:
                return []
            got = self._get_collection().get(
                ids=parent_ids, include=["documents", "metadatas"])
            by_id = {i: (d, m) for i, d, m in
                     zip(got["ids"], got["documents"], got["metadatas"])}
            return [(pid, *by_id[pid]) for pid in parent_ids if pid in by_id]
        except Exception as e:
            log.warning("HyPE lane failed soft: %s", e)
            return []

    # ---- route (generated_mode route, spec §5.4) ----

    def _route_cfg(self, scored: bool) -> dict:
        """The retrieval.route block. A scored route refuses tantivy first: its router and
        span passes reuse bm25s's whole score vector, which tantivy never computes."""
        if scored and self.sparse_backend == "tantivy":
            raise ValueError(ROUTE_TANTIVY)
        if self.route is None:
            raise ValueError("generated_mode route needs the retrieval.route block in config.yaml")
        return self.route

    def _span_rows(self, target: dict) -> list[int]:
        """Payload indices of target's primary rows: its file, pages overlapping its span
        (any page when it names none), never a row the rule calls generated."""
        rows, _ = self._file_rows()[norm_vault_path(target["source_file"])]
        metas, gen = self._get_bm25()["metadatas"], self._generated_rows()
        a, b = target["page_start"], target["page_end"]
        return [i for i in rows if i not in gen and (a is None or _in_pages(metas[i], a, b))]

    def servable(self, target: dict) -> bool:
        """The store can serve this route target: its file is indexed and a primary row
        lies in its pages. The routed lane counts any other in targets_missing."""
        return norm_vault_path(target["source_file"]) in self._file_rows() and bool(self._span_rows(target))

    def _filter_rows(self, target: dict) -> list[int]:
        """A file filter's rows: target's _span_rows, none for a file the payload lacks."""
        known = norm_vault_path(target["source_file"]) in self._file_rows()
        return self._span_rows(target) if known else []

    def span_ids(self, target: dict) -> set[str]:
        """The ids of a file filter's rows (_filter_rows): what retrieve() keeps of every
        lane, so a later stage (search()'s neighbour pages) can be held to the same span."""
        ids = self._get_bm25()["ids"]
        return {ids[i] for i in self._filter_rows(target)}

    def _span_rank(self, qvec, scores, target: dict, k: int) -> list[tuple[str, str, dict]]:
        """target's best k primary rows for this query: the dense span (_span_dense)
        interleaved with the bm25 span (_span_sparse), deduped. _span_rows already leaves
        out every row the rule calls generated."""
        rows = self._span_rows(target)
        out: dict[str, tuple[str, str, dict]] = {}
        for pair in zip_longest(self._span_dense(qvec, target, rows, k), self._span_sparse(scores, rows, k)):
            for row in pair:
                if row is not None:
                    out.setdefault(row[0], row)
        return list(out.values())[:k]

    def _span_dense(self, qvec, target: dict, rows: list[int], k: int) -> list[tuple[str, str, dict]]:
        """rows (target's _span_rows) ranked in-process by exact cosine over their cached
        stored vectors (_span_vectors: any Chroma query per span cost seconds on the live
        store), best k."""
        if not rows:
            return []
        vecs = self._span_vectors(target)
        q = np.asarray(qvec, dtype=np.float32)
        q /= np.linalg.norm(q) or 1.0
        sims = np.stack([vecs[i] for i in rows]) @ q
        return self._payload_rows([rows[j] for j in np.argsort(-sims, kind="stable")[:k]])

    def _span_sparse(self, scores, rows: list[int], k: int) -> list[tuple[str, str, dict]]:
        """rows ranked by this query's bm25 scores, the best k scoring above 0."""
        return self._payload_rows(sorted((i for i in rows if scores[i] > 0),
                                         key=lambda i: scores[i], reverse=True)[:k])

    def _span_vectors(self, target: dict) -> dict[int, "np.ndarray"]:
        """Unit stored vectors of target's file's primary rows, by payload index. The first
        span of a file fetches them all from Chroma in one get(ids=...); every later span
        of that file ranks in-process, with no Chroma call (a filtered query cost seconds
        to tens of seconds per span on the live store)."""
        payload = self._get_bm25()
        if self._vec_cache is None or self._vec_cache[0] is not payload:
            self._vec_cache = (payload, {})
        cache = self._vec_cache[1]
        # ponytail: unbounded, one vector per primary row of every file routed to since
        # boot (worst case the whole store, ~0.6 GB float32); add an LRU by file if :8051 grows.
        rows, _ = self._file_rows()[norm_vault_path(target["source_file"])]
        gen = self._generated_rows()
        missing = [i for i in rows if i not in cache and i not in gen]
        if missing:
            ids = [payload["ids"][i] for i in missing]
            got = self._get_collection().get(ids=ids, include=["embeddings"])
            at = {cid: v for cid, v in zip(got["ids"], got["embeddings"])}
            absent = [cid for cid in ids if cid not in at]
            if absent:
                raise ValueError(f"{len(absent)} bm25 rows of {target['source_file']!r} are not "
                                 f"in Chroma (e.g. {absent[0]!r}): the indexes are out of sync; "
                                 "rebuild bm25 (python rebuild_bm25.py), then restart :8051")
            for i, cid in zip(missing, ids):
                v = np.asarray(at[cid], dtype=np.float32)
                cache[i] = v / (np.linalg.norm(v) or 1.0)
        return cache

    def _span_pages(self, target: dict, k: int) -> list[tuple[str, str, dict]]:
        """target's first k primary rows in document order (by page, then payload order,
        which is the chunk JSONLs' parse order), with no embedding and no bm25."""
        metas = self._get_bm25()["metadatas"]
        rows = sorted(self._span_rows(target), key=lambda i: (metas[i].get("page_start") or 0, i))
        return self._payload_rows(rows[:k])

    def _routed_rows(self, hits: list[str], meta_of, span, taken, max_new: int | None,
                     within: dict | None = None):
        """Steps 3-6 of the routed lane, shared with routed_for: router ids (best first) ->
        (rows, found_via by primary id, summary).

        A hit with no targets (no sidecar loaded, or none for its id) is unrouted: counted,
        contributing nothing. Targets dedupe by (file, pages) and stop at max_targets in
        router order; one the store cannot serve (servable) is counted and skipped,
        spending no slot. span(target) ranks each target's primary rows. Under a file
        filter (within), each target is clipped to it first (_within), and one outside it
        is neither ranked nor counted, so it spends no slot and no max_new seat.

        Every row comes once, its found_via naming every router id that reached it, in
        router order. A row in `taken` (another lane has it) stays; at most max_new
        others join (None: no cap)."""
        from src.generation.generator import Generator      # generator.py imports this module
        summary = dict.fromkeys(ROUTE_KEYS, 0)
        summary["router_hits"] = len(hits)
        spans: dict[tuple, tuple[dict, list[str]]] = {}     # (file, a, b) -> (target, router ids)
        missing: set[tuple] = set()
        for rid in hits:
            targets = None if self.route_pointers is None else self.route_pointers.targets(rid)
            if targets is None:
                summary["unrouted"] += 1
                continue
            summary["routed"] += 1
            for t in targets:
                if within is not None:
                    t = _within(t, within)
                    if t is None:
                        continue
                key = (norm_vault_path(t["source_file"]), t["page_start"], t["page_end"])
                if key in spans:
                    spans[key][1].append(rid)
                elif key in missing or not self.servable(t):
                    missing.add(key)
                elif len(spans) < self.route["max_targets"]:
                    spans[key] = (t, [rid])
        rows: dict[str, tuple[str, str, dict]] = {}
        via: dict[str, set[str]] = {}
        for t, rids in spans.values():
            for row in span(t):
                if row[0] not in rows:
                    if row[0] not in taken:
                        if max_new is not None and summary["new"] >= max_new:
                            continue
                        summary["new"] += 1
                    rows[row[0]] = row
                    via[row[0]] = set()
                via[row[0]].update(rids)
        rank = {rid: n for n, rid in enumerate(hits)}
        labels = {rid: Generator._source_label(meta_of(rid)) for rid in set().union(*via.values())}
        found_via = {cid: [{"id": rid, "label": labels[rid]} for rid in sorted(rids, key=rank.__getitem__)]
                     for cid, rids in via.items()}
        summary["targets"], summary["targets_missing"] = len(spans), len(missing)
        return list(rows.values()), found_via, summary

    def _routed_lane(self, qvec, scores, hype_generated: list[tuple], taken: set[str],
                     within: dict | None = None):
        """(lane rows, found_via by primary id, summary). The router hits are the RRF
        fusion (weight 1) of a dense pass over the stored-generated rows, the top router_k
        generated rows by scores, and HyPE's generated parents; each brings in its targets'
        primary spans (_routed_rows), at most max_new of them net new. within is a file
        filter's span, which every target is clipped to."""
        cfg = self.route
        k = cfg["router_k"]
        sparse = sorted((i for i in self._generated_rows() if scores[i] > 0),
                        key=lambda i: (-scores[i], i))[:k]
        hits: dict[str, list] = {}                          # id -> [rrf score, meta]
        for rows in (self._dense_search(qvec, k, where={"provenance": "generated"}),
                     self._payload_rows(sparse), hype_generated):
            for rank, (cid, _, meta) in enumerate(rows):
                hits.setdefault(cid, [0.0, meta])[0] += 1.0 / (self.rrf_k + rank)
        ranked = sorted(hits, key=lambda cid: hits[cid][0], reverse=True)
        return self._routed_rows(
            ranked, lambda rid: hits[rid][1],
            lambda t: self._span_rank(qvec, scores, t, cfg["per_target"]), taken, cfg["max_new"], within)

    def routed_for(self, query: str | None, router_ids: list[str]) -> tuple[list[RetrievedDoc], dict]:
        """(the primary docs router_ids route to, summary), through the routed lane's own
        target lookup, so with no route_pointers every id is unrouted. Each primary comes
        once; its debug["found_via"] names every router id that reached it, in router order.

        A str query ranks each span by it (one embedding, one bm25 pass). None lists each
        span's primary rows in document order with neither (the console's map view). per_target
        and max_targets bound the output; max_new does not, since no other lane shares this
        pool, so every returned primary counts as new."""
        cfg = self._route_cfg(scored=query is not None)
        if query is None:
            span = lambda t: self._span_pages(t, cfg["per_target"])
        else:
            qvec, scores = self.embedder.embed_query(query), self._bm25_scores(query)
            span = lambda t: self._span_rank(qvec, scores, t, cfg["per_target"])
        def meta_of(rid: str) -> dict:
            # A routed id the loaded pickle lacks: the sidecar or Chroma is newer than bm25.
            if rid not in self._id_rows():
                raise ValueError(f"router id {rid!r} is not in the loaded bm25 index "
                                 f"({self.bm25_index}), which predates it: rebuild it (python "
                                 "rebuild_bm25.py, or the rebuild_bm25 job), then restart :8051")
            return self._get_bm25()["metadatas"][self._id_rows()[rid]]

        rows, found_via, summary = self._routed_rows(
            list(dict.fromkeys(router_ids)), meta_of, span, frozenset(), None)
        return [RetrievedDoc(id=cid, text=text, metadata=meta, debug={"found_via": found_via[cid]})
                for cid, text, meta in rows], summary

    # ---- RRF fusion ----

    def retrieve(
        self,
        query: str,
        dense_top_k: int | None = None,
        sparse_top_k: int | None = None,
        boost_code: bool = False,
        scope=None,
        omnisearch: bool | None = None,
        hype: bool | None = None,
        lane_weights: dict[str, float] | None = None,
        lanes: list[str] | None = None,
        metadata_boost: bool | None = None,
        trace: StageTrace | None = None,
        generated_mode: str = "off",
        file_filter: dict | None = None,
    ) -> list[RetrievedDoc]:
        """
        Hybrid retrieve. The optional args are per-query overrides (presets /
        API top_k field) and never mutate the configured defaults.

        boost_code additionally opens a CODE LANE: an extra dense pass filtered
        to code_file_types plus a filtered sparse pass join the RRF fusion.
        the author's scripts/notebooks are ~2.4% of the corpus, so without a
        reserved lane the candidate pool fills up with lecture PDFs that
        mention the same keywords, and the reranker never even sees the code.

        scope (a retrieval.scope.Scope) opens an analogous SCOPE LANE when the
        query names a domain or content type ("my statistics homework", "in
        the tech books"): chunks matching the named domain/path/file-type get
        guaranteed seats in the candidate pool. Soft routing — fusion and the
        reranker still decide the final order.

        omnisearch adds a LIVE-VAULT LANE via Obsidian's Omnisearch HTTP API
        (notes edited since the last ingest, filename/heading-weighted BM25).
        None = follow the client's configured default; True/False override per
        query. Fail-soft: Obsidian closed -> empty lane, never an error.

        hype adds the HYPOTHETICAL-QUESTION LANE (build_hype.py): the query is
        matched against questions generated per chunk, and hits map back to
        their parent chunks. None = follow retrieval.hype.enabled. Fail-soft
        when the collection has not been built.

        lane_weights reweights the RRF fusion per lane, merged over the
        configured defaults. A weight of 0 does NOT remove the lane: its
        candidates still reach the reranker, they just contribute nothing to
        the fused ORDER. To drop a lane entirely, turn the lane off.

        lanes restricts which lanes RUN. None keeps today's behaviour; a list
        runs only those lanes — one outside it is not executed at all, and no
        query embedding is computed when no dense, dense_code, dense_scope,
        hype or routed lane is left. It only ever RESTRICTS: dense_code/sparse_code
        still need boost_code, the scope lanes a detected scope, omnisearch
        and hype their own switch, routed generated_mode route, so naming a lane
        that has no trigger runs nothing. An empty list or an unknown lane name
        raises ValueError.

        metadata_boost overrides retrieval.metadata_boost for this call: None
        follows the config, True/False force the course/domain/tag boost on
        or off (boost_code's has_code boost is a separate switch).

        trace (a StageTrace) collects this call's per-stage timings — embed,
        one "lane.<name>" per lane that ran, fuse, boost; bm25 and route under
        route — each lane's candidate count, and trace.route. None = nobody is
        listening.

        generated_mode (retrieval.generated_mode, spec §5.1): off = generated
        chunks compete as evidence, and every lane helper is called exactly as
        before. drop = each lane leaves them out by the provenance rule: the
        dense lanes by their stored key from an over-fetch, the sparse lanes
        before top_k (so both refill; tantivy within its pool), HyPE parents and
        omnisearch rows after the fact (no refill). route (spec §5.4) = drop,
        plus the ROUTED LANE: generated hits (dense and bm25 passes over the
        generated rows, HyPE's generated parents) bring the primary pages they
        cite (route_pointers) into the pool, each with debug["found_via"]; bm25
        is scored once for every sparse pass, and trace.route sums what routed.
        route needs the retrieval.route block and sparse_backend bm25s.

        file_filter (a route target, {source_file, page_start, page_end}; spec §5.7)
        restricts every lane to that span's primary rows, _span_rows, in every mode: the
        dense lane ranks them by exact cosine over their stored vectors and the sparse
        lane by bm25, as a routed span is ranked; the routed lane's targets are clipped
        to it; every other lane keeps only them, and omnisearch (whose live excerpts are
        no stored rows) does not run. A file the payload lacks gives no candidates. It
        needs sparse_backend bm25s.
        """
        mode = check_mode(generated_mode, "generated_mode")
        route = mode == "route"
        if route:
            self._route_cfg(scored=True)          # refuses before any lane runs
        if file_filter is not None and self.sparse_backend == "tantivy":
            raise ValueError(FILTER_TANTIVY)
        drop = mode in ("drop", "route")          # route is drop plus the routed lane
        rule = self._drop_rule() if drop else None
        # Under off this is empty, so every helper gets exactly today's arguments.
        drop_kw = {"drop_generated": True} if drop else {}
        tr = trace if trace is not None else StageTrace()
        dk = dense_top_k or self.dense_top_k
        sk = sparse_top_k or self.sparse_top_k
        chosen = _clean_lane_set(lanes, "lanes")
        want = (lambda n: True) if chosen is None else chosen.__contains__

        # Which conditional lanes have a trigger this call. Decided up front
        # because the query embedding is only worth computing for a lane that
        # is both triggered and selected.
        code_on = bool(boost_code and self.code_file_types)
        omni_on = (self.omnisearch is not None and file_filter is None and
                   (self.omnisearch.enabled if omnisearch is None else omnisearch))
        hype_on = self.hype_enabled if hype is None else hype
        routed_on = route and want("routed")
        qvec = None
        if (want("dense") or (code_on and want("dense_code"))
                or (scope and want("dense_scope")) or (hype_on and want("hype")) or routed_on):
            with tr.span("embed"):
                qvec = self.embedder.embed_query(query)   # embed ONCE for every dense lane
        # The file filter's span: its payload rows and their ids (none for an unknown file).
        span_rows: list[int] = []
        span: set[str] | None = None
        if file_filter is not None:
            span_rows = self._filter_rows(file_filter)
            span = {self._get_bm25()["ids"][i] for i in span_rows}
        # The routed lane's router and spans reuse one bm25 pass, so it is scored ONCE,
        # and the sparse lanes take it too; so does a file filter's sparse span. Only under
        # route or a file filter: otherwise off and drop call the sparse helpers as before.
        sparse_kw = drop_kw
        if routed_on or (span is not None and want("sparse")):
            with tr.span("bm25"):
                scores = self._bm25_scores(query)
            sparse_kw = dict(drop_kw, scores=scores)

        # (name, ranked list) lanes; all fused with the same RRF formula. Each
        # one that `lanes` allows runs under its own "lane.<name>" span.
        active_lanes: list[tuple[str, list[tuple[str, str, dict]]]] = []
        if want("dense"):
            with tr.span("lane.dense"):
                active_lanes.append(("dense", self._dense_search(qvec, dk, **drop_kw) if span is None
                                     else self._span_dense(qvec, file_filter, span_rows, dk)))
        if want("sparse"):
            with tr.span("lane.sparse"):
                active_lanes.append(("sparse", self._sparse_search(query, sk, **sparse_kw) if span is None
                                     else self._span_sparse(scores, span_rows, sk)))
        if code_on:
            code_where = {"file_type": {"$in": list(self.code_file_types)}}
            allowed = set(self.code_file_types)
            if want("dense_code"):
                with tr.span("lane.dense_code"):
                    active_lanes.append(
                        ("dense_code", self._dense_search(
                            qvec, max(dk // 2, 10), where=code_where, **drop_kw))
                    )
            if want("sparse_code"):
                with tr.span("lane.sparse_code"):
                    active_lanes.append(
                        ("sparse_code",
                         self._sparse_search(
                             query, max(sk // 2, 10),
                             predicate=lambda m: str(m.get("file_type", "")).lower() in allowed,
                             **sparse_kw,
                         ))
                    )
        if scope and want("dense_scope"):
            with tr.span("lane.dense_scope"):
                active_lanes.append(("dense_scope", self._dense_scope_search(qvec, dk, scope, **drop_kw)))
        if scope and want("sparse_scope"):
            with tr.span("lane.sparse_scope"):
                active_lanes.append(
                    ("sparse_scope",
                     self._sparse_search(query, max(sk // 2, 10), predicate=scope.matches, **sparse_kw))
                )
        if omni_on and want("omnisearch"):
            with tr.span("lane.omnisearch"):
                rows = self.omnisearch.lane(query)
                if drop:      # path rule only: the rows carry source_file and file_type
                    rows = [r for r in rows if provenance_of(r[2], rule) != "generated"]
                active_lanes.append(("omnisearch", rows))
        # HyPE's generated parents are router hits, but only when the hype lane runs.
        hype_generated: list[tuple[str, str, dict]] = []
        if hype_on and want("hype"):
            with tr.span("lane.hype"):
                rows = self._hype_search(qvec, self.hype_top_k)
                # Outside _hype_search's fail-soft try, so a provenance error raises.
                if drop:
                    is_gen = [provenance_of(m, rule) == "generated" for _, _, m in rows]
                    hype_generated = [r for r, g in zip(rows, is_gen) if g]
                    rows = [r for r, g in zip(rows, is_gen) if not g]
                active_lanes.append(("hype", rows))
        found_via: dict[str, list[dict]] = {}
        if routed_on:
            taken = {cid for _, results in active_lanes for cid, _, _ in results}
            with tr.span("lane.routed"), tr.span("route"):
                # Under a file filter only: without one the lane is called as before.
                rows, found_via, summary = self._routed_lane(
                    qvec, scores, hype_generated, taken, **({} if span is None else {"within": file_filter}))
            active_lanes.append(("routed", rows))
            for key, n in summary.items():
                tr.route[key] += n
        if span is not None:          # every other lane keeps the file filter's rows only
            active_lanes = [(name, [r for r in results if r[0] in span]) for name, results in active_lanes]
            found_via = {cid: via for cid, via in found_via.items() if cid in span}
        # Accumulates, like the span timings: a decomposed search runs one
        # retrieve() per branch on the same trace.
        for lane_name, results in active_lanes:
            tr.lanes[lane_name] = tr.lanes.get(lane_name, 0) + len(results)

        # Per-call weights override the configured ones LANE BY LANE, so
        # {"sparse": 1.5} changes sparse and leaves the other lanes alone.
        weights = dict(self.lane_weights)
        weights.update(_clean_lane_weights(lane_weights, "lane_weights"))

        with tr.span("fuse"):
            fused: dict[str, RetrievedDoc] = {}
            for lane_name, results in active_lanes:
                # Unset = 1.0, which is the plain RRF term. So an install with no
                # lane_weights fuses exactly as it did before weighting existed.
                w = weights.get(lane_name, 1.0)
                for rank, (cid, text, meta) in enumerate(results):
                    doc = fused.setdefault(cid, RetrievedDoc(id=cid, text=text, metadata=meta))
                    if lane_name == "dense":
                        doc.dense_rank = rank
                    elif lane_name == "sparse":
                        doc.sparse_rank = rank
                    else:
                        doc.debug[lane_name + "_rank"] = rank
                    doc.score += w / (self.rrf_k + rank)
            for cid, via in found_via.items():
                fused[cid].debug["found_via"] = via

            ranked = sorted(fused.values(), key=lambda d: d.score, reverse=True)

        boost_on = self.metadata_boost if metadata_boost is None else bool(metadata_boost)
        if boost_on or boost_code:
            with tr.span("boost"):
                ranked = self._apply_metadata_boost(
                    query, ranked, boost_code=boost_code, metadata_boost=boost_on)

        log.info(
            "retrieve(%r): %s fused=%d",
            query[:48],
            " ".join(f"{name}={len(res)}" for name, res in active_lanes),
            len(ranked),
        )
        return ranked

    def _apply_metadata_boost(
        self, query: str, docs: list[RetrievedDoc], boost_code: bool = False,
        metadata_boost: bool | None = None,
    ) -> list[RetrievedDoc]:
        """
        Light heuristic: if the query names a course/domain keyword that matches
        a doc's metadata, nudge it up. Cheap precision win for queries like
        'explain ARIMA in time series' or 'my NLP capstone'.

        metadata_boost: whether that course/domain/tag half runs. None follows
        the configured retrieval.metadata_boost; the per-call override from
        retrieve() passes its decision here.

        boost_code: for code-intent queries, additionally nudge chunks the
        loaders flagged with has_code (notebooks, scripts, code-bearing PDF
        pages) so the ~2.4% code minority survives fusion against prose.
        """
        q = query.lower()
        use = self.metadata_boost if metadata_boost is None else metadata_boost
        # Pull candidate course/domain tokens from query
        boosted = 0
        code_boosted = 0
        for doc in docs:
            m = doc.metadata
            if use:
                # Read the fields the loaders/parser actually write. (Older code read
                # "course"/"canonical_course", which were never set — so the course
                # half of the boost never fired. "domain" was the only live signal.)
                course_name = str(m.get("course_name") or m.get("course") or "").lower()
                course_code = str(m.get("course_code") or "").lower()
                domain = str(m.get("domain", "")).lower()
                # User tags reach this function in TWO shapes: a real list
                # from the BM25 payload (which stores metadata verbatim) and a
                # joined string from Chroma (whose values must be scalars, so
                # Embedder._clean_meta flattens lists with ", ".join).
                #
                # Split on ", " — the exact inverse of that join, and the same
                # delimiter decode_canvas_edges uses. Splitting on "," alone
                # also cut a tag that merely CONTAINED a comma into two, which
                # is the canvas edge-label bug in miniature. Tags are
                # normalised comma-free at the point they are stamped (see
                # apply_forced_meta), so this round-trip is now lossless.
                raw_tags = m.get("tags") or ""
                if isinstance(raw_tags, str):
                    user_tags = [t.strip().lower() for t in raw_tags.split(", ")]
                else:
                    user_tags = [str(t).strip().lower() for t in raw_tags]
                for tag in (course_name, course_code, domain, *user_tags):
                    if (tag and tag not in ("unknown", "general")
                            and len(tag) >= 3 and tag in q):
                        doc.score *= 1.15
                        doc.debug["metadata_boost"] = tag
                        boosted += 1
                        break
            if boost_code and m.get("has_code"):
                doc.score *= 1.2
                doc.debug["code_boost"] = True
                code_boosted += 1
        if boosted or code_boosted:
            docs = sorted(docs, key=lambda d: d.score, reverse=True)
            log.info("  metadata boost: %d course/domain, %d has_code", boosted, code_boosted)
        return docs
