"""
retriever.py — Hybrid search with weighted Reciprocal Rank Fusion.

Pipeline (every lane is a ranked list of (id, text, metadata)):
    query -> dense        ChromaDB, always
          -> sparse       BM25, always
          -> dense_code   + sparse_code    when boost_code (the code preset)
          -> dense_scope  + sparse_scope   when a Scope was detected
          -> omnisearch   live vault over HTTP, per config or per call
          -> hype         hypothetical-question collection, per config or call
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
from pathlib import Path
from typing import Any

from src.embeddings.embedder import Embedder, _tokenize
from src.embeddings.sidecar import (
    check_collection, sidecar_label, sidecar_path, unstamped_warning)
from src.utils.chroma_client import persistent_client
from src.utils.config_loader import Config
from src.utils.logger import get_logger
from src.utils.timing import StageTrace

log = get_logger(__name__)


# Every lane that can reach RRF fusion. ONE list, so config, the API schema
# and the tests cannot drift from what retrieve() actually builds.
#
# There are EIGHT, not two: the code and scope lanes are separate filtered
# passes, and omnisearch/hype are their own retrievers. Weighting them
# individually is therefore a far finer instrument than "dense vs sparse".
LANES = ("dense", "sparse", "dense_code", "sparse_code",
         "dense_scope", "sparse_scope", "omnisearch", "hype")


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
    ):
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
        )
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

    # ---- individual searches ----

    def _dense_search(
        self, qvec: list[float], top_k: int, where: dict | None = None
    ) -> list[tuple[str, str, dict]]:
        kwargs: dict[str, Any] = {
            "query_embeddings": [qvec],
            "n_results": top_k,
            "include": ["documents", "metadatas"],
        }
        if where:
            kwargs["where"] = where
        res = self._get_collection().query(**kwargs)
        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        return list(zip(ids, docs, metas))

    def _sparse_search(
        self, query: str, top_k: int, predicate=None
    ) -> list[tuple[str, str, dict]]:
        """predicate(meta) -> bool restricts the ranked list (code/scope lanes)."""
        payload = self._get_bm25()
        bm25 = payload["bm25"]
        scores = bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        if predicate is not None:
            ranked = [i for i in ranked if predicate(payload["metadatas"][i])]
        top = ranked[:top_k]
        return [
            (payload["ids"][i], payload["documents"][i], payload["metadatas"][i])
            for i in top
            if scores[i] > 0
        ]

    def _dense_scope_search(
        self, qvec: list[float], top_k: int, scope
    ) -> list[tuple[str, str, dict]]:
        """
        Dense lane restricted to a Scope. Domain / file_type constraints are
        pushed into ChromaDB; path substrings can't be (no $contains on
        metadata), so those are filtered client-side from an oversampled fetch.
        """
        clauses: list[dict] = []
        if scope.domains:
            clauses.append({"domain": {"$in": list(scope.domains)}})
        # file_type is only pushed down when there's no path filter — the two
        # are OR'd inside a scope, and Chroma can only AND clauses.
        if scope.file_types and not scope.path_contains:
            clauses.append({"file_type": {"$in": list(scope.file_types)}})
        where = clauses[0] if len(clauses) == 1 else ({"$and": clauses} if clauses else None)

        lane_k = max(top_k // 2, 10)
        if scope.path_contains:
            rows = self._dense_search(qvec, min(200, max(top_k * 4, 80)), where=where)
            return [r for r in rows if scope.matches(r[2])][:lane_k]
        return self._dense_search(qvec, lane_k, where=where)

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
        query embedding is computed when no dense, dense_code, dense_scope or
        hype lane is left. It only ever RESTRICTS: dense_code/sparse_code
        still need boost_code, the scope lanes a detected scope, omnisearch
        and hype their own switch, so naming a lane that has no trigger runs
        nothing. An empty list or an unknown lane name raises ValueError.

        metadata_boost overrides retrieval.metadata_boost for this call: None
        follows the config, True/False force the course/domain/tag boost on
        or off (boost_code's has_code boost is a separate switch).

        trace (a StageTrace) collects this call's per-stage timings — embed,
        one "lane.<name>" per lane that ran, fuse, boost — and each lane's
        candidate count. None = nobody is listening.
        """
        tr = trace if trace is not None else StageTrace()
        dk = dense_top_k or self.dense_top_k
        sk = sparse_top_k or self.sparse_top_k
        chosen = _clean_lane_set(lanes, "lanes")
        want = (lambda n: True) if chosen is None else chosen.__contains__

        # Which conditional lanes have a trigger this call. Decided up front
        # because the query embedding is only worth computing for a lane that
        # is both triggered and selected.
        code_on = bool(boost_code and self.code_file_types)
        omni_on = (self.omnisearch is not None and
                   (self.omnisearch.enabled if omnisearch is None else omnisearch))
        hype_on = self.hype_enabled if hype is None else hype
        qvec = None
        if (want("dense") or (code_on and want("dense_code"))
                or (scope and want("dense_scope")) or (hype_on and want("hype"))):
            with tr.span("embed"):
                qvec = self.embedder.embed_query(query)   # embed ONCE for every dense lane

        # (name, ranked list) lanes; all fused with the same RRF formula. Each
        # one that `lanes` allows runs under its own "lane.<name>" span.
        active_lanes: list[tuple[str, list[tuple[str, str, dict]]]] = []
        if want("dense"):
            with tr.span("lane.dense"):
                active_lanes.append(("dense", self._dense_search(qvec, dk)))
        if want("sparse"):
            with tr.span("lane.sparse"):
                active_lanes.append(("sparse", self._sparse_search(query, sk)))
        if code_on:
            code_where = {"file_type": {"$in": list(self.code_file_types)}}
            allowed = set(self.code_file_types)
            if want("dense_code"):
                with tr.span("lane.dense_code"):
                    active_lanes.append(
                        ("dense_code", self._dense_search(qvec, max(dk // 2, 10), where=code_where))
                    )
            if want("sparse_code"):
                with tr.span("lane.sparse_code"):
                    active_lanes.append(
                        ("sparse_code",
                         self._sparse_search(
                             query, max(sk // 2, 10),
                             predicate=lambda m: str(m.get("file_type", "")).lower() in allowed,
                         ))
                    )
        if scope and want("dense_scope"):
            with tr.span("lane.dense_scope"):
                active_lanes.append(("dense_scope", self._dense_scope_search(qvec, dk, scope)))
        if scope and want("sparse_scope"):
            with tr.span("lane.sparse_scope"):
                active_lanes.append(
                    ("sparse_scope",
                     self._sparse_search(query, max(sk // 2, 10), predicate=scope.matches))
                )
        if omni_on and want("omnisearch"):
            with tr.span("lane.omnisearch"):
                active_lanes.append(("omnisearch", self.omnisearch.lane(query)))
        if hype_on and want("hype"):
            with tr.span("lane.hype"):
                active_lanes.append(("hype", self._hype_search(qvec, self.hype_top_k)))
        for lane_name, results in active_lanes:
            tr.lanes[lane_name] = len(results)

        # Per-call weights override the configured ones LANE BY LANE, so
        # {"sparse": 1.5} changes sparse and leaves the other seven alone.
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
