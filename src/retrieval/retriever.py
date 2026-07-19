"""
retriever.py — Hybrid search with Reciprocal Rank Fusion.

Pipeline:
    query -> [dense search over ChromaDB]  ┐
          -> [sparse BM25 search]           ┴-> RRF fuse -> ranked list

RRF (Reciprocal Rank Fusion) merges two ranked lists without needing the
scores to be on the same scale. For a doc appearing at rank r in a list:
    contribution = 1 / (rrf_k + r)
Summed across lists. Robust, parameter-light, the standard hybrid-fusion move.

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
from src.utils.config_loader import Config
from src.utils.logger import get_logger

log = get_logger(__name__)


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
        """Human-readable citation stub: 'Course (date)' or filename."""
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

        self._collection = None
        self._bm25_payload = None
        self._hype_col = None
        self._hype_missing_logged = False

    @classmethod
    def from_config(cls, cfg: Config, embedder: Embedder) -> "HybridRetriever":
        return cls(
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
        )

    # ---- lazy loaders ----

    def _get_collection(self):
        if self._collection is None:
            import chromadb
            client = chromadb.PersistentClient(path=str(self.chroma_dir))
            self._collection = client.get_collection(self.collection_name)
        return self._collection

    def _get_bm25(self):
        if self._bm25_payload is None:
            with open(self.bm25_index, "rb") as f:
                self._bm25_payload = pickle.load(f)
        return self._bm25_payload

    def _get_hype_collection(self):
        """The HyPE question collection, or None (never raises — the lane is
        optional and build_hype.py may simply not have been run yet)."""
        if self._hype_col is None:
            import chromadb
            client = chromadb.PersistentClient(path=str(self.chroma_dir))
            try:
                self._hype_col = client.get_collection(self.hype_collection)
            except Exception:
                if not self._hype_missing_logged:
                    log.info("HyPE lane requested but collection %r doesn't "
                             "exist — run build_hype.py first (lane skipped).",
                             self.hype_collection)
                    self._hype_missing_logged = True
                self._hype_col = False           # sentinel: checked, absent
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
        """
        dk = dense_top_k or self.dense_top_k
        sk = sparse_top_k or self.sparse_top_k
        qvec = self.embedder.embed_query(query)   # embed ONCE for every dense lane

        # (name, ranked list) lanes; all fused with the same RRF formula.
        lanes: list[tuple[str, list[tuple[str, str, dict]]]] = [
            ("dense", self._dense_search(qvec, dk)),
            ("sparse", self._sparse_search(query, sk)),
        ]
        if boost_code and self.code_file_types:
            code_where = {"file_type": {"$in": list(self.code_file_types)}}
            allowed = set(self.code_file_types)
            lanes.append(
                ("dense_code", self._dense_search(qvec, max(dk // 2, 10), where=code_where))
            )
            lanes.append(
                ("sparse_code",
                 self._sparse_search(
                     query, max(sk // 2, 10),
                     predicate=lambda m: str(m.get("file_type", "")).lower() in allowed,
                 ))
            )
        if scope:
            lanes.append(("dense_scope", self._dense_scope_search(qvec, dk, scope)))
            lanes.append(
                ("sparse_scope",
                 self._sparse_search(query, max(sk // 2, 10), predicate=scope.matches))
            )
        omni_on = (self.omnisearch is not None and
                   (self.omnisearch.enabled if omnisearch is None else omnisearch))
        if omni_on:
            lanes.append(("omnisearch", self.omnisearch.lane(query)))
        hype_on = self.hype_enabled if hype is None else hype
        if hype_on:
            lanes.append(("hype", self._hype_search(qvec, self.hype_top_k)))

        fused: dict[str, RetrievedDoc] = {}
        for lane_name, results in lanes:
            for rank, (cid, text, meta) in enumerate(results):
                doc = fused.setdefault(cid, RetrievedDoc(id=cid, text=text, metadata=meta))
                if lane_name == "dense":
                    doc.dense_rank = rank
                elif lane_name == "sparse":
                    doc.sparse_rank = rank
                else:
                    doc.debug[lane_name + "_rank"] = rank
                doc.score += 1.0 / (self.rrf_k + rank)

        ranked = sorted(fused.values(), key=lambda d: d.score, reverse=True)

        if self.metadata_boost or boost_code:
            ranked = self._apply_metadata_boost(query, ranked, boost_code=boost_code)

        log.info(
            "retrieve(%r): %s fused=%d",
            query[:48],
            " ".join(f"{name}={len(res)}" for name, res in lanes),
            len(ranked),
        )
        return ranked

    def _apply_metadata_boost(
        self, query: str, docs: list[RetrievedDoc], boost_code: bool = False
    ) -> list[RetrievedDoc]:
        """
        Light heuristic: if the query names a course/domain keyword that matches
        a doc's metadata, nudge it up. Cheap precision win for queries like
        'explain ARIMA in time series' or 'my NLP capstone'.

        boost_code: for code-intent queries, additionally nudge chunks the
        loaders flagged with has_code (notebooks, scripts, code-bearing PDF
        pages) so the ~2.4% code minority survives fusion against prose.
        """
        q = query.lower()
        # Pull candidate course/domain tokens from query
        boosted = 0
        code_boosted = 0
        for doc in docs:
            m = doc.metadata
            if self.metadata_boost:
                # Read the fields the loaders/parser actually write. (Older code read
                # "course"/"canonical_course", which were never set — so the course
                # half of the boost never fired. "domain" was the only live signal.)
                course_name = str(m.get("course_name") or m.get("course") or "").lower()
                course_code = str(m.get("course_code") or "").lower()
                domain = str(m.get("domain", "")).lower()
                # user tags (console tag lane): list in JSONL, comma-joined
                # string in the Chroma/BM25 metadata copies — accept both.
                raw_tags = m.get("tags") or ""
                if isinstance(raw_tags, str):
                    user_tags = [t.strip().lower() for t in raw_tags.split(",")]
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
