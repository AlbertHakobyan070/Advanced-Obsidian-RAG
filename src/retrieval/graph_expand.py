"""
graph_expand.py — Graph RAG traversal over the canvas edge metadata.

WHAT THIS IS
  A SEPARATE retrieval mode, not a change to the main query path. It seeds
  from a canvas-restricted search (or from chunk ids a previous result
  produced), then walks the hand-drawn edges Obsidian canvases already carry,
  and returns the nodes it reached PLUS the tree it walked — so a caller can
  show the traversal instead of a flat list.

WHY IT IS SEPARATE
  Nothing here runs during /query or /search. A depth knob on the main path
  would make every ordinary query pay for graph expansion it never asked for;
  here the caller asked for a traversal, so the cost is theirs to spend.

  A previous measurement is worth keeping in mind: a plain query already
  returned 20 canvas chunks out of 32 hits. Canvas content is short, dense
  and highly connected, so it competes perfectly well in ordinary retrieval.
  This mode exists for DEPTH — following an edge the ranker would never have
  surfaced — not to get canvas content into results at all.

THE TWO THINGS THAT WILL BITE
  1. CANVAS GRAPHS CONTAIN CYCLES. A walk without a visited set does not
     terminate. `visited` below is the single most load-bearing line here.
  2. A NODE ID IS ONLY UNIQUE WITHIN ITS CANVAS FILE. Identity is therefore
     (source_file, canvas_node_id), never the node id alone, and every hop
     query is scoped to the parent's source_file.

CAPS ARE REPORTED, NOT SILENT
  depth / max_per_hop / max_total each stop the walk. When one bites, the
  response says which (`truncated_at`) — a traversal that quietly returned
  less would look like a graph that is smaller than it is.
"""
from __future__ import annotations

from typing import Any

from src.ingestion.canvas_loader import decode_canvas_edges
from src.retrieval.retriever import RetrievedDoc
from src.utils.config_loader import Config
from src.utils.logger import get_logger

log = get_logger(__name__)


def _node_key(meta: dict) -> tuple[str, str]:
    """Graph identity. A canvas node id is unique only inside its own file."""
    return (str(meta.get("source_file") or ""),
            str(meta.get("canvas_node_id") or ""))


def _edges_of(meta: dict) -> tuple[list[dict], bool]:
    """This chunk's outgoing/incoming edges, and whether they carry labels.

    `canvas_edges` is the aligned list and the only one a label can be read
    from. Chunks indexed before that key existed still have
    `canvas_neighbors`, so the walk degrades to UNLABELLED edges rather than
    refusing to traverse — and says so, so nobody reads a missing label as
    "this edge has no label".
    """
    aligned = meta.get("canvas_edges")
    if aligned:
        return decode_canvas_edges(aligned), True
    raw = meta.get("canvas_neighbors")
    if not raw:
        return [], True
    if isinstance(raw, (list, tuple)):
        neighbors = [str(n) for n in raw if str(n).strip()]
    else:
        neighbors = [n for n in str(raw).split(", ") if n.strip()]
    return ([{"neighbor": n, "direction": "--", "label": ""} for n in neighbors],
            False)


class GraphExpander:
    """Walks canvas edges. Owns no models — it borrows the retriever's
    collection and embedder, and the pipeline's reranker for seed ordering."""

    def __init__(
        self,
        retriever,
        reranker=None,
        enabled: bool = True,
        max_depth: int = 2,
        max_per_hop: int = 8,
        max_total: int = 40,
        seed_top_k: int = 5,
        file_type: str = "canvas",
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.enabled = enabled
        self.max_depth = max_depth
        self.max_per_hop = max_per_hop
        self.max_total = max_total
        self.seed_top_k = seed_top_k
        self.file_type = file_type

    @classmethod
    def from_config(cls, cfg: Config, retriever, reranker=None) -> "GraphExpander":
        return cls(
            retriever=retriever,
            reranker=reranker,
            enabled=bool(cfg.get("graph.enabled", True)),
            max_depth=int(cfg.get("graph.max_depth", 2)),
            max_per_hop=int(cfg.get("graph.max_per_hop", 8)),
            max_total=int(cfg.get("graph.max_total", 40)),
            seed_top_k=int(cfg.get("graph.seed_top_k", 5)),
            file_type=str(cfg.get("graph.file_type", "canvas")),
        )

    # ---- seeding ----

    def seed_from_query(self, query: str, seed_top_k: int | None = None,
                        rerank: str | None = None) -> list[RetrievedDoc]:
        """Hybrid dense+sparse restricted to the graph's file_type, RRF-fused.

        Deliberately plain: no HyDE, no scope routing, no metadata boost. The
        seed set is the thing a human is about to steer, so it should be
        explainable — "the best canvas chunks for these words" — rather than
        the product of every heuristic the main path applies.
        """
        k = seed_top_k or self.seed_top_k
        pool = max(k * 4, 20)
        qvec = self.retriever.embedder.embed_query(query)
        ftype = self.file_type
        lanes = [
            self.retriever._dense_search(qvec, pool, where={"file_type": ftype}),
            self.retriever._sparse_search(
                query, pool,
                predicate=lambda m: str(m.get("file_type", "")).lower() == ftype),
        ]
        fused: dict[str, RetrievedDoc] = {}
        for results in lanes:
            for rank, (cid, text, meta) in enumerate(results):
                doc = fused.setdefault(
                    cid, RetrievedDoc(id=cid, text=text, metadata=meta))
                doc.score += 1.0 / (self.retriever.rrf_k + rank)
        ranked = sorted(fused.values(), key=lambda d: d.score, reverse=True)
        if self.reranker is not None:
            ranked = self.reranker.rerank(query, ranked, top_k=k, mode=rerank)
        log.info("graph seeds(%r): %d candidate(s) -> %d seed(s)",
                 query[:48], len(fused), min(k, len(ranked)))
        return ranked[:k]

    def seed_from_ids(self, ids: list[str]) -> tuple[list[RetrievedDoc], list[str]]:
        """Explicit seeds from a previous result. Returns (docs, missing_ids)
        — an id that is not in the index is reported, never dropped."""
        by_id = {d.id: d for d in self._fetch_ids(ids)}
        # Preserve the caller's order; it is the ranking they already saw.
        ordered = [by_id[i] for i in dict.fromkeys(ids) if i in by_id]
        return ordered, [i for i in dict.fromkeys(ids) if i not in by_id]

    # ---- the walk ----

    def expand(
        self,
        seeds: list[RetrievedDoc],
        depth: int | None = None,
        max_per_hop: int | None = None,
        max_total: int | None = None,
    ) -> dict[str, Any]:
        """Breadth-first walk from `seeds` along canvas edges.

        Returns {"nodes", "tree", "cross_edges", "stats"}. `tree` is a flat
        parent->child edge list carrying each hop's label and direction;
        `cross_edges` are edges onto nodes the walk had ALREADY reached —
        reported separately because dropping them would draw a cyclic graph
        as if it were a tree.
        """
        d_cap = self.max_depth if depth is None else int(depth)
        hop_cap = self.max_per_hop if max_per_hop is None else int(max_per_hop)
        total_cap = self.max_total if max_total is None else int(max_total)

        nodes: dict[str, dict] = {}
        tree: list[dict] = []
        cross: list[dict] = []
        # THE lines that make this terminate. Canvas graphs are cyclic by
        # nature (A -> B -> C -> A is an ordinary knowledge map), so identity
        # is tracked per canvas NODE, not per chunk: a node split across
        # several chunks must not be walked once per part.
        visited: set[tuple[str, str]] = set()
        by_key: dict[tuple[str, str], list[str]] = {}
        dangling = 0
        labelled = True
        caps_hit: list[str] = []

        def cap(name: str) -> None:
            if name not in caps_hit:
                caps_hit.append(name)

        def admit(doc: RetrievedDoc, hop: int) -> None:
            nodes[doc.id] = self._node_out(doc, hop)
            by_key.setdefault(_node_key(doc.metadata), []).append(doc.id)

        frontier: list[RetrievedDoc] = []
        for doc in seeds:
            key = _node_key(doc.metadata)
            if key in visited:
                by_key.setdefault(key, []).append(doc.id)
                nodes.setdefault(doc.id, self._node_out(doc, 0))
                continue
            visited.add(key)
            admit(doc, 0)
            frontier.append(doc)
        if len(nodes) >= total_cap:
            cap("max_total")

        for hop in range(1, d_cap + 1):
            if "max_total" in caps_hit or not frontier:
                break
            # Gather this hop's wanted (source_file -> node ids) and remember
            # which parent + edge asked for each, so the tree is built from
            # the same pass rather than a second walk.
            wanted: dict[str, set[str]] = {}
            asked: dict[tuple[str, str], list[dict]] = {}
            for parent in frontier:
                sf = str(parent.metadata.get("source_file") or "")
                edges, ok = _edges_of(parent.metadata)
                labelled = labelled and ok
                taken = 0
                for edge in edges:
                    nid = edge["neighbor"]
                    if not nid:
                        continue
                    if (sf, nid) in visited:
                        # A cycle or a diamond. Record it so the caller can
                        # SEE the loop; do not re-walk it.
                        for cid in by_key.get((sf, nid), []):
                            cross.append({"parent": parent.id, "child": cid,
                                          "label": edge["label"],
                                          "direction": edge["direction"],
                                          "depth": hop})
                        continue
                    if taken >= hop_cap:
                        cap("max_per_hop")
                        break
                    wanted.setdefault(sf, set()).add(nid)
                    asked.setdefault((sf, nid), []).append(
                        {"parent": parent.id, "label": edge["label"],
                         "direction": edge["direction"]})
                    taken += 1
            if not wanted:
                break

            reached: list[RetrievedDoc] = []
            for sf, ids in wanted.items():
                rows = self._fetch_neighbors(sf, sorted(ids))
                seen = {str(d.metadata.get("canvas_node_id") or "") for d in rows}
                # An edge can point at a node that never became a chunk: file
                # and group nodes never do, and a text node under
                # canvas.min_chunk_size does not either. Normal, not an error.
                dangling += len(ids - seen)
                reached.extend(rows)

            next_frontier: list[RetrievedDoc] = []
            for doc in reached:
                key = _node_key(doc.metadata)
                if key not in visited and len(nodes) >= total_cap:
                    cap("max_total")
                    break
                for link in asked.get(key, []):
                    tree.append({"parent": link["parent"], "child": doc.id,
                                 "label": link["label"],
                                 "direction": link["direction"], "depth": hop})
                if key in visited:
                    # Another part of a node already admitted this hop.
                    nodes.setdefault(doc.id, self._node_out(doc, hop))
                    by_key.setdefault(key, []).append(doc.id)
                    continue
                visited.add(key)
                admit(doc, hop)
                next_frontier.append(doc)
            frontier = next_frontier
        else:
            # Ran every allowed hop and still had somewhere to go: DEPTH is
            # what stopped the walk.
            if frontier and d_cap > 0:
                cap("depth")

        return {
            "nodes": list(nodes.values()),
            "tree": tree,
            "cross_edges": cross,
            "stats": {
                "seeds": len(seeds),
                "nodes": len(nodes),
                "edges": len(tree),
                "cross_edges": len(cross),
                "hops": d_cap,
                "visited": len(visited),
                "dangling_edges": dangling,
                "edges_labelled": labelled,
                # The FIRST cap that bit, plus every cap that bit at all —
                # one scalar would hide the others.
                "truncated_at": caps_hit[0] if caps_hit else None,
                "caps_hit": caps_hit,
                "caps": {"depth": d_cap, "max_per_hop": hop_cap,
                         "max_total": total_cap},
            },
        }

    # ---- chunk store access ----

    def _node_out(self, doc: RetrievedDoc, depth: int) -> dict:
        meta = doc.metadata
        return {
            "id": doc.id,
            "depth": depth,
            "label": doc.source_label,
            "source_file": meta.get("source_file"),
            "canvas_node_id": meta.get("canvas_node_id"),
            "canvas_group": meta.get("canvas_group"),
            "canvas_degree": meta.get("canvas_degree"),
            "canvas_part": meta.get("canvas_part"),
            "score": round(float(doc.score), 4) if doc.score else None,
            "text": doc.text,
        }

    def _fetch_ids(self, ids: list[str]) -> list[RetrievedDoc]:
        if not ids:
            return []
        got = self.retriever._get_collection().get(
            ids=list(dict.fromkeys(ids)), include=["documents", "metadatas"])
        return [RetrievedDoc(id=str(i), text=t or "", metadata=m or {})
                for i, t, m in zip(got["ids"], got["documents"], got["metadatas"])]

    def _fetch_neighbors(self, source_file: str, node_ids: list[str]) -> list[RetrievedDoc]:
        """One hop, scoped to ONE canvas file — node ids repeat across files."""
        if not node_ids:
            return []
        got = self.retriever._get_collection().get(
            where={"$and": [{"source_file": source_file},
                            {"canvas_node_id": {"$in": node_ids}}]},
            include=["documents", "metadatas"])
        return [RetrievedDoc(id=str(i), text=t or "", metadata=m or {})
                for i, t, m in zip(got["ids"], got["documents"], got["metadatas"])]
