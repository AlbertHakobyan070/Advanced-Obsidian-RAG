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
     query is scoped to ONE source_file (the parent's; a cross-canvas hop's
     is the other canvas's).

CAPS ARE REPORTED, NOT SILENT
  depth / max_per_hop / max_total each stop the walk. When one bites, the
  response says which (`truncated_at`) — a traversal that quietly returned
  less would look like a graph that is smaller than it is.
"""
from __future__ import annotations

import posixpath
import re
import unicodedata
from typing import Any

from src.ingestion.canvas_loader import decode_canvas_edges, decode_canvas_files, norm_vault_path as _norm_path
from src.ingestion.obsidian_parser import MAX_WIKILINKS as _WIKILINKS_CAP   # parse_file stores extract_wikilinks(body)[:MAX_WIKILINKS]; stdlib + yaml at import
from src.retrieval.retriever import RetrievedDoc
from src.utils.config_loader import Config
from src.utils.logger import get_logger
from src.utils.provenance import ProvenanceRule, provenance_of, provenance_rule

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


_MD_TYPES = ("note", "daily_note")
_DIRECTIONS = ("both", "up", "down")
_PAGE_RE = re.compile(r"#page=(\d+)")


def _norm_label(s) -> str:      # "Prerequisite of" == "prerequisite-of"
    return re.sub(r"[\s_]+", "-", str(s).strip().casefold())


def _direction(value) -> str:
    """graph.direction, checked the same way from the config and from a request."""
    if value not in _DIRECTIONS:
        raise ValueError(f"graph.direction must be one of {', '.join(_DIRECTIONS)}, not {value!r}")
    return value


def _edge_types(value) -> set[str] | None:
    """graph.edge_types as the walk compares it: None = every edge, else the set of
    _norm_label forms. An empty list would follow nothing, so it is refused."""
    if value is None:
        return None
    if not isinstance(value, list) or not value or not all(isinstance(t, str) and t.strip() for t in value):
        raise ValueError(f"graph.edge_types must be null or a non-empty list of edge labels, not {value!r}")
    return {_norm_label(t) for t in value}


def _forms(p) -> list[str]:
    """The exact strings a stored source_file can take for vault path p: NFC and NFD,
    each with '/' and '\\' (md/pdf store '\\', canvases '/'). Chroma's $in is exact."""
    out: list[str] = []
    for form in ("NFC", "NFD"):
        s = unicodedata.normalize(form, str(p)).replace("\\", "/").strip("/")
        out += [s, s.replace("/", "\\")]
    return list(dict.fromkeys(out))


# Marks that sit INSIDE a word, deleted rather than spaced: the Armenian emphasis,
# exclamation and question marks ("ինչո՞ւ" is "ինչու" stressed) and both apostrophes.
_IN_WORD = dict.fromkeys(map(ord, "\u055b\u055c\u055e'\u2019"))


def title_key(text: str) -> str:
    """The same-as key of a canvas chunk, from its BODY only: drop line 1 (the
    [Canvas: ...] header), cut at the first "\\n\\nConnections:", take the first
    non-empty line; NFKC, casefolded, in-word marks (_IN_WORD) deleted, every other
    Unicode P*/S* char -> space, whitespace collapsed. "" for an empty body: a footer
    word must never join two canvases."""
    body = str(text or "").partition("\n")[2].split("\n\nConnections:", 1)[0]
    line = next((ln for ln in body.split("\n") if ln.strip()), "")
    line = "".join(" " if unicodedata.category(c)[0] in "PS" else c
                   for c in unicodedata.normalize("NFKC", line).casefold().translate(_IN_WORD))
    return " ".join(line.split())


def _link_target(raw: str) -> tuple[str, str, str] | None:
    """[[Folder/Note#Heading|x]] target -> (stem, heading, folder). Drops '^block', and
    the '\\' a markdown table escapes the alias pipe with ([[Note\\|x]] is stored
    "Note\\"). A last segment ending .md is a note; one ending in a letter-led
    extension (.png .pdf .canvas .m4a ...) is not a note link -> None; anything else
    is a stem ("Lecture 1.2"). folder is "" for a bare link: Obsidian writes one when
    two notes share the stem."""
    # ponytail: "v1.final" reads as an attachment; Obsidian's own resolver if that matters
    path, *heads = str(raw).split("|", 1)[0].rstrip("\\").split("#")
    heads = [h.strip() for h in heads if h.strip() and not h.strip().startswith("^")]
    folder, _, stem = path.replace("\\", "/").rpartition("/")
    stem = stem.strip()
    if stem.casefold().endswith(".md"):
        stem = stem[:-3]
    elif re.search(r"\.[^\W\d_]\w*$", stem):
        return None
    return (stem, heads[-1] if heads else "", folder.strip().strip("/")) if stem else None


def _passes(label, direction, kind, types, want) -> bool:
    """E4: whether a typed walk follows this edge. `direction` is the stored one,
    which the loader takes from the arrowheads; legacy canvas_neighbors edges are
    '--', so they pass only under both."""
    if kind == "same-as":                     # identity, not a relation
        return True
    if types is not None and _norm_label(label) not in types:
        return False                          # unlabelled edges never pass a filter
    if want == "both":
        return True
    return direction in {"down": ("->", "<>"), "up": ("<-", "<>")}[want]   # '--' only under both


def _paths(tree: list[dict]) -> list[dict]:
    """How the walk reached each node: its FIRST tree entry (breadth-first, so a
    shortest route), followed up to a seed. One path per child, in tree order. A
    tree entry without a kind is the legacy walk's, which follows canvas edges only."""
    first: dict[str, dict] = {}
    for t in tree:
        first.setdefault(t["child"], t)
    out = []
    for child in first:
        chain, cur = [], child
        while cur in first and len(chain) < len(first):      # bounded: a cycle cannot spin
            chain.append(first[cur])
            cur = first[cur]["parent"]
        chain.reverse()
        out.append({"to": child, "nodes": [cur, *(e["child"] for e in chain)],
                    "edges": [{"label": e["label"], "kind": e.get("kind", "canvas"), "direction": e["direction"]}
                              for e in chain]})
    return out


def _seed_index(cfg: Config, retriever, file_type: str):
    """(embedder, collection, error) for graph.embedding_provider's seed index,
    <paths.collection_name>__graph (src/embeddings/graph_twin.py).

    A twin that cannot be served — not built, unfinished, built by another
    embedder, or holding other ids than the main collection's graph lane — is
    an ERROR, never a fallback to the main embedder. But it is scoped: it
    refuses query seeding only, and :8051, /query, /search and seeding by id
    keep working. The id check is exact; text edited under a kept id is not
    seen here (the console's status names chunk files newer than the build)."""
    from chromadb.errors import NotFoundError

    from src.embeddings.embedder import Embedder
    from src.embeddings.graph_twin import (
        BUILD_HINT, graph_collection_name, graph_spec, twin_id_diff)
    from src.embeddings.sidecar import EmbeddingMismatchError, check_collection
    from src.utils.chroma_client import persistent_client

    name = graph_collection_name(cfg)
    chroma_dir = cfg.path("paths.chroma_dir")
    try:
        embedder = Embedder.from_config(cfg, spec=graph_spec(cfg))
        if check_collection(embedder, chroma_dir, name, role="graph") is None:
            raise RuntimeError(f"{name!r} has not been built: {BUILD_HINT}")
        col = persistent_client(chroma_dir).get_collection(name)
        only_main, only_twin = twin_id_diff(retriever._get_collection(), col, file_type)
        if only_main or only_twin:
            raise RuntimeError(
                f"{name!r} is out of step with {retriever.collection_name!r}: {len(only_main)} "
                f"{file_type} chunk(s) are not in it and {len(only_twin)} are no longer in the "
                f"main collection. Rebuild it: {BUILD_HINT}")
    except (EmbeddingMismatchError, ValueError, RuntimeError, NotFoundError) as e:
        log.error("graph-mode query seeding is unavailable: %s", e)
        return None, None, str(e)
    log.info("graph seeds: %r embedded by %s", name, embedder.spec.label())
    return embedder, col, None


class GraphExpander:
    """Walks canvas edges. Owns no models — it borrows the retriever's
    collection and embedder, and the pipeline's reranker for seed ordering.
    With graph.embedding_provider set, query seeding ranks its dense lane in
    that embedder's own index instead (seed_embedder / seed_collection); a
    broken one sets seed_error, which only query seeding refuses on."""

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
        seed_embedder=None,
        seed_collection=None,
        seed_error: str | None = None,
        bridges: bool = False,
        cross_canvas: bool = False,
        wikilinks: bool = False,
        direction: str = "both",
        edge_types: list[str] | None = None,
        target_max_chunks: int = 4,
        bridge_max_total: int = 20,
        same_as_min_chars: int = 4,
        same_as_max_canvases: int = 4,
        provenance_rule: ProvenanceRule | None = None,
    ):
        for key, value in (("target_max_chunks", target_max_chunks), ("bridge_max_total", bridge_max_total),
                           ("same_as_min_chars", same_as_min_chars),
                           ("same_as_max_canvases", same_as_max_canvases)):
            if int(value) < 1:
                raise ValueError(f"graph.{key} must be at least 1, not {value!r}")
        self.retriever = retriever
        self.reranker = reranker
        self.enabled = enabled
        self.max_depth = max_depth
        self.max_per_hop = max_per_hop
        self.max_total = max_total
        self.seed_top_k = seed_top_k
        self.file_type = file_type
        self.seed_embedder = seed_embedder
        self.seed_collection = seed_collection
        self.seed_error = seed_error
        # The Navigator switches (graph.* in config.yaml), each also a per-call
        # expand() argument. All at these defaults = the walk before they existed.
        self.bridges = bool(bridges)
        self.cross_canvas = bool(cross_canvas)
        self.wikilinks = bool(wikilinks)
        self.direction = _direction(direction)
        self.edge_types = _edge_types(edge_types)
        self.target_max_chunks = int(target_max_chunks)
        self.bridge_max_total = int(bridge_max_total)
        self.same_as_min_chars = int(same_as_min_chars)
        self.same_as_max_canvases = int(same_as_max_canvases)
        # Labels each node's provenance (spec §5.5). None only in direct
        # constructions; from_config always passes the rule.
        self.provenance_rule = provenance_rule
        self._cindex = None
        self._sparse_links = None

    @classmethod
    def from_config(cls, cfg: Config, retriever, reranker=None) -> "GraphExpander":
        from src.embeddings.graph_twin import configured_provider
        enabled = bool(cfg.get("graph.enabled", True))
        file_type = str(cfg.get("graph.file_type", "canvas"))
        seed = (_seed_index(cfg, retriever, file_type)
                if enabled and configured_provider(cfg) is not None
                else (None, None, None))
        return cls(
            retriever=retriever,
            reranker=reranker,
            enabled=enabled,
            max_depth=int(cfg.get("graph.max_depth", 2)),
            max_per_hop=int(cfg.get("graph.max_per_hop", 8)),
            max_total=int(cfg.get("graph.max_total", 40)),
            seed_top_k=int(cfg.get("graph.seed_top_k", 5)),
            file_type=file_type,
            seed_embedder=seed[0],
            seed_collection=seed[1],
            seed_error=seed[2],
            bridges=bool(cfg.get("graph.bridges", False)),
            cross_canvas=bool(cfg.get("graph.cross_canvas", False)),
            wikilinks=bool(cfg.get("graph.wikilinks", False)),
            direction=cfg.get("graph.direction", "both"),
            edge_types=cfg.get("graph.edge_types", None),
            target_max_chunks=int(cfg.get("graph.target_max_chunks", 4)),
            bridge_max_total=int(cfg.get("graph.bridge_max_total", 20)),
            same_as_min_chars=int(cfg.get("graph.same_as_min_chars", 4)),
            same_as_max_canvases=int(cfg.get("graph.same_as_max_canvases", 4)),
            provenance_rule=provenance_rule(cfg),
        )

    # ---- seeding ----

    def seed_from_query(self, query: str, seed_top_k: int | None = None,
                        rerank: str | None = None) -> list[RetrievedDoc]:
        """Hybrid dense+sparse restricted to the graph's file_type, RRF-fused.

        Deliberately plain: no HyDE, no scope routing, no metadata boost. The
        seed set is the thing a human is about to steer, so it should be
        explainable — "the best canvas chunks for these words" — rather than
        the product of every heuristic the main path applies.

        With a graph seed index (graph.embedding_provider), the dense lane is
        ranked there, by that embedder, and each hit is read back from the
        main collection. A broken one refuses here, and only here.
        """
        if self.seed_error:
            raise RuntimeError(f"graph-mode query seeding is unavailable: {self.seed_error}")
        k = seed_top_k or self.seed_top_k
        pool = max(k * 4, 20)
        ftype = self.file_type
        if self.seed_collection is not None:
            dense = self._seed_index_search(query, pool)
        else:
            qvec = self.retriever.embedder.embed_query(query)
            dense = self.retriever._dense_search(qvec, pool, where={"file_type": ftype})
        lanes = [
            dense,
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

    def _seed_index_search(self, query: str, pool: int) -> list[tuple[str, str, dict]]:
        """The dense lane over the graph seed index: its ranking, the main
        collection's text and metadata (what the walk and the answer read)."""
        res = self.seed_collection.query(
            query_embeddings=[self.seed_embedder.embed_query(query)], n_results=pool,
            include=["distances"])
        ranked = res["ids"][0]
        by_id = {d.id: d for d in self._fetch_ids(ranked)}
        gone = [i for i in ranked if i not in by_id]
        if gone:
            # ponytail: an id deleted from main since the build is skipped per query;
            # the startup id check refuses the whole index at the next :8051 restart.
            log.warning("graph seed index returned %d id(s) the main collection no longer "
                        "holds (skipped; rebuild it with graph_embed), e.g. %r", len(gone), gone[0])
        return [(i, by_id[i].text, by_id[i].metadata) for i in ranked if i in by_id]

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
        *,
        bridges: bool | None = None,
        cross_canvas: bool | None = None,
        wikilinks: bool | None = None,
        direction: str | None = None,
        edge_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Breadth-first walk from `seeds` along canvas edges.

        Returns {"nodes", "tree", "cross_edges", "stats"}. `tree` is a flat
        parent->child edge list carrying each hop's label and direction;
        `cross_edges` are edges onto nodes the walk had ALREADY reached —
        reported separately because dropping them would draw a cyclic graph
        as if it were a tree.

        The Navigator switches (bridges ... edge_types) are per call; None = the
        graph.* config. An edge edge_types or direction rejects is never a hop
        nor a cross edge, and is counted in stats.filtered_edges. With any switch
        off its default (extended mode) the result also has "paths", and nodes,
        tree and cross entries carry a `kind`.
        """
        d_cap = self.max_depth if depth is None else int(depth)
        hop_cap = self.max_per_hop if max_per_hop is None else int(max_per_hop)
        total_cap = self.max_total if max_total is None else int(max_total)
        bridges = self.bridges if bridges is None else bool(bridges)
        cross_canvas = self.cross_canvas if cross_canvas is None else bool(cross_canvas)
        wikilinks = self.wikilinks if wikilinks is None else bool(wikilinks)
        direction = self.direction if direction is None else _direction(direction)
        types = self.edge_types if edge_types is None else _edge_types(edge_types)

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
            """A chunk joins the walk as a walk node (a new node, or a part of one). A
            bridge to it is dropped, so no chunk is listed twice."""
            nodes.setdefault(doc.id, self._node_out(doc, hop, ext))
            by_key.setdefault(_node_key(doc.metadata), []).append(doc.id)
            bridge_nodes.pop(doc.id, None)

        # Extended mode: a Navigator switch is off its pre-slice default. Only then
        # do entries carry `kind`; with every switch at its default the output is
        # the old walk's, key for key (tests/test_graph_engine.py snapshot).
        ext = bridges or cross_canvas or wikilinks or direction != "both" or types is not None
        # E3 links run along their arrow, so `up` never follows one.
        links = wikilinks and direction != "up" and (types is None or "links-to" in types)
        # Extended-mode counters, merged into stats; each bridge_*/wikilinks_* key is a miss.
        st = {"filtered_edges": 0, "bridges_unavailable": 0, "bridge_targets_without_chunks": 0,
              "bridge_heading_missing": 0, "bridge_pdf_without_page": 0,
              "bridges_matched_after_normalised": 0,
              "virtual_edges": {"same-as": 0, "shares-bridge": 0, "opens-canvas": 0, "links-to": 0},
              "virtual_edges_dropped": 0, "wikilinks_unresolved": 0, "wikilinks_ambiguous": 0,
              "wikilinks_truncated": 0}
        bridge_nodes: dict[str, dict] = {}

        def attach(docs: list[RetrievedDoc], depth: int) -> None:
            for doc in docs:
                if doc.metadata.get("file_type") != self.file_type:
                    continue
                if bridges:
                    self._attach_bridges(doc, depth, nodes, bridge_nodes, tree, cross, st, cap)
                elif (cross_canvas and "canvas_edge_files" not in doc.metadata
                      and doc.metadata.get("canvas_file_refs")):
                    st["bridges_unavailable"] += 1    # its file-node hops cannot be taken either

        def consider(parent: RetrievedDoc, key: tuple[str, str], label: str, direction: str,
                     kind: str, docs: list[RetrievedDoc] | None = None, virtual: bool = False) -> bool:
            """One edge from `parent` onto node `key`, this hop. False = this parent's
            per-hop budget is spent, so the caller stops. A real edge records the
            max_per_hop cap; a virtual one does not, its caller counts the drop. With
            `docs` (wikilink targets, already fetched) the link names its own chunks: a
            note is one node however many sections its links name, each chunk joins
            `direct` once however many parents link it, and each parent's link reaches
            only the chunks it names (`ids`)."""
            nonlocal taken
            if key in visited:
                # A cycle or a diamond. Record it so the caller can
                # SEE the loop; do not re-walk it. A link into a note the walk holds
                # names its own chunks: an edge onto those held, and the sections not
                # held yet join this hop as parts of that note.
                held = by_key.get(key, []) if docs is None else [r.id for r in docs if r.id in nodes]
                for cid in held:
                    cross.append({"parent": parent.id, "child": cid,
                                  "label": label,
                                  "direction": direction,
                                  "depth": hop, **({"kind": kind} if ext else {})})
                docs = [r for r in docs or [] if r.id not in nodes]
                if not docs:
                    return True
            elif taken >= hop_cap:
                if not virtual:
                    cap("max_per_hop")
                return False
            else:
                taken += 1
            if docs is None:
                wanted.setdefault(key[0], set()).add(key[1])
            else:
                have = {d.id for d in direct}
                direct.extend(r for r in docs if r.id not in have)
            asked.setdefault(key, []).append(
                {"parent": parent.id, "label": label,
                 "direction": direction, **({"kind": kind} if ext else {}),
                 **({"ids": {r.id for r in docs}} if docs is not None else {})})
            return True

        frontier: list[RetrievedDoc] = []
        for doc in seeds:
            key = _node_key(doc.metadata)
            if key in visited:
                by_key.setdefault(key, []).append(doc.id)
                nodes.setdefault(doc.id, self._node_out(doc, 0, ext))
                continue
            visited.add(key)
            admit(doc, 0)
            frontier.append(doc)
        attach(frontier, 0)
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
            direct: list[RetrievedDoc] = []
            for parent in frontier:
                sf = str(parent.metadata.get("source_file") or "")
                edges, ok = _edges_of(parent.metadata)
                labelled = labelled and ok
                # With bridges or cross_canvas on, an edge onto one of this chunk's file
                # nodes is its bridge (_attach_bridges) or its cross-canvas hop
                # (_virtual_edges): neither a hop nor a dangling edge.
                files = ({f["node"] for f in decode_canvas_files(parent.metadata.get("canvas_edge_files"))
                          if f["node"]} if bridges or cross_canvas else set())
                taken = 0
                for edge in edges:
                    nid = edge["neighbor"]
                    if not nid or nid in files:
                        continue
                    if not _passes(edge["label"], edge["direction"], "canvas", types, direction):
                        st["filtered_edges"] += 1
                        continue
                    if not consider(parent, (sf, nid), edge["label"], edge["direction"], "canvas"):
                        break
                ptype = parent.metadata.get("file_type")
                if cross_canvas and ptype == self.file_type:
                    # After the real edges, which keep their most-important-first order. A
                    # virtual edge over the shared budget is counted, never blamed on max_per_hop.
                    for sf2, nid2, kind in self._virtual_edges(parent):
                        if not _passes(kind, "--", kind, types, direction):
                            st["filtered_edges"] += 1
                            continue
                        fresh = (sf2, nid2) not in visited
                        if not consider(parent, (sf2, nid2), kind, "--", kind, virtual=True):
                            st["virtual_edges_dropped"] += 1
                        elif fresh:
                            st["virtual_edges"][kind] += 1
                if links and ptype in _MD_TYPES:
                    for k2, rows in self._wikilink_targets(parent, st):
                        fresh = k2 not in visited
                        if not consider(parent, k2, "links-to", "->", "links-to", docs=rows):
                            break
                        if fresh:
                            st["virtual_edges"]["links-to"] += 1
                elif wikilinks and ptype in _MD_TYPES:
                    # E3 is on, but direction or edge_types rejects links-to: each link is a
                    # filtered edge, counted like a canvas one (no fetch needed to count them).
                    st["filtered_edges"] += len({t for t in map(_link_target, self._links_of(parent)) if t})
            # A frontier of notes asks for no canvas rows: its links are in `direct`.
            if not wanted and not direct:
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
            reached.extend(direct)               # wikilink targets, fetched while gathering

            next_frontier: list[RetrievedDoc] = []
            for doc in reached:
                key = _node_key(doc.metadata)
                if len(nodes) >= total_cap:
                    cap("max_total")
                    if key not in visited:
                        break
                    continue                     # a part over the cap: skipped, and reported
                for link in asked.get(key, []):
                    if "ids" in link and doc.id not in link["ids"]:
                        continue                 # that link names other sections of this note
                    tree.append({"parent": link["parent"], "child": doc.id,
                                 "label": link["label"],
                                 "direction": link["direction"], "depth": hop,
                                 **({"kind": link["kind"]} if ext else {})})
                if key in visited:
                    # Another part of a node already admitted (this hop, or a note
                    # section a link names that the walk did not hold yet).
                    admit(doc, hop)
                    continue
                visited.add(key)
                admit(doc, hop)
                next_frontier.append(doc)
            attach(next_frontier, hop)
            frontier = next_frontier
        else:
            # Ran every allowed hop and still had somewhere to go: DEPTH is
            # what stopped the walk.
            if frontier and d_cap > 0:
                cap("depth")

        result = {
            "nodes": [*nodes.values(), *bridge_nodes.values()],     # bridges only exist in extended mode
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
                         "max_total": total_cap,
                         **({"bridge_max_total": self.bridge_max_total} if ext else {})},
                **({**st, "bridges": len(bridge_nodes),
                    "options": {"bridges": bridges, "cross_canvas": cross_canvas,
                                "wikilinks": wikilinks, "direction": direction,
                                "edge_types": sorted(types) if types is not None else None}}
                   if ext else {}),
            },
        }
        if ext:
            result["paths"] = _paths(tree)
        return result

    def _attach_bridges(self, doc: RetrievedDoc, depth: int, nodes: dict, bridge_nodes: dict,
                        tree: list, cross: list, st: dict, cap) -> None:
        """E1: attach the chunks this canvas chunk's file nodes and Source: lines cite.
        Bridges are TERMINAL - never expanded, never visited, never counted in
        max_total - so a deep walk still reaches its evidence; graph.target_max_chunks
        caps each target and graph.bridge_max_total the walk. A cited chunk the walk
        already holds (a seed, a walk node) gets a cross edge, never a route to it. A
        chunk indexed before canvas_edge_files existed is counted, never guessed from
        canvas_file_refs (unaligned, and lossy once Chroma has joined it)."""
        meta = doc.metadata
        if "canvas_edge_files" not in meta:
            if meta.get("canvas_file_refs"):
                st["bridges_unavailable"] += 1
            return
        labels = {e["neighbor"]: e["label"] for e in _edges_of(meta)[0]}
        for f in decode_canvas_files(meta["canvas_edge_files"]):
            if _norm_path(f["file"]).endswith(".canvas"):
                continue                                 # a canvas is a hop (E2), not evidence
            if len(bridge_nodes) >= self.bridge_max_total:
                cap("bridge_max_total")
                return
            rows, why = self._resolve_file(f["file"], f["subpath"], st)
            if why:
                st[why] += 1
            for r in rows[:self.target_max_chunks]:
                if len(bridge_nodes) >= self.bridge_max_total:
                    cap("bridge_max_total")
                    return
                entry = {"parent": doc.id, "child": r.id,
                         "label": labels.get(f["node"], "") if f["node"] else "source",
                         "direction": "->", "depth": depth + 1, "kind": "bridge"}
                if r.id in nodes:
                    cross.append(entry)
                    continue
                tree.append(entry)
                if r.id not in bridge_nodes:
                    bridge_nodes[r.id] = {**self._node_out(r, depth + 1, True), "kind": "bridge"}

    def _canvas_index(self) -> dict[str, dict]:
        """E2's lookups over every canvas chunk: title (same-as key -> nodes), title_of
        (node -> key), files (a FILE NODE's cited path -> the nodes citing it) and
        canvases (a canvas -> its nodes, highest canvas_degree first). Source-line
        entries stay out of files: a book cited on every node of several canvases
        would join them all. A title in more than graph.same_as_max_canvases
        canvases is generic ("python") and joins none."""
        # ponytail: built once per process; a canvas appended while :8051 runs joins at the restart the append already needs
        if self._cindex is not None:
            return self._cindex
        title: dict[str, list] = {}
        title_of: dict[tuple[str, str], str] = {}
        files: dict[str, list] = {}
        canvases: dict[str, list] = {}
        for d in self._get({"file_type": self.file_type}):
            m = d.metadata
            if m.get("canvas_part") not in (None, 1):
                continue                  # every part carries its node's metadata: index the node once
            key = _node_key(m)
            for f in decode_canvas_files(m.get("canvas_edge_files")):
                if f["node"]:
                    files.setdefault(_norm_path(f["file"]), []).append(key)
            canvases.setdefault(_norm_path(key[0]), []).append((-int(m.get("canvas_degree") or 0), key))
            t = title_key(d.text)
            if t:
                title.setdefault(t, []).append(key)
                title_of[key] = t
        self._cindex = {
            "title": {t: ks for t, ks in title.items() if len(t) >= self.same_as_min_chars
                      and len({sf for sf, _ in ks}) <= self.same_as_max_canvases},
            "title_of": title_of,
            "files": files,
            "canvases": {c: [k for _, k in sorted(v, key=lambda x: x[0])] for c, v in canvases.items()},
        }
        return self._cindex

    def _virtual_edges(self, parent: RetrievedDoc) -> list[tuple[str, str, str]]:
        """E2: (source_file, node id, kind) of the nodes this canvas node joins in other
        canvases, in this order, each node once: same-as (its title), opens-canvas (a
        file node on a .canvas: that canvas's hubs) and shares-bridge (another canvas's
        file node on the same file). File-node entries only, never Source lines."""
        idx = self._canvas_index()
        key = _node_key(parent.metadata)
        out: dict[tuple[str, str], str] = {}
        for k in idx["title"].get(idx["title_of"].get(key), []):
            if k[0] != key[0]:
                out.setdefault(k, "same-as")
        targets = [_norm_path(f["file"]) for f in decode_canvas_files(parent.metadata.get("canvas_edge_files"))
                   if f["node"]]
        for t in targets:
            if t.endswith(".canvas"):
                for k in idx["canvases"].get(t, []):
                    out.setdefault(k, "opens-canvas")
        for t in targets:
            if not t.endswith(".canvas"):
                for k in idx["files"].get(t, []):
                    if k[0] != key[0]:
                        out.setdefault(k, "shares-bridge")
        out.pop(key, None)
        return [(sf, nid, kind) for (sf, nid), kind in out.items()]

    def _links_of(self, doc: RetrievedDoc) -> list[str]:
        """doc's [[wikilinks]] as the parser stored them, a list. Chroma holds that list
        ', '-joined (Embedder._clean_meta), which cannot be split back once a target holds
        ', ' itself ([[Smith, 2020]]); the sparse lane keeps the JSONL list verbatim, so a
        joined string is read back from there by chunk id. A note chunk the sparse lane
        lacks, or whose sparse list does not join to the dense string, is dense/sparse
        drift and raises: a guessed split can hop to another note, a stale list to old links."""
        raw = doc.metadata.get("wikilinks") or []
        if isinstance(raw, (list, tuple)):
            return [str(t) for t in raw]
        if ", " not in str(raw):
            return [str(raw)]                       # one link: nothing was joined
        if self._sparse_links is None:
            # ponytail: md chunk id -> its sparse metadata, built once per process like _cindex
            payload = self.retriever._get_bm25()
            self._sparse_links = {i: m for i, m in zip(payload["ids"], payload["metadatas"])
                                  if (m or {}).get("file_type") in _MD_TYPES}
        links = (self._sparse_links.get(doc.id) or {}).get("wikilinks")
        if not isinstance(links, (list, tuple)):
            raise RuntimeError(
                f"note chunk {doc.id!r} has ', '-joined wikilinks in the dense collection but no "
                f"list of them in the sparse index, so its links cannot be read back: dense and "
                f"sparse disagree. Run python rebuild_bm25.py with the servers down, then restart "
                f"this API")
        if ", ".join(map(str, links)) != str(raw):  # the dense string is the order-kept join of the true list
            raise RuntimeError(
                f"note chunk {doc.id!r} lists its wikilinks as {links!r} in the sparse index but as "
                f"{str(raw)!r} in the dense collection: dense and sparse disagree. The usual cause is "
                f"its doc_id on two chunk files: the sparse union keeps the FIRST row per doc_id "
                f"(chunks.jsonl, then data/*_chunks.jsonl sorted), so a note re-parsed and appended "
                f"under an unchanged id keeps its old links there, and rebuild_bm25.py alone cannot "
                f"fix that. Make the append a REPLACE: POST /api/documents/delete of that note with "
                f"\"keep_jsonls\" naming the appended chunk file and \"rebuild\": false, then the same "
                f"append, then restart this API")
        return [str(t) for t in links]

    def _wikilink_targets(self, parent: RetrievedDoc, st: dict) -> list[tuple[tuple[str, str], list[RetrievedDoc]]]:
        """E3: (node key, chunks) of the notes this note's [[wikilinks]] name, one entry per
        note: the union of the sections its links name, graph.target_max_chunks per link. A
        stem matches a stored filename exactly after NFC, case included; a link with a
        folder ([[A/Note]]) keeps only the note under that folder, never another one, and a
        relative one ([[../B/Note]], [[./Note]]) only the note at that path from this note's
        folder; a stem two notes share is ambiguous and never guessed. Each miss is counted,
        never widened."""
        # ponytail: outgoing only; backlinks need a reverse index over md metadata
        meta = parent.metadata
        found = self._links_of(parent)
        if len(found) >= _WIKILINKS_CAP:
            st["wikilinks_truncated"] += 1          # the parser kept only the first MAX_WIKILINKS
        targets = list(dict.fromkeys(t for t in map(_link_target, found) if t))
        if not targets:
            return []
        # ponytail: Obsidian resolves links case-insensitively; a casefolded stem index over md metadata if wikilinks_unresolved shows it matters
        # ponytail: a note's first target_max_chunks rows in the order Chroma returns them (U3)
        notes: dict[str, dict[str, list[RetrievedDoc]]] = {}
        stems = list(dict.fromkeys(f for stem, _, _ in targets for f in _forms(stem)))
        for d in self._get({"$and": [{"filename": {"$in": stems}}, {"file_type": {"$in": list(_MD_TYPES)}}]}):
            notes.setdefault(unicodedata.normalize("NFC", str(d.metadata.get("filename") or "")), {}) \
                .setdefault(str(d.metadata.get("source_file") or ""), []).append(d)
        own = str(meta.get("source_file") or "")
        out: dict[str, list[RetrievedDoc]] = {}
        for stem, heading, folder in targets:
            hits = notes.get(unicodedata.normalize("NFC", stem), {})
            if folder.split("/")[0] in (".", ".."):  # relative: a whole vault path, so an exact match
                path = posixpath.normpath(posixpath.join(posixpath.dirname(_norm_path(own)), folder, stem))
                hits = {sf: rows for sf, rows in hits.items() if _norm_path(sf) == _norm_path(path) + ".md"}
            elif folder:                            # anchored on a '/' boundary: "A/Other" is not "BA/Other"
                tail = "/" + _norm_path(f"{folder}/{stem}") + ".md"
                hits = {sf: rows for sf, rows in hits.items() if ("/" + _norm_path(sf)).endswith(tail)}
            if len(hits) != 1:
                st["wikilinks_ambiguous" if hits else "wikilinks_unresolved"] += 1
                continue
            [(sf2, rows)] = hits.items()
            if sf2 == own:
                continue
            if heading:
                rows = [r for r in rows if str(r.metadata.get("heading") or "").strip().casefold() == heading.casefold()]
                if not rows:
                    st["wikilinks_unresolved"] += 1
                    continue
            note = out.setdefault(sf2, [])          # a second link into one note adds its sections, no second edge
            have = {r.id for r in note}
            note += [r for r in rows[:self.target_max_chunks] if r.id not in have]
        return [(_node_key(rows[0].metadata), rows) for rows in out.values()]

    def attach_routed_ids(self, nodes: list[dict]) -> None:
        """For the map view (POST /graph/expand) only: give each generated node
        routed_ids, the first retrieval.route.per_target primary chunks of each span it
        cites, in document order, for at most retrieval.route.max_targets spans
        (routed_for(None, ...): no embedding, one call per node so none loses its targets
        to max_targets). Nothing when no route sidecar is loaded. expand() and
        graph_search never call it, so walks pay nothing for it."""
        if self.retriever.route_pointers is None:
            return
        for node in nodes:
            if node.get("provenance") == "generated":
                node["routed_ids"] = [d.id for d in self.retriever.routed_for(None, [node["id"]])[0]]

    # ---- chunk store access ----

    def _node_out(self, doc: RetrievedDoc, depth: int, ext: bool = False) -> dict:
        meta = doc.metadata
        extra = ({"kind": "canvas" if meta.get("canvas_node_id") else "note",
                  "file_type": meta.get("file_type"), "heading": meta.get("heading"),
                  "page_start": meta.get("page_start"), "page_end": meta.get("page_end")}
                 if ext else {})
        if self.provenance_rule is not None:    # a map, not evidence: generated nodes stay, labelled
            extra["provenance"] = provenance_of(meta, self.provenance_rule)
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
            **extra,
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

    def _get(self, where: dict) -> list[RetrievedDoc]:
        got = self.retriever._get_collection().get(where=where, include=["documents", "metadatas"])
        return [RetrievedDoc(id=str(i), text=t or "", metadata=m or {})
                for i, t, m in zip(got["ids"], got["documents"], got["metadatas"])]

    def _resolve_file(self, file: str, subpath: str, st: dict) -> tuple[list[RetrievedDoc], str]:
        """(rows, why) for one cited file. why = the stats key of a miss, "" on a hit.
        A PDF bridges only through #page=N: a page-less PDF is front matter, never
        evidence (decision 14). A note #Heading keeps that section, and a missing
        heading bridges nothing rather than the whole note."""
        want = _norm_path(file)
        if want.endswith(".pdf"):
            m = _PAGE_RE.fullmatch(subpath or "")
            if not m:
                return [], "bridge_pdf_without_page"
            n = int(m.group(1))
            # The page is tested here, not in the where: Chroma answers a page_start/page_end
            # range by comparing every PDF chunk's pages (seconds per bridge on the live store),
            # a source_file $in by an index lookup. A missing bound is off the page, as Chroma's
            # $lte/$gte skip it. Collection.get has no order parameter: sort, so the first page
            # chunk is stable.
            on_page = [d for d in self._get({"source_file": {"$in": _forms(file)}})
                       if None not in (d.metadata.get("page_start"), d.metadata.get("page_end"))
                       and d.metadata["page_start"] <= n <= d.metadata["page_end"]]
            rows = sorted(on_page, key=lambda d: (d.metadata.get("page_start"), d.id))
            heading = ""
        else:
            # ponytail: a note's first target_max_chunks rows in the order Chroma returns
            # them (U3); sort on a chunk-order key if one is ever stored.
            rows = self._get({"source_file": {"$in": _forms(file)}})
            segs = [s.strip() for s in (subpath or "").split("#") if s.strip()]
            heading = segs[-1] if segs and not segs[-1].startswith("^") else ""   # ^block = whole note
        exact = str(file).replace("\\", "/")
        rows = [r for r in rows if _norm_path(r.metadata.get("source_file") or "") == want]
        st["bridges_matched_after_normalised"] += sum(
            str(r.metadata.get("source_file")).replace("\\", "/") != exact for r in rows)
        if heading:
            hit = [r for r in rows if str(r.metadata.get("heading") or "").strip().casefold() == heading.casefold()]
            if rows and not hit:
                return [], "bridge_heading_missing"
            rows = hit
        return rows, "" if rows else "bridge_targets_without_chunks"
