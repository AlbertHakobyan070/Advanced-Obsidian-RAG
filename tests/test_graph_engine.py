"""Navigator graph engine tests (slice D, E1-E4) - src/retrieval/graph_expand.py.

Fully offline. WhereCollection stands in for ChromaDB and REFUSES any filter it
does not implement: tests/test_graph_expand.py's FakeCollection reduces an
unknown clause to "no rows", so a walk that sent a wrong filter would pass there
by finding nothing.

The snapshot test is the lock on the walk as it was before slice D: every new
switch at its default must give this exact output.
"""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.ingestion.canvas_loader import encode_canvas_edge
from src.retrieval.graph_expand import GraphExpander
from src.retrieval.retriever import RetrievedDoc


# ---- a store fake that refuses what it does not implement ----

_KEYS = ("source_file", "file_type", "filename", "page_start", "page_end", "canvas_node_id")


def _match(meta: dict, where: dict | None) -> bool:
    """Chroma's where semantics for the clauses the walk sends: $and, $in, $lte,
    $gte and equality, in the shapes chromadb 1.5.9 accepts: one key per dict, one
    operator per expression, an $and of at least 2 clauses, a non-empty $in.
    Anything else raises, so a test cannot pass vacuously on a filter real Chroma
    refuses. Every clause is evaluated, so every clause is checked."""
    if where is None:
        return True
    if not isinstance(where, dict) or len(where) != 1:
        raise ValueError(f"WhereCollection: a where needs exactly one key, got {where!r}")
    ((key, cond),) = where.items()
    if key == "$and":
        if not isinstance(cond, list) or len(cond) < 2:
            raise ValueError(f"WhereCollection: $and needs a list of at least 2 clauses, got {cond!r}")
        return all([_match(meta, c) for c in cond])
    if key not in _KEYS:
        raise ValueError(f"WhereCollection does not implement key {key!r}")
    value = meta.get(key)
    if not isinstance(cond, dict):
        return value == cond
    if len(cond) != 1:
        raise ValueError(f"WhereCollection: an operator expression needs exactly one operator, got {cond!r}")
    ((op, arg),) = cond.items()
    if op == "$in":
        if not isinstance(arg, list) or not arg:
            raise ValueError(f"WhereCollection: $in needs a non-empty list, got {arg!r}")
        return value in arg
    if op == "$lte":
        return value is not None and value <= arg
    if op == "$gte":
        return value is not None and value >= arg
    raise ValueError(f"WhereCollection does not implement operator {op!r}")


class WhereCollection:
    def __init__(self, rows: dict[str, tuple[str, dict]]):
        self.rows = rows
        self.calls: list[dict] = []

    def get(self, ids=None, where=None, include=None, limit=None):
        self.calls.append({"ids": ids, "where": where, "limit": limit})
        _match({}, where)                    # check the shape even when no row is held
        hits = [(i, t, m) for i, (t, m) in self.rows.items()
                if (ids is None or i in ids) and _match(m, where)]
        hits = hits[:limit] if limit is not None else hits
        return {"ids": [h[0] for h in hits],
                "documents": [h[1] for h in hits],
                "metadatas": [h[2] for h in hits]}


class FakeRetriever:
    rrf_k = 60

    def __init__(self, collection, sparse=None):
        self.collection = collection
        self.embedder = None
        self.sparse = sparse

    def _get_collection(self):
        return self.collection

    def _get_bm25(self):
        """The sparse lane's payload: each row's JSONL metadata verbatim (lists stay
        lists). The store's own rows by default; `sparse` = id -> (text, meta) when
        the store holds them in Chroma's joined shape."""
        rows = self.collection.rows if self.sparse is None else self.sparse
        return {"ids": list(rows), "metadatas": [m for _, m in rows.values()]}


def _canvas_chunk(cid, source, node_id, title, edges, file_refs=()):
    """A canvas chunk shaped like canvas_loader.load_file writes it today, with
    no canvas_edge_files key (the chunks indexed before slice D)."""
    footer = "\n".join(f"  {d} [{l}] {n}" if l else f"  {d} {n}" for n, d, l in edges)
    meta = {"source_file": source, "filename": Path(source).stem, "file_type": "canvas",
            "canvas_node_id": node_id,
            "canvas_neighbors": [n for n, _, _ in edges],
            "canvas_edge_labels": [l for _, _, l in edges if l],
            "canvas_edges": [encode_canvas_edge(n, d, l) for n, d, l in edges],
            "canvas_file_refs": list(file_refs),
            "canvas_degree": len(edges)}
    return cid, (f"[Canvas: {Path(source).stem}]\n{title}\n\nConnections:\n{footer}", meta)


def _store(rows):
    return WhereCollection(dict(rows))


def _seed(store, cid):
    text, meta = store.rows[cid]
    return RetrievedDoc(id=cid, text=text, metadata=meta)


def test_fake_store_refuses_what_it_does_not_implement():
    with pytest.raises(ValueError, match="heading"):
        _match({"heading": "x"}, {"heading": "x"})
    with pytest.raises(ValueError, match=r"\$ne"):
        _match({"page_start": 2}, {"page_start": {"$ne": 1}})
    meta = {"source_file": "a", "canvas_node_id": "A", "page_start": 5}
    for where in ({"$and": [{"source_file": "a"}]},                          # a single-clause $and
                  {"source_file": "a", "canvas_node_id": {"$in": ["A"]}},      # two top-level keys
                  {"page_start": {"$lte": 5, "$gte": 5}},                      # two operators
                  {"canvas_node_id": {"$in": []}}):                            # an empty $in
        with pytest.raises(ValueError):                                       # chromadb 1.5.9 refuses each
            _match(meta, where)


# ---- the pre-slice lock ----

def _snapshot_rows():
    """A -> B -> C -> A in a.canvas, A also -> file node F1 (never a chunk). The
    b.canvas node shares A's title and B's node id: a default walk reaches it
    neither by title nor by node id."""
    return [
        _canvas_chunk("cA", "Maps/a.canvas", "A", "Soft margin SVM",
                      [("B", "->", "leads to"), ("C", "<-", "closes"), ("F1", "->", "cites")],
                      file_refs=["Books/svm.pdf"]),
        _canvas_chunk("cB", "Maps/a.canvas", "B", "Slack variables",
                      [("A", "<-", "leads to"), ("C", "->", "")]),
        _canvas_chunk("cC", "Maps/a.canvas", "C", "Hinge loss",
                      [("B", "<-", ""), ("A", "->", "closes")]),
        _canvas_chunk("cB2", "Maps/b.canvas", "B", "Soft margin SVM",
                      [("X", "->", "uses")]),
    ]


# Captured from expand([cA], depth=2) on the unmodified code (27ef688 walk, B tip ffcd50c).
EXPECTED = r"""{
  "cross_edges": [
    {
      "child": "cA",
      "depth": 2,
      "direction": "<-",
      "label": "leads to",
      "parent": "cB"
    },
    {
      "child": "cC",
      "depth": 2,
      "direction": "->",
      "label": "",
      "parent": "cB"
    },
    {
      "child": "cB",
      "depth": 2,
      "direction": "<-",
      "label": "",
      "parent": "cC"
    },
    {
      "child": "cA",
      "depth": 2,
      "direction": "->",
      "label": "closes",
      "parent": "cC"
    }
  ],
  "nodes": [
    {
      "canvas_degree": 3,
      "canvas_group": null,
      "canvas_node_id": "A",
      "canvas_part": null,
      "depth": 0,
      "id": "cA",
      "label": "a",
      "score": null,
      "source_file": "Maps/a.canvas",
      "text": "[Canvas: a]\nSoft margin SVM\n\nConnections:\n  -> [leads to] B\n  <- [closes] C\n  -> [cites] F1"
    },
    {
      "canvas_degree": 2,
      "canvas_group": null,
      "canvas_node_id": "B",
      "canvas_part": null,
      "depth": 1,
      "id": "cB",
      "label": "a",
      "score": null,
      "source_file": "Maps/a.canvas",
      "text": "[Canvas: a]\nSlack variables\n\nConnections:\n  <- [leads to] A\n  -> C"
    },
    {
      "canvas_degree": 2,
      "canvas_group": null,
      "canvas_node_id": "C",
      "canvas_part": null,
      "depth": 1,
      "id": "cC",
      "label": "a",
      "score": null,
      "source_file": "Maps/a.canvas",
      "text": "[Canvas: a]\nHinge loss\n\nConnections:\n  <- B\n  -> [closes] A"
    }
  ],
  "stats": {
    "caps": {
      "depth": 2,
      "max_per_hop": 8,
      "max_total": 40
    },
    "caps_hit": [],
    "cross_edges": 4,
    "dangling_edges": 1,
    "edges": 2,
    "edges_labelled": true,
    "hops": 2,
    "nodes": 3,
    "seeds": 1,
    "truncated_at": null,
    "visited": 3
  },
  "tree": [
    {
      "child": "cB",
      "depth": 1,
      "direction": "->",
      "label": "leads to",
      "parent": "cA"
    },
    {
      "child": "cC",
      "depth": 1,
      "direction": "<-",
      "label": "closes",
      "parent": "cA"
    }
  ]
}"""


def test_flags_off_walk_matches_the_pre_slice_snapshot():
    store = _store(_snapshot_rows())
    exp = GraphExpander(retriever=FakeRetriever(store), reranker=None)
    out = exp.expand([_seed(store, "cA")], depth=2)
    assert json.dumps(out, sort_keys=True, indent=2) == EXPECTED
    out = exp.expand([_seed(store, "cA")], depth=2, bridges=False, cross_canvas=False,
                     wikilinks=False, direction="both", edge_types=None)
    assert json.dumps(out, sort_keys=True, indent=2) == EXPECTED
    # a per-call False beats a config True (the bench's G-legacy relies on it)
    on = GraphExpander(retriever=FakeRetriever(store), reranker=None,
                       bridges=True, cross_canvas=True, wikilinks=True)
    out = on.expand([_seed(store, "cA")], depth=2, bridges=False, cross_canvas=False,
                    wikilinks=False, direction="both", edge_types=None)
    assert json.dumps(out, sort_keys=True, indent=2) == EXPECTED


# ---- options ----

def test_bad_options_raise():
    for kw, key in (({"direction": "sideways"}, "graph.direction"),
                    ({"edge_types": []}, "graph.edge_types"),
                    ({"target_max_chunks": 0}, "graph.target_max_chunks")):
        with pytest.raises(ValueError, match=key):
            GraphExpander(retriever=FakeRetriever(_store([])), **kw)
    exp = GraphExpander(retriever=FakeRetriever(_store([])))
    for kw, key in (({"direction": "sideways"}, "graph.direction"),
                    ({"edge_types": []}, "graph.edge_types")):
        with pytest.raises(ValueError, match=key):                  # a per-call value, checked the same way
            exp.expand([], **kw)


def test_from_config_reads_every_navigator_key(tmp_path):
    from src.utils.config_loader import Config
    graph = {"bridges": True, "cross_canvas": True, "wikilinks": True, "direction": "down",
             "edge_types": ["Prerequisite of", "uses"], "target_max_chunks": 2, "bridge_max_total": 7,
             "same_as_min_chars": 6, "same_as_max_canvases": 9}
    exp = GraphExpander.from_config(Config({"graph": graph}, tmp_path), FakeRetriever(_store([])))
    assert (exp.bridges, exp.cross_canvas, exp.wikilinks, exp.direction) == (True, True, True, "down")
    assert exp.edge_types == {"prerequisite-of", "uses"}
    assert (exp.target_max_chunks, exp.bridge_max_total, exp.same_as_min_chars,
            exp.same_as_max_canvases) == (2, 7, 6, 9)


def test_navigator_helpers():
    from src.retrieval.graph_expand import _forms, _link_target, _norm_label, title_key
    assert _norm_label(" Prerequisite of ") == _norm_label("prerequisite_of") == "prerequisite-of"
    assert _forms("Books/Café.pdf") == ["Books/Café.pdf", "Books\\Café.pdf",
                                             "Books/Café.pdf", "Books\\Café.pdf"]
    assert title_key("[Canvas: a]\nSoft Margin SVM:\nmore\n\nConnections:\n  -> x") == "soft margin svm"
    assert title_key("[Canvas: a]\n\n\nConnections:\n  -> [leads to] x") == ""
    assert _link_target("Folder/Other#H1#Kernels|alias") == ("Other", "Kernels", "Folder")
    assert _link_target("Other.md#^block") == ("Other", "", "")
    assert _link_target("Lecture 1.2") == ("Lecture 1.2", "", "")
    assert _link_target("img.png") is None and _link_target("#Heading") is None


def test_title_key_deletes_in_word_marks():
    """An Armenian question mark sits inside its word: "ինչո՞ւ է այդպես" is
    "ինչու է այդպես" with emphasis, and an apostrophe joins "don't" the same way."""
    from src.retrieval.graph_expand import title_key
    armenian = "\u056b\u0576\u0579\u0578{}\u0582 \u0567 \u0561\u0575\u0564\u057a\u0565\u057d"
    assert title_key("[Canvas: x]\n" + armenian.format("\u055e")) == title_key(
        "[Canvas: y]\n" + armenian.format("")) == armenian.format("")
    assert title_key("[Canvas: x]\nDon't stop") == title_key("[Canvas: x]\ndon\u2019t stop") == "dont stop"
    assert title_key("[Canvas: x]\nsoft-margin SVM") == "soft margin svm"     # every other mark still spaces


# ---- E4: typed traversal ----

def test_edge_types_follow_only_listed_labels():
    store = _store([
        _canvas_chunk("cA", "Maps/t.canvas", "A", "Kernels",
                      [("B", "->", "Prerequisite of"), ("C", "->", "uses"), ("D", "->", "")]),
        _canvas_chunk("cB", "Maps/t.canvas", "B", "SVM", [("A", "<-", "Prerequisite of")]),
        _canvas_chunk("cC", "Maps/t.canvas", "C", "libsvm", [("A", "<-", "uses")]),
        _canvas_chunk("cD", "Maps/t.canvas", "D", "Notes", [("A", "<-", "")]),
    ])
    exp = GraphExpander(retriever=FakeRetriever(store))
    out = exp.expand([_seed(store, "cA")], depth=1, edge_types=["prerequisite-of"])
    assert [n["id"] for n in out["nodes"]] == ["cA", "cB"]
    assert [(t["parent"], t["child"], t["label"]) for t in out["tree"]] == [("cA", "cB", "Prerequisite of")]
    assert out["stats"]["filtered_edges"] == 2                       # "uses" and the unlabelled edge


def test_direction_up_follows_incoming_only_and_skips_undirected():
    """B's edges as T2 stores them from B's side: A->B is '<-', B->C '->', an
    arrowless B-E '--', a double-headed B-D '<>'. L predates canvas_edges, so its
    one edge is legacy (undirected, '--')."""
    from src.retrieval.graph_expand import _passes
    legacy = ("cL", ("[Canvas: d]\nLegacy", {"source_file": "Maps/d.canvas", "file_type": "canvas",
                                            "canvas_node_id": "L", "canvas_neighbors": ["M"]}))
    store = _store([
        _canvas_chunk("cB", "Maps/d.canvas", "B", "Middle",
                      [("A", "<-", ""), ("C", "->", ""), ("E", "--", ""), ("D", "<>", "")]),
        *(_canvas_chunk(f"c{n}", "Maps/d.canvas", n, f"Node {n}", []) for n in "ACDEM"),
        legacy])
    exp = GraphExpander(retriever=FakeRetriever(store))
    seeds = [_seed(store, "cB"), _seed(store, "cL")]
    up = exp.expand(seeds, depth=1, direction="up")
    assert sorted(n["id"] for n in up["nodes"]) == ["cA", "cB", "cD", "cL"]
    assert up["stats"]["filtered_edges"] == 3                        # B->C, B-E and the legacy L-M
    down = exp.expand(seeds, depth=1, direction="down")
    assert sorted(n["id"] for n in down["nodes"]) == ["cB", "cC", "cD", "cL"]
    both = exp.expand(seeds, depth=1)
    assert sorted(n["id"] for n in both["nodes"]) == ["cA", "cB", "cC", "cD", "cE", "cL", "cM"]
    assert _passes("", "--", "same-as", {"uses"}, "up")              # identity passes any filter


# ---- E1: bridges ----

def _pdf_chunk(cid, source, page):
    """A pdf chunk as pdf_loader writes it: one page, page_start == page_end."""
    return cid, (f"page {page} of {source}",
                 {"source_file": source, "file_type": "pdf", "page_start": page, "page_end": page})


def _note_chunk(cid, source, heading, **extra):
    return cid, (f"{heading} in {source}",
                 {"source_file": source, "file_type": "note", "heading": heading, **extra})


def _with_files(row, *entries):
    """canvas_edge_files as T2's loader writes it, from (node id, vault path, subpath)."""
    cid, (text, meta) = row
    return cid, (text, {**meta, "canvas_edge_files": [encode_canvas_edge(*e) for e in entries]})


def _bridged(out):
    return [n["id"] for n in out["nodes"] if n.get("kind") == "bridge"]


def _bridging(rows, **kw):
    store = _store(rows)
    return store, GraphExpander(retriever=FakeRetriever(store), bridges=True, **kw)


def test_bridges_on_in_config_reach_the_pdf_page_a_file_node_cites(tmp_path):
    from src.utils.config_loader import Config
    store = _store([
        _with_files(_canvas_chunk("cA", "Maps/a.canvas", "A", "Soft margin SVM",
                                  [("F1", "->", "cites")], file_refs=["Books/svm.pdf"]),
                    ("F1", "Books/svm.pdf", "#page=5")),
        *(_pdf_chunk(f"p{n}", "Books\\svm.pdf", n) for n in (4, 5, 6))])
    exp = GraphExpander.from_config(Config({"graph": {"bridges": True}}, tmp_path), FakeRetriever(store))
    out = exp.expand([_seed(store, "cA")])
    assert "p5" in _bridged(out)
    assert "p4" not in _bridged(out) and "p6" not in _bridged(out)
    assert out["stats"]["dangling_edges"] == 0       # the file-node edge is the bridge, not a dangling hop


def test_bridge_md_heading_takes_only_that_section():
    store, exp = _bridging([
        _with_files(_canvas_chunk("cA", "Maps/a.canvas", "A", "Kernels", [("F1", "->", "explained in")]),
                    ("F1", "Notes/svm.md", "#Kernels")),
        _note_chunk("n1", "Notes\\svm.md", "Intro"),
        _note_chunk("n2", "Notes\\svm.md", "KERNELS"),
        _note_chunk("n3", "Notes\\svm.md", "kernels"),
        _note_chunk("n4", "Notes\\svm.md", "Soft margin")])
    out = exp.expand([_seed(store, "cA")], depth=1)
    assert _bridged(out) == ["n2", "n3"]
    assert [(t["child"], t["label"], t["kind"]) for t in out["tree"]] == [
        ("n2", "explained in", "bridge"), ("n3", "explained in", "bridge")]


def test_bridge_heading_missing_is_counted_not_widened():
    store, exp = _bridging([
        _with_files(_canvas_chunk("cA", "Maps/a.canvas", "A", "Duality", [("F1", "->", "")]),
                    ("F1", "Notes/svm.md", "#Duality")),
        _note_chunk("n1", "Notes\\svm.md", "Intro"),
        _note_chunk("n2", "Notes\\svm.md", "Kernels")])
    out = exp.expand([_seed(store, "cA")], depth=1)
    assert _bridged(out) == []
    assert out["stats"]["bridge_heading_missing"] == 1
    assert out["stats"]["bridge_targets_without_chunks"] == 0


def test_bridge_pdf_without_page_is_counted_not_bridged():
    store, exp = _bridging([
        _with_files(_canvas_chunk("cA", "Maps/a.canvas", "A", "SVM", [("F1", "->", ""), ("F2", "->", "")]),
                    ("F1", "Books/svm.pdf", ""), ("F2", "Books/svm.pdf", "#Kernels")),
        _pdf_chunk("p1", "Books\\svm.pdf", 1),
        _pdf_chunk("p2", "Books\\svm.pdf", 2)])
    out = exp.expand([_seed(store, "cA")], depth=1)
    assert _bridged(out) == []
    assert out["stats"]["bridge_pdf_without_page"] == 2


def test_bridge_pdf_page_rows_are_sorted_in_python():
    store, exp = _bridging([
        _with_files(_canvas_chunk("cA", "Maps/a.canvas", "A", "SVM", [("F1", "->", "")]),
                    ("F1", "Books/svm.pdf", "#page=5")),
        _pdf_chunk("p5b", "Books\\svm.pdf", 5),
        _pdf_chunk("p5a", "Books\\svm.pdf", 5)], target_max_chunks=1)
    out = exp.expand([_seed(store, "cA")], depth=1)
    assert _bridged(out) == ["p5a"]


def test_bridge_pdf_page_is_tested_in_python_never_sent_to_the_store():
    """The store gets source_file alone: a page_start/page_end range in the where makes
    Chroma compare every PDF chunk's pages (seconds per bridge on the live store), while
    a source_file $in is an index lookup. A chunk whose pages span the cited page is on
    it; one with no page bounds never is, as Chroma's $lte/$gte skip a missing key."""
    store, exp = _bridging([
        _with_files(_canvas_chunk("cA", "Maps/a.canvas", "A", "SVM", [("F1", "->", "")]),
                    ("F1", "Books/svm.pdf", "#page=5")),
        _pdf_chunk("a5", "Books\\svm.pdf", 5),
        _pdf_chunk("b4", "Books\\svm.pdf", 4),
        ("z36", ("pages 3-6", {"source_file": "Books\\svm.pdf", "file_type": "pdf",
                               "page_start": 3, "page_end": 6})),
        ("c0", ("front matter", {"source_file": "Books\\svm.pdf", "file_type": "pdf"}))])
    out = exp.expand([_seed(store, "cA")], depth=1)
    assert _bridged(out) == ["z36", "a5"]
    sent = json.dumps([c["where"] for c in store.calls])
    assert "page_start" not in sent and "page_end" not in sent


def test_bridges_are_terminal_and_capped():
    """o2 carries an edge to node Z, and a Z chunk sits in o2's file: a bridge is
    never expanded, so Z is never reached."""
    store, exp = _bridging([
        _with_files(_canvas_chunk("cA", "Maps/a.canvas", "A", "Hub",
                                  [("F1", "->", ""), ("F2", "->", ""), ("F3", "->", "")]),
                    ("F1", "Notes/one.md", ""), ("F2", "Notes/two.md", ""), ("F3", "Notes/three.md", "")),
        _note_chunk("o1", "Notes\\one.md", "a"),
        _note_chunk("o2", "Notes\\one.md", "b", canvas_node_id="Y",
                    canvas_edges=[encode_canvas_edge("Z", "->", "next")]),
        _note_chunk("o3", "Notes\\one.md", "c"),
        _note_chunk("o4", "Notes\\one.md", "d"),
        _note_chunk("oZ", "Notes\\one.md", "z", canvas_node_id="Z"),
        _note_chunk("t1", "Notes\\two.md", "a"),
        _note_chunk("h1", "Notes\\three.md", "a")], target_max_chunks=4, bridge_max_total=2)
    out = exp.expand([_seed(store, "cA")], depth=2)
    assert _bridged(out) == ["o1", "o2"]
    assert "bridge_max_total" in out["stats"]["caps_hit"]
    assert "oZ" not in [n["id"] for n in out["nodes"]]
    assert all("oZ" not in (t["parent"], t["child"]) for t in out["tree"])
    assert out["stats"]["nodes"] == 1


def test_chunk_without_canvas_edge_files_is_unavailable_not_guessed():
    store, exp = _bridging([
        _canvas_chunk("cA", "Maps/a.canvas", "A", "SVM", [("F1", "->", "cites")],
                      file_refs=["Books/svm.pdf"]),
        _pdf_chunk("p1", "Books\\svm.pdf", 1)])
    out = exp.expand([_seed(store, "cA")], depth=1)
    assert _bridged(out) == []
    assert out["stats"]["bridges_unavailable"] == 1
    out = exp.expand([_seed(store, "cA")], depth=1, bridges=False, cross_canvas=True)
    assert out["stats"]["bridges_unavailable"] == 1                  # its file-node hops are skipped too (§3)


def test_source_line_bridge_reaches_the_cited_page():
    store, exp = _bridging([
        _with_files(_canvas_chunk("cA", "Maps/a.canvas", "A", "SVM\nSource: Books/svm.pdf p. 5", []),
                    ("", "Books/svm.pdf", "#page=5")),
        *(_pdf_chunk(f"p{n}", "Books\\svm.pdf", n) for n in (4, 5, 6))])
    out = exp.expand([_seed(store, "cA")], depth=1)
    assert _bridged(out) == ["p5"]
    assert out["tree"] == [{"parent": "cA", "child": "p5", "label": "source",
                            "direction": "->", "depth": 1, "kind": "bridge"}]


def test_bridge_follows_a_relocated_pdf():
    from src.index_ops import relocate_meta
    rows = [_with_files(_canvas_chunk("cA", "Maps/a.canvas", "A", "SVM", [("F1", "->", "cites")],
                                      file_refs=["Old/Book.pdf"]), ("F1", "Old/Book.pdf", "#page=5")),
            _pdf_chunk("p5", "Old\\Book.pdf", 5)]
    moved = [(cid, (text, relocate_meta(meta, "Old\\Book.pdf", "New\\Book.pdf"))) for cid, (text, meta) in rows]
    assert all(meta is not None for _, (_, meta) in moved)
    store, exp = _bridging(moved)
    out = exp.expand([_seed(store, "cA")], depth=1)
    assert [(n["id"], n["source_file"]) for n in out["nodes"] if n.get("kind") == "bridge"] == [
        ("p5", "New\\Book.pdf")]
    assert out["stats"]["bridge_targets_without_chunks"] == 0


def test_bridge_matches_an_nfd_source_file_and_counts_it():
    store, exp = _bridging([
        _with_files(_canvas_chunk("cA", "Maps/a.canvas", "A", "SVM", [("F1", "->", "")]),
                    ("F1", "Books/Café.pdf", "#page=3")),
        _pdf_chunk("p3", "Books\\Café.pdf", 3)])
    out = exp.expand([_seed(store, "cA")], depth=1)
    assert _bridged(out) == ["p3"]
    assert out["stats"]["bridges_matched_after_normalised"] == 1


def test_a_chunk_is_listed_once_and_a_bridge_never_routes_to_a_seed():
    """cA's file node cites Other.md, and s1 links [[Other]]: o1 is a bridge first,
    then a walk node, and is listed once. A bridge onto a seed is an edge onto it,
    never a path to it."""
    rows = [_with_files(_canvas_chunk("cA", "Maps/a.canvas", "A", "Hub", [("F1", "->", "")]),
                        ("F1", "Notes/Other.md", "")),
            _md("s1", "Notes\\s.md", "a", ["Other"]), _md("o1", "Notes\\Other.md", "x")]
    store = _store(rows)
    exp = GraphExpander(retriever=FakeRetriever(store), bridges=True, wikilinks=True)
    out = exp.expand([_seed(store, "cA"), _seed(store, "s1")], depth=1)
    ids = [n["id"] for n in out["nodes"]]
    assert len(ids) == len(set(ids)) and sorted(ids) == ["cA", "o1", "s1"]
    assert out["stats"]["nodes"] + out["stats"]["bridges"] == 3
    out = exp.expand([_seed(store, "o1"), _seed(store, "cA")], depth=0)
    assert out["tree"] == [] and out["paths"] == []
    assert [(c["parent"], c["child"], c["kind"]) for c in out["cross_edges"]] == [("cA", "o1", "bridge")]


# ---- E2: cross-canvas hops ----

def _crossing(rows, **kw):
    store = _store(rows)
    return store, GraphExpander(retriever=FakeRetriever(store), cross_canvas=True, **kw)


def _kinds(out, kind):
    return [(t["parent"], t["child"]) for t in out["tree"] if t["kind"] == kind]


def _pairs(out):
    return [(t["parent"], t["child"]) for t in out["tree"]]


def test_same_as_joins_two_canvases_on_normalised_title():
    store, exp = _crossing([
        _canvas_chunk("cA", "Maps/a.canvas", "A", "Soft Margin SVM:", []),
        _canvas_chunk("cB", "Maps/b.canvas", "B", "soft-margin svm", []),
        _canvas_chunk("cC", "Maps/b.canvas", "C", "Hinge loss", [])])
    out = exp.expand([_seed(store, "cA")], depth=1)
    assert _kinds(out, "same-as") == [("cA", "cB")]
    assert next(n for n in out["nodes"] if n["id"] == "cB")["source_file"] == "Maps/b.canvas"
    assert out["stats"]["virtual_edges"]["same-as"] == 1
    up = exp.expand([_seed(store, "cA")], depth=1, direction="up", edge_types=["uses"])
    assert _kinds(up, "same-as") == [("cA", "cB")]                   # identity passes any typed walk


def test_title_comes_from_the_body_not_the_footer():
    def part(row, n):
        cid, (text, meta) = row
        return cid, (text, {**meta, "canvas_part": n})
    rows = [_canvas_chunk("e1", "Maps/a.canvas", "E1", "", [("X", "->", "leads to")]),
            _canvas_chunk("e2", "Maps/b.canvas", "E2", "", [("Y", "->", "leads to")]),
            _canvas_chunk("k1", "Maps/a.canvas", "K", "Kernel trick", []),
            part(_canvas_chunk("k2", "Maps/c.canvas", "K2", "Kernel trick", []), 2)]
    store, exp = _crossing(rows)
    out = exp.expand([_seed(store, "e1"), _seed(store, "k1")], depth=1)
    assert _kinds(out, "same-as") == []
    store, exp = _crossing([*rows, _canvas_chunk("k3", "Maps/d.canvas", "K3", "Kernel trick", [])])
    out = exp.expand([_seed(store, "k1")], depth=1)
    assert _kinds(out, "same-as") == [("k1", "k3")]


def test_generic_titles_join_nothing():
    rows = [_canvas_chunk(f"c{i}", f"Maps/{i}.canvas", "P", "Python", []) for i in range(5)]
    store, exp = _crossing(rows, same_as_max_canvases=4)
    assert exp.expand([_seed(store, "c0")], depth=1)["tree"] == []
    store, exp = _crossing(rows[:4], same_as_max_canvases=4)       # at the cap it still joins
    assert len(_kinds(exp.expand([_seed(store, "c0")], depth=1), "same-as")) == 3
    store, exp = _crossing([_canvas_chunk("s1", "Maps/1.canvas", "S", "SVM", []),
                            _canvas_chunk("s2", "Maps/2.canvas", "S", "SVM", [])])
    assert exp.expand([_seed(store, "s1")], depth=1)["tree"] == []   # shorter than same_as_min_chars


def test_shares_bridge_reaches_the_other_canvas_citing_the_same_file():
    store, exp = _crossing([
        _with_files(_canvas_chunk("cA", "Maps/a.canvas", "A", "Margins", [("F1", "->", "")]),
                    ("F1", "Books/svm.pdf", "#page=5")),
        _with_files(_canvas_chunk("cB", "Maps/b.canvas", "B", "Kernels", [("F9", "->", "")]),
                    ("F9", "Books/svm.pdf", "")),
        _with_files(_canvas_chunk("cC", "Maps/c.canvas", "C", "Duality\nSource: Books/svm.pdf p. 9", []),
                    ("", "Books/svm.pdf", "#page=9"))])
    out = exp.expand([_seed(store, "cA")], depth=1)
    assert _kinds(out, "shares-bridge") == [("cA", "cB")]
    assert [n["id"] for n in out["nodes"]] == ["cA", "cB"]
    assert out["stats"]["dangling_edges"] == 0       # the file-node edge is the hop, not a dangling edge
    assert exp.expand([_seed(store, "cC")], depth=1)["tree"] == []   # a Source line joins no canvas


def test_file_node_to_canvas_opens_its_hubs_by_degree():
    store, exp = _crossing([
        _with_files(_canvas_chunk("ix", "Maps/index.canvas", "I", "Index", [("F", "->", "")]),
                    ("F", "Maps/a.canvas", "")),
        _canvas_chunk("h1", "Maps/a.canvas", "H1", "Leaf", [("H3", "<-", "")]),
        _canvas_chunk("h3", "Maps/a.canvas", "H3", "Hub", [("H1", "->", ""), ("H2", "->", ""), ("X", "->", "")]),
        _canvas_chunk("h2", "Maps/a.canvas", "H2", "Middle", [("H3", "<-", ""), ("H1", "->", "")])])
    out = exp.expand([_seed(store, "ix")], depth=1, max_per_hop=2)
    assert sorted(_kinds(out, "opens-canvas")) == [("ix", "h2"), ("ix", "h3")]
    assert out["stats"]["virtual_edges_dropped"] == 1
    assert out["stats"]["virtual_edges"]["opens-canvas"] == 2


def test_virtual_edges_over_budget_are_counted_not_capped():
    store, exp = _crossing([
        _canvas_chunk("cA", "Maps/a.canvas", "A", "Perceptron", [("B", "->", "leads to")]),
        _canvas_chunk("cB", "Maps/a.canvas", "B", "Logistic regression", [("A", "<-", "leads to")]),
        _canvas_chunk("cX", "Maps/x.canvas", "X", "Perceptron", []),
        _canvas_chunk("cY", "Maps/y.canvas", "Y", "Perceptron", [])])
    out = exp.expand([_seed(store, "cA")], depth=1, max_per_hop=1)
    assert [t["child"] for t in out["tree"]] == ["cB"]
    assert out["stats"]["virtual_edges_dropped"] == 2
    assert "max_per_hop" not in out["stats"]["caps_hit"]


def test_cross_canvas_cycle_terminates():
    store, exp = _crossing([
        _canvas_chunk("cA", "Maps/a.canvas", "A", "Hard margin SVM", []),
        _canvas_chunk("cB", "Maps/b.canvas", "B", "Hard margin SVM", [])])
    out = exp.expand([_seed(store, "cA")], depth=3)
    assert [n["id"] for n in out["nodes"]] == ["cA", "cB"]
    assert out["stats"]["visited"] == 2
    assert [(c["parent"], c["child"], c["kind"]) for c in out["cross_edges"]] == [("cB", "cA", "same-as")]


def test_two_same_titled_nodes_in_one_canvas_are_not_same_as():
    store, exp = _crossing([_canvas_chunk("c1", "Maps/a.canvas", "A", "Kernel trick", []),
                            _canvas_chunk("c2", "Maps/a.canvas", "B", "Kernel trick", [])])
    out = exp.expand([_seed(store, "c1")], depth=1)
    assert out["tree"] == [] and out["cross_edges"] == []


# ---- E3: wikilinks ----

def _linking(rows, **kw):
    store = _store(rows)
    return store, GraphExpander(retriever=FakeRetriever(store), wikilinks=True, **kw)


def _md(cid, source, heading, links=()):
    return _note_chunk(cid, source, heading, filename=Path(source.replace("\\", "/")).stem,
                       wikilinks=list(links))


def test_wikilinks_follow_outgoing_links_between_notes():
    store, exp = _linking([
        _md("s1", "Notes\\svm.md", "Intro", ["Other#Kernels", "img.png"]),
        _md("o1", "Notes\\Other.md", "Intro"),
        _md("o2", "Notes\\Other.md", "Kernels")])
    out = exp.expand([_seed(store, "s1")], depth=1)
    assert [(t["parent"], t["child"], t["label"], t["kind"]) for t in out["tree"]] == [
        ("s1", "o2", "links-to", "links-to")]
    assert out["stats"]["wikilinks_unresolved"] == 0                  # img.png is not a note link
    assert out["stats"]["virtual_edges"]["links-to"] == 1


def test_two_parents_linking_one_note_yield_one_tree_entry_each():
    store, exp = _linking([
        _md("n1", "Notes\\one.md", "a", ["Other"]),
        _md("n2", "Notes\\two.md", "a", ["Other"]),
        _md("o1", "Notes\\Other.md", "x"),
        _md("o2", "Notes\\Other.md", "y")])
    out = exp.expand([_seed(store, "n1"), _seed(store, "n2")], depth=1)
    assert sorted((t["parent"], t["child"]) for t in out["tree"]) == [
        ("n1", "o1"), ("n1", "o2"), ("n2", "o1"), ("n2", "o2")]
    assert out["stats"]["visited"] == 3
    ids = [n["id"] for n in out["nodes"]]
    assert sorted(ids) == ["n1", "n2", "o1", "o2"] and len(ids) == len(set(ids))
    assert out["cross_edges"] == []


def test_full_wikilinks_list_is_counted_truncated():
    full = [f"Missing {i}" for i in range(20)]
    store, exp = _linking([
        _md("f", "Notes\\full.md", "a", full),
        _md("s", "Notes\\short.md", "a", full[:3])])
    assert exp.expand([_seed(store, "f")], depth=1)["stats"]["wikilinks_truncated"] == 1
    assert exp.expand([_seed(store, "s")], depth=1)["stats"]["wikilinks_truncated"] == 0
    text, meta = store.rows["f"]
    joined = RetrievedDoc(id="f", text=text, metadata={**meta, "wikilinks": ", ".join(full)})   # Chroma's shape
    assert exp.expand([joined], depth=1)["stats"]["wikilinks_truncated"] == 1


def test_lowercase_link_to_a_capitalised_note_is_unresolved():
    store, exp = _linking([
        _md("s", "Notes\\svm.md", "a", ["other"]),
        _md("o", "Notes\\Other.md", "a")])
    out = exp.expand([_seed(store, "s")], depth=1)
    assert out["tree"] == []
    assert out["stats"]["wikilinks_unresolved"] == 1


def test_ambiguous_wikilink_is_skipped_and_counted():
    store, exp = _linking([
        _md("s", "Notes\\svm.md", "a", ["Other"]),
        _md("o1", "A\\Other.md", "a"),
        _md("o2", "B\\Other.md", "a")])
    out = exp.expand([_seed(store, "s")], depth=1)
    assert out["tree"] == []
    assert out["stats"]["wikilinks_ambiguous"] == 1


def _sections():
    """n1 links [[Other#Alpha]], n2 [[Other#Beta]]; Other.md holds one chunk per section."""
    return [_md("n1", "Notes\\one.md", "a", ["Other#Alpha"]), _md("n2", "Notes\\two.md", "a", ["Other#Beta"]),
            _md("oA", "Notes\\Other.md", "Alpha"), _md("oB", "Notes\\Other.md", "Beta")]


def test_heading_links_into_one_note_reach_each_named_section():
    store, exp = _linking(_sections())
    out = exp.expand([_seed(store, "n1"), _seed(store, "n2")], depth=1)
    assert sorted(_pairs(out)) == [("n1", "oA"), ("n2", "oB")]
    assert [n["id"] for n in out["nodes"]] == ["n1", "n2", "oA", "oB"]
    assert out["stats"]["visited"] == 3 and out["cross_edges"] == []          # one note, one node
    store, exp = _linking([_md("n", "Notes\\n.md", "a", ["Other#Alpha", "Other#Beta"]), *_sections()[2:]])
    assert sorted(_pairs(exp.expand([_seed(store, "n")], depth=1))) == [("n", "oA"), ("n", "oB")]


def test_a_heading_link_into_a_note_the_walk_holds_adds_its_section():
    store, exp = _linking(_sections())
    out = exp.expand([_seed(store, "n2"), _seed(store, "oA")], depth=1)        # oA: a search hit, say
    assert _pairs(out) == [("n2", "oB")] and out["cross_edges"] == []
    out = exp.expand([_seed(store, "n1"), _seed(store, "oA")], depth=1)
    assert out["tree"] == [] and [(c["parent"], c["child"]) for c in out["cross_edges"]] == [("n1", "oA")]


def test_a_comma_in_a_link_target_survives_chromas_join():
    """Chroma holds wikilinks ', '-joined (Embedder._clean_meta), which cannot be split
    back for [[Smith, 2020]]; the sparse lane keeps the parser's list verbatim."""
    from src.embeddings.embedder import Embedder
    rows = [_md("s1", "Notes\\s.md", "a", ["Smith, 2020", "Kernels"]), _md("t", "Notes\\Smith, 2020.md", "a"),
            _md("sm", "Notes\\Smith.md", "a"), _md("k", "Notes\\Kernels.md", "a")]
    chroma = WhereCollection({cid: (text, Embedder._clean_meta(meta)) for cid, (text, meta) in rows})
    assert chroma.rows["s1"][1]["wikilinks"] == "Smith, 2020, Kernels"
    exp = GraphExpander(retriever=FakeRetriever(chroma, sparse=dict(rows)), wikilinks=True)
    out = exp.expand([_seed(chroma, "s1")], depth=1)
    assert sorted(_pairs(out)) == [("s1", "k"), ("s1", "t")]                    # never Smith.md
    assert out["stats"]["wikilinks_unresolved"] == 0
    exp = GraphExpander(retriever=FakeRetriever(chroma, sparse={}), wikilinks=True)
    with pytest.raises(RuntimeError, match="rebuild_bm25"):                    # drift: no lossless copy
        exp.expand([_seed(chroma, "s1")], depth=1)


def test_a_sparse_link_list_older_than_chromas_raises():
    """An md id hashes a chunk's first 500 chars and its links are file-wide, so a link added
    lower down keeps the id; the sparse union keeps the FIRST row per id, so its list can be
    older than the string Chroma was upserted with. The walk must not follow the old list."""
    from src.embeddings.embedder import Embedder
    rows = [_md("s1", "Notes\\s.md", "a", ["Other", "Third", "New"]), _md("o", "Notes\\Other.md", "a"),
            _md("t", "Notes\\Third.md", "a"), _md("n", "Notes\\New.md", "a")]
    chroma = WhereCollection({cid: (text, Embedder._clean_meta(meta)) for cid, (text, meta) in rows})
    text, meta = dict(rows)["s1"]
    stale = {**dict(rows), "s1": (text, {**meta, "wikilinks": ["Other", "Third"]})}
    exp = GraphExpander(retriever=FakeRetriever(chroma, sparse=stale), wikilinks=True)
    with pytest.raises(RuntimeError, match="FIRST row"):
        exp.expand([_seed(chroma, "s1")], depth=1)


def test_a_table_escaped_link_resolves():
    from src.ingestion.obsidian_parser import extract_wikilinks
    links = extract_wikilinks("| 1 | [[Kernel Methods\\|Kernels]] |\n| 2 | [[SVM#Duality\\|dual]] |")
    assert links == ["Kernel Methods\\", "SVM#Duality\\"]                      # what the parser stores
    store, exp = _linking([_md("s", "Notes\\s.md", "a", links), _md("k", "Notes\\Kernel Methods.md", "a"),
                           _md("d", "Notes\\SVM.md", "Duality")])
    out = exp.expand([_seed(store, "s")], depth=1)
    assert sorted(_pairs(out)) == [("s", "d"), ("s", "k")]
    assert out["stats"]["wikilinks_unresolved"] == 0


def test_max_total_caps_the_parts_of_a_linked_note_and_says_so():
    store, exp = _linking([_md("s", "Notes\\s.md", "a", ["Other"]),
                           *(_md(f"o{i}", "Notes\\Other.md", f"h{i}") for i in range(4))])
    out = exp.expand([_seed(store, "s")], depth=1, max_total=2)
    assert [n["id"] for n in out["nodes"]] == ["s", "o0"] and _pairs(out) == [("s", "o0")]
    assert "max_total" in out["stats"]["caps_hit"]


def test_links_a_typed_walk_rejects_are_counted_as_filtered():
    store, exp = _linking([_md("s1", "Notes\\s.md", "a", ["Other", "Third", "img.png"]),
                           _md("o", "Notes\\Other.md", "a")])
    for kw in ({"direction": "up"}, {"edge_types": ["uses"]}):
        out = exp.expand([_seed(store, "s1")], depth=1, **kw)
        assert out["tree"] == [] and out["stats"]["filtered_edges"] == 2, kw    # img.png is no link


def test_a_path_link_keeps_its_folder():
    rows = [_md("s", "Notes\\s.md", "a", ["A/Other"]), _md("oa", "A\\Other.md", "a"), _md("ob", "B\\Other.md", "a")]
    store, exp = _linking(rows)
    out = exp.expand([_seed(store, "s")], depth=1)
    assert _pairs(out) == [("s", "oa")] and out["stats"]["wikilinks_ambiguous"] == 0
    store, exp = _linking([rows[0], rows[2]])                                  # only B's note is indexed
    out = exp.expand([_seed(store, "s")], depth=1)
    assert out["tree"] == [] and out["stats"]["wikilinks_unresolved"] == 1
    # Obsidian's relative link format: from the linking note's own folder, a whole vault path.
    # Three Other.md, so a bare-stem guess would be ambiguous; ../../ climbs out of the vault.
    store, exp = _linking([_md("s", "A\\s.md", "a", ["../B/Other", "./Other", "../../Other"]),
                           _md("r", "r.md", "a", ["./Other"]), _md("oa", "A\\Other.md", "a"),
                           _md("ob", "B\\Other.md", "a"), _md("o", "Other.md", "a")])
    out = exp.expand([_seed(store, "s"), _seed(store, "r")], depth=1)
    assert sorted(_pairs(out)) == [("r", "o"), ("s", "oa"), ("s", "ob")]
    assert out["stats"]["wikilinks_unresolved"] == 1 and out["stats"]["wikilinks_ambiguous"] == 0


def test_a_link_resolves_to_the_note_not_a_same_stem_canvas_or_pdf():
    store, exp = _linking([
        _md("s", "Notes\\s.md", "a", ["SVM"]), _md("n", "Notes\\SVM.md", "a"),
        ("c", ("[Canvas: SVM]\nx", {"source_file": "Maps/SVM.canvas", "filename": "SVM",
                                    "file_type": "canvas", "canvas_node_id": "X"})),
        ("p", ("page 1", {"source_file": "Books\\SVM.pdf", "filename": "SVM", "file_type": "pdf",
                          "page_start": 1, "page_end": 1}))])
    out = exp.expand([_seed(store, "s")], depth=1)
    assert _pairs(out) == [("s", "n")] and out["stats"]["wikilinks_ambiguous"] == 0


def test_a_note_linking_itself_adds_nothing():
    store, exp = _linking([_md("s", "Notes\\SVM.md", "a", ["SVM"]), _md("s2", "Notes\\SVM.md", "b", ["SVM"])])
    out = exp.expand([_seed(store, "s")], depth=1)
    assert out["tree"] == [] and out["cross_edges"] == []
    assert out["stats"]["wikilinks_unresolved"] == out["stats"]["wikilinks_ambiguous"] == 0


# ---- extended output ----

def _source_chain():
    """cA -> cB by a canvas edge; each cites a pdf page through a Source: line."""
    return _bridging([
        _with_files(_canvas_chunk("cA", "Maps/a.canvas", "A", "SVM\nSource: Books/svm.pdf p. 5",
                                  [("B", "->", "leads to")]), ("", "Books/svm.pdf", "#page=5")),
        _with_files(_canvas_chunk("cB", "Maps/a.canvas", "B", "Kernels\nSource: Books/svm.pdf p. 7",
                                  [("A", "<-", "leads to")]), ("", "Books/svm.pdf", "#page=7")),
        _pdf_chunk("p5", "Books\\svm.pdf", 5),
        _pdf_chunk("p7", "Books\\svm.pdf", 7)])


def test_paths_carry_labels_from_seed_to_bridge():
    store, exp = _source_chain()
    paths = {p["to"]: p for p in exp.expand([_seed(store, "cA")], depth=1)["paths"]}
    assert paths["p5"] == {"to": "p5", "nodes": ["cA", "p5"],
                           "edges": [{"label": "source", "kind": "bridge", "direction": "->"}]}
    assert paths["p7"] == {"to": "p7", "nodes": ["cA", "cB", "p7"],
                           "edges": [{"label": "leads to", "kind": "canvas", "direction": "->"},
                                     {"label": "source", "kind": "bridge", "direction": "->"}]}


def test_extended_nodes_and_stats_name_kinds_and_options():
    store, exp = _source_chain()
    out = exp.expand([_seed(store, "cA")], depth=1)
    by_id = {n["id"]: n for n in out["nodes"]}
    assert [(i, n["kind"]) for i, n in by_id.items()] == [
        ("cA", "canvas"), ("cB", "canvas"), ("p5", "bridge"), ("p7", "bridge")]
    assert (by_id["p5"]["file_type"], by_id["p5"]["page_start"], by_id["p5"]["page_end"]) == ("pdf", 5, 5)
    assert "heading" in by_id["cA"]
    st = out["stats"]
    assert st["options"] == {"bridges": True, "cross_canvas": False, "wikilinks": False,
                             "direction": "both", "edge_types": None}
    assert st["caps"]["bridge_max_total"] == 20 and st["bridges"] == 2
    assert st["virtual_edges"] == {"same-as": 0, "shares-bridge": 0, "opens-canvas": 0, "links-to": 0}
    assert {"virtual_edges_dropped", "wikilinks_unresolved", "wikilinks_ambiguous",
            "wikilinks_truncated", "filtered_edges"} <= set(st)
    note = GraphExpander(retriever=FakeRetriever(_store([_md("n", "Notes\\n.md", "a")])), wikilinks=True)
    n_out = note.expand([RetrievedDoc(id="n", text="a", metadata=_md("n", "Notes\\n.md", "a")[1][1])], depth=1)
    assert n_out["nodes"][0]["kind"] == "note"
    assert exp.expand([_seed(store, "cA")], depth=1, edge_types=["leads-to", "Uses"])["stats"][
        "options"]["edge_types"] == ["leads-to", "uses"]
