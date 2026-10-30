"""RAGPipeline.graph_search: the one search + walk + union rerank path the bench
and POST /evidence_pack share. The pipeline is built with __new__ and stub
attributes, so no model, index or backend loads."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.ingestion.canvas_loader import encode_canvas_edge
from src.pipeline import RAGPipeline
from src.retrieval.retriever import RetrievedDoc


def _search_docs() -> list[RetrievedDoc]:
    """What search() returns: child texts, already reranked. n1 is a note child
    with a parent section, c1 a canvas chunk (a walk seed), p1 a pdf page."""
    return [
        RetrievedDoc("n1", "child of N", {"file_type": "note", "parent_id": "N"}, rerank_score=3.0),
        RetrievedDoc("c1", "canvas node", {"file_type": "canvas", "source_file": "a.canvas",
                                           "canvas_node_id": "A", "provenance": "generated"},
                     rerank_score=2.0),
        RetrievedDoc("p1", "pdf page", {"file_type": "pdf", "page_start": 4, "page_end": 4},
                     rerank_score=1.0),
    ]


class _Search:
    """search() stand-in: records every call's kwargs, returns fresh docs."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, question, **kw):
        self.calls.append(kw)
        docs = _search_docs()
        return docs[:kw.get("top_k") or 3], {
            "rerank_top_k": kw.get("top_k") or 3, "rerank_mode": "cross_encoder",
            "rerank_instruction": "prefer primary sources",
            "parent_context": bool(kw.get("parent_context")), "parent_swaps": 0,
            "parent_siblings_dropped": 0, "neighbor_context": False, "neighbors_added": 0,
            "timings": {"rerank": 4.0, "total": 10.0}}


class _Reranker:
    def __init__(self):
        self.calls: list[dict] = []

    def rerank(self, question, docs, top_k=None, mode=None, instruction=None):
        self.calls.append({"texts": [d.text for d in docs], "ids": [d.id for d in docs],
                           "top_k": top_k, "mode": mode, "instruction": instruction})
        return docs[:top_k]


class _Graph:
    file_type = "canvas"
    wikilinks = False

    def __init__(self, walked_ids=(), query_seeds=()):
        self.walked_ids = list(walked_ids)
        self.query_seeds = list(query_seeds)
        self.expand_calls: list[tuple[list[str], dict]] = []
        self.seed_calls: list[dict] = []

    def seed_from_query(self, question, seed_top_k=None, rerank=None):
        self.seed_calls.append({"seed_top_k": seed_top_k, "rerank": rerank})
        return [RetrievedDoc(i, f"seed {i}", {"file_type": "canvas", "source_file": "s.canvas",
                                              "canvas_node_id": i}) for i in self.query_seeds]

    def expand(self, seeds, **opts):
        self.expand_calls.append(([d.id for d in seeds], opts))
        nodes = [{"id": d.id, "depth": 0} for d in seeds] + [{"id": i, "depth": 1} for i in self.walked_ids]
        return {"nodes": nodes, "cross_edges": [], "stats": {"nodes": len(nodes)},
                "tree": [{"parent": "c1", "child": i, "label": "", "direction": "->", "depth": 1}
                         for i in self.walked_ids]}

    def _fetch_ids(self, ids):
        return [RetrievedDoc(i, f"walked {i}", {"file_type": "canvas", "source_file": "b.canvas",
                                                "canvas_node_id": i}) for i in ids]


def _parent_apply(docs):
    """The fake ParentContext.apply: swaps every child with a parent_id."""
    swaps = 0
    for d in docs:
        if d.metadata.get("parent_id"):
            d.debug["parent_swap"] = d.metadata["parent_id"]
            d.text = "PARENT"
            swaps += 1
    return docs, swaps, 0


def _rag(graph=None):
    rag = RAGPipeline.__new__(RAGPipeline)
    rag.search = _Search()
    rag.presets = {}
    rag.hyde = SimpleNamespace(code_intent_signal=lambda q: None)
    rag.graph = graph if graph is not None else _Graph()
    rag.reranker = _Reranker()
    rag.parent_ctx = SimpleNamespace(enabled=False, apply=_parent_apply)
    rag.neighbor_ctx = None
    rag.provenance_map = {"pdf": "source"}
    rag.generated_mode = "off"
    return rag


def test_graph_none_is_the_search_echo_plus_provenance():
    rag = _rag()
    docs, info = rag.graph_search("q", graph=None, top_k=3, parent_context=True)
    assert [d.id for d in docs] == [d.id for d in _search_docs()]
    assert rag.search.calls == [{"top_k": 3, "parent_context": True}]     # unchanged
    assert info["graph"]["walk"] is False
    assert info["graph"]["seeds"] == 0 and info["graph"]["added_ids"] == []
    assert info["graph"]["stats"] is None and info["graph"]["paths"] == []
    assert info["graph"]["provenance"] == ["authored", "generated", "source"]
    assert rag.reranker.calls == [] and rag.graph.expand_calls == []


def test_walk_adds_chunks_and_reranks_to_the_same_top_k():
    rag = _rag(_Graph(walked_ids=["w1", "w2"]))
    docs, info = rag.graph_search("q", graph={"depth": 1, "bridges": True, "edge_types": None},
                                  top_k=3)
    assert len(docs) == info["rerank_top_k"] == 3
    (call,) = rag.reranker.calls
    assert call["ids"] == ["n1", "c1", "p1", "w1", "w2"]            # search + walked, once
    assert call["top_k"] == 3
    assert call["mode"] == "cross_encoder" and call["instruction"] == "prefer primary sources"
    seeds, opts = rag.graph.expand_calls[0]
    assert seeds == ["c1"]                                            # the canvas hit seeds the walk
    assert opts == {"depth": 1, "bridges": True}                      # None dropped
    assert info["graph"]["walk"] is True and info["graph"]["seeds"] == 1
    assert info["graph"]["added_ids"] == ["w1", "w2"]
    assert [p["to"] for p in info["graph"]["paths"]] == ["w1", "w2"]
    assert len(info["graph"]["provenance"]) == len(docs)
    t = info["timings"]
    assert {"graph_seed", "graph_expand", "graph_rerank"} <= set(t)
    assert t["total"] == pytest.approx(10.0 + t["graph_seed"] + t["graph_expand"] + t["graph_rerank"],
                                       abs=0.05)


def test_walk_with_no_additions_returns_the_search_result_unreranked():
    rag = _rag(_Graph(walked_ids=[]))
    docs, info = rag.graph_search("q", graph={}, top_k=3, parent_context=True)
    assert rag.search.calls == [{"top_k": 3, "parent_context": False, "neighbor_context": False}]
    assert rag.reranker.calls == []                                   # no union rerank
    want, swaps, _ = _parent_apply(_search_docs())
    assert [(d.id, d.text) for d in docs] == [(d.id, d.text) for d in want]
    assert info["parent_context"] is True and info["parent_swaps"] == swaps == 1
    assert "graph_rerank" not in info["timings"]


def test_parent_swap_runs_after_the_union_rerank():
    rag = _rag(_Graph(walked_ids=["w1"]))
    docs, info = rag.graph_search("q", graph={}, top_k=4, parent_context=True)
    (call,) = rag.reranker.calls
    assert "PARENT" not in call["texts"]                              # child texts were scored
    swapped = [d for d in docs if d.debug.get("parent_swap")]
    assert [d.id for d in swapped] == ["n1"]
    assert swapped[0].text == "PARENT" and swapped[0].debug["parent_swap"] == "N"
    assert info["parent_context"] is True and info["parent_swaps"] == 1


def test_unknown_graph_option_raises():
    from src.pipeline import WALK_KEYS
    rag = _rag()
    with pytest.raises(ValueError) as exc_info:
        rag.graph_search("q", graph={"bridgez": True})
    assert "bridgez" in str(exc_info.value)
    assert all(k in str(exc_info.value) for k in WALK_KEYS)
    assert rag.search.calls == []                                     # refused before any search


def test_wikilinks_seed_the_walk_from_note_hits_and_seed_top_k_arrives():
    rag = _rag()
    rag.graph_search("q", graph={"wikilinks": True, "seed_top_k": 7}, top_k=3)
    seeds, opts = rag.graph.expand_calls[0]
    assert seeds == ["n1", "c1"] and opts == {"wikilinks": True}     # the note hit seeds the walk too
    assert rag.graph.seed_calls == [{"seed_top_k": 7, "rerank": "cross_encoder"}]
    rag = _rag()
    rag.graph_search("q", graph={}, top_k=3)
    assert rag.graph.expand_calls[0][0] == ["c1"]                     # wikilinks off: no note seeds


def test_query_seeds_are_kept_apart_from_walked_chunks():
    """seed_from_query's canvas hits enter the union rerank, but no edge reached them:
    they are seed_ids, and added_ids holds only what the walk brought in (hop >= 1,
    bridges). search_metas is the search's own list, before the union rerank."""
    rag = _rag(_Graph(walked_ids=["w1"], query_seeds=["s9"]))
    docs, info = rag.graph_search("q", graph={}, top_k=5)
    assert info["graph"]["added_ids"] == ["w1"]
    assert info["graph"]["seed_ids"] == ["s9"] and info["graph"]["seeds"] == 2
    (call,) = rag.reranker.calls
    assert call["ids"] == ["n1", "c1", "p1", "s9", "w1"]              # both enter the union rerank
    assert info["graph"]["search_metas"] == [d.metadata for d in _search_docs()]
    _, info = rag.graph_search("q", graph=None, top_k=3)
    assert info["graph"]["seed_ids"] == [] and info["graph"]["search_metas"] == [
        d.metadata for d in _search_docs()]


class _MixedGraph(_Graph):
    """_Graph whose walked node w1 is stored generated."""

    def _fetch_ids(self, ids):
        return [RetrievedDoc(i, f"walked {i}", {"file_type": "canvas", "source_file": "b.canvas",
                                                "canvas_node_id": i,
                                                **({"provenance": "generated"} if i == "w1" else {})})
                for i in ids]


def test_graph_search_drops_generated_walked_nodes_under_drop():
    """Spec §5.5: under drop a generated walked node leaves before the union rerank;
    the seeds (the generated canvas hit c1 among them) are unchanged."""
    rag = _rag(_MixedGraph(walked_ids=["w1", "w2"]))
    rag.generated_mode = "drop"
    docs, info = rag.graph_search("q", graph={}, top_k=5)
    (call,) = rag.reranker.calls
    assert call["ids"] == ["n1", "c1", "p1", "w2"]                    # w1 never reached the rerank
    assert info["graph"]["added_ids"] == ["w2"]
    assert rag.graph.expand_calls[0][0] == ["c1"]                     # seeds unchanged
    assert [d.debug["provenance"] for d in docs] == info["graph"]["provenance"]
    # a per-call mode beats the pipeline's and reaches search() unchanged
    rag = _rag(_MixedGraph(walked_ids=["w1", "w2"]))
    rag.graph_search("q", graph={}, top_k=5, generated_mode="drop")
    assert rag.reranker.calls[0]["ids"] == ["n1", "c1", "p1", "w2"]
    assert rag.search.calls == [{"top_k": 5, "generated_mode": "drop",
                                 "parent_context": False, "neighbor_context": False}]
    # off keeps them
    rag = _rag(_MixedGraph(walked_ids=["w1", "w2"]))
    rag.graph_search("q", graph={}, top_k=5)
    assert rag.reranker.calls[0]["ids"] == ["n1", "c1", "p1", "w1", "w2"]
    # graph=None: search()'s own list, stamped too
    docs, _ = _rag().graph_search("q", graph=None, top_k=3)
    assert [d.debug["provenance"] for d in docs] == ["authored", "generated", "source"]


class _Canvas:
    """A two-node canvas, cA -> cB, as Chroma serves the legacy walk: by id, or one
    hop's {"$and": [{"source_file": sf}, {"canvas_node_id": {"$in": ids}}]}."""
    ROWS = {"cA": ("[Canvas: a]\nSoft margin SVM",
                   {"file_type": "canvas", "source_file": "Maps/a.canvas", "canvas_node_id": "A",
                    "canvas_edges": [encode_canvas_edge("B", "->", "leads to")]}),
            "cB": ("[Canvas: a]\nSlack variables",
                   {"file_type": "canvas", "source_file": "Maps/a.canvas", "canvas_node_id": "B",
                    "canvas_edges": [encode_canvas_edge("A", "<-", "leads to")]})}

    def get(self, ids=None, where=None, include=None):
        if where is not None:
            sf, nids = where["$and"][0]["source_file"], where["$and"][1]["canvas_node_id"]["$in"]
            ids = [i for i, (_, m) in self.ROWS.items() if m["source_file"] == sf and m["canvas_node_id"] in nids]
        ids = [i for i in ids if i in self.ROWS]
        return {"ids": ids, "documents": [self.ROWS[i][0] for i in ids],
                "metadatas": [self.ROWS[i][1] for i in ids]}


def test_a_default_walk_gives_walked_evidence_its_path():
    """Every walk switch at its default (the shipped config): expand() itself returns
    no paths, so graph_search derives them from the tree, and the evidence pack can
    tell a walked chunk from a search hit."""
    from src.retrieval.evidence_pack import build_evidence_pack
    from src.retrieval.graph_expand import GraphExpander
    col = _Canvas()
    graph = GraphExpander(SimpleNamespace(rrf_k=60, _get_collection=lambda: col))
    graph.seed_from_query = lambda *a, **k: []
    rag = _rag(graph)
    rag.search = lambda q, **kw: ([RetrievedDoc("cA", *_Canvas.ROWS["cA"], rerank_score=9.0)], {
        "rerank_top_k": kw.get("top_k") or 5, "rerank_mode": "cross_encoder",
        "rerank_instruction": None, "timings": {"total": 1.0}})
    walk = [{"label": "leads to", "kind": "canvas", "direction": "->"}]
    _, info = rag.graph_search("q", graph={}, top_k=5)
    assert info["graph"]["paths"] == [{"to": "cB", "nodes": ["cA", "cB"], "edges": walk}]
    pack = build_evidence_pack(rag, "q", None, search_kwargs={}, graph={}, top_k=5,
                               min_score=None, include_text=0, max_sub_questions=4)
    by_id = {e["id"]: e for e in pack["evidence"]}
    assert by_id["cB"]["via_path"] == ["cA", "cB"] and by_id["cA"]["via_path"] == []
    assert pack["paths"] == [{"to": "cB", "nodes": ["cA", "cB"], "edges": walk}]
