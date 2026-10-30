"""POST /evidence_pack and the walk fields on /graph/expand. build_evidence_pack
runs over a fake rag.graph_search (RAGPipeline.graph_search's contract: docs plus
an info echo whose graph block carries paths, added_ids and one provenance per
doc), so no model, index or backend loads."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.retrieval.retriever import RetrievedDoc
from src.utils.config_loader import Config
from src.utils.provenance import provenance_of


def _doc(i, ft="canvas", score=1.0, **meta):
    return RetrievedDoc(i, f"text of {i}", {"file_type": ft, "source_file": f"{i}.{ft}", **meta},
                        rerank_score=score)


class _Rag:
    """graph_search stand-in: one canned result per sub-question, every call recorded."""

    def __init__(self, by_q, paths=(), added=()):
        self.by_q, self.paths, self.added = by_q, list(paths), list(added)
        self.provenance_map = {"pdf": "source"}
        self.calls: list[tuple[str, dict | None, dict]] = []

    def graph_search(self, question, graph=None, **kw):
        self.calls.append((question, graph, kw))
        docs = list(self.by_q.get(question, []))
        return docs, {"rerank_mode": "cross_encoder", "timings": {"total": 1.0},
                      "graph": {"walk": graph is not None, "stats": {"nodes": 1} if graph else None,
                                "paths": self.paths, "added_ids": self.added,
                                "provenance": [provenance_of(d.metadata, self.provenance_map)
                                               for d in docs]}}


def _pack(rag, question="q", subs=None, **kw):
    from src.retrieval.evidence_pack import build_evidence_pack
    args = dict(search_kwargs={}, graph=None, top_k=5, min_score=None, include_text=600,
                max_sub_questions=4)
    args.update(kw)
    return build_evidence_pack(rag, question, subs, **args)


def test_coverage_gaps_and_generated_only():
    gen = {"provenance": "generated"}
    rag = _Rag({"s1": [_doc("g1", **gen), _doc("g2", **gen)], "s2": []})
    pack = _pack(rag, subs=["s1", "s2"])
    assert list(pack["coverage"]) == ["s1", "s2"] and pack["sub_questions"] == ["s1", "s2"]
    assert pack["coverage"] == {"s1": ["g1", "g2"], "s2": []}
    assert pack["gaps"] == ["s2"]                       # an empty sub is a gap, not generated-only
    assert pack["generated_only"] == ["s1"]
    assert pack["provenance_mix"] == {"authored": 0, "source": 0, "generated": 2}
    assert sum(pack["provenance_mix"].values()) == len(pack["evidence"])

    # One item per id across subs: sub_questions unioned, the higher score kept.
    rag = _Rag({"s1": [_doc("p1", ft="pdf", score=2.0, page_start=4, page_end=4), _doc("g1", **gen)],
                "s2": [_doc("n1", ft="note"), _doc("p1", ft="pdf", score=5.0, page_start=4, page_end=4)]})
    pack = _pack(rag, subs=["s1", "s2"])
    assert [e["id"] for e in pack["evidence"]] == ["p1", "g1", "n1"]
    p1 = pack["evidence"][0]
    assert p1["sub_questions"] == ["s1", "s2"] and p1["score"] == 5.0
    assert p1["pages"] == [4, 4] and p1["provenance"] == "source" and p1["file_type"] == "pdf"
    assert pack["gaps"] == [] and pack["generated_only"] == []
    assert pack["provenance_mix"] == {"authored": 1, "source": 1, "generated": 1}
    assert sum(pack["provenance_mix"].values()) == len(pack["evidence"]) == 3
    assert pack["graph"] is False
    assert [r["q"] for r in pack["retrieval"]] == ["s1", "s2"]
    assert pack["retrieval"][0]["rerank_mode"] == "cross_encoder"


def test_min_score_drops_and_needs_scores():
    rag = _Rag({"s1": [_doc("a", score=3.0), _doc("b", score=0.5)]})
    pack = _pack(rag, subs=["s1"], min_score=1.0)
    assert [e["id"] for e in pack["evidence"]] == ["a"]
    assert pack["dropped_below_min_score"] == {"s1": 1}

    rag = _Rag({"s1": [_doc("a", score=None)]})
    with pytest.raises(ValueError, match="min_score needs a scoring rerank mode"):
        _pack(rag, subs=["s1"], min_score=1.0)


def test_omnisearch_and_too_many_subs_refused():
    rag = _Rag({})
    with pytest.raises(ValueError, match="omnisearch"):
        _pack(rag, search_kwargs={"omnisearch": True})
    with pytest.raises(ValueError, match="evidence_pack.max_sub_questions"):
        _pack(rag, subs=["a", "b", "c", "d", "e"])
    with pytest.raises(ValueError, match="empty"):
        _pack(rag, subs=["a", "  "])
    assert rag.calls == []                              # refused before any search


def test_parent_swap_id_and_via_path():
    swapped = _doc("n1", ft="note", parent_id="N", course_name="unknown")
    swapped.debug["parent_swap"] = "N"
    used = {"to": "w1", "nodes": ["c1", "w1"], "edges": [{"label": "uses", "kind": "canvas",
                                                          "direction": "->"}]}
    unused = {"to": "w9", "nodes": ["c1", "w9"], "edges": []}
    rag = _Rag({"s1": [swapped, _doc("c1", course_name="ML"), _doc("w1")]},
               paths=[used, unused], added=["w1", "w9"])
    pack = _pack(rag, subs=["s1"], graph={"bridges": True})
    by_origin = {e["origin_id"]: e for e in pack["evidence"]}
    assert by_origin["n1"]["id"] == "parent:N" and by_origin["n1"]["course"] is None
    assert by_origin["c1"]["via_path"] == [] and by_origin["c1"]["course"] == "ML"   # a direct hit
    assert by_origin["w1"]["via_path"] == ["c1", "w1"]
    assert pack["paths"] == [used]                      # only the paths some evidence uses
    assert pack["coverage"]["s1"] == ["parent:N", "c1", "w1"]
    assert pack["graph"] is True and pack["retrieval"][0]["graph_stats"] == {"nodes": 1}
    assert rag.calls[0][1] == {"bridges": True}


def _route(monkeypatch, rag, cfg_data=None):
    import serve_api
    monkeypatch.setitem(serve_api._STATE, "rag", rag)
    monkeypatch.setitem(serve_api._STATE, "cfg", Config(cfg_data or {}, Path.cwd()))
    return serve_api


def test_route_503_when_graph_off_and_400_for_max_sources(monkeypatch):
    from fastapi import HTTPException
    from src.pipeline import WALK_KEYS
    rag = _Rag({"a": [_doc("x" * 2, score=2.0)], "b": []})
    rag.graph = SimpleNamespace(enabled=False, file_type="canvas")
    api = _route(monkeypatch, rag, {"evidence_pack": {"top_k": 7, "include_text": 3,
                                                      "max_sub_questions": 2}})
    assert set(api._WALK_FIELDS) == WALK_KEYS

    with pytest.raises(HTTPException) as exc_info:
        api.evidence_pack(api.EvidencePackIn(q="x", graph=True))
    assert exc_info.value.status_code == 503 and "graph.enabled" in str(exc_info.value.detail)
    with pytest.raises(HTTPException) as exc_info:
        api.evidence_pack(api.EvidencePackIn(q="x", max_sources=3))
    assert exc_info.value.status_code == 400 and "max_sources" in str(exc_info.value.detail)
    with pytest.raises(HTTPException) as exc_info:
        api.evidence_pack(api.EvidencePackIn(q="x", sub_questions=["a", "b", "c"]))
    assert exc_info.value.status_code == 400
    assert "evidence_pack.max_sub_questions" in str(exc_info.value.detail)
    assert rag.calls == []

    pack = api.evidence_pack(api.EvidencePackIn(q="the question", sub_questions=["a", "b"]))
    assert pack["question"] == "the question" and pack["gaps"] == ["b"] and "ms" in pack
    assert pack["evidence"][0]["text"] == "te…"                    # evidence_pack.include_text
    (q, graph, kw), _ = rag.calls
    assert q == "a" and graph is None
    assert kw["top_k"] == 7 and kw["omnisearch"] is False             # evidence_pack.top_k

    rag.graph.enabled = True
    rag.calls.clear()
    api.evidence_pack(api.EvidencePackIn(q="x", graph=True, bridges=True, top_k=2, include_text=0))
    (_, graph, kw), = rag.calls
    assert graph == {**{k: None for k in WALK_KEYS}, "bridges": True} and kw["top_k"] == 2


def test_blank_edge_labels_are_refused_by_the_request_model():
    """A blank label would pass the list's min_length, seed the walk (an embedding and a
    rerank), then fail in the walk as a 200 "Traversal failed": refused up front (422)."""
    import pydantic
    import serve_api as api
    for model in (api.GraphExpandIn, api.EvidencePackIn):
        for bad in (["  "], [""], ["uses", " "], []):
            with pytest.raises(pydantic.ValidationError):
                model(q="svm", edge_types=bad)
        assert model(q="svm", edge_types=["uses", "leads to"]).edge_types == ["uses", "leads to"]
        assert model(q="svm").edge_types is None


def test_graph_expand_route_passes_new_fields(monkeypatch):
    calls = []

    def expand(seeds, **kw):
        calls.append(kw)
        return {"nodes": [{"id": "c1", "text": "t"}], "tree": [], "cross_edges": [],
                "stats": {"edges_labelled": labelled, "bridges_unavailable": 2}}

    graph = SimpleNamespace(enabled=True, file_type="canvas", expand=expand,
                            seed_from_ids=lambda ids: ([_doc("c1")], []),
                            attach_routed_ids=lambda nodes: None)
    api = _route(monkeypatch, SimpleNamespace(graph=graph))
    labelled = True
    out = api.graph_expand(api.GraphExpandIn(seeds=["c1"], bridges=True, cross_canvas=False,
                                             direction="up", edge_types=["uses"]))
    assert calls[-1] == {"depth": None, "max_per_hop": None, "max_total": None, "bridges": True,
                         "cross_canvas": False, "wikilinks": None, "direction": "up",
                         "edge_types": ["uses"]}
    assert out["note"] == ("2 canvas chunk(s) predate canvas_edge_files, so their file nodes were "
                           "not bridged: run canvas_refresh.py, then rebuild_bm25.py, then restart "
                           "this API")
    labelled = False
    out = api.graph_expand(api.GraphExpandIn(seeds=["c1"], bridges=True))
    assert "UNLABELLED" in out["note"] and out["note"].endswith("then restart this API")
