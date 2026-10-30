"""retrieval.generated_mode on the warm API (route-then-cite S1, spec §5.1, §5.5, §5.6).

Every retrieval request shape takes a per-call generated_mode, but only "drop":
off is config-only, so a per-call off is a 422. Every source says its
provenance. /answer fetches the caller's ids instead of retrieving, so under any
mode but off it refuses generated ids with a 400 that names only them.
/graph/expand is a map, not evidence: its nodes still come back, each with its
provenance. /health and /schema report the mode.

Fully offline: a fake chunk store and fake pipelines (copies of
tests/test_graph_expand.py's helpers: private test helpers are not imported
across files); no model, index or generation backend.
"""
import re
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

import serve_api as S
from src.ingestion.canvas_loader import encode_canvas_edge
from src.retrieval.graph_expand import GraphExpander
from src.retrieval.retriever import RetrievedDoc
from src.utils.config_loader import Config
from src.utils.provenance import GENERATED_MODES, ProvenanceRule

REFUSAL = "generated chunks are maps, not evidence: answer from the primary pages they cite"


class FakeCollection:
    def __init__(self, rows: dict[str, tuple[str, dict]]):
        self.rows = rows

    def get(self, ids=None, where=None, include=None):
        if ids is not None:
            hits = [(i, *self.rows[i]) for i in ids if i in self.rows]
        else:
            clauses = (where or {}).get("$and", [where or {}])
            source = next((c["source_file"] for c in clauses if "source_file" in c), None)
            wanted = next((c["canvas_node_id"]["$in"] for c in clauses
                           if "canvas_node_id" in c), [])
            hits = [(i, t, m) for i, (t, m) in self.rows.items()
                    if m.get("source_file") == source and m.get("canvas_node_id") in wanted]
        return {"ids": [h[0] for h in hits], "documents": [h[1] for h in hits],
                "metadatas": [h[2] for h in hits]}


class FakeRetriever:
    rrf_k = 60
    route_pointers = None       # as on HybridRetriever: no sidecar loaded

    def __init__(self, collection):
        self.collection = collection
        self.embedder = None

    def _get_collection(self):
        return self.collection


def _chunk(cid, node_id, edges, source="map.canvas", text=None, **extra):
    meta = {"source_file": source, "file_type": "canvas",
            "canvas_node_id": node_id, "canvas_degree": len(edges),
            "canvas_edges": [encode_canvas_edge(n, d, l) for n, d, l in edges]}
    meta.update(extra)
    return cid, (text or f"body of {node_id}", meta)


def _expander(rows, **kw):
    return GraphExpander(retriever=FakeRetriever(FakeCollection(dict(rows))), reranker=None, **kw)


class _Gen:
    """A generator stand-in: records the docs it was handed and answers "ok"."""
    llm = SimpleNamespace(provider="openai", model="fake", backend="fake")

    def __init__(self):
        self.seen = None

    def generate(self, q, docs, max_tokens=None):
        self.seen = [d.id for d in docs]
        return SimpleNamespace(text="ok", confidence="HIGH", citations=[], sources=docs, usage=None)


def _api(monkeypatch, rows=(), generated_mode="drop", provenance_map=None):
    """serve_api with a fake warm pipeline (tests/test_graph_expand.py's _api, plus
    the mode, the rule and a generator)."""
    exp = _expander(list(rows))
    monkeypatch.setattr(S, "_HISTORY", deque(maxlen=50))
    monkeypatch.setitem(S._STATE, "rag", SimpleNamespace(
        graph=exp, retriever=exp.retriever, generator=_Gen(),
        generated_mode=generated_mode, provenance_map=provenance_map or {}))
    return S


# ---- the knob ----

def test_generated_mode_is_on_every_retrieval_model_and_forwarded(monkeypatch):
    for model in (S.QueryIn, S.SearchIn, S.CompareBranchIn, S.EvidencePackIn):
        assert "generated_mode" in model.model_fields, model.__name__
    assert "generated_mode" in S._RETRIEVAL_FIELDS
    seen = {}

    def search(q, **kw):
        seen.update(kw)
        return [], {}

    monkeypatch.setattr(S, "_HISTORY", deque(maxlen=50))
    monkeypatch.setitem(S._STATE, "rag", SimpleNamespace(search=search))
    S.search(S.SearchIn(q="x", generated_mode="drop"))
    assert seen["generated_mode"] == "drop"
    # /compare shares one evidence set per retrieval key, so the mode is part of the key
    assert (S._retrieval_cache_key(S.CompareBranchIn(id="a"))
            != S._retrieval_cache_key(S.CompareBranchIn(id="a", generated_mode="drop")))


def test_a_per_call_off_is_a_422(monkeypatch):
    with pytest.raises(ValidationError):
        S.SearchIn(q="x", generated_mode="off")
    monkeypatch.setattr(S.app.state, "guard_settings", None)
    client = TestClient(S.app, base_url="http://127.0.0.1")    # the request guard refuses Host: testserver
    for path, body in (("/search", {"q": "x", "generated_mode": "off"}),
                       ("/query", {"q": "x", "generated_mode": "off"}),
                       ("/evidence_pack", {"q": "x", "generated_mode": "off"}),
                       ("/compare", {"q": "x", "branches": [{"id": "a", "generated_mode": "off"},
                                                            {"id": "b"}]})):
        assert client.post(path, json=body).status_code == 422, path


def test_sources_carry_provenance():
    doc = RetrievedDoc(id="p1", text="t", metadata={"file_type": "pdf"}, debug={"provenance": "source"})
    unstamped = RetrievedDoc(id="p2", text="t", metadata={"file_type": "pdf"})
    out = S._sources_out([doc, unstamped], [], include_text=0, cap=None)
    # read from the stamp, never recomputed here: None says no pipeline path stamped it
    assert [s.provenance for s in out] == ["source", None]


# ---- /answer ----

def test_answer_refuses_generated_ids_and_names_only_them(monkeypatch):     # Review Focus 5
    api = _api(monkeypatch, [_chunk("g1", "n1", [], provenance="generated"),
                             _chunk("p1", "n2", [])], generated_mode="drop")
    with pytest.raises(HTTPException) as e:
        api.answer(api.AnswerIn(q="why?", docs=["p1", "g1"]))
    assert e.value.status_code == 400 and e.value.detail["ids"] == ["g1"]
    assert e.value.detail["error"] == REFUSAL
    assert api._STATE["rag"].generator.seen is None              # nothing was generated
    # an unknown id is still named first: the refusal sits after that check
    with pytest.raises(HTTPException) as e:
        api.answer(api.AnswerIn(q="why?", docs=["g1", "ghost"]))
    assert e.value.status_code == 400 and e.value.detail["unknown"] == ["ghost"]


def test_answer_judges_ids_by_the_whole_rule_not_the_stored_key(monkeypatch):
    rule = ProvenanceRule(generated_paths=("maps",))
    api = _api(monkeypatch, [_chunk("m1", "n1", [], source="Maps/a.canvas", provenance="authored"),
                             _chunk("p1", "n2", [])], generated_mode="drop", provenance_map=rule)
    with pytest.raises(HTTPException) as e:
        api.answer(api.AnswerIn(q="why?", docs=["m1", "p1"]))
    assert e.value.status_code == 400 and e.value.detail["ids"] == ["m1"]


def test_answer_with_primary_ids_still_answers(monkeypatch):
    api = _api(monkeypatch, [_chunk("g1", "n1", [], provenance="generated"),
                             _chunk("p1", "n2", [], source="Books/b.pdf", file_type="pdf"),
                             _chunk("p2", "n3", [])],
               generated_mode="drop", provenance_map={"pdf": "source"})
    out = api.answer(api.AnswerIn(q="why?", docs=["p1", "p2"]))
    assert out.answer == "ok" and api._STATE["rag"].generator.seen == ["p1", "p2"]
    assert [s.provenance for s in out.sources] == ["source", "authored"]


def test_answer_under_off_answers_generated_ids(monkeypatch):
    api = _api(monkeypatch, [_chunk("g1", "n1", [], provenance="generated"),
                             _chunk("p1", "n2", [])], generated_mode="off")
    out = api.answer(api.AnswerIn(q="why?", docs=["g1", "p1"]))
    assert out.answer == "ok" and api._STATE["rag"].generator.seen == ["g1", "p1"]
    assert [s.provenance for s in out.sources] == ["generated", "authored"]


# ---- /graph/expand: a map, so generated nodes stay, each labelled ----

def test_graph_expand_nodes_carry_provenance(monkeypatch, tmp_path):
    rows = [_chunk("c1", "A", [("B", "->", "x")], provenance="generated"), _chunk("c2", "B", [])]
    exp = GraphExpander.from_config(Config({"provenance": {"generated_paths": ["Maps"]}}, tmp_path),
                                    FakeRetriever(FakeCollection(dict(rows))))
    assert exp.provenance_rule.generated_paths == ("maps",)       # from_config passes the whole rule
    monkeypatch.setitem(S._STATE, "rag", SimpleNamespace(graph=exp, retriever=exp.retriever))
    out = S.graph_expand(S.GraphExpandIn(seeds=["c1"], depth=1))
    assert {n["id"]: n["provenance"] for n in out["nodes"]} == {"c1": "generated", "c2": "authored"}
    # with no rule (direct constructions only) a node keeps its pre-slice shape
    bare = _expander(rows)
    seed = RetrievedDoc(id="c1", text=rows[0][1][0], metadata=rows[0][1][1])
    assert all("provenance" not in n for n in bare.expand([seed], depth=1)["nodes"])


# ---- /schema and /health ----

def test_schema_documents_the_mode_and_chunk_origin(monkeypatch):
    monkeypatch.setitem(S._STATE, "rag", SimpleNamespace(presets={}, generated_mode="drop"))
    contract = S.schema()
    knob = ("drop|route? (generated chunks out of the evidence; off is config-only; route is a "
            "409 while the store held generated rows at startup and no sidecar is loaded)")
    ends = contract["endpoints"]
    for body in (ends["POST /search"]["body"], ends["POST /query"]["body"],
                 ends["POST /evidence_pack"]["body"], ends["POST /compare"]["body"]["branch"]):
        assert body["generated_mode"] == knob
    prov = contract["provenance"]
    assert {"retrieval", "generation", "sources", "chunk_origin", "generated_mode"} <= set(prov)
    assert prov["chunk_origin"]["values"] == ["authored", "source", "generated"]
    assert prov["generated_mode"]["active"] == "drop"
    assert set(prov["generated_mode"]["modes"]) == set(GENERATED_MODES)
    drop = prov["generated_mode"]["modes"]["drop"]           # as SKILL.md and docs/agents.md say it
    assert "dense and sparse lanes refill" in drop and "tantivy_filter_pool" in drop
    assert "HyPE and Omnisearch results lose theirs with no refill" in drop
    note = contract["experimental"]["features"]["graph_rag"]["unmeasured"]
    assert "labelled question sets" in note and "never" not in note and not re.search(r"\d", note)
    assert "has not been scored against the shipped generated_mode" in note and "slice" not in note
    # the SimpleNamespace(presets={}) fakes other suites use still get a schema
    monkeypatch.setitem(S._STATE, "rag", SimpleNamespace(presets={}))
    assert S.schema()["provenance"]["generated_mode"]["active"] is None


def test_health_reports_the_mode(monkeypatch):
    monkeypatch.setitem(S._STATE, "boot_error", None)
    monkeypatch.setitem(S._STATE, "rag", SimpleNamespace(generated_mode="drop"))
    monkeypatch.setitem(S._STATE, "ready", True)
    assert S.health()["generated_mode"] == "drop"
    monkeypatch.setitem(S._STATE, "ready", False)
    assert "generated_mode" not in S.health()


def test_health_hint_sends_an_unstamped_store_to_the_error(monkeypatch):
    monkeypatch.setitem(S._STATE, "ready", False)
    monkeypatch.setitem(S._STATE, "boot_error", "UnstampedProvenanceError: 1 stored rows ... Fix: ...")
    hint = S.health()["hint"]
    assert "error field" in hint and "stamp-provenance" in hint and "model ids" not in hint
    monkeypatch.setitem(S._STATE, "boot_error", "OSError: no such model")   # every other boot error
    assert "check the model ids and index paths" in S.health()["hint"]
