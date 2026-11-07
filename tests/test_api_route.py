"""generated_mode route on the warm API (route-then-cite S2, spec §5.4-§5.6).

A per-call route is accepted. Without a route sidecar while the store holds generated
rows, search() raises RouteUnavailableError, and every retrieval route turns it into a
409 naming the fix, never the 200 "Retrieval failed" of its catch-all. Each source and
evidence item carries found_via (the maps that routed it), and each response the call's
route block. /answer still refuses a generated id, now with routes_to naming the pages
the sidecar routes it to. /graph/expand (the map view) gives each generated node its
routed_ids in page order; graph_search walks never pay for that. /health and /schema
report the sidecar's coverage.

Fully offline: fake pipelines, and for the map view the real HybridRetriever and
GraphExpander over a bm25s payload and a Chroma stand-in (copies of the helpers in
tests/test_api_generated.py and tests/test_routed_lane.py: private test helpers are not
imported across files).
"""
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import serve_api as S
from src.embeddings.embedder import BM25Index, _tokenize
from src.ingestion.canvas_loader import encode_canvas_edge
from src.pipeline import RAGPipeline
from src.retrieval.evidence_pack import build_evidence_pack
from src.retrieval.graph_expand import GraphExpander
from src.retrieval.retriever import HybridRetriever, RetrievedDoc
from src.retrieval.route import RoutePointers
from src.utils.config_loader import Config
from src.utils.provenance import ProvenanceRule, RouteUnavailableError

ROUTED = {"router_hits": 2, "routed": 1, "unrouted": 1, "targets": 1, "targets_missing": 0, "new": 1}
VIA = [{"id": "g1", "label": "Maps/m.canvas"}]
B_SPAN = {"source_file": "Books/b.pdf", "page_start": 59, "page_end": 63}
GEN = {"provenance": "generated"}


def _pointers(pointers):
    return RoutePointers({"version": 1, "built_at": "2026-10-29T00:00:00Z",
                          "pointers": pointers, "unrouted": {}})


def _client_with_rag(monkeypatch, raises=None, search=None):
    """A TestClient (no lifespan) over a fake warm pipeline. With `raises`, search and
    graph_search raise it whatever the mode: the per-call check lives in search()."""
    def fail(*args, **kw):
        raise raises

    monkeypatch.setattr(S, "_HISTORY", deque(maxlen=50))
    monkeypatch.setitem(S._STATE, "rag", SimpleNamespace(search=search or fail, graph_search=fail))
    monkeypatch.setitem(S._STATE, "cfg", Config({}, Path.cwd()))    # /evidence_pack reads it
    monkeypatch.setitem(S.app.state._state, "guard_settings", None)
    return TestClient(S.app, base_url="http://127.0.0.1")         # the guard refuses Host: testserver


# ---- the 409 ----

ROUTES = [("/search", {"q": "x"}), ("/query", {"q": "x"}), ("/query", {"q": "x", "retrieve_only": True}),
          ("/evidence_pack", {"q": "x"}),
          ("/compare", {"q": "x", "branches": [{"id": "a", "generated_mode": "route"}, {"id": "b"}]})]


@pytest.mark.parametrize("path,body", ROUTES)
def test_a_route_without_a_sidecar_is_a_409_on_every_retrieval_route(monkeypatch, path, body):
    client = _client_with_rag(monkeypatch, raises=RouteUnavailableError("run python main.py route-pointers"))
    r = client.post(path, json={"generated_mode": "route", **body})
    assert r.status_code == 409 and "route-pointers" in r.json()["detail"]["error"]
    # any other failure still reaches the route's own catch-all, as before
    other = _client_with_rag(monkeypatch, raises=RuntimeError("boom"))
    assert other.post(path, json={"generated_mode": "route", **body}).status_code == 200


# ---- found_via and the route block ----

def test_sources_carry_found_via(monkeypatch):
    routed = RetrievedDoc(id="p1", text="t", metadata={}, debug={"provenance": "source", "found_via": VIA})
    plain = RetrievedDoc(id="p2", text="t", metadata={}, debug={"provenance": "authored"})
    out = S._sources_out([routed, plain], [], include_text=0, cap=None)
    assert [[f.model_dump() for f in s.found_via] for s in out] == [VIA, []]
    # through the routes: each source's trail, and the call's route block (retrieval.route)
    client = _client_with_rag(monkeypatch, search=lambda q, **kw: ([routed, plain], {"route": dict(ROUTED)}))
    for path, extra, key in (("/search", {}, "results"), ("/query", {"retrieve_only": True}, "sources")):
        r = client.post(path, json={"q": "x", "generated_mode": "route", **extra})
        assert r.status_code == 200, r.text
        assert [s["found_via"] for s in r.json()[key]] == [VIA, []]
        assert r.json()["retrieval"]["route"] == ROUTED


class _PackRag:
    """graph_search stand-in: canned docs per sub-question; info carries `route` when given."""

    def __init__(self, by_q, route=None):
        self.by_q, self.route = by_q, route

    def graph_search(self, question, graph=None, **kw):
        docs = self.by_q[question]
        info = {"rerank_mode": "cross_encoder", "timings": {"total": 1.0},
                "graph": {"stats": None, "paths": [], "added_ids": [],
                          "provenance": ["source"] * len(docs)}}
        if self.route is not None:
            info["route"] = self.route
        return docs, info


def _pack(rag, subs):
    return build_evidence_pack(rag, "q", subs, search_kwargs={}, graph=None, top_k=5,
                               min_score=None, include_text=0, max_sub_questions=4)


def _page(cid, via=None):
    return RetrievedDoc(cid, "page", {"file_type": "pdf", "source_file": "b.pdf"}, rerank_score=1.0,
                        debug={} if via is None else {"found_via": via})


def test_evidence_pack_items_carry_found_via():
    g2 = {"id": "g2", "label": "Maps/n.canvas"}
    rag = _PackRag({"s1": [_page("p1", [dict(f) for f in VIA]), _page("n1")],
                    "s2": [_page("p1", [g2, VIA[0]]), _page("n1", [g2])]})
    items = {e["id"]: e for e in _pack(rag, ["s1", "s2"])["evidence"]}
    # merged across sub-questions: unioned by id, first seen first
    assert items["p1"]["found_via"] == [VIA[0], g2]
    assert items["n1"]["found_via"] == [g2]
    assert items["p1"]["via_path"] == []                    # a routed page: found_via is its trail
    assert rag.by_q["s1"][0].debug["found_via"] == VIA      # the doc's own list is not mutated


def test_evidence_pack_retrieval_entries_carry_the_route_block():
    pack = _pack(_PackRag({"s1": [_page("p1")], "s2": []}, route=dict(ROUTED)), ["s1", "s2"])
    assert [r["route"] for r in pack["retrieval"]] == [ROUTED, ROUTED]
    assert [r["graph_route"] for r in pack["retrieval"]] == [None, None]    # no walk routed
    (entry,) = _pack(_PackRag({"s1": [_page("p1")]}), ["s1"])["retrieval"]
    assert entry["route"] is None and entry["graph_route"] is None


# ---- /answer ----

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
        out = {"ids": [h[0] for h in hits], "documents": [h[1] for h in hits],
               "metadatas": [h[2] for h in hits]}
        if include and "embeddings" in include:     # the span's stored vectors: cosine falls with row order
            order = list(self.rows)
            out["embeddings"] = [[1.0, 0.01 * order.index(h[0])] for h in hits]
        return out


class FakeRetriever:
    rrf_k = 60
    route_pointers = None       # as on HybridRetriever: no sidecar loaded

    def __init__(self, collection):
        self.collection = collection
        self.embedder = None

    def _get_collection(self):
        return self.collection


class _Gen:
    llm = SimpleNamespace(provider="openai", model="fake", backend="fake")

    def generate(self, q, docs, max_tokens=None):
        raise AssertionError("a refused /answer generates nothing")


def test_answer_400_lists_each_generated_ids_servable_targets(monkeypatch):     # Review Focus 5
    # On the real retriever (the map world below). g1 cites pp. 59-63 and the whole of b.pdf,
    # which the store serves, pp. 30-31 (b.pdf is indexed, but no row lies there) and a file
    # not in the store; g2 cites only those two. routes_to lists only what the store serves.
    whole = {"source_file": "Books/b.pdf", "page_start": None, "page_end": None}
    empty = {"source_file": "Books/b.pdf", "page_start": 30, "page_end": 31}
    gone = {"source_file": "Books/zzz.pdf", "page_start": 1, "page_end": 2}
    _, r = _map_world({"g1": [B_SPAN, empty, gone, whole], "g2": [empty, gone]})
    monkeypatch.setitem(S._STATE, "rag", SimpleNamespace(
        retriever=r, generator=_Gen(), generated_mode="route", provenance_map={}))
    with pytest.raises(HTTPException) as e:
        S.answer(S.AnswerIn(q="why?", docs=["b-p60", "g1", "g2"]))
    assert e.value.status_code == 400 and e.value.detail["ids"] == ["g1", "g2"]
    # g2 routes to nothing the store serves, so it is named in ids but absent here
    assert e.value.detail["routes_to"] == {"g1": [B_SPAN, whole]}
    # A generated id the loaded sidecar does not route at all (targets() is None) is
    # absent too: the same 400, never a 500
    _, r = _map_world({"g1": [B_SPAN]})
    monkeypatch.setitem(S._STATE, "rag", SimpleNamespace(
        retriever=r, generator=_Gen(), generated_mode="route", provenance_map={}))
    with pytest.raises(HTTPException) as e:
        S.answer(S.AnswerIn(q="why?", docs=["g1", "g2"]))
    assert e.value.status_code == 400 and e.value.detail["ids"] == ["g1", "g2"]
    assert e.value.detail["routes_to"] == {"g1": [B_SPAN]}


def test_answer_400_without_a_sidecar_has_no_routes_to(monkeypatch):
    rows = {"g1": ("map one", {"source_file": "Maps/a.canvas", **GEN}),
            "p1": ("page", {"source_file": "Books/b.pdf", "file_type": "pdf"})}
    monkeypatch.setitem(S._STATE, "rag", SimpleNamespace(
        retriever=FakeRetriever(FakeCollection(rows)), generator=_Gen(), generated_mode="route",
        provenance_map={}))
    with pytest.raises(HTTPException) as e:
        S.answer(S.AnswerIn(q="why?", docs=["g1"]))
    assert e.value.status_code == 400 and e.value.detail["ids"] == ["g1"]
    assert "routes_to" not in e.value.detail


# ---- /health and /schema ----

COVERAGE = {"generated_rows": 3, "with_pointers": 2, "sidecar": "<data-dir>/route_pointers.json"}


def test_health_and_schema_report_route_coverage(monkeypatch):
    monkeypatch.setitem(S._STATE, "boot_error", None)
    monkeypatch.setitem(S._STATE, "rag", SimpleNamespace(presets={}, generated_mode="route",
                                                         route_coverage=dict(COVERAGE)))
    monkeypatch.setitem(S._STATE, "ready", True)
    assert S.health()["route"] == COVERAGE
    monkeypatch.setitem(S._STATE, "ready", False)
    assert "route" not in S.health()                       # like generated_mode: only when ready
    gm = S.schema()["provenance"]["generated_mode"]
    assert gm["route"] == COVERAGE
    route = gm["modes"]["route"]
    assert "route-pointers" in route and "not available yet" not in route
    assert "409" in route and "zero generated rows" in route and "found_via" in route
    assert "held generated rows at startup and no sidecar is loaded" in route
    assert "Needs sparse_backend: bm25s." in route
    ends = S.schema()["endpoints"]
    assert ("while the store held generated rows at startup and no sidecar is loaded"
            in ends["POST /search"]["body"]["generated_mode"])
    docs = ends["POST /answer"]["body"]["docs"]
    assert "routes_to" in docs and "only the spans the store serves" in docs and "targets_missing" in docs
    assert "an id left with none is omitted" in docs
    assert ("the full cited spans, as pages, not chunk ids (send a /graph/expand node's routed_ids, "
            "each span's first chunks, in the map's place)") in docs
    expand = ends["POST /graph/expand"]["response"]
    assert "routed_ids" in expand and "first retrieval.route.per_target primary chunks of each span" in expand
    assert "retrieval.route.max_targets spans" in expand
    assert "(a cite with no pages gives the file's first chunks, in document order)" in expand
    assert "graph_route (the walk's own routing" in ends["POST /evidence_pack"]["response"]
    # the fakes other suites use (no route_coverage) still get both
    monkeypatch.setitem(S._STATE, "rag", SimpleNamespace(presets={}))
    monkeypatch.setitem(S._STATE, "ready", True)
    assert S.health()["route"] is None and S.schema()["provenance"]["generated_mode"]["route"] is None


def test_health_hint_sends_a_missing_route_sidecar_to_route_pointers(monkeypatch):
    monkeypatch.setitem(S._STATE, "ready", False)
    monkeypatch.setitem(S._STATE, "boot_error", "RouteUnavailableError: generated_mode route needs ...")
    hint = S.health()["hint"]
    assert "python main.py route-pointers" in hint and "route_pointers job" in hint
    assert "model ids" not in hint and "stamp-provenance" not in hint


# ---- the map view: routed_ids on /graph/expand ----

def _matches(meta, where):
    if where is None:
        return True
    if "$and" in where:
        return all(_matches(meta, w) for w in where["$and"])
    ((key, cond),) = where.items()
    value = meta.get(key)
    if not isinstance(cond, dict):
        return value == cond
    ((op, arg),) = cond.items()
    return value is not None and {"$in": lambda: value in arg, "$lte": lambda: value <= arg,
                                  "$gte": lambda: value >= arg}[op]()


class _Store(FakeCollection):
    """FakeCollection's get (ids, and the walk's one-canvas where) plus Chroma's query,
    ranked in row order."""

    def query(self, query_embeddings, n_results, include, where=None, ids=None):
        hit = [(i, t, m) for i, (t, m) in self.rows.items()
               if _matches(m, where) and (ids is None or i in ids)][:n_results]
        return {"ids": [[h[0] for h in hit]], "documents": [[h[1] for h in hit]],
                "metadatas": [[h[2] for h in hit]]}


class _Embedder:
    calls = 0

    def embed_query(self, text):
        self.calls += 1
        return [1.0, 0.0]


def _canvas(cid, node_id, edges, **extra):
    return cid, (f"map node {node_id}", {
        "source_file": "Maps/m.canvas", "file_type": "canvas", "canvas_node_id": node_id,
        "canvas_degree": len(edges), "canvas_edges": [encode_canvas_edge(n, d, "") for n, d in edges],
        **extra})


def _pdf(cid, page):
    return cid, (f"page {page}", {"source_file": "Books\\b.pdf", "file_type": "pdf",
                                  "provenance": "source", "page_start": page, "page_end": page})


def _map_world(pointers, per_target=3):
    """(GraphExpander, its HybridRetriever). c1 (authored) links to g1 and g2 (generated);
    g1 cites b.pdf pp. 59-63, g2 nothing. The store holds b-p62 before b-p60: Chroma's order
    is not page order. pointers None = no sidecar loaded."""
    rows = dict([_canvas("c1", "A", [("B", "->"), ("C", "->")]),
                 _canvas("g1", "B", [("A", "<-")], **GEN), _canvas("g2", "C", [("A", "<-")], **GEN),
                 _pdf("b-p62", 62), _pdf("b-p60", 60), _pdf("b-p10", 10)])
    rule = ProvenanceRule(by_file_type={"pdf": "source"})
    r = HybridRetriever(
        embedder=_Embedder(), chroma_dir=Path("unused"), bm25_index=Path("unused"),
        collection_name="unused", dense_top_k=2, sparse_top_k=2, metadata_boost=False,
        code_file_types=["py"], provenance_rule=rule,
        route={"router_k": 5, "per_target": per_target, "max_targets": 8, "max_new": 10},
        route_pointers=None if pointers is None else _pointers(pointers))
    r._bm25_payload = {"bm25": BM25Index([_tokenize(t) for t, _ in rows.values()]),
                       "ids": list(rows), "documents": [t for t, _ in rows.values()],
                       "metadatas": [m for _, m in rows.values()]}
    r._collection = _Store(rows)
    return GraphExpander(retriever=r, reranker=None, provenance_rule=rule), r


def _spy_routed_for(monkeypatch, r):
    calls = []
    real = r.routed_for

    def spy(query, router_ids):
        calls.append((query, list(router_ids)))
        return real(query, router_ids)

    monkeypatch.setattr(r, "routed_for", spy)
    return calls


def test_graph_expand_gives_generated_nodes_their_routed_ids_in_page_order(monkeypatch):
    exp, r = _map_world({"g1": [B_SPAN]})
    monkeypatch.setitem(S._STATE, "rag", SimpleNamespace(graph=exp, retriever=r))
    calls = _spy_routed_for(monkeypatch, r)
    seeded = S.graph_expand(S.GraphExpandIn(seeds=["c1"], depth=1))
    by_id = {n["id"]: n for n in seeded["nodes"]}
    assert by_id["g1"]["routed_ids"] == ["b-p60", "b-p62"]      # page order, not the store's
    assert by_id["g2"]["routed_ids"] == []                      # no targets: the console keeps it off
    assert "routed_ids" not in by_id["c1"]                      # only generated nodes route
    # one page-order call per generated node, so none loses its targets to max_targets
    assert calls == [(None, ["g1"]), (None, ["g2"])]
    assert r.embedder.calls == 0
    # seeded by a query, the same
    monkeypatch.setattr(exp, "seed_from_query", lambda q, seed_top_k=None, rerank=None:
                        exp.seed_from_ids(["c1"])[0])
    queried = S.graph_expand(S.GraphExpandIn(q="caching", depth=1))
    assert {n["id"]: n.get("routed_ids") for n in queried["nodes"]} == {
        "c1": None, "g1": ["b-p60", "b-p62"], "g2": []}


def test_routed_ids_are_each_spans_first_per_target_chunks(monkeypatch):
    # pp. 59-63 hold b-p60 and b-p62, more than per_target 1: the first in page order only.
    # A cite with no pages spans the whole file, so it gives the file's first chunk.
    whole = {"source_file": "Books/b.pdf", "page_start": None, "page_end": None}
    exp, r = _map_world({"g1": [B_SPAN], "g2": [whole]}, per_target=1)
    monkeypatch.setitem(S._STATE, "rag", SimpleNamespace(graph=exp, retriever=r))
    out = S.graph_expand(S.GraphExpandIn(seeds=["c1"], depth=1))
    assert {n["id"]: n.get("routed_ids") for n in out["nodes"]} == {
        "c1": None, "g1": ["b-p60"], "g2": ["b-p10"]}


def test_a_stale_bm25_names_its_fix_in_the_traversal_error(monkeypatch):
    # g1 is in the store and the sidecar but not in the loaded bm25 payload (it predates g1):
    # routed_for's ValueError comes back inside the expand try, never as a bare 500.
    exp, r = _map_world({"g1": [B_SPAN]})
    p = r._bm25_payload
    keep = [i for i, d in enumerate(p["ids"]) if d != "g1"]
    r._bm25_payload = {"bm25": p["bm25"], **{k: [p[k][i] for i in keep]
                                              for k in ("ids", "documents", "metadatas")}}
    monkeypatch.setitem(S._STATE, "rag", SimpleNamespace(graph=exp, retriever=r))
    out = S.graph_expand(S.GraphExpandIn(seeds=["c1"], depth=1))
    assert "rebuild_bm25" in out["error"] and out["error"].startswith("Traversal failed (ValueError")
    assert out["nodes"] == []


def test_graph_expand_without_a_sidecar_attaches_nothing(monkeypatch):
    exp, r = _map_world(None)
    monkeypatch.setitem(S._STATE, "rag", SimpleNamespace(graph=exp, retriever=r))
    calls = _spy_routed_for(monkeypatch, r)
    out = S.graph_expand(S.GraphExpandIn(seeds=["c1"], depth=1))
    assert [n["id"] for n in out["nodes"]] == ["c1", "g1", "g2"]
    assert all("routed_ids" not in n for n in out["nodes"]) and calls == []


def _never(self, nodes):
    raise AssertionError("attach_routed_ids ran outside POST /graph/expand")


def _walk_rag(monkeypatch, exp, r):
    """A route pipeline over the map world for graph_search: search() is a stand-in whose
    one hit is c1 (the walk's seed) and whose echo carries the search-time block, ROUTED;
    seed_from_query adds nothing; the rerank keeps the union's order."""
    monkeypatch.setattr(exp, "seed_from_query", lambda q, seed_top_k=None, rerank=None: [])
    seed = exp.seed_from_ids(["c1"])[0]

    def search(question, **kw):
        return list(seed), {"rerank_top_k": 5, "rerank_mode": "none", "rerank_instruction": None,
                            "timings": {"total": 1.0}, "route": dict(ROUTED)}

    rag = RAGPipeline.__new__(RAGPipeline)
    rag.search, rag.presets, rag.graph, rag.retriever = search, {}, exp, r
    rag.hyde = SimpleNamespace(code_intent_signal=lambda q: None)
    rag.reranker = SimpleNamespace(rerank=lambda q, docs, top_k=None, mode=None, instruction=None: docs[:top_k])
    rag.parent_ctx = rag.neighbor_ctx = None
    rag.provenance_map = {"pdf": "source"}
    rag.generated_mode = "route"
    return rag


# The walk routes g1 (to b-p60 and b-p62) and finds nothing for g2.
WALK_ROUTED = {"router_hits": 2, "routed": 1, "unrouted": 1, "targets": 1, "targets_missing": 0, "new": 2}


def test_graph_search_walks_never_attach_routed_ids(monkeypatch):
    monkeypatch.setattr(GraphExpander, "attach_routed_ids", _never)
    exp, r = _map_world({"g1": [B_SPAN]})
    calls = _spy_routed_for(monkeypatch, r)
    docs, info = _walk_rag(monkeypatch, exp, r).graph_search("caching", graph={"depth": 1}, top_k=5)
    assert calls == [("caching", ["g1", "g2"])]            # the walk routed, scored, in one call
    assert {"b-p60", "b-p62"} <= {d.id for d in docs} and info["graph"]["route"] == WALK_ROUTED


def test_evidence_pack_forwards_the_walks_own_route_block(monkeypatch):
    exp, r = _map_world({"g1": [B_SPAN]})
    pack = build_evidence_pack(_walk_rag(monkeypatch, exp, r), "caching", None, search_kwargs={},
                               graph={"depth": 1}, top_k=5, min_score=None, include_text=0,
                               max_sub_questions=4)
    (entry,) = pack["retrieval"]
    # each block under its own key, never summed: search()'s, then the walk's
    assert entry["route"] == ROUTED and entry["graph_route"] == WALK_ROUTED
    items = {e["id"]: e for e in pack["evidence"]}
    assert [f["id"] for f in items["b-p60"]["found_via"]] == ["g1"]


def test_a_per_call_route_walk_on_a_drop_pipeline_routes(monkeypatch):
    # The live path after the merge: drop configured, route per call (graph=true).
    exp, r = _map_world({"g1": [B_SPAN]})
    rag = _walk_rag(monkeypatch, exp, r)
    rag.generated_mode = "drop"
    pack = build_evidence_pack(rag, "caching", None, search_kwargs={"generated_mode": "route"},
                               graph={"depth": 1}, top_k=5, min_score=None, include_text=0,
                               max_sub_questions=4)
    (entry,) = pack["retrieval"]
    assert entry["graph_route"] == WALK_ROUTED
    items = {e["id"]: e for e in pack["evidence"]}
    assert not {"g1", "g2"} & set(items)                   # the maps routed, never evidence
    assert [f["id"] for f in items["b-p60"]["found_via"]] == ["g1"]
