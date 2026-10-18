"""Graph RAG traversal tests — src/retrieval/graph_expand.py.

Fully offline: a fake collection stands in for ChromaDB, so no index, model
or generation backend is constructed.

The properties that matter are the ones that make a graph walk safe rather
than the ones that make it clever:
  - it TERMINATES on a cyclic graph (canvas graphs are cyclic by nature)
  - every cap is reported when it bites, never silently applied
  - a node id is scoped to its canvas file, never global
  - an edge to a node that never became a chunk is skipped, not fatal
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.ingestion.canvas_loader import encode_canvas_edge
from src.retrieval.graph_expand import GraphExpander, _edges_of, _node_key
from src.retrieval.retriever import RetrievedDoc


# ---- a fake chunk store shaped exactly like the Chroma calls we make ----

class FakeCollection:
    def __init__(self, rows: dict[str, tuple[str, dict]]):
        self.rows = rows
        self.calls: list[dict] = []

    def get(self, ids=None, where=None, include=None):
        self.calls.append({"ids": ids, "where": where})
        if ids is not None:
            hits = [(i, *self.rows[i]) for i in ids if i in self.rows]
        else:
            clauses = (where or {}).get("$and", [where or {}])
            source = next((c["source_file"] for c in clauses
                           if "source_file" in c), None)
            wanted = next((c["canvas_node_id"]["$in"] for c in clauses
                           if "canvas_node_id" in c), [])
            hits = [(i, t, m) for i, (t, m) in self.rows.items()
                    if m.get("source_file") == source
                    and m.get("canvas_node_id") in wanted]
        return {"ids": [h[0] for h in hits],
                "documents": [h[1] for h in hits],
                "metadatas": [h[2] for h in hits]}


class FakeRetriever:
    rrf_k = 60

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
    return GraphExpander(retriever=FakeRetriever(FakeCollection(dict(rows))),
                         reranker=None, **kw)


def _seed(expander, cid):
    text, meta = expander.retriever.collection.rows[cid]
    return RetrievedDoc(id=cid, text=text, metadata=meta)


# ---- THE property: a cyclic graph terminates ----

def test_a_three_node_cycle_terminates_and_is_walked_once():
    """A -> B -> C -> A is an ordinary knowledge map, not a pathological case.
    Without a visited set this walk never returns."""
    rows = [
        _chunk("cA", "A", [("B", "->", "leads to"), ("C", "<-", "closes")]),
        _chunk("cB", "B", [("A", "<-", "leads to"), ("C", "->", "then")]),
        _chunk("cC", "C", [("B", "<-", "then"), ("A", "->", "closes")]),
    ]
    exp = _expander(rows)
    out = exp.expand([_seed(exp, "cA")], depth=5)
    assert {n["id"] for n in out["nodes"]} == {"cA", "cB", "cC"}
    assert out["stats"]["visited"] == 3


def test_a_self_loop_terminates():
    rows = [_chunk("cA", "A", [("A", "->", "recurses")])]
    exp = _expander(rows)
    out = exp.expand([_seed(exp, "cA")], depth=3)
    assert [n["id"] for n in out["nodes"]] == ["cA"]


def test_the_cycle_edge_is_reported_rather_than_hidden():
    """Dropping edges back onto reached nodes would draw a cyclic graph as a
    tree — the reader would never learn the loop exists."""
    rows = [
        _chunk("cA", "A", [("B", "->", "leads to")]),
        _chunk("cB", "B", [("A", "->", "loops back"), ("A", "<-", "leads to")]),
    ]
    exp = _expander(rows)
    out = exp.expand([_seed(exp, "cA")], depth=3)
    assert [(e["parent"], e["child"]) for e in out["tree"]] == [("cA", "cB")]
    back = [e for e in out["cross_edges"] if e["label"] == "loops back"]
    assert back and back[0]["parent"] == "cB" and back[0]["child"] == "cA"
    assert out["stats"]["cross_edges"] == len(out["cross_edges"])


# ---- caps: each one reported when it bites ----

def _chain(n: int):
    rows = []
    for i in range(n):
        edges = []
        if i + 1 < n:
            edges.append((f"N{i+1}", "->", f"step {i+1}"))
        if i:
            edges.append((f"N{i-1}", "<-", f"step {i}"))
        rows.append(_chunk(f"c{i}", f"N{i}", edges))
    return rows


def test_depth_cap_stops_the_walk_and_says_so():
    exp = _expander(_chain(6))
    out = exp.expand([_seed(exp, "c0")], depth=2)
    assert {n["id"] for n in out["nodes"]} == {"c0", "c1", "c2"}
    assert max(n["depth"] for n in out["nodes"]) == 2
    assert out["stats"]["truncated_at"] == "depth"


def test_depth_cap_is_not_reported_when_the_graph_simply_ends():
    exp = _expander(_chain(3))
    out = exp.expand([_seed(exp, "c0")], depth=5)
    assert len(out["nodes"]) == 3
    assert out["stats"]["truncated_at"] is None


def test_depth_zero_returns_the_seeds_untouched():
    exp = _expander(_chain(4))
    out = exp.expand([_seed(exp, "c0")], depth=0)
    assert [n["id"] for n in out["nodes"]] == ["c0"]
    assert out["tree"] == []


def test_max_total_truncation_is_reported_honestly():
    exp = _expander(_chain(10))
    out = exp.expand([_seed(exp, "c0")], depth=9, max_total=4)
    assert len(out["nodes"]) <= 4
    assert out["stats"]["truncated_at"] == "max_total"
    assert "max_total" in out["stats"]["caps_hit"]


def test_max_per_hop_truncation_is_reported_honestly():
    hub_edges = [(f"L{i}", "->", f"leaf {i}") for i in range(6)]
    rows = [_chunk("hub", "H", hub_edges)]
    rows += [_chunk(f"c{i}", f"L{i}", [("H", "<-", f"leaf {i}")])
             for i in range(6)]
    exp = _expander(rows)
    out = exp.expand([_seed(exp, "hub")], depth=1, max_per_hop=2)
    assert len(out["nodes"]) == 3          # the hub plus two leaves
    assert out["stats"]["truncated_at"] == "max_per_hop"


def test_every_cap_that_bites_is_listed_not_just_the_first():
    hub_edges = [(f"L{i}", "->", f"leaf {i}") for i in range(6)]
    rows = [_chunk("hub", "H", hub_edges)]
    rows += [_chunk(f"c{i}", f"L{i}",
                    [("H", "<-", f"leaf {i}"), (f"M{i}", "->", "deeper")])
             for i in range(6)]
    rows += [_chunk(f"m{i}", f"M{i}", [(f"L{i}", "<-", "deeper")])
             for i in range(6)]
    exp = _expander(rows)
    out = exp.expand([_seed(exp, "hub")], depth=1, max_per_hop=3)
    assert set(out["stats"]["caps_hit"]) == {"max_per_hop", "depth"}


# ---- identity is per canvas FILE, never a bare node id ----

def test_the_same_node_id_in_two_canvases_is_not_the_same_node():
    rows = [
        _chunk("mine", "shared", [("other", "->", "mine")], source="mine.canvas"),
        _chunk("theirs", "shared", [], source="theirs.canvas"),
        _chunk("target", "other", [], source="mine.canvas"),
    ]
    exp = _expander(rows)
    out = exp.expand([_seed(exp, "mine")], depth=2)
    assert {n["id"] for n in out["nodes"]} == {"mine", "target"}
    assert "theirs" not in {n["id"] for n in out["nodes"]}


def test_node_key_pairs_the_file_with_the_node_id():
    assert _node_key({"source_file": "a.canvas", "canvas_node_id": "n1"}) \
        != _node_key({"source_file": "b.canvas", "canvas_node_id": "n1"})


def test_split_parts_of_one_node_are_walked_once_and_all_returned():
    """canvas.max_chunk_size can split a node across chunks. A hop onto that
    node must reach every part, but must expand the node only once."""
    rows = [
        _chunk("seed", "S", [("BIG", "->", "explains")]),
        _chunk("big1", "BIG", [("S", "<-", "explains")], text="part one"),
        _chunk("big2", "BIG", [("S", "<-", "explains")], text="part two",
               canvas_part=2),
    ]
    exp = _expander(rows)
    out = exp.expand([_seed(exp, "seed")], depth=2)
    assert {n["id"] for n in out["nodes"]} == {"seed", "big1", "big2"}
    assert out["stats"]["visited"] == 2, "one canvas node, not two"


# ---- dangling edges: skipped, counted, never fatal ----

def test_an_edge_to_a_node_that_is_not_a_chunk_is_counted_not_raised():
    """file and group nodes never become chunks, and neither does a text node
    under canvas.min_chunk_size. Their edges are still real edges."""
    rows = [_chunk("cA", "A", [("GONE", "->", "see also"),
                               ("B", "->", "leads to")]),
            _chunk("cB", "B", [("A", "<-", "leads to")])]
    exp = _expander(rows)
    out = exp.expand([_seed(exp, "cA")], depth=2)
    assert {n["id"] for n in out["nodes"]} == {"cA", "cB"}
    assert out["stats"]["dangling_edges"] == 1


def test_a_seed_with_no_edges_at_all_returns_just_itself():
    rows = [_chunk("lonely", "L", [])]
    exp = _expander(rows)
    out = exp.expand([_seed(exp, "lonely")], depth=3)
    assert [n["id"] for n in out["nodes"]] == ["lonely"]
    assert out["stats"]["dangling_edges"] == 0


# ---- edge labels, and the honest fallback for pre-canvas_edges chunks ----

def test_each_hop_carries_the_label_of_the_edge_it_followed():
    rows = [_chunk("cA", "A", [("B", "->", "outputs"), ("C", "<-", "derives")]),
            _chunk("cB", "B", []), _chunk("cC", "C", [])]
    exp = _expander(rows)
    out = exp.expand([_seed(exp, "cA")], depth=1)
    by_child = {e["child"]: e for e in out["tree"]}
    assert by_child["cB"]["label"] == "outputs"
    assert by_child["cB"]["direction"] == "->"
    assert by_child["cC"]["label"] == "derives"
    assert by_child["cC"]["direction"] == "<-"
    assert out["stats"]["edges_labelled"] is True


def test_chunks_predating_canvas_edges_still_traverse_but_say_labels_are_lost():
    """Chunks indexed before the aligned key exists only have
    canvas_neighbors. Refusing to walk them would be worse than walking them
    unlabelled — but pretending the edges have no labels would be worse still."""
    legacy = {"source_file": "old.canvas", "file_type": "canvas",
              "canvas_node_id": "A", "canvas_neighbors": "B, C",
              "canvas_edge_labels": "one label for two edges"}
    edges, labelled = _edges_of(legacy)
    assert [e["neighbor"] for e in edges] == ["B", "C"]
    assert labelled is False
    assert all(e["label"] == "" and e["direction"] == "--" for e in edges)

    rows = [("cA", ("old body", legacy)), _chunk("cB", "B", [], source="old.canvas")]
    exp = _expander(rows)
    out = exp.expand([_seed(exp, "cA")], depth=1)
    assert {n["id"] for n in out["nodes"]} == {"cA", "cB"}
    assert out["stats"]["edges_labelled"] is False


def test_a_node_with_no_edge_metadata_at_all_is_not_an_error():
    rows = [("cA", ("plain body", {"source_file": "x.canvas",
                                   "canvas_node_id": "A"}))]
    exp = _expander(rows)
    out = exp.expand([_seed(exp, "cA")], depth=2)
    assert out["stats"]["edges"] == 0


# ---- seeding from explicit ids ----

def test_seed_from_ids_keeps_the_callers_order_and_names_what_is_missing():
    rows = [_chunk("c1", "A", []), _chunk("c2", "B", []), _chunk("c3", "C", [])]
    exp = _expander(rows)
    docs, missing = exp.seed_from_ids(["c3", "nope", "c1"])
    assert [d.id for d in docs] == ["c3", "c1"]
    assert missing == ["nope"]


def test_duplicate_seed_ids_do_not_duplicate_the_walk():
    rows = [_chunk("c1", "A", [("B", "->", "x")]), _chunk("c2", "B", [])]
    exp = _expander(rows)
    docs, missing = exp.seed_from_ids(["c1", "c1"])
    assert [d.id for d in docs] == ["c1"]
    out = exp.expand(docs, depth=1)
    assert len(out["nodes"]) == 2


def test_caps_come_from_config_when_the_request_omits_them():
    rows = _chain(8)
    exp = _expander(rows, max_depth=1, max_per_hop=3, max_total=9)
    out = exp.expand([_seed(exp, "c0")])
    assert out["stats"]["caps"] == {"depth": 1, "max_per_hop": 3, "max_total": 9}
    assert len(out["nodes"]) == 2


# ---- endpoint contract: the errors that must never be silent ----

def _api(monkeypatch, rows=()):
    """serve_api with a fake warm pipeline — no model, index or backend."""
    import serve_api
    from types import SimpleNamespace
    exp = _expander(list(rows))
    monkeypatch.setitem(serve_api._STATE, "rag",
                        SimpleNamespace(graph=exp, retriever=exp.retriever))
    return serve_api


def test_graph_expand_rejects_q_and_seeds_together(monkeypatch):
    from fastapi import HTTPException
    api = _api(monkeypatch)
    with pytest.raises(HTTPException) as exc_info:
        api.graph_expand(api.GraphExpandIn(q="anything", seeds=["c1"]))
    assert exc_info.value.status_code == 400
    assert "both were given" in str(exc_info.value.detail)


def test_graph_expand_rejects_neither_q_nor_seeds(monkeypatch):
    from fastapi import HTTPException
    api = _api(monkeypatch)
    with pytest.raises(HTTPException) as exc_info:
        api.graph_expand(api.GraphExpandIn())
    assert exc_info.value.status_code == 400
    assert "neither was given" in str(exc_info.value.detail)


def test_graph_expand_names_an_unknown_seed_id(monkeypatch):
    from fastapi import HTTPException
    api = _api(monkeypatch, [_chunk("c1", "A", [])])
    with pytest.raises(HTTPException) as exc_info:
        api.graph_expand(api.GraphExpandIn(seeds=["c1", "ghost"]))
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["unknown"] == ["ghost"]


def test_graph_expand_truncates_node_text_to_include_text(monkeypatch):
    api = _api(monkeypatch, [_chunk("c1", "A", [], text="x" * 900)])
    out = api.graph_expand(api.GraphExpandIn(seeds=["c1"], include_text=100))
    assert len(out["nodes"][0]["text"]) == 100
    bare = api.graph_expand(api.GraphExpandIn(seeds=["c1"], include_text=0))
    assert bare["nodes"][0]["text"] is None


def test_graph_expand_flags_unlabelled_edges_in_the_response(monkeypatch):
    legacy = {"source_file": "old.canvas", "canvas_node_id": "A",
              "canvas_neighbors": "B"}
    api = _api(monkeypatch, [("c1", ("body", legacy))])
    out = api.graph_expand(api.GraphExpandIn(seeds=["c1"]))
    assert "UNLABELLED" in out["note"]


def test_graph_expand_is_503_when_the_mode_is_disabled(monkeypatch):
    from fastapi import HTTPException
    from types import SimpleNamespace
    import serve_api
    exp = _expander([], enabled=False)
    monkeypatch.setitem(serve_api._STATE, "rag", SimpleNamespace(graph=exp))
    with pytest.raises(HTTPException) as exc_info:
        serve_api.graph_expand(serve_api.GraphExpandIn(seeds=["c1"]))
    assert exc_info.value.status_code == 503
    assert "graph.enabled" in str(exc_info.value.detail)


def test_answer_refuses_an_unknown_chunk_id_rather_than_shrinking_context(monkeypatch):
    """An answer grounded on fewer documents than the caller chose is a
    DIFFERENT answer. Silently dropping one is the worst possible outcome."""
    from fastapi import HTTPException
    api = _api(monkeypatch, [_chunk("c1", "A", [])])
    with pytest.raises(HTTPException) as exc_info:
        api.answer(api.AnswerIn(q="why?", docs=["c1", "missing"]))
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["unknown"] == ["missing"]


def test_answer_refuses_evidence_ids_that_are_not_indexed_records(monkeypatch):
    from fastapi import HTTPException
    api = _api(monkeypatch, [_chunk("c1", "A", [])])
    with pytest.raises(HTTPException) as exc_info:
        api.answer(api.AnswerIn(q="why?", docs=["c1", "live:abc", "parent:xyz"]))
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["ids"] == ["live:abc", "parent:xyz"]


def test_answer_requires_at_least_one_document():
    import serve_api
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        serve_api.AnswerIn(q="why?", docs=[])


# ---- graph.embedding_provider: query seeds from <collection>__graph ----
#
# These use a REAL Chroma store under tmp_path (the twin is built by
# src/embeddings/graph_twin.py) and a fake embedding backend: no model, no network.

TWIN_ENTRY = {"kind": "openai", "base_url": "http://127.0.0.1:8104/v1", "model": "fake/g2",
              "dimensions": 8}                       # no prefixes: a query equals its own node


@pytest.fixture
def release_chroma():
    """A store's loaded collections hold C-runtime file streams (4 per HNSW
    segment, of 512 per process) until its System stops, and nothing here closes
    them. Stop the stores this test opened, so later fopen users in the same run
    (PyMuPDF in test_web_import_ocr.py) are not starved."""
    from chromadb.api.shared_system_client import SharedSystemClient
    before = set(SharedSystemClient._identifier_to_system)
    yield
    for key in set(SharedSystemClient._identifier_to_system) - before:
        SharedSystemClient._identifier_to_refcount.pop(key, None)
        SharedSystemClient._identifier_to_system.pop(key).stop()


def _twin_vec(text):
    import hashlib
    import numpy as np
    raw = np.frombuffer(hashlib.sha256(text.encode()).digest()[:8], dtype=np.uint8).astype(float) - 127.5
    return raw / np.linalg.norm(raw)


class _TwinBackend:
    dimension = 8

    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [_twin_vec("g2" + t).tolist() for t in texts]


class _TwinRetriever(FakeRetriever):
    """FakeRetriever over a real main collection. Its embedder stays None, so any
    call to the main embedder fails the test, and its dense lane must not run."""
    collection_name = "vault"

    def _dense_search(self, *args, **kwargs):
        raise AssertionError("the main collection's dense lane ran despite a graph seed index")

    def _sparse_search(self, query, top_k, predicate=None):
        return []


def _twin_cfg(root, provider="g2", entry=None, enabled=True):
    from src.utils.config_loader import Config
    return Config({"paths": {"chunks_file": str(root / "data" / "chunks.jsonl"),
                             "chroma_dir": str(root / "chroma"),
                             "bm25_index": str(root / "bm25.pkl"), "collection_name": "vault"},
                   "embedding": {"provider": "local", "local_model": "fake/main", "batch_size": 4,
                                 "providers": {"g2": entry or TWIN_ENTRY}},
                   "graph": {"enabled": enabled, "file_type": "canvas",
                             "embedding_provider": provider}}, root)


def _twin_embedder(cfg, spec, backend):
    from src.embeddings.embedder import Embedder
    return Embedder(backend=backend, chunks_file=cfg.path("paths.chunks_file"),
                    chroma_dir=cfg.path("paths.chroma_dir"), bm25_index=cfg.path("paths.bm25_index"),
                    collection_name="vault", batch_size=4, spec=spec)


def _twin_build(w, **kw):
    from src.embeddings.graph_twin import build_twin, graph_spec
    return build_twin(w.cfg, _twin_embedder(w.cfg, graph_spec(w.cfg), _TwinBackend()),
                      client=w.client, progress=lambda _: None, **kw)


def _twin_world(tmp_path, monkeypatch, build=True):
    """Main collection 'vault': 6 canvas + 3 markdown chunks whose stored text
    starts "MAIN " (the chunk files hold it without), so a seed's text says which
    collection it was read from. Embedder.from_config hands out the fake g2."""
    import json
    from types import SimpleNamespace
    from src.embeddings.embedder import Embedder
    from src.utils.chroma_client import persistent_client

    data = tmp_path / "data"
    data.mkdir()
    canvas = [(f"c{i}", f"canvas node {i}") for i in range(6)]
    md = [(f"m{i}", f"md text {i}") for i in range(3)]
    for name, rows, ft in (("chunks.jsonl", md, "md"), ("canvas_chunks.jsonl", canvas, "canvas")):
        (data / name).write_text("".join(json.dumps({"doc_id": i, "text": t, "metadata": {
            "file_type": ft}}) + "\n" for i, t in rows), encoding="utf-8")
    cfg = _twin_cfg(tmp_path)
    client = persistent_client(cfg.path("paths.chroma_dir"))
    main = client.get_or_create_collection("vault", metadata={"hnsw:space": "cosine"})
    main.add(ids=[i for i, _ in canvas + md],
             embeddings=[_twin_vec("main" + t).tolist() for _, t in canvas + md],
             documents=["MAIN " + t for _, t in canvas + md],
             metadatas=[{"source_file": "map.canvas", "file_type": "canvas",
                         "canvas_node_id": f"N{i[1:]}"} for i, _ in canvas]
             + [{"source_file": "a.md", "file_type": "md"} for _ in md])
    backend = _TwinBackend()
    monkeypatch.setattr(Embedder, "from_config", classmethod(
        lambda cls, c, spec=None: _twin_embedder(c, spec, backend)))
    w = SimpleNamespace(cfg=cfg, client=client, main=main, backend=backend,
                        retriever=_TwinRetriever(main))
    if build:
        _twin_build(w)
    return w


def test_query_seeds_come_from_the_graph_index_refetched_from_main(tmp_path, monkeypatch, caplog,
                                                                   release_chroma):
    w = _twin_world(tmp_path, monkeypatch)
    w.main.update(ids=["c3"], metadatas=[{"source_file": "map.canvas", "file_type": "canvas",
                                          "canvas_node_id": "N3", "marker": "read from main"}])
    exp = GraphExpander.from_config(w.cfg, w.retriever)
    assert exp.seed_error is None and exp.seed_collection.name == "vault__graph"

    seeds = exp.seed_from_query("canvas node 3", seed_top_k=3)
    assert seeds[0].id == "c3"                                   # the graph embedder's ranking
    assert seeds[0].text == "MAIN canvas node 3"                 # main's text, not the index's
    assert seeds[0].metadata["marker"] == "read from main"       # main's current metadata
    assert len(seeds) == 3 and all(d.metadata["file_type"] == "canvas" for d in seeds)
    assert w.backend.calls[-1] == ["canvas node 3"]              # the query went to the graph embedder
    assert w.retriever.embedder is None                          # the main one was never called

    w.main.delete(ids=["c3"])                                    # gone from main since the build
    with caplog.at_level("WARNING"):
        seeds = exp.seed_from_query("canvas node 3", seed_top_k=3)
    assert "c3" not in [d.id for d in seeds] and len(seeds) == 3
    assert len([r for r in caplog.records if r.levelname == "WARNING"
                and "no longer holds" in r.getMessage()]) == 1


def test_a_seed_error_refuses_query_seeding_not_seeding_by_id(monkeypatch):
    import serve_api
    from types import SimpleNamespace
    rows = [_chunk("c1", "A", [("B", "->", "x")]), _chunk("c2", "B", [("A", "<-", "x")])]
    exp = GraphExpander(retriever=FakeRetriever(FakeCollection(dict(rows))),
                        seed_error="'vault__graph' has not been built")
    with pytest.raises(RuntimeError, match="graph-mode query seeding is unavailable: "
                                           "'vault__graph' has not been built"):
        exp.seed_from_query("anything")
    docs, missing = exp.seed_from_ids(["c1"])
    assert [d.id for d in docs] == ["c1"] and missing == []

    monkeypatch.setitem(serve_api._STATE, "rag", SimpleNamespace(graph=exp, retriever=exp.retriever))
    out = serve_api.graph_expand(serve_api.GraphExpandIn(q="anything"))   # the existing error path
    assert out["nodes"] == [] and "graph-mode query seeding is unavailable" in out["error"]
    out = serve_api.graph_expand(serve_api.GraphExpandIn(seeds=["c1"]))
    assert {n["id"] for n in out["nodes"]} == {"c1", "c2"} and "error" not in out


def test_from_config_turns_unbuilt_stale_or_mismatched_twins_into_seed_errors(
        tmp_path, monkeypatch, caplog, release_chroma):
    w = _twin_world(tmp_path, monkeypatch, build=False)

    def seed_error(cfg=None):
        exp = GraphExpander.from_config(cfg or w.cfg, w.retriever)
        assert exp.seed_error is None or (exp.seed_collection, exp.seed_embedder) == (None, None)
        return exp.seed_error

    with caplog.at_level("ERROR"):
        assert "'vault__graph' has not been built" in seed_error()
    assert any(r.levelname == "ERROR" and "graph-mode query seeding is unavailable" in r.getMessage()
               for r in caplog.records)
    _twin_build(w, limit=2)
    assert "unfinished re-embed" in seed_error()                       # a --limit smoke run
    _twin_build(w)
    assert seed_error() is None
    assert "differ in model" in seed_error(_twin_cfg(tmp_path, entry={**TWIN_ENTRY, "model": "fake/g3"}))
    assert "is not in embedding.providers" in seed_error(_twin_cfg(tmp_path, provider="nope"))
    assert "main collection's embedder" in seed_error(_twin_cfg(tmp_path, provider="local"))
    w.main.delete(ids=["c0"])
    assert "out of step" in seed_error()                               # ids, both directions
    w.client.delete_collection("vault__graph")
    assert "vault__graph] does not exist" in seed_error()             # a fingerprint, no collection


def test_graph_off_or_null_provider_opens_nothing(tmp_path, monkeypatch):
    import src.retrieval.graph_expand as G
    monkeypatch.setattr(G, "_seed_index", lambda *a: pytest.fail("the seed index was opened"))
    for cfg in (_twin_cfg(tmp_path, enabled=False), _twin_cfg(tmp_path, provider=None),
                _twin_cfg(tmp_path, provider="")):                 # empty = unset, like null
        exp = GraphExpander.from_config(cfg, FakeRetriever(FakeCollection({})))
        assert (exp.seed_embedder, exp.seed_collection, exp.seed_error) == (None, None, None)
    assert not (tmp_path / "chroma").exists()
