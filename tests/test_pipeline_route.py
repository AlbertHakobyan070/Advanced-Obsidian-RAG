"""retrieval.generated_mode route in the pipeline (route-then-cite S2, spec §5.4, §5.5).

At build the pipeline refuses a configured route with no retrieval.route block, loads the
pointer sidecar whenever the block names an existing file (in every mode, so a drop
pipeline can serve a per-call route), and refuses a configured route while the store
holds generated rows and no sidecar is loaded. A per-call route refuses the same way,
on the count taken at startup. search() echoes the routed lane's summary; graph_search
swaps walked generated nodes for the primaries they cite.

No model loads: the build tests use the real Embedder around a fake
sentence_transformers module, the real HybridRetriever, a pickled payload and a real
Chroma collection under tmp_path; the search and walk tests use stand-ins.
"""
import json
import pickle
import sys
import types
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from src.pipeline import RAGPipeline, _route_coverage
from src.retrieval.decompose import Decomposer
from src.retrieval.retriever import RetrievedDoc
from src.retrieval.route import RoutePointers
from src.utils.config_loader import Config
from src.utils.provenance import RouteUnavailableError, UnstampedProvenanceError
from src.utils.timing import ROUTE_KEYS

chromadb = pytest.importorskip("chromadb")

DIM = 16
ROUTE = {"pointers_file": "route_pointers.json", "router_k": 5, "per_target": 3,
         "max_targets": 8, "max_new": 10}
GEN_META = {"provenance": "generated", "source_file": "m.canvas"}
PRIMARY = {"source_file": "Books\\b.pdf", "provenance": "source", "file_type": "pdf"}
TARGET = {"source_file": "Books/b.pdf", "page_start": 4, "page_end": 6}
ROUTED = {"router_hits": 3, "routed": 2, "unrouted": 1, "targets": 2, "targets_missing": 0, "new": 4}


# ---- the build: the real retriever (copies of tests/test_pipeline_generated.py's
# fake_st, _cfg and _setup: private test helpers are not imported across files)

def _fake_st(monkeypatch):
    class FakeSentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, texts, *args, **kwargs):
            return np.stack([np.ones(DIM) / np.sqrt(DIM) for _ in texts])

        def get_embedding_dimension(self):
            return DIM

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def _boot(tmp_path, monkeypatch, mode, metas=(), route=ROUTE, sidecar=None, bm25=True):
    """RAGPipeline.from_config over a config with retrieval.generated_mode `mode` and,
    unless route is None, the retrieval.route block; sidecar (a dict) is written to its
    pointers_file. With bm25, the pickled payload p0.. over `metas` and a real collection
    holding the same rows."""
    import src.pipeline as P
    _fake_st(monkeypatch)
    tmp_path.mkdir(parents=True, exist_ok=True)
    retrieval = {"generated_mode": mode}
    if route is not None:
        retrieval["route"] = route
    cfg = Config({"paths": {"chunks_file": "chunks.jsonl", "chroma_dir": "chroma_db",
                            "bm25_index": "bm25.pkl", "collection_name": "obsidian_vault"},
                  "embedding": {"provider": "local", "local_model": "fake/model-a"},
                  "retrieval": retrieval}, tmp_path)
    if sidecar is not None:
        cfg.path("retrieval.route.pointers_file").write_text(json.dumps(sidecar), encoding="utf-8")
    if bm25:
        metas = list(metas)
        ids = [f"p{i}" for i in range(len(metas))]
        with open(cfg.path("paths.bm25_index"), "wb") as f:
            pickle.dump({"ids": ids, "documents": ["text"] * len(ids), "metadatas": metas,
                         "bm25": None}, f)
        col = chromadb.PersistentClient(path=str(cfg.path("paths.chroma_dir"))).get_or_create_collection(
            "obsidian_vault", metadata={"hnsw:space": "cosine"})
        col.add(ids=ids, documents=["text"] * len(ids), metadatas=metas,
                embeddings=[[float(j == n % DIM) for j in range(DIM)] for n in range(len(ids))])
    stand_in = SimpleNamespace(from_config=lambda *args, **kwargs: SimpleNamespace())
    for name in ("LLMClient", "HyDE", "Generator", "GraphExpander"):
        monkeypatch.setattr(P, name, stand_in)
    monkeypatch.setattr(P, "configure_logging", lambda **kwargs: None)
    return RAGPipeline.from_config(cfg)


def _sidecar(pointers):
    return {"version": 1, "built_at": "2026-10-29T00:00:00Z", "pointers": pointers, "unrouted": {}}


def test_route_configured_with_generated_rows_and_no_sidecar_refuses_to_start(tmp_path, monkeypatch):
    with pytest.raises(RouteUnavailableError, match="python main.py route-pointers"):
        _boot(tmp_path, monkeypatch, mode="route", metas=[GEN_META])


def test_route_with_no_generated_rows_starts_without_a_sidecar(tmp_path, monkeypatch):
    rag = _boot(tmp_path / "a", monkeypatch, mode="route", metas=[PRIMARY])
    assert rag.generated_mode == "route" and rag.retriever.route_pointers is None
    assert rag.route_coverage == {"generated_rows": 0, "with_pointers": 0, "sidecar": None}
    assert rag.outline == {"groups": [{"domain": "unknown", "course": "unknown", "sources": [
        {"source_file": "Books\\b.pdf", "file_type": "pdf", "chunks": 1, "pages": None}]}], "maps": []}
    rag = _boot(tmp_path / "b", monkeypatch, mode="route", bm25=False)     # a fresh install
    assert rag.route_coverage == {"generated_rows": 0, "with_pointers": 0, "sidecar": None}
    assert rag.outline == {"groups": [], "maps": []}


def test_a_configured_route_with_no_route_block_is_refused_before_the_sidecar_check(tmp_path, monkeypatch):
    # generated rows and no sidecar too: the block refusal comes first, and it is no
    # RouteUnavailableError, whose fix (route-pointers) itself needs the block
    with pytest.raises(ValueError, match=r"^generated_mode route needs the retrieval\.route block "
                                         r"in config\.yaml$"):
        _boot(tmp_path, monkeypatch, mode="route", metas=[GEN_META], route=None)


def test_a_configured_route_with_a_sidecar_starts_and_reports_its_coverage(tmp_path, monkeypatch):
    rag = _boot(tmp_path, monkeypatch, mode="route", metas=[GEN_META, GEN_META, PRIMARY],
                sidecar=_sidecar({"p0": [TARGET]}))
    assert rag.retriever.route_pointers.targets("p0") == [TARGET]
    assert rag.route_coverage == {"generated_rows": 2, "with_pointers": 1,
                                  "sidecar": str(tmp_path / "route_pointers.json")}


@pytest.mark.parametrize("mode", ["off", "drop"])
def test_the_sidecar_loads_in_every_mode(tmp_path, monkeypatch, mode):
    """So a drop pipeline can serve a per-call route."""
    rag = _boot(tmp_path, monkeypatch, mode=mode, metas=[GEN_META], sidecar=_sidecar({"p0": [TARGET]}))
    assert rag.retriever.route_pointers.targets("p0") == [TARGET]
    assert rag.route_coverage["with_pointers"] == 1


@pytest.mark.parametrize("loaded", [True, False])
def test_a_loaded_sidecar_builds_the_routed_lanes_row_maps_at_startup(tmp_path, monkeypatch, loaded):
    """So the first routed or map-view request after a start does not pay for them; with
    no sidecar loaded the build touches neither."""
    rag = _boot(tmp_path, monkeypatch, mode="drop", metas=[GEN_META, PRIMARY],
                sidecar=_sidecar({"p0": [TARGET]}) if loaded else None)
    r = rag.retriever
    assert (r._file_cache is not None, r._id_cache is not None) == (loaded, loaded)
    if loaded:          # built over the payload the coverage read, so never rebuilt
        assert r._file_cache[0] is r._get_bm25() and r._id_cache[0] is r._get_bm25()
        assert r._id_rows() == {"p0": 0, "p1": 1}


@pytest.mark.parametrize("mode", ["off", "drop", "route"])
def test_a_bad_sidecar_fails_the_boot_in_every_mode(tmp_path, monkeypatch, mode):
    with pytest.raises(ValueError, match="sidecar version 2, expected 1"):
        _boot(tmp_path, monkeypatch, mode=mode, metas=[GEN_META],
              sidecar={"version": 2, "pointers": {}, "unrouted": {}})


def test_a_per_call_route_without_a_sidecar_raises(tmp_path, monkeypatch):
    rag = _boot(tmp_path, monkeypatch, mode="drop", metas=[GEN_META])
    assert rag.generated_rows == 1 and rag.retriever.route_pointers is None
    # raised before HyDE: the stand-in here has no methods at all
    with pytest.raises(RouteUnavailableError, match="python main.py route-pointers"):
        rag.search("q", generated_mode="route")


def test_a_configured_route_refuses_unstamped_rows_before_the_missing_sidecar(tmp_path, monkeypatch):
    # stamping comes first: a sidecar built over unstamped rows would need a second restart
    unstamped = {"source_file": "Maps/x (generated).canvas", "provenance": "authored"}
    with pytest.raises(UnstampedProvenanceError, match="stamp-provenance"):
        _boot(tmp_path, monkeypatch, mode="route", metas=[GEN_META, unstamped])


@pytest.mark.parametrize("mode, value", [("off", None), ("drop", ""), ("route", None)])
def test_a_route_block_without_its_pointers_file_fails_every_boot(tmp_path, monkeypatch, mode, value):
    route = {k: v for k, v in ROUTE.items() if k != "pointers_file"}
    if value is not None:
        route["pointers_file"] = value
    with pytest.raises(ValueError, match=r"^retrieval\.route\.pointers_file is not set: .*config\.yaml"):
        _boot(tmp_path, monkeypatch, mode=mode, metas=[GEN_META], route=route)


@pytest.mark.parametrize("mode", ["off", "drop"])
def test_off_and_drop_with_no_route_block_touch_nothing_new_on_the_retriever(tmp_path, monkeypatch, mode):
    """A retriever stand-in holding only bm25_index, as tests/test_relevance_gate.py builds."""
    import src.pipeline as P
    stand_in = SimpleNamespace(from_config=lambda *args, **kwargs: SimpleNamespace())
    for name in ("Embedder", "LLMClient", "HyDE", "Generator", "GraphExpander"):
        monkeypatch.setattr(P, name, stand_in)
    retriever = SimpleNamespace(bm25_index=tmp_path / "no-such-bm25.pkl")
    monkeypatch.setattr(P, "HybridRetriever", SimpleNamespace(from_config=lambda *a, **k: retriever))
    monkeypatch.setattr(P, "configure_logging", lambda **kwargs: None)
    rag = RAGPipeline.from_config(Config({"retrieval": {"generated_mode": mode}}, tmp_path))
    assert rag.route_coverage == {"generated_rows": 0, "with_pointers": 0, "sidecar": None}
    assert vars(retriever) == {"bm25_index": tmp_path / "no-such-bm25.pkl"}     # nothing assigned


# ---- search(): stand-ins (copies of tests/test_pipeline_generated.py's helpers)

class _Reranker:
    mode = "none"
    model_name = "test/reranker"
    max_length = 512
    instruction = None

    @staticmethod
    def rerank(question, docs, top_k=None, mode=None, instruction=None):
        return docs[:top_k]

    @staticmethod
    def scoring_query(query, mode, instruction=None):
        return query


class _Hyde:
    @staticmethod
    def expand_with_info(question, enabled=None):
        return question, "off"

    @staticmethod
    def code_intent_signal(question):
        return None


class _LLM:
    model = "stub"

    def __init__(self, reply):
        self.reply = reply

    def complete(self, system, user, temperature=None, max_tokens=None):
        return SimpleNamespace(text=self.reply)


class _RoutingRetriever:
    """Each retrieve() adds `adds` (ROUTED) to the trace's route block, as the routed lane does."""
    dense_top_k = sparse_top_k = 20
    omnisearch = None
    hype_enabled = False
    lane_weights: dict = {}
    metadata_boost = True
    rrf_k = 60
    sparse_backend = "bm25s"
    route_pointers = None
    route = {k: v for k, v in ROUTE.items() if k != "pointers_file"}

    def __init__(self, adds=ROUTED):
        self.modes, self.adds = [], adds

    def retrieve(self, query, generated_mode="off", trace=None, **kwargs):
        self.modes.append(generated_mode)
        for key, n in self.adds.items():
            trace.route[key] += n
        return [RetrievedDoc("a", "t", {"source_file": "Books\\b.pdf", "file_type": "pdf"})]

    def _get_collection(self):
        return None


def _pipeline(retriever, **kw):
    return RAGPipeline(retriever=retriever, reranker=_Reranker(), hyde=_Hyde(),
                       generator=None, rerank_top_k=3, **kw)


def test_search_echoes_the_route_block():
    retr = _RoutingRetriever()
    _, info = _pipeline(retr, generated_mode="route").search("q")
    assert info["route"] == ROUTED
    # every retrieve() of a decomposed search adds to the one block
    rag = _pipeline(retr, generated_mode="route",
                    decomposer=Decomposer(_LLM('["part one", "part two"]'), enabled=True))
    _, info = rag.search("q")
    assert info["route"] == {k: 3 * n for k, n in ROUTED.items()}
    for mode in ("off", "drop"):
        _, info = _pipeline(retr).search("q", generated_mode=mode)
        assert "route" not in info
    # a route call nothing routed still carries the six keys, at 0 (the bench averages them)
    _, info = _pipeline(_RoutingRetriever(adds={}), generated_mode="route").search("q")
    assert info["route"] == dict.fromkeys(ROUTE_KEYS, 0)


def test_the_per_call_check_reads_the_startup_count_and_follows_the_unstamped_one():
    retr = _RoutingRetriever()
    rag = _pipeline(retr)
    assert rag.route_coverage is None                      # only from_config computes it
    rag.search("q", generated_mode="route")                # zero generated rows at startup: runs
    rag.generated_rows = 1
    with pytest.raises(RouteUnavailableError, match="python main.py route-pointers"):
        rag.search("q", generated_mode="route")
    rag.search("q", generated_mode="drop")                 # drop needs no sidecar
    rag.unstamped = {"Maps\\a.md": 1}
    with pytest.raises(ValueError, match="stamp-provenance"):   # the unstamped check first
        rag.search("q", generated_mode="route")
    rag.unstamped = {}
    retr.route = None              # no route block: its refusal first, since route-pointers needs it
    with pytest.raises(ValueError, match=r"^generated_mode route needs the retrieval\.route block "
                                         r"in config\.yaml$"):
        rag.search("q", generated_mode="route")
    del retr.route                 # back to the class's block
    retr.route_pointers = RoutePointers(_sidecar({}))
    _, info = rag.search("q", generated_mode="route")
    assert info["route"] == ROUTED
    assert retr.modes == ["route", "drop", "route"]        # no refused call reached retrieve()


def _no_draft(question, enabled=None):
    raise AssertionError("HyDE drafted for a call that was going to be refused")


def test_a_per_call_route_on_tantivy_is_refused_before_the_409_and_before_hyde():
    # Generated rows and no sidecar too: the 409's fix (build a sidecar) would not make a
    # tantivy server route, so tantivy is refused first.
    retr = _RoutingRetriever()
    retr.sparse_backend = "tantivy"
    rag = _pipeline(retr)
    rag.generated_rows = 1
    rag.hyde = SimpleNamespace(expand_with_info=_no_draft)
    with pytest.raises(ValueError, match=r"^generated_mode route needs retrieval\.sparse_backend: "
                                         r"bm25s in config\.yaml$"):
        rag.search("q", generated_mode="route")
    assert retr.modes == []


class _BranchRetriever(_RoutingRetriever):
    """Each retrieve() builds its own doc objects, as the real one does: x in every branch,
    routed only in the branches `routes` names (branch text -> its router ids)."""

    def __init__(self, routes):
        super().__init__(adds={})
        self.routes = routes

    def retrieve(self, query, generated_mode="off", trace=None, **kwargs):
        super().retrieve(query, generated_mode, trace, **kwargs)
        via = {"found_via": [{"id": g, "label": f"Maps/{g}.canvas"} for g in self.routes[query]]} \
            if query in self.routes else {}
        return [RetrievedDoc("x", "t", {"source_file": "Books\\b.pdf", "file_type": "pdf"}, debug=via)]


@pytest.mark.parametrize("routes, via", [
    ({"part two": ["g1"]}, ["g1"]),                     # only a sub-query's branch routed x
    ({"q": ["g1"]}, ["g1"]),                            # only the original question's
    ({"q": ["g2"], "part two": ["g1", "g2"]}, ["g2", "g1"])])   # unioned by id, first seen first
def test_a_decomposed_route_keeps_found_via_whichever_branch_routed(routes, via):
    rag = _pipeline(_BranchRetriever(routes), generated_mode="route",
                    decomposer=Decomposer(_LLM('["part one", "part two"]'), enabled=True))
    docs, _ = rag.search("q")
    assert [d.id for d in docs] == ["x"]
    assert docs[0].debug["found_via"] == [{"id": g, "label": f"Maps/{g}.canvas"} for g in via]


# ---- graph_search(): stand-ins. Not tests/test_graph_search.py's: its search already
# returns p1, and its pipeline has no retriever.

def _search(question, **kw):
    """search() stand-in: n1 a note, c1 a generated canvas hit (the walk's seed), p0 a page.
    Its echo carries the search-time route block, ROUTED."""
    return [RetrievedDoc("n1", "note", {"file_type": "note", "source_file": "Notes/n.md"}),
            RetrievedDoc("c1", "map", {"file_type": "canvas", "source_file": "Maps/a.canvas",
                                       "provenance": "generated"}),
            RetrievedDoc("p0", "page 0", PRIMARY)], {
        "rerank_top_k": kw.get("top_k") or 3, "rerank_mode": "cross_encoder",
        "rerank_instruction": None, "parent_context": False, "parent_swaps": 0,
        "parent_siblings_dropped": 0, "neighbor_context": False, "neighbors_added": 0,
        "timings": {"total": 1.0}, "route": dict(ROUTED)}


class _Walk:
    """GraphExpander stand-in. The walk reaches `nodes` beyond its seeds; query_seeds are
    seed_from_query's hits. _fetch_ids serves them REVERSED: Chroma's order is not the walk's."""
    file_type = "canvas"
    wikilinks = False

    def __init__(self, nodes, query_seeds=()):
        self.nodes, self.query_seeds = list(nodes), list(query_seeds)

    @staticmethod
    def _meta(i):
        return {"file_type": "canvas", "source_file": "Maps/b.canvas",
                **({"provenance": "generated"} if i[0] in "gs" else {})}

    def seed_from_query(self, question, seed_top_k=None, rerank=None):
        return [RetrievedDoc(i, f"seed {i}", self._meta(i)) for i in self.query_seeds]

    def expand(self, seeds, **opts):
        nodes = [{"id": d.id, "depth": 0} for d in seeds] + self.nodes
        return {"nodes": nodes, "cross_edges": [], "stats": {"nodes": len(nodes)}, "tree": []}

    def _fetch_ids(self, ids):
        return [RetrievedDoc(i, f"walked {i}", self._meta(i)) for i in reversed(ids)]


class _Router:
    """routed_for stand-in, with a sidecar loaded: routes maps a router id to the primary ids
    it cites; each primary comes once, found_via naming every router id that reached it, in
    router order."""

    def __init__(self, routes):
        self.routes, self.calls = routes, []
        self.route_pointers = RoutePointers(_sidecar({}))

    def routed_for(self, query, router_ids):
        self.calls.append((query, list(router_ids)))
        via: dict[str, list[str]] = {}
        for rid in router_ids:
            for pid in self.routes.get(rid, []):
                via.setdefault(pid, []).append(rid)
        docs = [RetrievedDoc(pid, f"page {pid}", dict(PRIMARY),
                             debug={"found_via": [{"id": r, "label": f"map {r}"} for r in rids]})
                for pid, rids in via.items()]
        return docs, {"router_hits": len(router_ids), "routed": len(router_ids), "unrouted": 0,
                      "targets": len(docs), "targets_missing": 0, "new": len(docs)}


class _NoRouter:
    def __init__(self, sidecar=True):
        self.route_pointers = RoutePointers(_sidecar({})) if sidecar else None

    def routed_for(self, query, router_ids):
        raise AssertionError("routed_for must not run here")


class _Rerank:
    def __init__(self):
        self.calls = []

    def rerank(self, question, docs, top_k=None, mode=None, instruction=None):
        self.calls.append([d.id for d in docs])
        return docs[:top_k]


def _graph_rag(walk, retriever, mode="route"):
    rag = RAGPipeline.__new__(RAGPipeline)
    rag.search = _search
    rag.presets = {}
    rag.hyde = SimpleNamespace(code_intent_signal=lambda q: None)
    rag.graph = walk
    rag.retriever = retriever
    rag.reranker = _Rerank()
    rag.parent_ctx = rag.neighbor_ctx = None
    rag.provenance_map = {"pdf": "source"}
    rag.generated_mode = mode
    return rag


# (configured mode, per-call mode): after the merge drop is configured, so a route walk
# is a per-call route (/evidence_pack graph=true generated_mode=route).
WALK_MODES = [("route", None), ("drop", "route"), ("off", "route")]


@pytest.mark.parametrize("configured, per_call", WALK_MODES)
def test_graph_search_replaces_walked_generated_nodes_with_their_routed_primaries(configured, per_call):
    # walk: generated g1 (depth 1) routes to p1, to p0 (already a search hit) and to w2
    # (walked itself); generated g2 (depth 2) routes nowhere
    router = _Router({"g1": ["p1", "p0", "w2"]})
    rag = _graph_rag(_Walk([{"id": "g1", "depth": 1}, {"id": "w2", "depth": 1},
                            {"id": "g2", "depth": 2}]), router, mode=configured)
    docs, info = rag.graph_search("q", graph={}, top_k=5, generated_mode=per_call)
    assert router.calls == [("q", ["g1", "g2"])]            # walk order, not _fetch_ids'
    (ids,) = rag.reranker.calls
    assert ids == ["n1", "c1", "p0", "w2", "p1"]            # p0 and w2 once; g1, g2 never
    assert "p1" in info["graph"]["added_ids"]
    assert "g1" not in info["graph"]["added_ids"] + info["graph"]["seed_ids"]
    assert info["graph"]["added_ids"] == ["w2", "p1"] and info["graph"]["seed_ids"] == []
    assert [d.id for d in docs].count("p0") <= 1
    assert next(d for d in docs if d.id == "p1").debug["found_via"][0]["id"] == "g1"
    # new counts what the walk added: p0 (a search hit) and w2 (walked) were in the pool
    assert info["graph"]["route"] == {"router_hits": 2, "routed": 2, "unrouted": 0,
                                      "targets": 3, "targets_missing": 0, "new": 1}
    assert info["route"] == ROUTED          # the walk's summary is not summed into search()'s


@pytest.mark.parametrize("configured, per_call", WALK_MODES)
def test_with_no_sidecar_walked_generated_nodes_just_leave(configured, per_call):
    # no sidecar (zero generated rows at startup, g1 appended since): g1 just leaves
    rag = _graph_rag(_Walk([{"id": "g1", "depth": 1}, {"id": "w2", "depth": 1}]),
                     _NoRouter(sidecar=False), mode=configured)
    _, info = rag.graph_search("q", graph={}, top_k=5, generated_mode=per_call)
    assert rag.reranker.calls == [["n1", "c1", "p0", "w2"]]
    assert info["graph"]["added_ids"] == ["w2"] and "route" not in info["graph"]


def test_a_primary_routed_only_from_a_depth_zero_seed_stays_a_seed():
    # s9: a generated query seed (depth 0) the search lacked; g1 a generated walked node
    router = _Router({"s9": ["p5", "p6"], "g1": ["p6"]})
    rag = _graph_rag(_Walk([{"id": "g1", "depth": 1}], query_seeds=["s9"]), router)
    _, info = rag.graph_search("q", graph={}, top_k=5)
    assert router.calls == [("q", ["s9", "g1"])]
    assert info["graph"]["seed_ids"] == ["p5"]              # reached only through the seed
    assert info["graph"]["added_ids"] == ["p6"]             # also reached by an edge


def test_routed_for_runs_only_under_route_with_generated_walked_nodes_and_a_sidecar():
    rag = _graph_rag(_Walk([{"id": "w2", "depth": 1}]), _NoRouter())
    _, info = rag.graph_search("q", graph={}, top_k=5)
    assert info["graph"]["added_ids"] == ["w2"] and "route" not in info["graph"]
    walk = [{"id": "g1", "depth": 1}, {"id": "w2", "depth": 1}]
    for mode, kept in (("drop", ["w2"]), ("off", ["w2", "g1"])):    # _fetch_ids order
        rag = _graph_rag(_Walk(walk), _NoRouter(), mode=mode)
        _, info = rag.graph_search("q", graph={}, top_k=5)
        assert info["graph"]["added_ids"] == kept and "route" not in info["graph"]
    rag = _graph_rag(_Walk(walk), _NoRouter())
    _, info = rag.graph_search("q", graph=None, top_k=5)
    assert "route" not in info["graph"]


def test_coverage_counts_generated_rows_with_and_without_pointers():
    ids = ["g1", "g2", "g3", "p1"]
    metas = [{"provenance": "generated"}] * 3 + [{"provenance": "source"}]
    # g2 has no targets; p1 is primary and "gone" is not stored, so neither counts
    pointers = RoutePointers(_sidecar({"g1": [TARGET], "p1": [TARGET], "gone": [TARGET]}))
    assert _route_coverage(ids, metas, pointers, "x/route_pointers.json") == {
        "generated_rows": 3, "with_pointers": 1, "sidecar": "x/route_pointers.json"}
    assert _route_coverage(ids, metas, None, None) == {
        "generated_rows": 3, "with_pointers": 0, "sidecar": None}
