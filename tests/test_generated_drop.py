"""retrieval.generated_mode drop: generated chunks never ground an answer.

The filter sits inside each lane, at retrieve()'s call sites, so each lane
refills from primary rows. Graph seeding calls the lane helpers directly and
must keep seeing generated canvases, so the helpers filter only when asked.
Under off, retrieve() calls every helper exactly as before: the narrow lane
stubs other test files use must keep working.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import yaml

from src.embeddings.embedder import BM25Index, _tokenize
from src.retrieval.retriever import HybridRetriever
from src.retrieval.scope import Scope
from src.utils.chroma_client import persistent_client
from src.utils.provenance import GENERATED_MODES, ProvenanceRule, check_mode

ROOT = Path(__file__).resolve().parents[1]
RULE = ProvenanceRule(by_file_type={"pdf": "source"}, generated_paths=("maps",),
                      instruction_files=frozenset({"claude.md"}),
                      generated_markers=("(generated)",))
SCOPE = Scope(domains=["stats"])
GEN_CANVAS = {"source_file": "x/Map (generated).canvas", "file_type": "live_note", "live": True}
# The Chroma filter the dense drop replaces; here it is the oracle the drop must match.
NE_GENERATED = {"provenance": {"$ne": "generated"}}


# ---- helpers (the _retriever / _stub_lanes pattern of tests/test_lane_weights.py) ----

def _retriever(rule=RULE):
    r = HybridRetriever.__new__(HybridRetriever)
    r.embedder = type("E", (), {"embed_query": staticmethod(lambda t: [0.0])})()
    r.rrf_k = 60
    r.dense_top_k = r.sparse_top_k = 20
    r.metadata_boost = False
    r.code_file_types = ["py"]
    r.omnisearch = None
    r.hype_enabled = False
    r.hype_top_k = 15
    r.lane_weights = {}
    r.provenance_rule = rule
    return r


def _empty_lanes(r, monkeypatch):
    monkeypatch.setattr(r, "_dense_search", lambda qvec, k, where=None, drop_generated=False: [])
    monkeypatch.setattr(r, "_sparse_search", lambda q, k, predicate=None, drop_generated=False: [])


def _recording_dense(monkeypatch):
    """(k, where, drop_generated) each dense lane passes to _dense_search, in
    retrieve()'s call order; the real _dense_scope_search runs, so its clause is
    the one it builds."""
    r = _retriever()
    seen, names = {}, ("dense", "dense_code", "dense_scope")

    def dense(qvec, k, where=None, drop_generated=False):
        seen[names[len(seen)]] = (k, where, drop_generated)
        return []
    monkeypatch.setattr(r, "_dense_search", dense)
    monkeypatch.setattr(r, "_sparse_search", lambda q, k, predicate=None, drop_generated=False: [])
    return r, seen


class _RankedCollection:
    """A Chroma stand-in that returns `metas` in order (ignoring `where`) and
    records each query's (n_results, where)."""

    def __init__(self, metas):
        self.metas, self.calls = metas, []

    def query(self, query_embeddings, n_results, include, where=None):
        self.calls.append((n_results, where))
        hit = list(enumerate(self.metas))[:n_results]
        return {"ids": [[f"r{i}" for i, _ in hit]], "documents": [["t"] * len(hit)],
                "metadatas": [[m for _, m in hit]]}


def _bm25s_retriever(rows, rule=RULE):
    """A real bm25s payload over `rows` = [(id, text, meta)]."""
    r = _retriever(rule)
    r._bm25_payload = {"bm25": BM25Index([_tokenize(t) for _, t, _ in rows]),
                       "ids": [i for i, _, _ in rows], "documents": [t for _, t, _ in rows],
                       "metadatas": [m for _, _, m in rows]}
    return r


# ---- dense lanes: an over-fetch filtered here, never a Chroma $ne ----

def test_the_dense_drop_keeps_what_a_chroma_ne_filter_keeps(tmp_path):
    pytest.importorskip("chromadb")
    # Four generated rows nearest the query, then a keyless pdf, an authored note,
    # and a generated and a primary py row. (chromadb 1.5.9 rejects names under 3 chars.)
    col = persistent_client(tmp_path).get_or_create_collection("drop", metadata={"hnsw:space": "cosine"})
    rows = [("g1", 0.00, {"provenance": "generated"}), ("g2", 0.01, {"provenance": "generated"}),
            ("g3", 0.02, {"provenance": "generated"}), ("g4", 0.03, {"provenance": "generated"}),
            ("pdf", 0.10, {"file_type": "pdf"}), ("note", 0.20, {"provenance": "authored"}),
            ("gpy", 0.30, {"file_type": "py", "provenance": "generated"}), ("py", 0.40, {"file_type": "py"})]
    col.add(ids=[i for i, _, _ in rows], embeddings=[[1.0, y] for _, y, _ in rows],
            documents=[i for i, _, _ in rows], metadatas=[m for _, _, m in rows])
    r = _retriever()
    r._collection = col

    def ids(got):
        return [i for i, _, _ in got]
    # k=8 asks for more primary rows than exist: the lane stops when Chroma runs out.
    for k, where in ((2, None), (8, None), (1, {"file_type": "py"})):
        oracle = {"$and": [where, NE_GENERATED]} if where else NE_GENERATED
        want = col.query(query_embeddings=[[1.0, 0.0]], n_results=k, where=oracle)["ids"][0]
        assert ids(r._dense_search([1.0, 0.0], k, where=where, drop_generated=True)) == want, (k, where)
    # Spelled out: the lane refilled past the four generated rows and kept the keyless pdf.
    assert ids(r._dense_search([1.0, 0.0], 2, drop_generated=True)) == ["pdf", "note"]
    assert ids(r._dense_search([1.0, 0.0], 2)) == ["g1", "g2"]                 # off: unfiltered


def test_the_dense_drop_sends_chroma_only_the_lanes_where_and_doubles_the_fetch():
    gen, prim = {"provenance": "generated"}, {"file_type": "pdf"}
    r = _retriever()
    # Enough primary rows in the first fetch, exactly top_k of them included:
    # ONE query, of 2 * top_k (a None metadata row has no key, so it stays).
    for metas in ([gen, prim, None, prim, prim], [gen, prim, None, gen, prim]):
        r._collection = col = _RankedCollection(metas)
        assert [i for i, _, _ in r._dense_search([0.0], 2, drop_generated=True)] == ["r1", "r2"]
        assert col.calls == [(4, None)]
    # A crowd of generated rows: 4, 8, 16, stopping when the 12-row collection runs out.
    where = {"file_type": {"$in": ["pdf"]}}
    r._collection = col = _RankedCollection([gen] * 9 + [prim] * 3)
    assert [i for i, _, _ in r._dense_search([0.0], 2, where=where, drop_generated=True)] == ["r9", "r10"]
    assert col.calls == [(4, where), (8, where), (16, where)]


def test_drop_asks_every_dense_lane_to_drop_and_keeps_its_own_where(monkeypatch):
    r, seen = _recording_dense(monkeypatch)
    r.retrieve("q", generated_mode="drop", boost_code=True, scope=SCOPE)
    assert seen == {"dense": (20, None, True),
                    "dense_code": (10, {"file_type": {"$in": ["py"]}}, True),
                    "dense_scope": (10, {"domain": {"$in": ["stats"]}}, True)}
    # A path scope filters client-side from its oversampled fetch, which drops too.
    seen.clear()
    r._dense_scope_search([0.0], 20, Scope(path_contains=["homework"]), drop_generated=True)
    assert seen == {"dense": (80, None, True)}


# ---- sparse lanes: each is asked to drop, and bm25s refills ----

def test_drop_asks_every_sparse_lane_to_drop(monkeypatch):
    r = _retriever()
    asked = []
    monkeypatch.setattr(r, "_dense_search", lambda qvec, k, where=None, drop_generated=False: [])
    monkeypatch.setattr(r, "_sparse_search", lambda q, k, predicate=None, drop_generated=False:
                        asked.append(drop_generated) or [])
    r.retrieve("q", generated_mode="drop", boost_code=True, scope=SCOPE)
    assert asked == [True, True, True]          # sparse, sparse_code, sparse_scope


def test_drop_refills_the_bm25s_lane(monkeypatch):
    r = _bm25s_retriever(rows=[("g1", "svm margin", {"provenance": "generated"}),
                               ("p1", "svm margin", {"file_type": "pdf"}),
                               ("p2", "svm", {"file_type": "pdf"})])
    assert [i for i, _, _ in r._sparse_search("svm margin", 2)] == ["g1", "p1"]
    assert [i for i, _, _ in r._sparse_search("svm margin", 2, drop_generated=True)] == ["p1", "p2"]


def test_the_path_rule_drops_an_unstamped_row_from_the_sparse_lane():
    # No stored "generated" on any of them: a stored authored under a generated
    # path, a marked canvas with no key, an agent-instruction file.
    r = _bm25s_retriever(rows=[("m1", "svm", {"source_file": "Maps/svm.md", "provenance": "authored"}),
                               ("c1", "svm", {"source_file": "x/SVM (generated).canvas"}),
                               ("a1", "svm", {"source_file": "repo/CLAUDE.md"}),
                               ("p1", "svm", {"source_file": "books/svm.pdf", "file_type": "pdf"})])
    assert {i for i, _, _ in r._sparse_search("svm", 10)} == {"m1", "c1", "a1", "p1"}
    assert [i for i, _, _ in r._sparse_search("svm", 10, drop_generated=True)] == ["p1"]


def test_generated_rows_are_computed_once_per_payload():
    r = _bm25s_retriever(rows=[("g1", "svm", {"provenance": "generated"}),
                               ("p1", "svm", {"file_type": "pdf"})])
    first = r._generated_rows()
    assert first == {0} and r._generated_rows() is first
    r._bm25_payload = dict(r._bm25_payload, metadatas=[{"file_type": "pdf"}, {"provenance": "generated"}])
    assert r._generated_rows() == {1}


# ---- hype and omnisearch: post-filtered by the rule ----

def test_hype_generated_parents_leave_and_a_hype_failure_still_raises(monkeypatch):
    r = _retriever()
    r.hype_enabled = True
    _empty_lanes(r, monkeypatch)
    parents = [("h1", "t", {"provenance": "generated"}), ("h2", "t", {"source_file": "Maps/a.md"}),
               ("h3", "t", {"file_type": "pdf"})]
    monkeypatch.setattr(r, "_hype_search", lambda qvec, k: parents)
    assert [d.id for d in r.retrieve("q")] == ["h1", "h2", "h3"]
    assert [d.id for d in r.retrieve("q", generated_mode="drop")] == ["h3"]
    # A bad stored value raises: the post-filter is not inside the fail-soft try.
    monkeypatch.setattr(r, "_hype_search", lambda qvec, k: [("hb", "t", {"provenance": "bogus"})])
    with pytest.raises(ValueError, match="bogus"):
        r.retrieve("q", generated_mode="drop")


def test_omnisearch_rows_of_a_generated_canvas_or_path_leave(monkeypatch):
    r = _retriever()
    rows = [("o1", "t", GEN_CANVAS),
            ("o2", "t", {"source_file": "Maps/svm.md", "file_type": "live_note", "live": True}),
            ("o3", "t", {"source_file": "notes/svm.md", "file_type": "live_note", "live": True})]
    r.omnisearch = SimpleNamespace(enabled=True, lane=lambda q: rows)
    _empty_lanes(r, monkeypatch)
    assert [d.id for d in r.retrieve("q")] == ["o1", "o2", "o3"]
    assert [d.id for d in r.retrieve("q", generated_mode="drop")] == ["o3"]


# ---- off: today's calls, byte for byte ----

def test_off_calls_every_lane_helper_exactly_as_before(monkeypatch):
    # No provenance_rule set: the class default (None) is what a __new__ test
    # retriever has, and off never needs it.
    r = HybridRetriever.__new__(HybridRetriever)
    r.embedder = type("E", (), {"embed_query": staticmethod(lambda t: [0.0])})()
    r.rrf_k, r.dense_top_k, r.sparse_top_k = 60, 20, 20
    r.metadata_boost, r.code_file_types, r.lane_weights = False, ["py"], {}
    r.hype_enabled, r.hype_top_k = True, 15
    gen = {"provenance": "generated"}
    calls = []

    def dense(qvec, k, where=None):              # the narrow stubs of the other files
        calls.append(("dense", where))
        return [(f"d{len(calls)}", "t", gen)]

    def sparse(q, k, predicate=None):
        calls.append(("sparse", predicate is not None))
        return [(f"s{len(calls)}", "t", gen)]
    monkeypatch.setattr(r, "_dense_search", dense)
    monkeypatch.setattr(r, "_sparse_search", sparse)
    monkeypatch.setattr(r, "_dense_scope_search", lambda qvec, k, scope: [("ds", "t", gen)])
    monkeypatch.setattr(r, "_hype_search", lambda qvec, k: [("h", "t", gen)])
    r.omnisearch = SimpleNamespace(enabled=True, lane=lambda q: [("o", "t", GEN_CANVAS)])

    out = {d.id for d in r.retrieve("q", boost_code=True, scope=SCOPE)}
    assert calls == [("dense", None), ("sparse", False),
                     ("dense", {"file_type": {"$in": ["py"]}}), ("sparse", True), ("sparse", True)]
    assert {"ds", "h", "o"} <= out and len(out) == 8, "off dropped a generated row"


def test_off_scope_search_calls_dense_search_exactly_as_before(monkeypatch):
    # The real _dense_scope_search over the narrow stub: an explicit
    # drop_generated=False would raise TypeError here.
    r = _retriever()
    calls = []
    monkeypatch.setattr(r, "_dense_search", lambda qvec, k, where=None: calls.append((k, where)) or [])
    r._dense_scope_search([0.0], 20, SCOPE)
    r._dense_scope_search([0.0], 20, Scope(path_contains=["homework"]))
    assert calls == [(10, {"domain": {"$in": ["stats"]}}), (80, None)]


# ---- errors ----

def test_drop_without_a_rule_raises(monkeypatch):
    r = _retriever(rule=None)
    _empty_lanes(r, monkeypatch)
    with pytest.raises(ValueError, match="needs the provenance rule: build the retriever with from_config"):
        r.retrieve("q", generated_mode="drop")
    r = _bm25s_retriever([("p1", "svm", {})], rule=None)
    with pytest.raises(ValueError, match="needs the provenance rule"):
        r._sparse_search("svm", 1, drop_generated=True)


def test_an_unknown_mode_raises_naming_it(monkeypatch):
    with pytest.raises(ValueError, match="generated_mode") as exc:
        check_mode("of", "generated_mode")
    assert "'of'" in str(exc.value) and "drop" in str(exc.value)
    assert [check_mode(m, "x") for m in GENERATED_MODES] == list(GENERATED_MODES)
    r = _retriever()
    _empty_lanes(r, monkeypatch)
    with pytest.raises(ValueError, match="generated_mode"):
        r.retrieve("q", generated_mode="of")


def test_a_yaml_bare_off_is_refused_naming_the_quotes():
    # yaml.safe_load reads `generated_mode: off` as False; false could as well
    # mean "no generated chunks", so it is refused rather than read as off.
    assert yaml.safe_load("m: off")["m"] is False
    with pytest.raises(ValueError, match=r'quote it \("off"\)'):
        check_mode(False, "retrieval.generated_mode")


# ---- config ----

def test_the_mode_ships_in_both_configs():
    example = yaml.safe_load((ROOT / "config.example.yaml").read_text(encoding="utf-8"))
    assert example["retrieval"]["generated_mode"] == "drop"
    if (ROOT / "config.yaml").exists():
        live = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
        assert live["retrieval"]["generated_mode"] in ("off", "drop", "route")


def test_from_config_builds_the_full_rule(tmp_path, monkeypatch):
    import src.retrieval.retriever as mod
    from src.utils.config_loader import Config
    # The fingerprint guard needs a real store; this test is about the rule only.
    monkeypatch.setattr(mod, "check_collection", lambda *a, **k: None)
    monkeypatch.setattr(mod, "unstamped_warning", lambda *a, **k: "unstamped")
    cfg = Config({"paths": {"chroma_dir": str(tmp_path), "bm25_index": str(tmp_path / "b.pkl")},
                  "provenance": {"generated_paths": ["Maps"]}}, tmp_path)
    rule = HybridRetriever.from_config(cfg, SimpleNamespace(spec=None)).provenance_rule
    assert rule.generated_by_path("Maps/a.md") and rule.instruction_file("repo/AGENTS.md")
