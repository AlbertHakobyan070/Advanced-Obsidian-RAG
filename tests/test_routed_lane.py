"""generated_mode route: the routed lane (route-then-cite §5.4).

Generated chunks never ground an answer, but under route they still ROUTE: a router
hit (a dense or bm25 pass over the generated rows, or a HyPE generated parent) brings
the primary pages it cites (the pointer sidecar) into the pool, and the reranker
decides. The world is a real bm25s payload and a Chroma stand-in that reads `where`
and `ids` the way chromadb 1.5.9 does; one test runs the same queries on a real Chroma.
The router searches by where; a span ranks its own primary rows by ids, with no where.
"""
import re
import sys
import unicodedata
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import yaml

from src.embeddings.embedder import BM25Index, _tokenize
from src.retrieval.retriever import HybridRetriever, _clean_lane_weights
from src.retrieval.route import RoutePointers
from src.utils.chroma_client import persistent_client
from src.utils.config_loader import Config
from src.utils.provenance import ProvenanceRule
from src.utils.timing import StageTrace

ROOT = Path(__file__).resolve().parents[1]
RULE = ProvenanceRule(by_file_type={"pdf": "source"}, generated_paths=("maps",))
ROUTE = {"router_k": 5, "per_target": 3, "max_targets": 8, "max_new": 10}
ZERO = dict.fromkeys(("router_hits", "routed", "unrouted", "targets", "targets_missing", "new"), 0)
GEN = {"provenance": "generated"}
B_SPAN = {"source_file": "Books/b.pdf", "page_start": 59, "page_end": 63}
Q = "prompt caching"
TANTIVY = "generated_mode route needs retrieval.sparse_backend: bm25s in config.yaml"


def _pdf(path, page, **extra):
    return {"source_file": path, "file_type": "pdf", "provenance": "source",
            "page_start": page, "page_end": page, **extra}


def _whole(path):
    return {"source_file": path, "page_start": None, "page_end": None}


def _rows(src="Books\\b.pdf"):
    """g1 is the map that cites b.pdf; b-p60 shares no word with Q and ranks below the
    dense lane's cut, so only routing brings it in. g2 is a generated row Q never names."""
    return [("g1", "prompt caching map", {"source_file": "Maps/caching.canvas", "file_type": "canvas", **GEN}),
            ("n1", "prompt caching in my notes",
             {"source_file": "notes\\cache.md", "file_type": "note", "provenance": "authored"}),
            ("c-p5", "caching layers", _pdf("Books\\c.pdf", 5)),
            ("b-p60", "the prefix is reused across calls", _pdf(src, 60)),
            ("b-p10", "tokenizer basics", _pdf(src, 10)),
            ("g2", "attention map", {"source_file": "Maps/attention.canvas", "file_type": "canvas", **GEN})]


def _matches(meta, where):
    if where is None:
        return True
    if "$and" in where:
        assert len(where["$and"]) >= 2, "chromadb rejects an $and of fewer than two clauses"
        return all(_matches(meta, w) for w in where["$and"])
    ((key, cond),) = where.items()
    value = meta.get(key)
    if not isinstance(cond, dict):
        return value == cond
    ((op, arg),) = cond.items()        # an operator outside these three ($ne included) is a KeyError
    return value is not None and {"$in": lambda: value in arg, "$lte": lambda: value <= arg,
                                  "$gte": lambda: value >= arg}[op]()


class _Store:
    """Chroma's query over `rows`, ranked in row order: exact equality and $in, a
    comparison a missing key fails, an $and of two clauses or more; `ids` keeps only
    those rows. Records each where, and each ids list (None when a query sent none).
    get(ids, include=["embeddings"]) gives stored vectors whose cosine with the query
    falls in row order, and records each id list it was asked for."""

    def __init__(self, rows):
        self.rows, self.wheres, self.ids, self.gets = rows, [], [], []

    def get(self, ids, include):
        self.gets.append(list(ids))
        order = [r[0] for r in self.rows]
        return {"ids": [i for i in ids if i in order],
                "embeddings": [[1.0, 0.01 * order.index(i)] for i in ids if i in order]}

    def query(self, query_embeddings, n_results, include, where=None, ids=None):
        self.wheres.append(where)
        self.ids.append(ids)
        hit = [r for r in self.rows if _matches(r[2], where)
               and (ids is None or r[0] in ids)][:n_results]
        return {"ids": [[i for i, _, _ in hit]], "documents": [[t for _, t, _ in hit]],
                "metadatas": [[m for _, _, m in hit]]}


class _Embedder:
    calls = 0

    def embed_query(self, text):
        self.calls += 1
        return [1.0, 0.0]


def _route_world(pointers, rows=None, route=None, store=None):
    """(retriever, the wheres its store was sent). pointers None = no sidecar loaded."""
    rows = rows or _rows()
    r = HybridRetriever(
        embedder=_Embedder(), chroma_dir=Path("unused"), bm25_index=Path("unused"),
        collection_name="unused", dense_top_k=2, sparse_top_k=2, metadata_boost=False,
        code_file_types=["py"], provenance_rule=RULE, route=dict(ROUTE, **(route or {})),
        route_pointers=None if pointers is None else RoutePointers(
            {"version": 1, "built_at": "2026-10-29T00:00:00Z", "pointers": pointers, "unrouted": {}}))
    r._bm25_payload = {"bm25": BM25Index([_tokenize(t) for _, t, _ in rows]),
                       "ids": [i for i, _, _ in rows], "documents": [t for _, t, _ in rows],
                       "metadatas": [m for _, _, m in rows]}
    r._collection = _Store(rows) if store is None else store
    return r, (r._collection.wheres if store is None else None)


def _ids(rows):
    return [row[0] for row in rows]


# ---- the lane ----

def test_a_generated_hit_brings_its_cited_span_into_the_pool():
    r, _ = _route_world(pointers={"g1": [B_SPAN]})
    tr = StageTrace()
    docs = r.retrieve(Q, generated_mode="route", trace=tr)
    ids = [d.id for d in docs]
    assert "g1" not in ids and "b-p60" in ids
    # The label is the ROUTER's (the map's file), not the page's own.
    assert next(d for d in docs if d.id == "b-p60").debug["found_via"] == [
        {"id": "g1", "label": "Maps/caching.canvas"}]
    assert tr.route == dict(ZERO, router_hits=2, routed=1, unrouted=1, targets=1, new=1)
    assert {"route", "lane.routed"} <= set(tr.ms) and tr.lanes["routed"] == 1


def test_span_lookup_matches_mac_path_forms():                       # Review Focus 1
    stored = "Books\\Café.pdf"
    r, wheres = _route_world(rows=_rows(stored), pointers={"g1": [
        {"source_file": unicodedata.normalize("NFD", "books/CAFÉ.pdf"), "page_start": 59, "page_end": 63}]})
    assert "b-p60" in [d.id for d in r.retrieve(Q, generated_mode="route")]
    assert r._collection.gets == [["b-p60", "b-p10"]]   # the file's primary rows, by the stored spelling


def test_the_span_ids_and_router_wheres_run_on_a_real_chroma(tmp_path):
    pytest.importorskip("chromadb")
    rows = _rows("Books\\Café.pdf")
    col = persistent_client(tmp_path).get_or_create_collection("route", metadata={"hnsw:space": "cosine"})
    col.add(ids=_ids(rows), embeddings=[[1.0, 0.01 * n] for n in range(len(rows))],
            documents=[t for _, t, _ in rows], metadatas=[m for _, _, m in rows])
    nfd = unicodedata.normalize("NFD", "books/CAFÉ.pdf")
    r, _ = _route_world(rows=rows, store=col, pointers={
        "g1": [{"source_file": nfd, "page_start": 59, "page_end": 63}]})
    assert "b-p60" in [d.id for d in r.retrieve(Q, generated_mode="route")]
    # A target with no pages spans its whole file: every primary row's id.
    assert set(_ids(r._span_rank([1.0, 0.0], r._bm25_scores(Q), _whole(nfd), 3))) == {"b-p60", "b-p10"}


def test_a_generated_row_without_pointers_is_counted_unrouted_and_never_returned():   # Review Focus 4
    tr = StageTrace()
    r, _ = _route_world(pointers={})
    docs = r.retrieve(Q, generated_mode="route", trace=tr)
    assert all(d.metadata.get("provenance") != "generated" for d in docs)
    assert tr.route == dict(ZERO, router_hits=2, unrouted=2)


def test_no_sidecar_loaded_means_every_router_hit_is_unrouted():
    r, _ = _route_world(pointers=None)
    assert StageTrace().route == ZERO                    # the six keys, at 0, before any call
    assert r.routed_for(Q, ["g1"]) == ([], dict(ZERO, router_hits=1, unrouted=1))
    tr = StageTrace()
    for _ in range(2):                                   # summed across calls on one trace
        r.retrieve(Q, generated_mode="route", trace=tr)
    assert tr.route == dict(ZERO, router_hits=4, unrouted=4)


def _never(*args):
    raise AssertionError("bm25 scored")


def test_routed_for_without_a_query_lists_spans_in_page_order_without_embedding(monkeypatch):
    r, wheres = _route_world(pointers={"g1": [_whole("Books/b.pdf")]}, route={"per_target": 1})
    monkeypatch.setattr(r._bm25_payload["bm25"], "get_scores", _never)
    docs, summary = r.routed_for(None, ["g1"])
    # b-p60 comes first in the store, b-p10 first in page order. The label is the
    # router's own row, looked up by id in the payload.
    assert [d.id for d in docs] == ["b-p10"]
    assert docs[0].debug["found_via"] == [{"id": "g1", "label": "Maps/caching.canvas"}]
    assert r.embedder.calls == 0 and wheres == []
    assert summary == dict(ZERO, router_hits=1, routed=1, targets=1, new=1)
    # A router id named twice counts once (routed_for dedupes its input).
    twice, summary_twice = r.routed_for(None, ["g1", "g1"])
    assert [f["id"] for f in twice[0].debug["found_via"]] == ["g1"] and summary_twice == summary


@pytest.mark.parametrize("unservable", [
    {"source_file": "Books/gone.pdf", "page_start": 1, "page_end": 2},     # the file is not indexed
    {"source_file": "Books/b.pdf", "page_start": 30, "page_end": 31}])     # indexed, no row in its pages
def test_a_target_the_store_cannot_serve_is_counted_not_queried(unservable):
    # Under max_targets 1 too: the unservable target, listed first, spends no slot (T11-c5).
    r, wheres = _route_world(pointers={"g1": [unservable, B_SPAN]}, route={"max_targets": 1})
    tr = StageTrace()
    assert "b-p60" in [d.id for d in r.retrieve(Q, generated_mode="route", trace=tr)]
    assert (tr.route["targets"], tr.route["targets_missing"]) == (1, 1)
    assert "gone" not in repr(wheres)
    assert r._collection.gets == [["b-p60", "b-p10"]]    # only the served file's vectors
    assert all(i is None for i in r._collection.ids)     # and no span query
    assert [d.id for d in r.routed_for(None, ["g1"])[0]] == ["b-p60"]


def test_net_new_candidates_are_capped_by_max_new():
    r, _ = _route_world(pointers={"g1": [_whole("Books/c.pdf"), _whole("Books/b.pdf")]}, route={"max_new": 1})
    tr = StageTrace()
    docs = {d.id: d for d in r.retrieve(Q, generated_mode="route", trace=tr)}
    # c-p5 is already a dense hit: it stays in the routed lane, uncounted, and one new seat is left.
    assert docs["c-p5"].debug["routed_rank"] == 0 and docs["c-p5"].debug["found_via"][0]["id"] == "g1"
    assert "b-p60" in docs and "b-p10" not in docs and tr.route["new"] == 1


def test_targets_are_capped_and_deduped_in_router_order():
    r, _ = _route_world(route={"max_targets": 2}, pointers={
        "g1": [B_SPAN, {"source_file": "Books\\c.pdf", "page_start": 5, "page_end": 5}],
        "g2": [{"source_file": "BOOKS/B.PDF", "page_start": 59, "page_end": 63},       # B_SPAN again
               {"source_file": "Books/b.pdf", "page_start": 10, "page_end": 10}]})   # a third: capped
    docs, summary = r.routed_for(None, ["g1", "g2"])
    assert [d.id for d in docs] == ["b-p60", "c-p5"]
    assert [f["id"] for f in docs[0].debug["found_via"]] == ["g1", "g2"]
    assert summary == dict(ZERO, router_hits=2, routed=2, targets=2, new=2)


def test_each_primary_comes_once_with_every_router_that_reached_it_in_router_order():
    # Task 12's walk credit and its no-dedupe rerank rely on this. In router order g3, g1,
    # g2 (neither id order nor span order), b-p60 is in two spans: the first cited by g3
    # and g2, the second by g1.
    rows = _rows() + [("g3", "x", {"source_file": "Maps/x.canvas", "file_type": "canvas", **GEN})]
    p60 = {"source_file": "Books/b.pdf", "page_start": 60, "page_end": 60}
    r, _ = _route_world(rows=rows, pointers={"g3": [B_SPAN], "g1": [p60], "g2": [B_SPAN]})
    for query in (None, Q):
        docs, _ = r.routed_for(query, ["g3", "g1", "g2"])
        assert [d.id for d in docs] == ["b-p60"], query
        assert [f["id"] for f in docs[0].debug["found_via"]] == ["g3", "g1", "g2"], query


def test_span_ranking_never_returns_a_generated_row():
    # b-g sits in the span, matches Q, and is the store's nearest row.
    rows = [("b-g", "prompt caching summary", _pdf("Books\\b.pdf", 61, provenance="generated"))] + _rows()
    r, wheres = _route_world(rows=rows, pointers={"g1": [B_SPAN]})
    assert _ids(r._span_rank([1.0, 0.0], r._bm25_scores(Q), B_SPAN, 3)) == ["b-p60"]
    assert "b-g" not in [d.id for d in r.retrieve(Q, generated_mode="route")]
    assert [d.id for d in r.routed_for(None, ["g1"])[0]] == ["b-p60"]
    assert "$ne" not in repr(wheres)
    assert not any("b-g" in ids for ids in r._collection.gets)            # never asked for


def test_the_score_vector_is_computed_once_per_call(monkeypatch):
    r, _ = _route_world(pointers={"g1": [B_SPAN]})
    bm25, calls = r._bm25_payload["bm25"], []
    real = bm25.get_scores
    monkeypatch.setattr(bm25, "get_scores", lambda toks: calls.append(1) or real(toks))
    # sparse, sparse_code, the router's bm25 pass and the span all share one vector
    r.retrieve(Q, generated_mode="route", boost_code=True)
    assert len(calls) == 1
    r.routed_for(Q, ["g1"])
    assert len(calls) == 2


def test_route_pool_is_a_superset_of_drop():
    r, _ = _route_world(pointers={"g1": [B_SPAN]})
    drop = {d.id for d in r.retrieve(Q, generated_mode="drop")}
    route = {d.id for d in r.retrieve(Q, generated_mode="route")}
    assert drop < route and route - drop == {"b-p60"}


def test_routed_weight_and_lane_selection_apply():
    r, wheres = _route_world(pointers={"g1": [B_SPAN]})
    weighed = {d.id: d for d in r.retrieve(Q, generated_mode="route", lane_weights={"routed": 0})}
    assert weighed["b-p60"].score == 0                  # still in the pool, no fusion score
    wheres.clear()
    ids = [d.id for d in r.retrieve(Q, generated_mode="route", lanes=["dense", "sparse"])]
    assert "b-p60" not in ids and {"provenance": "generated"} not in wheres     # no router search
    assert [d.id for d in r.retrieve(Q, generated_mode="route", lanes=["routed"])] == ["b-p60"]


def test_hype_generated_parents_are_router_hits(monkeypatch):
    r, _ = _route_world(pointers={"h1": [B_SPAN]})
    r.hype_enabled = True
    monkeypatch.setattr(r, "_hype_search", lambda qvec, k: [
        ("h1", "t", {"source_file": "Maps/h.md", "provenance": "generated"})])
    docs = {d.id: d for d in r.retrieve(Q, generated_mode="route")}
    assert "h1" not in docs and docs["b-p60"].debug["found_via"] == [{"id": "h1", "label": ANY}]
    assert "b-p60" not in {d.id for d in r.retrieve(Q, generated_mode="route", hype=False)}


def _router_world(router_k):
    """Five maps, each citing its own book. bm25 for Q: m2 > m3 > m4 > 0 = m0 = m1. The
    store (the dense order) ranks m1, m3, m2, m4, m0: not the payload order, not the ids."""
    maps = {"m0": "unrelated text here", "m1": "dense favourite node", "m2": "prompt caching",
            "m3": "caching layers", "m4": "caching appendix with many filler words to make it long"}
    gen = {m: (m, t, {"source_file": f"Maps/{m}.canvas", "file_type": "canvas", **GEN}) for m, t in maps.items()}
    books = [(f"p-{m}", "x", _pdf(f"Books\\{m}.pdf", 1)) for m in maps]
    store = _Store([gen[m] for m in ("m1", "m3", "m2", "m4", "m0")] + books)
    r, _ = _route_world(rows=list(gen.values()) + books, store=store,
                        route={"router_k": router_k, "max_targets": 1},
                        pointers={m: [_whole(f"Books/{m}.pdf")] for m in maps})
    return r


@pytest.mark.parametrize("router_k, winner, hits", [(2, "m3", 3), (3, "m2", 4), (4, "m2", 4)])
def test_router_hits_are_the_rrf_fusion_of_router_k_dense_and_bm25_hits(router_k, winner, hits):
    # k=2: dense m1, m3 + bm25 m2, m3 -> m3 leads on two lanes. k=3: m2 (dense 2, bm25 0)
    # edges m3 (1, 1). k=4: bm25 has three rows above 0, so the zero-score m0 is no hit.
    # Under max_targets 1 only the top router's book is routed.
    tr = StageTrace()
    docs = _router_world(router_k).retrieve(Q, generated_mode="route", lanes=["routed"], trace=tr)
    assert [d.id for d in docs] == [f"p-{winner}"]
    assert docs[0].debug["found_via"] == [{"id": winner, "label": f"Maps/{winner}.canvas"}]
    assert tr.route["router_hits"] == hits == tr.route["routed"]


def _span_world(route=None):
    """One map citing all of s.pdf. Store order gs, s5, s1, s2, s3, s4; bm25 for Q: s5 > s4,
    the rest 0. So the dense span is s5, s1, s2, ... and the bm25 span s5, s4."""
    s = lambda cid, text, page: (cid, text, _pdf("Books\\s.pdf", page))
    rows = [("gs", "prompt caching map", {"source_file": "Maps/s.canvas", "file_type": "canvas", **GEN}),
            s("s5", "prompt caching", 5), s("s1", "filler one", 1), s("s2", "filler two", 2),
            s("s3", "filler three", 3), s("s4", "prompt caching deep dive", 4)]
    return _route_world(rows=rows, route=route, pointers={"gs": [_whole("Books/s.pdf")]})[0]


def test_a_span_interleaves_its_dense_and_bm25_halves_and_keeps_per_target():
    # dense s5, s1, s2 / bm25 s5, s4: s5 once, then s1, then s4 (last in the store, so only
    # the bm25 half finds it), cut to per_target 3 before s2.
    r = _span_world()
    assert _ids(r._span_rank([1.0, 0.0], r._bm25_scores(Q), _whole("Books/s.pdf"), 3)) == ["s5", "s1", "s4"]
    assert [d.id for d in r.routed_for(Q, ["gs"])[0]] == ["s5", "s1", "s4"]
    assert [d.id for d in r.retrieve(Q, generated_mode="route", lanes=["routed"])] == ["s5", "s1", "s4"]
    r = _span_world(route={"per_target": 2})
    assert [d.id for d in r.routed_for(Q, ["gs"])[0]] == ["s5", "s1"]
    assert [d.id for d in r.retrieve(Q, generated_mode="route", lanes=["routed"])] == ["s5", "s1"]


def test_the_span_ranks_in_process_under_every_stored_spelling_with_no_chroma_query():
    # b-p61 is stored under a second spelling of b.pdf and matches no query word, so only
    # the dense span, ranking every primary row of the span by its stored vector, can find
    # it. A span sends Chroma no query at all (any per-span query cost seconds on the live
    # store): the file's vectors come from one get, cached for every later call.
    rows = _rows() + [("b-p61", "the cache key", _pdf("books/b.pdf", 61))]
    r, wheres = _route_world(rows=rows, pointers={"g1": [B_SPAN]})
    assert {"b-p60", "b-p61"} <= {d.id for d in r.retrieve(Q, generated_mode="route")}
    assert r._collection.gets == [["b-p60", "b-p10", "b-p61"]]
    assert all(i is None for i in r._collection.ids)
    assert not any(key in repr(w) for w in wheres for key in ("page_start", "page_end", "source_file"))
    r.retrieve(Q, generated_mode="route")
    r.routed_for(Q, ["g1"])
    assert len(r._collection.gets) == 1                  # cached: no second fetch


def test_a_span_row_missing_from_chroma_names_the_rebuild():
    r, _ = _route_world(pointers={"g1": [B_SPAN]})
    r._collection.rows = [row for row in r._collection.rows if row[0] != "b-p10"]
    with pytest.raises(ValueError, match="not in Chroma.*rebuild bm25"):
        r.retrieve(Q, generated_mode="route")


def test_the_page_order_path_sorts_by_page_then_document_order_and_reads_overlap():
    # Pages 9-50: p-5-10 and p-40-120 straddle the edges and are in; p-1 and p-100 are out.
    # Page order disagrees with id order, and p-9a / p-9b tie on page 9: the payload
    # (document order) has b first, so b comes first.
    def p(cid, start, end):
        return (cid, "x", dict(_pdf("Books\\p.pdf", start), page_end=end))
    rows = [("gp", "map", {"source_file": "Maps/p.canvas", "file_type": "canvas", **GEN}),
            p("p-100", 100, 100), p("p-9b", 9, 9), p("p-40-120", 40, 120), p("p-9a", 9, 9),
            p("p-5-10", 5, 10), p("p-1", 1, 1)]
    r, _ = _route_world(rows=rows, route={"per_target": 10},
                        pointers={"gp": [{"source_file": "Books/p.pdf", "page_start": 9, "page_end": 50}]})
    in_span = ["p-5-10", "p-9b", "p-9a", "p-40-120"]
    assert [d.id for d in r.routed_for(None, ["gp"])[0]] == in_span
    assert {d.id for d in r.routed_for(Q, ["gp"])[0]} == set(in_span)     # the dense ids agree


def test_a_pageless_note_cite_gives_the_notes_first_chunks_in_document_order():
    # A note's chunks carry no pages, and its ids (content hashes) sort against the
    # order it was written in: the first per_target come in payload order.
    note = lambda cid: (cid, "x", {"source_file": "notes\\guide.md", "file_type": "note",
                                   "provenance": "authored"})
    rows = [("gn", "map", {"source_file": "Maps/n.canvas", "file_type": "canvas", **GEN}),
            note("n-z-preamble"), note("n-m-start-here"), note("n-a-second"), note("n-b-third")]
    r, _ = _route_world(rows=rows, pointers={"gn": [_whole("notes/guide.md")]})
    assert [d.id for d in r.routed_for(None, ["gn"])[0]] == ["n-z-preamble", "n-m-start-here", "n-a-second"]


def test_a_generated_row_appended_after_startup_is_unrouted_and_never_returned():   # Review Focus 4
    # In Chroma, not in the loaded payload, the generated-row cache or the sidecar.
    late = ("g-late", "prompt caching late map", {"source_file": "Maps/late.canvas", "file_type": "canvas", **GEN})
    r, _ = _route_world(pointers={"g1": [B_SPAN]}, store=_Store([late] + _rows()))
    tr = StageTrace()
    ids = [d.id for d in r.retrieve(Q, generated_mode="route", trace=tr)]
    assert "g-late" not in ids and "b-p60" in ids
    assert tr.route == dict(ZERO, router_hits=3, routed=1, unrouted=2, targets=1, new=1)


def test_routed_for_names_the_stale_bm25_index_for_a_router_it_lacks():
    # The sidecar (rebuilt from the JSONLs) knows g-late; the loaded pickle does not.
    r, _ = _route_world(pointers={"g-late": [B_SPAN]})
    with pytest.raises(ValueError, match=r"'g-late' is not in the loaded bm25 index .*rebuild_bm25\.py"):
        r.routed_for(None, ["g-late"])


def test_the_file_and_id_maps_are_built_once_per_payload():
    r, _ = _route_world(pointers={})
    files, ids = r._file_rows(), r._id_rows()
    assert r._file_rows() is files and r._id_rows() is ids
    assert files["books/b.pdf"] == ([3, 4], frozenset({"Books\\b.pdf"})) and ids["g2"] == 5
    r._bm25_payload = dict(r._bm25_payload)          # a reloaded payload rebuilds both
    assert r._file_rows() is not files and r._file_rows() == files
    assert r._id_rows() is not ids and r._id_rows() == ids


# ---- refusals ----

def test_route_without_a_route_block_raises():
    r, _ = _route_world(pointers={"g1": [B_SPAN]})
    r.route = None
    msg = "generated_mode route needs the retrieval.route block in config.yaml"
    # A route call refuses even when its lanes leave routed out: the mode, not the lane, asks.
    for lanes in (None, ["dense"]):
        with pytest.raises(ValueError, match=msg):
            r.retrieve(Q, generated_mode="route", lanes=lanes)
    with pytest.raises(ValueError, match=msg):
        r.routed_for(None, ["g1"])
    assert "b-p60" not in {d.id for d in r.retrieve(Q, generated_mode="drop")}     # drop needs no block


def test_route_with_tantivy_refuses(tmp_path):
    paths = {"chroma_dir": str(tmp_path), "bm25_index": str(tmp_path / "b.pkl")}
    cfg = Config({"paths": paths, "retrieval": {
        "generated_mode": "route", "sparse_backend": "tantivy", "route": ROUTE}}, tmp_path)
    # Raised before the tantivy index check: there is no index here, and that error would hide this one.
    with pytest.raises(ValueError, match=f"^{re.escape(TANTIVY)}$"):
        HybridRetriever.from_config(cfg, SimpleNamespace(spec=None))
    # A drop-built tantivy retriever (no route block either) asked for route per call.
    r = HybridRetriever(embedder=None, chroma_dir=tmp_path, bm25_index=tmp_path / "b.pkl",
                        collection_name="c", sparse_backend="tantivy", provenance_rule=RULE)
    for call in (lambda: r.retrieve(Q, generated_mode="route"),
                 lambda: r.retrieve(Q, generated_mode="route", lanes=["dense"]),
                 lambda: r.routed_for(Q, ["g1"])):
        with pytest.raises(ValueError, match=f"^{re.escape(TANTIVY)}$"):
            call()


def test_the_page_order_path_runs_on_tantivy_and_the_scored_calls_do_not():     # T11-c2
    # Task 13 calls routed_for(None, ...) whenever a sidecar is loaded, in every mode, and
    # the page-order path reads the payload alone.
    r, _ = _route_world(pointers={"g1": [B_SPAN]})
    r.sparse_backend = "tantivy"
    docs, summary = r.routed_for(None, ["g1"])
    assert [d.id for d in docs] == ["b-p60"] and summary == dict(ZERO, router_hits=1, routed=1, targets=1, new=1)
    for call in (lambda: r.routed_for(Q, ["g1"]), lambda: r.retrieve(Q, generated_mode="route")):
        with pytest.raises(ValueError, match=f"^{re.escape(TANTIVY)}$"):
            call()


@pytest.mark.parametrize("bad", [{"max_new": None}, {"per_target": -1}, {"router_k": True},
                                 {"max_targets": "3"}, {"max_new": 0}])
def test_a_bad_route_block_fails_the_build_naming_its_key(bad):
    key = next(iter(bad))
    block = {k: v for k, v in dict(ROUTE, **bad).items() if v is not None}
    with pytest.raises(ValueError, match=rf"retrieval\.route\.{key} must be a positive integer"):
        HybridRetriever(embedder=None, chroma_dir=Path("unused"), bm25_index=Path("unused"),
                        collection_name="unused", route=block)
    with pytest.raises(ValueError, match="retrieval.route must be a mapping"):
        HybridRetriever(embedder=None, chroma_dir=Path("unused"), bm25_index=Path("unused"),
                        collection_name="unused", route=[10])


def test_from_config_takes_the_route_block_and_never_reads_the_sidecar(tmp_path, monkeypatch):
    import src.retrieval.retriever as mod
    monkeypatch.setattr(mod, "check_collection", lambda *a, **k: None)
    monkeypatch.setattr(mod, "unstamped_warning", lambda *a, **k: "unstamped")
    sidecar = tmp_path / "route_pointers.json"
    sidecar.write_text("not json", encoding="utf-8")          # loading it would raise
    block = dict(ROUTE, pointers_file=str(sidecar))
    cfg = Config({"paths": {"chroma_dir": str(tmp_path), "bm25_index": str(tmp_path / "b.pkl")},
                  "retrieval": {"generated_mode": "route", "route": block}}, tmp_path)
    r = HybridRetriever.from_config(cfg, SimpleNamespace(spec=None))
    assert r.route == block and r.route_pointers is None      # pipeline.from_config assigns it


# ---- config ----

@pytest.mark.parametrize("name", ["config.yaml", "config.example.yaml"])
def test_routed_ships_in_both_configs_lane_weights(name):
    path = ROOT / name
    if not path.exists():
        pytest.skip(f"{name} is not in this checkout")
    weights = yaml.safe_load(path.read_text(encoding="utf-8"))["retrieval"]["lane_weights"]
    assert weights["routed"] == 1.0 and _clean_lane_weights(weights, name)["routed"] == 1.0
