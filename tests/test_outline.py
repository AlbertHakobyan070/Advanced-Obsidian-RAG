"""The outline for cold agents (route-then-cite S3, spec §5.7).

GET /outline lists what the vault holds: the primary sources grouped by domain and
course, with chunk and page counts, paged; the generated maps apart and whole; the
agent-instruction files nowhere. GET /outline/{source} outlines one file: a source's
chapters with page ranges, a map's concepts with the primary pages each cites. The
source_file and pages fields of every retrieval model restrict every lane of a search
to one file's primary chunks.

Fully offline: a payload literal, a real bm25s index over it, a Chroma stand-in and
stub reranker / HyDE (copies of tests/test_routed_lane.py's and
tests/test_pipeline_route.py's helpers: private test helpers are not imported across
files).
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

import serve_api as S
from src.embeddings.embedder import BM25Index, _tokenize
from src.ingestion.obsidian_parser import ObsidianParser
from src.pipeline import RAGPipeline
from src.retrieval.context_expand import NeighborContext
from src.retrieval.outline import build_outline, source_outline
from src.retrieval.retriever import HybridRetriever
from src.retrieval.route import RoutePointers
from src.retrieval.scope import Scope
from src.utils.config_loader import Config
from src.utils.provenance import ProvenanceRule

RULE = ProvenanceRule(by_file_type={"pdf": "source"}, generated_paths=("blueprints",),
                      instruction_files=frozenset({"agents.md"}), generated_markers=("(generated)",))
AI = {"domain": "ai", "course_name": "AI Engineering"}
MAP = "Maps/m (generated).canvas"
T1 = {"source_file": "Books/b.pdf", "page_start": 2, "page_end": 3}
T2 = {"source_file": "Books/b.pdf", "page_start": 5, "page_end": 5}
Q = "prompt caching"


def _pdf(page, chapter=None, end=None):
    meta = {"source_file": "Books/b.pdf", "file_type": "pdf", "provenance": "source",
            "page_start": page, "page_end": end or page, **AI}
    if chapter:
        meta["chapter"] = chapter
    return meta


def _note(heading, path, level):
    return {"source_file": "Notes/n.md", "file_type": "note", "provenance": "authored",
            "domain": "ml", "heading": heading, "heading_path": path, "heading_level": level}


def _map_part(part):
    return {"source_file": MAP, "file_type": "canvas", "provenance": "generated",
            "canvas_node_id": "n1", "canvas_part": part, "canvas_part_count": 2, **AI}


# Payload order is not page order. b.pdf's p1 has no ToC chapter, and "Summary" closes
# both of its chapters; the map node is split in two and only part 1 carries the
# Source line; AGENTS.md is an agent-instruction file.
ROWS = [
    ("b-p5", "prompt caching costs", _pdf(5, "2 Prompting")),
    ("n-1", "prompt caching in my notes", _note("Intro", "Intro", 1)),
    ("b-p1", "title page", _pdf(1)),
    ("m-n1", "[Canvas: m (generated)]\nPrompt caching\nReuse the prompt prefix.\n"
             "Source: Books/b.pdf pp. 2-3, 5\n\nConnections:\n-> Other node", _map_part(1)),
    ("b-p2", "prompt caching basics", _pdf(2, "1 Foundations")),
    ("n-2", "the cache key", _note("Detail", "Intro > Detail", 2)),
    ("b-p3", "caching the prompt prefix", _pdf(3, "1 Foundations")),
    ("m-n1#p2", "[Canvas: m (generated)]\nThe cache key is the prefix.\n\nConnections:\n-> Other node",
     _map_part(2)),
    ("b-p4", "what chapter one said", _pdf(4, "Summary")),
    ("a-1", "always run the tests", {"source_file": "proj/AGENTS.md", "file_type": "note",
                                     "provenance": "authored", **AI}),
    ("b-p6", "what chapter two said", _pdf(6, "Summary", end=7)),
    ("n-3", "how it was measured", _note("Methods", "Methods", 1)),
]
PAYLOAD = {"ids": [r[0] for r in ROWS], "documents": [r[1] for r in ROWS],
           "metadatas": [r[2] for r in ROWS]}


def _pointers(pointers):
    return RoutePointers({"version": 1, "built_at": "2026-10-30T00:00:00Z",
                          "pointers": pointers, "unrouted": {}})


# The sidecar gives every part of a node the node's targets.
POINTERS = _pointers({"m-n1": [T1, T2], "m-n1#p2": [T1, T2]})


# ---- the outline ----

def test_outline_groups_sources_by_course_and_lists_maps_apart():
    out = build_outline(PAYLOAD, RULE, POINTERS)
    assert [s["source_file"] for g in out["groups"] for s in g["sources"]] == ["Books/b.pdf", "Notes/n.md"]
    assert out["maps"] == [{"source_file": "Maps/m (generated).canvas", "chunks": 2}]
    assert all("AGENTS.md" not in s["source_file"] for g in out["groups"] for s in g["sources"])
    assert all("AGENTS.md" not in m["source_file"] for m in out["maps"])
    # pages counts the distinct pages the chunks cover (b-p6 spans 6-7); a note has none.
    assert out["groups"] == [
        {"domain": "ai", "course": "AI Engineering",
         "sources": [{"source_file": "Books/b.pdf", "file_type": "pdf", "chunks": 6, "pages": 7}]},
        {"domain": "ml", "course": "unknown",
         "sources": [{"source_file": "Notes/n.md", "file_type": "note", "chunks": 3, "pages": None}]}]


def test_a_note_spanning_two_courses_is_listed_under_each():
    # A daily note carries one course per lecture section, so rows group, not files.
    metas = [{"source_file": "Daily/d.md", "file_type": "daily_note", "domain": d, "course_name": c}
             for d, c in (("ml", "RL"), ("nlp", "NLP"), ("ml", "RL"))]
    out = build_outline({"ids": ["d1", "d2", "d3"], "documents": ["x"] * 3, "metadatas": metas}, RULE, None)
    assert [(g["course"], g["sources"][0]["chunks"]) for g in out["groups"]] == [("RL", 2), ("NLP", 1)]


def test_a_file_stored_under_several_spellings_is_one_entry():
    # Keyed by norm_vault_path, as the filter and source_outline are; named by the first.
    metas = [_pdf(1), dict(_pdf(2), source_file="Books\\b.pdf"), dict(_pdf(3), source_file="BOOKS/B.PDF"),
             _map_part(1), dict(_map_part(2), source_file=MAP.replace("/", "\\"))]
    out = build_outline({"ids": [f"r{i}" for i in range(5)], "documents": ["x"] * 5, "metadatas": metas},
                        RULE, None)
    assert out == {"groups": [{"domain": "ai", "course": "AI Engineering", "sources": [
                       {"source_file": "Books/b.pdf", "file_type": "pdf", "chunks": 3, "pages": 3}]}],
                   "maps": [{"source_file": MAP, "chunks": 2}]}


def test_a_pdf_outline_gives_chapters_with_page_ranges():
    # Any spelling of the path finds the file; the answer names the stored one. A title
    # that recurs is one entry per run, in page order; pages before the first ToC entry
    # are a run with no chapter.
    assert source_outline(PAYLOAD, "books/B.PDF", RULE, POINTERS) == {
        "source_file": "Books/b.pdf",
        "chapters": [{"chapter": None, "page_start": 1, "page_end": 1, "chunks": 1},
                     {"chapter": "1 Foundations", "page_start": 2, "page_end": 3, "chunks": 2},
                     {"chapter": "Summary", "page_start": 4, "page_end": 4, "chunks": 1},
                     {"chapter": "2 Prompting", "page_start": 5, "page_end": 5, "chunks": 1},
                     {"chapter": "Summary", "page_start": 6, "page_end": 7, "chunks": 1}]}
    # A note's top-level headings, in document order, with no pages.
    assert source_outline(PAYLOAD, "Notes\\n.md", RULE, POINTERS) == {
        "source_file": "Notes/n.md",
        "chapters": [{"chapter": "Intro", "page_start": None, "page_end": None, "chunks": 2},
                     {"chapter": "Methods", "page_start": None, "page_end": None, "chunks": 1}]}


def test_a_note_opening_at_h2_has_a_chapter_per_h2(tmp_path):
    # The parser stacks raw levels, so here the first H2 heads every later heading_path;
    # the chapters follow the shallowest heading level instead.
    note = tmp_path / "Notes" / "h2.md"
    note.parent.mkdir()
    note.write_text("## Section A\nalpha text here\n### Detail\ndetail text here\n"
                    "## Section B\nbeta text here\n## Section C\ngamma text here\n", encoding="utf-8")
    docs = ObsidianParser(str(tmp_path), {"min_chunk_size": 5}).parse_file(note)
    payload = {"ids": [f"h{i}" for i in range(len(docs))], "documents": [d.text for d in docs],
               "metadatas": [d.metadata for d in docs]}
    chapters = source_outline(payload, "Notes/h2.md", RULE, None)["chapters"]
    assert [(c["chapter"], c["chunks"]) for c in chapters] == [("Section A", 2), ("Section B", 1),
                                                               ("Section C", 1)]


def test_a_map_outline_gives_each_concept_its_label_and_targets():
    # The two parts are one concept, labelled by part 1's first body line.
    assert source_outline(PAYLOAD, MAP, RULE, POINTERS) == {
        "source_file": MAP, "map": True, "concepts": [{"label": "Prompt caching", "targets": [T1, T2]}]}
    assert source_outline(PAYLOAD, MAP, RULE, None)["concepts"] == [{"label": "Prompt caching", "targets": []}]
    # A generated note: a concept per heading, its targets the deduped union over its rows,
    # in order; a row without a sidecar entry adds nothing.
    note = lambda heading: {"source_file": "Blueprints/g.md", "file_type": "note",
                            "provenance": "generated", "heading": heading}
    payload = {"ids": ["g1", "g2", "g3", "g4"], "documents": ["x"] * 4,
               "metadatas": [note("Few-shot"), note("Few-shot"), note("Chain of thought"), note("Few-shot")]}
    out = source_outline(payload, "Blueprints/g.md", RULE, _pointers({"g1": [T2], "g2": [T1, T2], "g3": [T1]}))
    assert out == {"source_file": "Blueprints/g.md", "map": True,
                   "concepts": [{"label": "Few-shot", "targets": [T2, T1]},
                                {"label": "Chain of thought", "targets": [T1]}]}


def _client(monkeypatch, outline=None):
    """A TestClient (no lifespan) over a warm-pipeline stand-in."""
    rag = SimpleNamespace(outline=outline or build_outline(PAYLOAD, RULE, POINTERS), provenance_map=RULE,
                          retriever=SimpleNamespace(_get_bm25=lambda: PAYLOAD, route_pointers=POINTERS))
    monkeypatch.setitem(S._STATE, "rag", rag)
    monkeypatch.setattr(S.app.state, "guard_settings", None)
    return TestClient(S.app, base_url="http://127.0.0.1")         # the guard refuses Host: testserver


def _files(page):
    return [s["source_file"] for g in page["groups"] for s in g["sources"]]


def test_outline_pages_over_sources_and_clamps(monkeypatch):
    client = _client(monkeypatch)
    first = client.get("/outline", params={"limit": 1}).json()
    assert (first["total"], _files(first)) == (2, ["Books/b.pdf"])
    assert first["maps"] == [{"source_file": MAP, "chunks": 2}]          # whole, on every page
    second = client.get("/outline", params={"limit": 1, "offset": 1}).json()
    assert (second["total"], _files(second), second["groups"][0]["course"]) == (2, ["Notes/n.md"], "unknown")
    assert _files(client.get("/outline", params={"limit": 0}).json()) == ["Books/b.pdf"]     # 0 -> 1
    # domain matches exactly, course as a substring in any case; total counts the matches
    assert _files(client.get("/outline", params={"domain": "ml"}).json()) == ["Notes/n.md"]
    assert client.get("/outline", params={"domain": "m"}).json()["total"] == 0
    assert _files(client.get("/outline", params={"course": "ai eng"}).json()) == ["Books/b.pdf"]
    many = {"groups": [{"domain": "d", "course": "c", "sources": [
        {"source_file": f"f{i:03}.md", "file_type": "note", "chunks": 1, "pages": None} for i in range(600)]}],
        "maps": []}
    page = _client(monkeypatch, many).get("/outline", params={"limit": 999}).json()
    assert page["total"] == 600 and len(_files(page)) == 500                            # 999 -> 500
    assert len(_files(_client(monkeypatch, many).get("/outline").json())) == 50          # the default


def test_unknown_source_and_agent_files_are_404(monkeypatch):
    for name in ("Books/none.pdf", "proj/AGENTS.md", "PROJ/agents.md"):
        with pytest.raises(KeyError):
            source_outline(PAYLOAD, name, RULE, POINTERS)
    client = _client(monkeypatch)
    ok = client.get("/outline/Books/b.pdf")
    assert ok.status_code == 200 and ok.json()["chapters"][1]["chapter"] == "1 Foundations"
    assert client.get("/outline/Books%2Fb.pdf").json()["source_file"] == "Books/b.pdf"   # as MCP quotes it
    assert client.get(f"/outline/{MAP}").json()["map"] is True
    for path, why in (("/outline/Books/none.pdf", "not in the index"),
                      ("/outline/proj/AGENTS.md", "agent-instruction file")):
        r = client.get(path)
        assert r.status_code == 404 and why in r.json()["detail"], path


def test_a_file_outline_with_no_index_built_is_503(monkeypatch, tmp_path):
    client = _client(monkeypatch, {"groups": [], "maps": []})
    S._STATE["rag"].retriever = HybridRetriever(
        embedder=_Embedder(), chroma_dir=Path("unused"), bm25_index=tmp_path / "bm25_index.pkl",
        collection_name="unused", provenance_rule=RULE)
    r = client.get("/outline/Books/b.pdf")
    assert r.status_code == 503 and "BM25 payload unavailable (FileNotFoundError" in r.json()["detail"]


# ---- the file filter ----

def _matches(meta, where):
    if where is None:
        return True
    if "$and" in where:
        return all(_matches(meta, w) for w in where["$and"])
    ((key, cond),) = where.items()
    return meta.get(key) in cond["$in"] if isinstance(cond, dict) else meta.get(key) == cond


class _Store:
    """Chroma's query over `rows`, ranked in row order, honouring equality, $in and $and;
    get(ids, include=["embeddings"]) gives stored vectors whose cosine with the query falls
    in row order, and get(where=, limit=) the matching rows (NeighborContext's call)."""

    def __init__(self, rows):
        self.rows = rows

    def get(self, ids=None, include=None, where=None, limit=None):
        if ids is None:
            hit = [r for r in self.rows if _matches(r[2], where)][:limit]
            return {"ids": [i for i, _, _ in hit], "documents": [t for _, t, _ in hit],
                    "metadatas": [m for _, _, m in hit]}
        order = [r[0] for r in self.rows]
        return {"ids": [i for i in ids if i in order],
                "embeddings": [[1.0, 0.01 * order.index(i)] for i in ids if i in order]}

    def query(self, query_embeddings, n_results, include, where=None):
        hit = [r for r in self.rows if _matches(r[2], where)][:n_results]
        return {"ids": [[i for i, _, _ in hit]], "documents": [[t for _, t, _ in hit]],
                "metadatas": [[m for _, _, m in hit]]}


class _Embedder:
    def embed_query(self, text):
        return [1.0, 0.0]


class _Omnisearch:
    enabled = True

    @staticmethod
    def lane(query):
        return [("Notes/n.md", "prompt caching, live", {"source_file": "Notes/n.md", "live": True})]


def _retriever(sparse_backend="bm25s"):
    """b.pdf's pages 2-3 are the filter's span. Every other lane finds rows outside it:
    the code lane (code_file_types note) and the scope lane (domain ml) the notes, HyPE
    n-1 beside b-p3, omnisearch a live excerpt (a filter skips that lane), and the routed
    lane b-p5, which the map cites but which lies outside the pages."""
    r = HybridRetriever(
        embedder=_Embedder(), chroma_dir=Path("unused"), bm25_index=Path("unused"),
        collection_name="unused", dense_top_k=4, sparse_top_k=4, metadata_boost=False,
        code_file_types=["note"], omnisearch=_Omnisearch(), hype_enabled=True, provenance_rule=RULE,
        sparse_backend=sparse_backend,
        route={"router_k": 5, "per_target": 3, "max_targets": 8, "max_new": 10},
        route_pointers=_pointers({"m-n1": [T2], "m-n1#p2": [T2]}))
    r._bm25_payload = {"bm25": BM25Index([_tokenize(t) for t in PAYLOAD["documents"]]), **PAYLOAD}
    r._collection = _Store(ROWS)
    r._hype_search = lambda qvec, k: [ROWS[1], ROWS[6]]                  # n-1, b-p3
    return r


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


def _pipeline(retriever):
    return RAGPipeline(retriever=retriever, reranker=_Reranker(), hyde=_Hyde(), generator=None,
                       rerank_top_k=10, presets={"code": {"boost_code": True}}, provenance_map=RULE,
                       scope_router=SimpleNamespace(detect=lambda q: Scope(domains=["ml"], labels=["ml"])),
                       neighbor_ctx=NeighborContext())          # off unless a call asks


def test_search_restricts_every_lane_to_the_file_and_pages(monkeypatch):
    rag = _pipeline(_retriever())
    calls = []
    real = rag.retriever.retrieve
    monkeypatch.setattr(rag.retriever, "retrieve", lambda q, **kw: calls.append(kw) or real(q, **kw))
    docs, info = rag.search(Q, preset="code", source_file="books/B.PDF", pages=[2, 3],
                            generated_mode="route", omnisearch=True)
    assert sorted(d.id for d in docs) == ["b-p2", "b-p3"]
    assert all("found_via" not in d.debug for d in docs)                 # b-p5 is outside the pages
    assert {"dense_code", "sparse_code", "dense_scope", "sparse_scope", "hype",
            "routed"} <= set(info["lanes_run"])                         # every lane ran, restricted
    assert "omnisearch" not in info["lanes_run"] and info["omnisearch"] is False   # no stored rows
    assert info["lanes_run"]["dense"] == info["lanes_run"]["sparse"] == 2 and info["lanes_run"]["hype"] == 1
    assert info["route"]["routed"] >= 1 and info["route"]["targets"] == 0   # it routed outside the pages
    assert info["file_filter"] == {"source_file": "books/B.PDF", "page_start": 2, "page_end": 3,
                                   "matched": True}
    assert calls[0]["file_filter"] == {"source_file": "books/B.PDF", "page_start": 2, "page_end": 3}
    # No pages: the whole file's primary chunks, still nothing from another file.
    docs, _ = rag.search(Q, source_file="Books/b.pdf")
    assert {d.metadata["source_file"] for d in docs} == {"Books/b.pdf"} and "b-p5" in {d.id for d in docs}
    # A file the store does not hold, and a map (no primary chunk): no candidates, matched false.
    for name in ("Books/none.pdf", MAP):
        docs, info = rag.search(Q, source_file=name)
        assert docs == [] and info["file_filter"]["matched"] is False, name
    # Without the fields nothing changes: the echo has no file_filter and other files return.
    docs, info = rag.search(Q)
    assert "file_filter" not in info and "Notes/n.md" in {d.metadata["source_file"] for d in docs}


def test_neighbour_pages_stay_inside_the_pages():
    # The neighbour step fetches pages by number after retrieve(): p1 and p4 border [2, 3].
    rag = _pipeline(_retriever())
    docs, info = rag.search(Q, source_file="Books/b.pdf", pages=[2, 3], neighbor_context=True)
    assert sorted(d.id for d in docs) == ["b-p2", "b-p3"] and info["neighbors_added"] == 0
    # A neighbour inside the range stays, and is counted.
    docs, info = rag.search(Q, source_file="Books/b.pdf", pages=[2, 4], top_k=1, neighbor_context=True)
    assert {d.id for d in docs} <= {"b-p2", "b-p3", "b-p4"}
    assert info["neighbors_added"] == sum("neighbor_of" in d.debug for d in docs) >= 1


def test_a_routed_row_inside_the_pages_keeps_its_found_via():
    r = _retriever()
    r.route_pointers = _pointers({"m-n1": [T1], "m-n1#p2": [T1]})
    docs, _ = _pipeline(r).search(Q, source_file="Books/b.pdf", pages=[2, 3], generated_mode="route")
    assert {d.id: [v["id"] for v in d.debug["found_via"]] for d in docs} == {
        "b-p2": ["m-n1", "m-n1#p2"], "b-p3": ["m-n1", "m-n1#p2"]}


def test_a_target_outside_the_filter_spends_no_seat():
    # m-n1 cites a note first, then b.pdf pp. 6-7, with one seat for new rows: unclipped,
    # the note's rows took the seat, the filter then dropped them, and b-p6 never got in.
    r = _retriever()
    r.route = dict(r.route, max_new=1)
    cited = [{"source_file": "Notes/n.md", "page_start": None, "page_end": None},
             {"source_file": "Books/b.pdf", "page_start": 6, "page_end": 7}]
    r.route_pointers = _pointers({"m-n1": cited, "m-n1#p2": cited})
    docs, info = _pipeline(r).search(Q, source_file="Books/b.pdf", generated_mode="route",
                                     scope_lanes=False, hype=False)
    assert "b-p6" in {d.id for d in docs}
    assert (info["route"]["targets"], info["route"]["new"]) == (1, 1)


def test_a_graph_walk_with_a_file_filter_is_refused(monkeypatch):
    # The walk seeds and steps across the vault, so it would leave the file: refused
    # before any retrieval, and /evidence_pack answers it with its 400.
    rag = _pipeline(_retriever())
    rag.graph = SimpleNamespace(enabled=True)
    monkeypatch.setattr(rag.retriever, "retrieve", lambda q, **kw: pytest.fail("retrieved"))
    for kw in ({"source_file": "Books/b.pdf"}, {"source_file": "Books/b.pdf", "pages": [2, 3]}):
        with pytest.raises(ValueError, match="^graph true cannot take source_file or pages"):
            rag.graph_search(Q, graph={}, **kw)
    monkeypatch.setitem(S._STATE, "rag", rag)
    monkeypatch.setitem(S._STATE, "cfg", Config({}, Path.cwd()))      # /evidence_pack reads it
    with pytest.raises(HTTPException) as exc_info:
        S.evidence_pack(S.EvidencePackIn(q=Q, source_file="Books/b.pdf", pages=[2, 3], graph=True))
    assert exc_info.value.status_code == 400 and "graph true cannot take source_file" in exc_info.value.detail


def test_pages_without_a_source_file_or_reversed_raise(monkeypatch):
    rag = _pipeline(_retriever())
    # Raised first: neither HyDE nor the preset lookup (which reads the HyDE object) runs.
    rag.hyde = SimpleNamespace(expand_with_info=lambda q, enabled=None: pytest.fail("HyDE drafted"))
    for kw, why in (({"pages": [2, 3]}, "pages needs source_file"),
                    ({"source_file": "Books/b.pdf", "pages": [3, 2]}, "1 <= first <= last"),
                    ({"source_file": "Books/b.pdf", "pages": [0, 2]}, "1 <= first <= last")):
        with pytest.raises(ValueError, match=why):
            rag.search(Q, **kw)
    # The API answers it as any other bad knob value: HTTP 200 with the error.
    monkeypatch.setitem(S._STATE, "rag", rag)
    assert "pages needs source_file" in S.search(S.SearchIn(q=Q, pages=[2, 3]))["error"]


def test_a_file_filter_on_tantivy_is_refused():
    r = _retriever(sparse_backend="tantivy")
    with pytest.raises(ValueError, match=r"^source_file needs retrieval\.sparse_backend: bm25s in config\.yaml$"):
        r.retrieve(Q, file_filter=T1)
    # search() refuses first: neither HyDE nor the preset lookup (which reads the HyDE object) runs.
    rag = _pipeline(r)
    rag.hyde = SimpleNamespace(expand_with_info=lambda q, enabled=None: pytest.fail("HyDE drafted"))
    with pytest.raises(ValueError, match=r"^source_file needs retrieval\.sparse_backend"):
        rag.search(Q, source_file="Books/b.pdf")


def test_source_file_and_pages_are_on_every_retrieval_model():
    for model in (S.SearchIn, S.QueryIn, S.CompareBranchIn, S.EvidencePackIn):
        assert {"source_file", "pages"} <= set(model.model_fields), model.__name__
    assert {"source_file", "pages"} <= set(S._RETRIEVAL_FIELDS)
    kwargs = S._retrieval_kwargs(S.SearchIn(q="x", source_file="Books/b.pdf", pages=[2, 3]))
    assert (kwargs["source_file"], kwargs["pages"]) == ("Books/b.pdf", [2, 3])
    for bad in ([2], [1, 2, 3]):
        with pytest.raises(ValidationError):
            S.SearchIn(q="x", source_file="Books/b.pdf", pages=bad)
    # query() hands both to search()
    rag = _pipeline(_retriever())
    seen = {}
    rag.search = lambda q, **kw: seen.update(kw) or ([], {})
    rag.generator = SimpleNamespace(generate=lambda q, docs, max_tokens=None: SimpleNamespace(retrieval=None))
    rag.query(Q, source_file="Books/b.pdf", pages=[2, 3])
    assert (seen["source_file"], seen["pages"]) == ("Books/b.pdf", [2, 3])
