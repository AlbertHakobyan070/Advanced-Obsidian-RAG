"""src/index_ops.py: the foundation of the metadata-only index maintenance (slice B).

Run:  python -m pytest tests/test_index_ops.py -q

Covers what every later writer leans on: rewrite_jsonl (byte-for-byte pass-through,
the line's own terminator, DROP, the .bak scheme), the file and port discovery, the
building refusal, and the duplicate-safe append's conflict check; then the
metadata-only relocate (load_map, relocate_meta, plan, apply, verify and the
main.py relocate command). Real Chroma stores under tmp_path with explicit vectors
and no model.
"""
import json
import os
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import yaml

from src import index_ops
from src.embeddings.embedder import Chunk, Embedder
from src.embeddings.registry import EmbeddingSpec
from src.embeddings.sidecar import make_sidecar, sidecar_path, write_sidecar
from src.index_ops import DROP, rewrite_jsonl
from src.ingestion.canvas_loader import decode_canvas_files, encode_canvas_edge
from src.ocr_studio.pages import doc_key
from src.utils.chroma_client import persistent_client
from src.utils.config_loader import Config

chromadb = pytest.importorskip("chromadb")


@pytest.fixture(autouse=True)
def _release_chroma_stores():
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


def _row(doc_id, text="t", **meta):
    return json.dumps({"doc_id": doc_id, "text": text, "metadata": {"source_file": "x.md", **meta}},
                      ensure_ascii=False).encode("utf-8")


def _fix(rec):
    """r1 gets a new tag, r2 goes, everything else is left alone."""
    if rec["doc_id"] == "r1":
        return dict(rec, text="café changed")
    if rec["doc_id"] == "r2":
        return DROP
    return None


# ---- rewrite_jsonl ----------------------------------------------------------------

def test_rewrite_jsonl_passes_untouched_lines_through_byte_for_byte(tmp_path):
    path = tmp_path / "a_chunks.jsonl"
    keep_a, keep_b = _row("k1", "  stays inside"), _row("k2")
    unparseable = b'{"doc_id": "broken", '
    lines = [keep_a, _row("r1"), unparseable, b"", _row("r2"), b"[1, 2]", keep_b]   # last: no newline
    original = b"\n".join(lines)
    path.write_bytes(original)

    # Nothing to change: no write, no .bak, no tmp. dry_run counts but writes nothing either.
    assert rewrite_jsonl(path, lambda r: None) == 0
    assert rewrite_jsonl(path, _fix, dry_run=True) == 2
    assert path.read_bytes() == original
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a_chunks.jsonl"]

    assert rewrite_jsonl(path, _fix, tag="t") == 2                # one replaced, one dropped
    expected = [keep_a, json.dumps({"doc_id": "r1", "text": "café changed",
                                    "metadata": {"source_file": "x.md"}},
                                   ensure_ascii=False).encode("utf-8"),
                unparseable, b"", b"[1, 2]", keep_b]
    assert path.read_bytes() == b"\n".join(expected)              # r2 gone, the rest verbatim
    (bak,) = [p for p in tmp_path.iterdir() if p.name.startswith("a_chunks.jsonl.t-")]
    assert bak.name.endswith(".bak") and bak.read_bytes() == original
    assert not (tmp_path / "a_chunks.jsonl.tmp").exists()
    # the backup name can never join the sparse union
    assert not bak.name.endswith("_chunks.jsonl")


def test_rewrite_jsonl_keeps_each_lines_own_terminator(tmp_path):
    path = tmp_path / "chunks.jsonl"
    original = b"\r\n".join([_row("k1"), _row("r1"), _row("r2"), _row("k2")]) + b"\r\n"
    path.write_bytes(original)

    assert rewrite_jsonl(path, _fix) == 2

    out = path.read_bytes()
    assert out.count(b"\n") == out.count(b"\r\n") == 3             # no bare \n anywhere
    assert out.startswith(_row("k1") + b"\r\n") and out.endswith(_row("k2") + b"\r\n")
    # a replaced LAST line without a terminator stays without one
    last = tmp_path / "b_chunks.jsonl"
    last.write_bytes(_row("k1") + b"\n" + _row("r1"))
    assert rewrite_jsonl(last, _fix) == 1
    assert not last.read_bytes().endswith(b"\n") and last.read_bytes().count(b"\n") == 1


def test_a_second_rewrite_in_the_same_second_raises_and_leaves_the_file(tmp_path, monkeypatch):
    path = tmp_path / "chunks.jsonl"
    path.write_bytes(_row("r1") + b"\n" + _row("k1") + b"\n")
    monkeypatch.setattr(index_ops, "time", SimpleNamespace(strftime=lambda fmt: "20260101-000000"))
    assert rewrite_jsonl(path, lambda r: dict(r, text="one") if r["doc_id"] == "r1" else None) == 1
    after_first = path.read_bytes()

    with pytest.raises(FileExistsError, match="chunks.jsonl.bak-20260101-000000.bak"):
        rewrite_jsonl(path, lambda r: dict(r, text="two") if r["doc_id"] == "r1" else None)

    assert path.read_bytes() == after_first                       # the failed rewrite changed nothing
    assert not (tmp_path / "chunks.jsonl.tmp").exists()


def test_a_fix_that_returns_a_non_record_is_an_error(tmp_path):
    path = tmp_path / "chunks.jsonl"
    path.write_bytes(_row("r1") + b"\n")
    with pytest.raises(TypeError, match="dict, None or DROP"):
        rewrite_jsonl(path, lambda r: "oops")
    assert path.read_bytes() == _row("r1") + b"\n"
    assert not (tmp_path / "chunks.jsonl.tmp").exists()


# ---- discovery --------------------------------------------------------------------

def test_chunk_jsonls_is_the_sparse_union_and_never_a_backup(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    cfg = Config({"paths": {"chunks_file": str(data / "chunks.jsonl")}}, tmp_path)
    for name in ("pdf_chunks.jsonl", "a_chunks.jsonl", "chunks.jsonl.delete-20260101-000000.bak",
                 "a_chunks.jsonl.recal-20260101-000000.bak", "hype_questions.jsonl"):
        (data / name).write_bytes(b"")
    assert [p.name for p in index_ops.chunk_jsonls(cfg)] == ["a_chunks.jsonl", "pdf_chunks.jsonl"]
    (data / "chunks.jsonl").write_bytes(b"")
    assert [p.name for p in index_ops.chunk_jsonls(cfg)] == [
        "chunks.jsonl", "a_chunks.jsonl", "pdf_chunks.jsonl"]


def test_parents_path_follows_the_retrieval_rule(tmp_path):
    set_ = Config({"retrieval": {"parents_file": "elsewhere/p.jsonl"}}, tmp_path)
    assert index_ops.parents_path(set_) == tmp_path / "elsewhere" / "p.jsonl"
    assert index_ops.parents_path(Config({}, tmp_path)) == tmp_path / "data" / "parents_md.jsonl"


def _closed_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _listener():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    return s, s.getsockname()[1]


def test_answering_ports_probes_streamlit(monkeypatch):
    listener, port = _listener()
    try:
        monkeypatch.setattr(index_ops, "STREAMLIT_PORT", port)
        cfg = Config({"webui": {"port": _closed_port(),
                                "rag_api": f"http://127.0.0.1:{_closed_port()}"}}, Path("."))
        assert index_ops.answering_ports(cfg) == [port]
    finally:
        listener.close()


def test_answering_ports_reports_the_api_and_the_console(monkeypatch):
    api, api_port = _listener()
    console, console_port = _listener()
    try:
        monkeypatch.setattr(index_ops, "STREAMLIT_PORT", _closed_port())
        cfg = Config({"webui": {"port": console_port, "rag_api": f"http://127.0.0.1:{api_port}"}},
                     Path("."))
        assert index_ops.answering_ports(cfg) == [api_port, console_port]
    finally:
        api.close()
        console.close()


# ---- the building refusal -----------------------------------------------------------

def test_refuse_if_building_names_the_collection_and_never_reads_a_bad_sidecar_as_idle(tmp_path):
    chroma = tmp_path / "chroma"
    client = persistent_client(chroma)
    for name in ("main", "eg2"):
        client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
    spec = EmbeddingSpec("local", "local", "fake/m")

    def sidecar(name, status):
        write_sidecar(chroma, name, make_sidecar(
            spec, collection=name, role="chunks", status=status, dimension=3, count=1,
            written_by="rebuild-into"))

    index_ops.refuse_if_building(chroma, client)                  # no sidecar at all: idle
    sidecar("main", "complete")
    sidecar("eg2", "complete")
    index_ops.refuse_if_building(chroma, client)

    sidecar("eg2", "building")
    with pytest.raises(index_ops.StoreBusy, match=r"eg2.*mid re-embed \(status building\).*"
                                                  r"--rebuild-into eg2"):
        index_ops.refuse_if_building(chroma, client)

    sidecar_path(chroma, "eg2").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not readable JSON"):    # unreadable is an error, not "idle"
        index_ops.refuse_if_building(chroma, client)


def test_a_building_graph_twin_gets_a_remedy_that_can_be_run(tmp_path):
    """REGRESSION (review B2-3): the twin got "main.py index --rebuild-into <twin> resumes",
    which reembed refuses for the reserved __graph suffix; the graph_embed job builds it."""
    chroma = tmp_path / "chroma"
    client = persistent_client(chroma)
    client.get_or_create_collection("main__graph", metadata={"hnsw:space": "cosine"})
    write_sidecar(chroma, "main__graph", make_sidecar(
        EmbeddingSpec("local", "local", "fake/m"), collection="main__graph", role="graph",
        status="building", dimension=3, count=1, written_by="graph-embed"))

    with pytest.raises(index_ops.StoreBusy) as busy:
        index_ops.refuse_if_building(chroma, client)

    assert "main.py graph-embed" in str(busy.value)
    assert "--rebuild-into main__graph" not in str(busy.value)


# ---- the duplicate-safe append ------------------------------------------------------

def _chunks(source, ids, file_type="note", page=None):
    meta = {"source_file": source, "file_type": file_type}
    if page is not None:
        meta["page_start"] = page
    return [Chunk(i, "text " + i, dict(meta)) for i in ids]


@pytest.fixture
def active(tmp_path):
    col = persistent_client(tmp_path / "chroma").get_or_create_collection(
        "active", metadata={"hnsw:space": "cosine"})
    rows = [("x1", {"source_file": "a.md", "file_type": "note"}),
            ("x2", {"source_file": "a.md", "file_type": "note"}),
            ("p3a", {"source_file": "b.pdf", "file_type": "pdf", "page_start": 3})]
    col.add(ids=[i for i, _ in rows], embeddings=[[1.0, 0.0, 0.0]] * len(rows),
            documents=[i for i, _ in rows], metadatas=[m for _, m in rows])
    return col


def test_append_conflicts_are_the_rows_a_replace_would_remove(active):
    # a whole-file lane: the file's ids are the only ids it may have
    assert index_ops.append_conflicts(active, _chunks("a.md", ["x1"])) == [
        "a.md: 1 indexed row(s) are not in this file (e.g. x2)"]
    assert index_ops.append_conflicts(active, _chunks("a.md", ["x1", "x2", "x3"])) == []
    # a pdf: only the pages the new file covers
    assert index_ops.append_conflicts(active, _chunks("b.pdf", ["p3b"], "pdf", page=3)) == [
        "b.pdf: 1 indexed row(s) are not in this file (e.g. p3a)"]
    assert index_ops.append_conflicts(active, _chunks("b.pdf", ["p4"], "pdf", page=4)) == []
    assert index_ops.append_conflicts(active, _chunks("c.md", ["n1"])) == []
    # several sources at once: one line per conflicting source
    both = _chunks("a.md", ["x1"]) + _chunks("b.pdf", ["p3b"], "pdf", page=3) + _chunks("c.md", ["n1"])
    assert [line.split(":")[0] for line in index_ops.append_conflicts(active, both)] == ["a.md", "b.pdf"]


def test_append_conflicts_name_an_id_indexed_under_another_source(tmp_path):
    """REGRESSION (review B2-4): relocate keeps the ids it moves (they hash the OLD path), so
    the same text ingested again at the old path mints exactly those ids, and the upsert
    silently moved the rows back while the JSONLs still said the new path."""
    col = persistent_client(tmp_path / "chroma").get_or_create_collection(
        "active", metadata={"hnsw:space": "cosine"})
    col.add(ids=["r1"], embeddings=[[1.0, 0.0, 0.0]], documents=["t"],
            metadatas=[{"source_file": "Books\\X.pdf", "file_type": "pdf", "page_start": 1}])

    (line,) = index_ops.append_conflicts(col, _chunks("Inbox\\X.pdf", ["r1"], "pdf", page=1))

    assert line.startswith("Inbox\\X.pdf:") and "Books\\X.pdf" in line and "r1" in line


def test_append_conflicts_say_when_a_pdf_file_lacks_indexed_pages(active):
    """The REPLACE recipe deletes the whole source: for a file that holds only some pages,
    that drops the indexed pages it does not bring (review B2-1), so the line says so."""
    active.add(ids=["p5a"], embeddings=[[1.0, 0.0, 0.0]], documents=["p5a"],
               metadatas=[{"source_file": "b.pdf", "file_type": "pdf", "page_start": 5}])

    (line,) = index_ops.append_conflicts(active, _chunks("b.pdf", ["p3b"], "pdf", page=3))

    assert line.startswith("b.pdf: 1 indexed row(s) are not in this file (e.g. p3a)")
    assert "1 more on pages this file does not cover" in line
    assert index_ops.append_conflicts(active, _chunks("b.pdf", ["p4"], "pdf", page=4)) == []


def test_append_conflicts_look_past_the_first_hundred_sources(tmp_path):
    col = persistent_client(tmp_path / "chroma").get_or_create_collection(
        "many", metadata={"hnsw:space": "cosine"})
    ids = [f"s{n:03d}" for n in range(205)]
    col.add(ids=ids, embeddings=[[1.0, 0.0, 0.0]] * 205, documents=ids,
            metadatas=[{"source_file": f"f{n:03d}.md", "file_type": "note"} for n in range(205)])
    incoming = [Chunk(i, i, {"source_file": f"f{n:03d}.md", "file_type": "note"})
                for n, i in enumerate(ids)]
    assert index_ops.append_conflicts(col, incoming) == []
    incoming[203] = Chunk("s203-new", "x", {"source_file": "f203.md", "file_type": "note"})
    assert index_ops.append_conflicts(col, incoming) == [
        "f203.md: 1 indexed row(s) are not in this file (e.g. s203)"]


# ---- relocate: the metadata-only move of indexed files --------------------------------
# One world: three moves (a pdf rename, an md rename, a code rename) over every store a
# chunk lives in, plus a canvas and rows that must not move. Real Chroma stores under
# tmp_path; nothing here touches data/ or a live store.

ACTIVE, EG2, HYPE = "obsidian_vault", "obsidian_vault_eg2_l1024", "hype_questions"
NOTE, BOOK, CODE = "Old\\note.md", "Old\\Book.pdf", "Old\\s.py"
NOTE2, BOOK2, CODE2 = "Old\\note2.md", "Old\\Book Renamed.pdf", "Old\\s2.py"
MOVED = {BOOK: BOOK2, NOTE: NOTE2, CODE: CODE2}


def _rec(doc_id, sf, file_type, **meta):
    return {"doc_id": doc_id, "text": "text " + doc_id,
            "metadata": {"source_file": sf, "file_type": file_type, "course_name": "Course", **meta}}


def _encode(row):
    return json.dumps(row, ensure_ascii=False).encode("utf-8")


JSONLS = {
    "chunks.jsonl": [
        _rec("m1", NOTE, "note", filename="note", heading="Sub", heading_path="note > Sub"),
        _rec("k1", "Keep\\k.md", "note", filename="k", heading="k", heading_path="k")],
    "pdf_chunks.jsonl": [
        _rec("p1", BOOK, "pdf", filename="Book", heading="Book", heading_path="Book > Ch 1"),
        _rec("p2", BOOK, "pdf", filename="Book", heading="Ch 2", heading_path="Book > Ch 2")],
    "code_chunks.jsonl": [_rec("c1", CODE, "code", filename="s.py", heading="s.py", heading_path="s.py")],
    "canvas_chunks.jsonl": [_rec("v1", "Old/c.canvas", "canvas", filename="c")],
    "other_chunks.jsonl": [_rec("o1", "Keep\\o.md", "note", filename="o", heading="o", heading_path="o")],
    "parents_md.jsonl": [{"parent_id": "pm1", "source_file": NOTE, "text": "parent"}],
    "hype_questions.jsonl": [{"doc_id": "m1", "source_file": NOTE, "questions": ["a?", "b?"]}],
}
REWRITTEN = ["chunks.jsonl", "pdf_chunks.jsonl", "code_chunks.jsonl", "parents_md.jsonl",
             "hype_questions.jsonl"]            # the files that hold a moved source


def _move(old, new, lane, n_chunks):
    return {"op": "rename", "kind": "file", "old": old, "new": new, "old_sf": old.replace("/", "\\"),
            "new_sf": new.replace("/", "\\"), "lane": lane, "indexed": True, "n_chunks": n_chunks,
            "approved": True}


def _write_map(path, moves, extra=()):
    path.write_text("\n".join(json.dumps(r) for r in [*moves, *extra]) + "\n", encoding="utf-8")


@pytest.fixture
def servers_down(monkeypatch):
    """The dev machine may have :8051 up; a write that is meant to proceed says none answers."""
    monkeypatch.setattr(index_ops, "answering_ports", lambda cfg: [])


@pytest.fixture
def world(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    for name, rows in JSONLS.items():             # CRLF, like the live files
        (data / name).write_bytes(b"".join(_encode(r) + b"\r\n" for r in rows))
    vault = tmp_path / "vault"                    # the files sit at their NEW paths only
    (vault / "Old").mkdir(parents=True)
    for name in ("Book Renamed.pdf", "note2.md", "s2.py"):
        (vault / "Old" / name).write_bytes(b"x")
    ocr = data / "ocr_studio" / doc_key("Old/Book.pdf")
    ocr.mkdir(parents=True)
    (ocr / "doc.json").write_text(json.dumps({"source": "Old/Book.pdf", "pages": [1]}), encoding="utf-8")
    (ocr / "page_1.json").write_text("{}", encoding="utf-8")
    sets = tmp_path / "eval_sets"
    sets.mkdir()
    (sets / "gold.yaml").write_text("questions:\n  - file: " + NOTE + "\n  - file: Keep\\k.md\n",
                                    encoding="utf-8")
    bm25 = tmp_path / "bm25_index.pkl"
    bm25.write_bytes(b"bm25")
    os.utime(bm25, (time.time() - 3600, time.time() - 3600))      # older than every rewrite

    cfgd = {"paths": {"chunks_file": str(data / "chunks.jsonl"), "chroma_dir": str(tmp_path / "chroma"),
                      "bm25_index": str(bm25), "collection_name": ACTIVE},
            "retrieval": {"parents_file": str(data / "parents_md.jsonl"), "hype": {"collection": HYPE}},
            "parser": {"vault_path": str(vault)},
            "eval": {"sets_dir": str(sets)},
            "webui": {"port": _closed_port(), "rag_api": f"http://127.0.0.1:{_closed_port()}"}}
    client = persistent_client(tmp_path / "chroma")
    chunk_rows = [r for name, rows in JSONLS.items() if name.endswith("chunks.jsonl") for r in rows]
    for name, dim in ((ACTIVE, 3), (EG2, 4)):
        col = client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
        col.add(ids=[r["doc_id"] for r in chunk_rows],
                embeddings=[[1.0 + k] + [0.0] * (dim - 1) for k in range(len(chunk_rows))],
                documents=[r["text"] for r in chunk_rows],
                metadatas=[{**r["metadata"], "tags": "t"} for r in chunk_rows])   # tags: a Chroma-only key
    client.get_or_create_collection(HYPE, metadata={"hnsw:space": "cosine"}).add(
        ids=["m1::q0", "m1::q1"], embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], documents=["a?", "b?"],
        metadatas=[{"doc_id": "m1", "source_file": NOTE}] * 2)
    moves = [_move("Old/Book.pdf", "Old/Book Renamed.pdf", "pdf", 2),
             _move("Old/note.md", "Old/note2.md", "md", 1),
             _move("Old/s.py", "Old/s2.py", "code", 1)]
    return SimpleNamespace(root=tmp_path, data=data, vault=vault, ocr=ocr, bm25=bm25, cfgd=cfgd,
                           cfg=Config(cfgd, tmp_path), client=client, moves=moves,
                           map=tmp_path / "rename_map.jsonl", sockets=[])


def _files(w):
    """Every file outside the Chroma folder: relative name -> (bytes, mtime_ns)."""
    return {str(p.relative_to(w.root)): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in sorted(w.root.rglob("*"))
            if p.is_file() and "chroma" not in p.relative_to(w.root).parts}


def _dump(w):
    """Every collection: id -> (metadata, document)."""
    out = {}
    for c in w.client.list_collections():
        g = w.client.get_collection(c.name).get(include=["metadatas", "documents"])
        out[c.name] = {i: (m, d) for i, m, d in zip(g["ids"], g["metadatas"], g["documents"])}
    return out


def _vectors(w):
    """Every collection: id -> embedding."""
    out = {}
    for c in w.client.list_collections():
        g = w.client.get_collection(c.name).get(include=["embeddings"])
        out[c.name] = {i: tuple(float(x) for x in e) for i, e in zip(g["ids"], g["embeddings"])}
    return out


# ---- the map and the per-row transform --------------------------------------------------

def test_load_map_keeps_indexed_file_moves_and_names_a_bad_line(tmp_path):
    good = _move("a/x.md", "b/x.md", "md", 3)
    rows = [good,
            {**_move("c/y.md", "d/y.md", "md", 1), "op": "move"},
            {**good, "kind": "dir"},                                   # filesystem only
            {**good, "indexed": False},                                # not in the index
            {**good, "op": "quarantine", "old_sf": "q\\z.md"},         # reported, not moved
            {**good, "op": "quarantine", "indexed": False}]
    path = tmp_path / "map.jsonl"
    _write_map(path, rows)
    moves, quarantined = index_ops.load_map(path)
    assert [m["old_sf"] for m in moves] == ["a\\x.md", "c\\y.md"]
    assert [q["old_sf"] for q in quarantined] == ["q\\z.md"]

    path.write_text(json.dumps(good) + "\n\n{not json\n", encoding="utf-8")
    with pytest.raises(index_ops.RelocateError, match="line 3"):
        index_ops.load_map(path)
    path.write_text(json.dumps([1, 2]) + "\n", encoding="utf-8")
    with pytest.raises(index_ops.RelocateError, match="line 1.*JSON object"):
        index_ops.load_map(path)
    _write_map(path, [{k: v for k, v in good.items() if k != "new_sf"}])
    with pytest.raises(index_ops.RelocateError, match="line 1.*new_sf"):
        index_ops.load_map(path)


def test_relocate_meta_rules():
    pdf1 = {"source_file": BOOK, "file_type": "pdf", "filename": "Book", "heading": "Book",
            "heading_path": "Book > Ch 1", "course_name": "C"}
    pdf2 = {**pdf1, "heading": "Ch 2", "heading_path": "Book > Ch 2"}
    out = index_ops.relocate_meta(pdf1, BOOK, BOOK2)
    assert out == {**pdf1, "source_file": BOOK2, "filename": "Book Renamed", "heading": "Book Renamed",
                   "heading_path": "Book Renamed > Ch 1"}
    assert pdf1["source_file"] == BOOK                                  # the input is not mutated
    out2 = index_ops.relocate_meta(pdf2, BOOK, BOOK2)
    assert out2["heading"] == "Ch 2" and out2["heading_path"] == "Book Renamed > Ch 2"
    bare = index_ops.relocate_meta({**pdf1, "heading_path": "Book"}, BOOK, BOOK2)
    assert bare["heading_path"] == "Book Renamed"
    other = index_ops.relocate_meta({**pdf1, "heading_path": "Booking > x"}, BOOK, BOOK2)
    assert other["heading_path"] == "Booking > x"                       # a prefix of the title, not the title

    code = {"source_file": CODE, "file_type": "code", "filename": "s.py", "heading": "s.py",
            "heading_path": "s.py"}
    assert index_ops.relocate_meta(code, CODE, CODE2) == {
        **code, "source_file": CODE2, "filename": "s2.py", "heading": "s2.py", "heading_path": "s2.py"}

    md = {"source_file": NOTE, "file_type": "note", "filename": "note", "heading": "note",
          "heading_path": "note > Sub"}
    assert index_ops.relocate_meta(md, NOTE, NOTE2) == {**md, "source_file": NOTE2, "filename": "note2"}

    hype = {"doc_id": "m1", "source_file": NOTE}
    assert index_ops.relocate_meta(hype, NOTE, NOTE2) == {"doc_id": "m1", "source_file": NOTE2}
    assert index_ops.relocate_meta(pdf1, "Elsewhere\\x.pdf", BOOK2) is None


def test_relocate_meta_rewrites_canvas_file_refs_in_both_shapes():
    meta = {"source_file": "Maps/m.canvas", "file_type": "canvas", "filename": "m", "canvas_node_id": "t",
            "canvas_edge_files": [encode_canvas_edge("f1", "Old/Book.pdf", "#page=5"),
                                  encode_canvas_edge("", "Old/Book.pdf", "#page=9"),
                                  encode_canvas_edge("f2", "Keep/k.md", "")],
            "canvas_file_refs": ["Old/Book.pdf", "Keep/k.md"], "tags": ["x"]}
    want = [{"node": "f1", "file": "New/Book Renamed.pdf", "subpath": "#page=5"},
            {"node": "", "file": "New/Book Renamed.pdf", "subpath": "#page=9"},
            {"node": "f2", "file": "Keep/k.md", "subpath": ""}]
    refs = ("canvas_edge_files", "canvas_file_refs")

    def rest(m):
        return {k: v for k, v in m.items() if k not in refs}

    out = index_ops.relocate_meta(meta, "Old\\Book.pdf", "New\\Book Renamed.pdf")
    assert decode_canvas_files(out["canvas_edge_files"]) == want
    assert out["canvas_file_refs"] == ["New/Book Renamed.pdf", "Keep/k.md"]
    assert rest(out) == rest(meta)                          # source_file and every other key stay
    assert meta["canvas_file_refs"] == ["Old/Book.pdf", "Keep/k.md"]       # the input is not mutated

    stored = Embedder._clean_meta(meta)                     # Chroma's ", "-joined shape
    out = index_ops.relocate_meta(stored, "Old\\Book.pdf", "New\\Book Renamed.pdf")
    assert isinstance(out["canvas_edge_files"], str) and decode_canvas_files(out["canvas_edge_files"]) == want
    assert out["canvas_file_refs"] == "New/Book Renamed.pdf, Keep/k.md"
    assert rest(out) == rest(stored)

    cites_nothing_moved = {**meta, "canvas_edge_files": [encode_canvas_edge("f2", "Keep/k.md", "")],
                           "canvas_file_refs": ["Keep/k.md"]}
    assert index_ops.relocate_meta(cites_nothing_moved, "Old\\Book.pdf", "New\\Book Renamed.pdf") is None
    not_a_canvas = {"source_file": "Keep\\k.md", "file_type": "note", "canvas_file_refs": ["Old/Book.pdf"]}
    assert index_ops.relocate_meta(not_a_canvas, "Old\\Book.pdf", "New\\Book Renamed.pdf") is None


# ---- plan: read-only --------------------------------------------------------------------

def test_dry_run_counts_every_store_and_changes_nothing(world, servers_down):
    files, dump = _files(world), _dump(world)
    p = index_ops.plan(world.cfg, world.moves, [], world.client)

    assert p["blockers"] == []
    book, note, code = p["rows"]
    assert (book["jsonl"], book["ids"], book["parents"], book["hype"]) == ({"pdf_chunks.jsonl": 2}, 2, 0, 0)
    assert book["cols"] == {ACTIVE: 2, EG2: 2, HYPE: 0}
    assert book["ocr_dir"] == world.ocr and book["eval"] == 0
    assert (note["jsonl"], note["ids"], note["parents"], note["hype"]) == ({"chunks.jsonl": 1}, 1, 1, 1)
    assert note["cols"] == {ACTIVE: 1, EG2: 1, HYPE: 2}
    assert note["ocr_dir"] is None and note["eval"] == 1
    assert code["jsonl"] == {"code_chunks.jsonl": 1} and code["cols"] == {ACTIVE: 1, EG2: 1, HYPE: 0}
    assert p["totals"] == {"jsonl": 4, "parents": 1, "hype": 1, ACTIVE: 4, EG2: 4, HYPE: 2}
    # nothing else to warn about: the HyPE collection holds questions, so it is not "lagging"
    assert len(p["warnings"]) == 1 and "eval gold" in p["warnings"][0] and NOTE in p["warnings"][0]
    assert _files(world) == files and _dump(world) == dump


def test_dry_run_warns_about_references_to_the_moved_file(world, servers_down):
    from src.ingestion.obsidian_parser import extract_wikilinks
    with open(world.data / "canvas_chunks.jsonl", "ab") as f:
        f.write(_encode(_rec("v2", "Old/other.canvas", "canvas", canvas_file_refs=["Old/Book.pdf"])) + b"\r\n")
    with open(world.data / "other_chunks.jsonl", "ab") as f:
        f.write(_encode(_rec("o2", "Keep\\w.md", "note", wikilinks=["Some Folder/NOTE#Part"])) + b"\r\n")
        # Obsidian needs the extension to link a pdf, code or notebook file, and allows it for a
        # note (review B2-LINK-WARNING-MISSES-EXTENSIONS): these were never matched
        f.write(_encode(_rec("o3", "Keep\\pdflink.md", "note", wikilinks=["Old/Book.pdf#page=3"])) + b"\r\n")
        f.write(_encode(_rec("o4", "Keep\\mdlink.md", "note", wikilinks=["note.md|alias"])) + b"\r\n")
        # a table row escapes the alias pipe, so the stored link keeps a trailing '\' (D-GW-4)
        table = extract_wikilinks("| a | [[note\\|the note]] |")
        f.write(_encode(_rec("o5", "Keep\\table.md", "note", wikilinks=table)) + b"\r\n")
    quarantine = [{"op": "quarantine", "kind": "file", "indexed": True, "old_sf": "Gone\\q.md"}]

    p = index_ops.plan(world.cfg, world.moves, quarantine, world.client)

    assert p["blockers"] == []
    (canvas,) = [w for w in p["warnings"] if "Old/other.canvas" in w]
    assert "cites" in canvas and "REPLACEd" in canvas
    assert "are rewritten to" in canvas
    assert len([w for w in p["warnings"] if "cites" in w]) == 1
    assert not any("stay stale unless slice D" in w for w in p["warnings"])
    (note,) = [w for w in p["warnings"] if "Keep\\w.md" in w]
    assert "links [[note]]" in note
    (pdflink,) = [w for w in p["warnings"] if "Keep\\pdflink.md" in w]
    assert "links [[Book]]" in pdflink
    (mdlink,) = [w for w in p["warnings"] if "Keep\\mdlink.md" in w]
    assert "links [[note]]" in mdlink
    (tablelink,) = [w for w in p["warnings"] if "Keep\\table.md" in w]
    assert "links [[note]]" in tablelink
    (quar,) = [w for w in p["warnings"] if "quarantine" in w]
    assert "Gone\\q.md" in quar and "documents/delete" in quar


def test_dry_run_finds_a_link_past_the_wikilink_metadata_cap(world, servers_down):
    """parse_file keeps only the first links of a note in its metadata (a cap, for size), so
    a link to the renamed file that comes later is invisible there: the note is read instead."""
    from src.ingestion.obsidian_parser import ObsidianParser
    note = world.vault / "Keep" / "w.md"
    note.parent.mkdir()
    note.write_text(
        " ".join(f"[[Other {n}]]" for n in range(21)) + " [[Some Folder/NOTE#Part]] "
        + " ".join(f"[[More {n}]]" for n in range(3)) + "\n\n"
        + "Filler sentence to clear the minimum chunk size. " * 6, encoding="utf-8")
    (doc,) = ObsidianParser(str(world.vault)).parse_file(note)
    links = doc.metadata["wikilinks"]                       # what the index really holds
    assert "Some Folder/NOTE#Part" not in links and len(links) < 25
    with open(world.data / "other_chunks.jsonl", "ab") as f:      # gone.md: indexed, not on disk
        for doc_id, name in (("o2", "w"), ("o3", "gone")):
            f.write(_encode(_rec(doc_id, f"Keep\\{name}.md", "note", wikilinks=links)) + b"\r\n")

    p = index_ops.plan(world.cfg, world.moves, [], world.client)

    assert p["blockers"] == []
    (found,) = [w for w in p["warnings"] if "Keep\\w.md" in w]
    assert "links [[note]]" in found
    (unread,) = [w for w in p["warnings"] if "Keep\\gone.md" in w]
    assert str(len(links)) in unread and "not on disk" in unread      # said out loud, never skipped


# ---- apply ------------------------------------------------------------------------------

def test_apply_moves_path_metadata_in_every_store_and_nothing_else(world, servers_down, monkeypatch):
    monkeypatch.setattr(index_ops, "PAGE", 1)                  # the update pages: 2 rows, 2 pages
    original = {n: (world.data / n).read_bytes() for n in JSONLS}
    dump, vectors = _dump(world), _vectors(world)

    res = index_ops.apply(world.cfg, world.moves, [], world.client)

    assert res["jsonl"] == {"chunks.jsonl": 1, "pdf_chunks.jsonl": 2, "code_chunks.jsonl": 1}
    assert res["parents"] == 1 and res["hype"] == 1 and res["ocr_studio"] == 1
    assert res["collections"] == {ACTIVE: 4, EG2: 4, HYPE: 2}
    # every JSONL row: moved rows get the new path keys only, the rest is byte-identical
    for name, rows in JSONLS.items():
        lines = (world.data / name).read_bytes().split(b"\r\n")
        assert lines.pop() == b""                              # CRLF kept, no bare \n
        assert len(lines) == len(rows)
        for line, row in zip(lines, rows):
            meta = row.get("metadata", row)
            if meta["source_file"] not in MOVED:
                assert line == _encode(row), name
            elif "metadata" in row:
                want = dict(row, metadata=index_ops.relocate_meta(
                    meta, meta["source_file"], MOVED[meta["source_file"]]))
                assert json.loads(line) == want and want["metadata"]["course_name"] == "Course"
                assert json.loads(line)["doc_id"] == row["doc_id"] and json.loads(line)["text"] == row["text"]
            else:                                              # parents / HyPE sidecar: top level
                assert json.loads(line) == dict(row, source_file=MOVED[meta["source_file"]])
    # a .bak beside every rewritten file and nowhere else
    baks = {p.name.split(".relocate-")[0]: p for p in world.data.glob("*.bak")}
    assert sorted(baks) == sorted(REWRITTEN)
    assert all(baks[n].read_bytes() == original[n] for n in REWRITTEN)
    # every collection: same ids, documents and vectors; metadata moved, Chroma-only keys kept
    after = _dump(world)
    for name, rows in dump.items():
        assert set(after[name]) == set(rows)
        for i, (meta, doc) in rows.items():
            a_meta, a_doc = after[name][i]
            assert a_doc == doc
            if meta["source_file"] in MOVED:
                assert a_meta == index_ops.relocate_meta(meta, meta["source_file"], MOVED[meta["source_file"]])
            else:
                assert a_meta == meta
    assert _vectors(world) == vectors
    assert after[ACTIVE]["m1"][0]["tags"] == "t"
    assert after[HYPE]["m1::q0"][0] == {"doc_id": "m1", "source_file": NOTE2}
    # OCR Studio: the dir follows the new key, doc.json names the new path, the page set stays
    moved = world.data / "ocr_studio" / doc_key("Old/Book Renamed.pdf")
    assert not world.ocr.exists() and (moved / "page_1.json").exists()
    assert json.loads((moved / "doc.json").read_text(encoding="utf-8")) == {
        "source": "Old/Book Renamed.pdf", "pages": [1]}


def _listen(w):
    s, port = _listener()
    w.sockets.append(s)
    w.cfg = Config({**w.cfgd, "webui": {**w.cfgd["webui"], "port": port}}, w.root)


def _index_new_note(w):
    with open(w.data / "other_chunks.jsonl", "ab") as f:
        f.write(_encode(_rec("o9", NOTE2, "note")) + b"\r\n")


def _clash(w):
    w.moves[2].update(new="Old/NOTE2.MD", new_sf="Old\\NOTE2.MD")


def _canvas(w):
    w.moves.append({**_move("Old/c.canvas", "Old/c2.canvas", "canvas", 1),
                    "old_sf": "Old/c.canvas", "new_sf": "Old/c2.canvas"})


def _building(w):
    write_sidecar(w.root / "chroma", EG2, make_sidecar(
        EmbeddingSpec("local", "local", "fake/m"), collection=EG2, role="chunks", status="building",
        dimension=4, count=1, written_by="rebuild-into"))


def _canvas_by_type(w):
    """A canvas whose map row misnames its lane and carries a valid backslash new_sf: only
    the STORED file_type can refuse it (review T3)."""
    (w.vault / "Old" / "c2.canvas").write_bytes(b"x")
    w.moves.append({**_move("Old/c.canvas", "Old/c2.canvas", "md", 1), "old_sf": "Old/c.canvas"})


REFUSALS = {
    "port": (_listen, r":\d+ answers"),
    "unapproved": (lambda w: w.moves[0].update(approved=False), "not approved"),
    "n_chunks": (lambda w: w.moves[1].update(n_chunks=5), "n_chunks 5"),
    "new_missing": (lambda w: (w.vault / "Old" / "note2.md").unlink(), "missing on disk"),
    "old_present": (lambda w: (w.vault / "Old" / "note.md").write_bytes(b"x"), "still on disk"),
    "new_sf_indexed": (_index_new_note, "new_sf already present"),
    "case_clash": (_clash, "differ only in case"),
    "slash_form": (lambda w: w.moves[0].update(new_sf="Old/Book Renamed.pdf"), "backslash"),
    "canvas": (_canvas, "canvas moves are REPLACE"),
    "canvas_by_type": (_canvas_by_type, "canvas moves are REPLACE"),
    "building": (_building, "building"),
    # review T4: with n_chunks 0 a wrong old_sf passes every count check; only this stops it
    "no_store": (lambda w: w.moves[0].update(old_sf="Old/Book.pdf", n_chunks=0), "found in no store"),
    # review T5: the only refusal standing between a clash and a non-resumable half-apply
    "ocr_target": (lambda w: (w.data / "ocr_studio" / doc_key("Old/Book Renamed.pdf")).mkdir(),
                   "OCR Studio target folder exists"),
    # review T6: verify would catch it only after every store was written
    "dense_sparse": (lambda w: w.client.get_collection(ACTIVE).delete(ids=["p2"]),
                     "dense and sparse disagree"),
}


@pytest.mark.parametrize("case", list(REFUSALS))
def test_apply_refuses_before_writing_anything(case, world, monkeypatch):
    mutate, reason = REFUSALS[case]
    if case == "port":                                  # the real probe, against a real listener
        monkeypatch.setattr(index_ops, "STREAMLIT_PORT", _closed_port())
    else:
        monkeypatch.setattr(index_ops, "answering_ports", lambda cfg: [])
    mutate(world)
    files, dump = _files(world), _dump(world)
    try:
        with pytest.raises(index_ops.RelocateError, match=reason):
            index_ops.apply(world.cfg, world.moves, [], world.client)
    finally:
        for s in world.sockets:
            s.close()
    assert _files(world) == files and _dump(world) == dump


def test_apply_failure_leaves_restorable_backups(world, servers_down, monkeypatch):
    """A failure in the Chroma step leaves the JSONLs already rewritten: every one has a
    relocate .bak that restores it byte-for-byte, and a second apply refuses (new_sf is
    already present). Chroma metadata written before the failure is NOT restored by these
    .bak files; the runbook's store backup (6.3 step 1) covers it, so this asserts only the
    JSONL restore and the refusal. Relocate is not resumable by design."""
    from chromadb.api.models.Collection import Collection
    real_update = Collection.update

    def update(self, *a, **kw):
        if self.name == EG2:
            raise RuntimeError("injected eg2 failure")
        return real_update(self, *a, **kw)

    monkeypatch.setattr(Collection, "update", update)
    original = {n: (world.data / n).read_bytes() for n in JSONLS}

    with pytest.raises(RuntimeError, match="injected eg2 failure"):
        index_ops.apply(world.cfg, world.moves, [], world.client)
    for name in REWRITTEN:
        (bak,) = world.data.glob(name + ".relocate-*.bak")
        assert bak.read_bytes() == original[name]
        assert (world.data / name).read_bytes() != original[name]
    assert len(list(world.data.glob("*.bak"))) == len(REWRITTEN)     # none beside the untouched files

    with pytest.raises(index_ops.RelocateError, match="new_sf already present"):
        index_ops.apply(world.cfg, world.moves, [], world.client)
    for name in REWRITTEN:                                # the restore the runbook performs
        (bak,) = world.data.glob(name + ".relocate-*.bak")
        (world.data / name).write_bytes(bak.read_bytes())
        assert (world.data / name).read_bytes() == original[name]


def _citing_canvas(w):
    """A canvas row outside the map that cites TWO moved files: Old/Book.pdf in both reference
    keys, Old/note.md through a second canvas_edge_files entry. In its JSONL (CRLF) and in both
    chunk collections, Chroma-shaped, as the world fixture builds them."""
    row = _rec("v3", "Maps/m.canvas", "canvas", filename="m", canvas_node_id="n1",
               canvas_edge_files=[encode_canvas_edge("f1", "Old/Book.pdf", "#page=5"),
                                  encode_canvas_edge("f2", "Old/note.md", "")],
               canvas_file_refs=["Old/Book.pdf"])
    with open(w.data / "canvas_chunks.jsonl", "ab") as f:
        f.write(_encode(row) + b"\r\n")
    for name, dim in ((ACTIVE, 3), (EG2, 4)):
        w.client.get_collection(name).add(ids=["v3"], embeddings=[[9.0] + [0.0] * (dim - 1)],
                                          documents=[row["text"]],
                                          metadatas=[{**Embedder._clean_meta(row["metadata"]), "tags": "t"}])
    return row


CITED = [{"node": "f1", "file": "Old/Book Renamed.pdf", "subpath": "#page=5"},
         {"node": "f2", "file": "Old/note2.md", "subpath": ""}]


def test_apply_rewrites_canvas_refs_in_jsonl_and_every_chunk_collection(world, servers_down):
    row = _citing_canvas(world)
    dump, vectors = _dump(world), _vectors(world)

    p = index_ops.plan(world.cfg, world.moves, [], world.client)

    assert p["blockers"] == []
    assert p["canvas_ref_rows"] == [("canvas_chunks.jsonl", "v3")]      # one row, though it cites two moves
    assert [r["canvas_refs"] for r in p["rows"]] == [1, 1, 0]           # Book, note, code
    for r, old, new in zip(p["rows"], ("Old/Book.pdf", "Old/note.md"), ("Old/Book Renamed.pdf", "Old/note2.md")):
        (cite,) = [w for w in r["warnings"] if "cites" in w]
        assert cite.startswith(f"Maps/m.canvas cites {old}: ") and f"rewritten to {new};" in cite
    assert not [w for w in p["rows"][2]["warnings"] if "cites" in w]

    res = index_ops.apply(world.cfg, world.moves, [], world.client)     # 4 own rows + the canvas once

    assert res["jsonl"] == {"chunks.jsonl": 1, "pdf_chunks.jsonl": 2, "code_chunks.jsonl": 1,
                            "canvas_chunks.jsonl": 1}
    assert res["collections"] == {ACTIVE: 4, EG2: 4, HYPE: 2}
    (line,) = [x for x in (world.data / "canvas_chunks.jsonl").read_bytes().split(b"\r\n") if b'"v3"' in x]
    got = json.loads(line)
    assert decode_canvas_files(got["metadata"]["canvas_edge_files"]) == CITED
    assert got["metadata"]["canvas_file_refs"] == ["Old/Book Renamed.pdf"]
    assert (got["doc_id"], got["text"]) == ("v3", row["text"])
    assert dict(got["metadata"], canvas_edge_files=None, canvas_file_refs=None) == dict(
        row["metadata"], canvas_edge_files=None, canvas_file_refs=None)
    after = _dump(world)
    for name in (ACTIVE, EG2):
        meta, doc = after[name]["v3"]
        assert decode_canvas_files(meta["canvas_edge_files"]) == CITED
        assert meta["canvas_file_refs"] == "Old/Book Renamed.pdf" and meta["tags"] == "t"
        assert doc == dump[name]["v3"][1]
    assert after[HYPE]["m1::q0"][0] == {"doc_id": "m1", "source_file": NOTE2}   # B's own-row move only
    assert _vectors(world) == vectors


def test_a_canvas_id_on_two_lines_is_planned_twice_and_applied(world, servers_down):
    """rewrite_jsonl rewrites LINES, so the plan counts citing lines, not ids: a canvas id on
    two lines of one JSONL must pass the dry run AND the apply's changed-rows check."""
    row = _citing_canvas(world)
    with open(world.data / "canvas_chunks.jsonl", "ab") as f:
        f.write(_encode(row) + b"\r\n")                                # the same id on a second line
    p = index_ops.plan(world.cfg, world.moves, [], world.client)
    assert p["blockers"] == []
    assert p["canvas_ref_rows"] == [("canvas_chunks.jsonl", "v3")] * 2
    res = index_ops.apply(world.cfg, world.moves, [], world.client)
    assert res["jsonl"]["canvas_chunks.jsonl"] == 2


def test_apply_refuses_when_changed_rows_differ_from_the_plan(world, servers_down, monkeypatch):
    real = index_ops.rewrite_jsonl

    def one_short(path, fix, **kw):
        n = real(path, fix, **kw)
        return n - 1 if Path(path).name == "pdf_chunks.jsonl" else n

    monkeypatch.setattr(index_ops, "rewrite_jsonl", one_short)
    dump = _dump(world)
    with pytest.raises(index_ops.RelocateError, match="planned rows"):
        index_ops.apply(world.cfg, world.moves, [], world.client)
    assert _dump(world) == dump                           # step 4 never ran


# ---- verify -----------------------------------------------------------------------------

def test_verify_fails_before_and_passes_after(world, servers_down):
    fails = index_ops.verify(world.cfg, world.moves, [], world.client)
    for old in MOVED:
        assert any(old in f for f in fails), old

    index_ops.apply(world.cfg, world.moves, [], world.client)
    fails = index_ops.verify(world.cfg, world.moves, [], world.client)
    assert len(fails) == 1 and "rebuild_bm25" in fails[0]       # the sparse index predates the rewrite
    os.utime(world.bm25)
    assert index_ops.verify(world.cfg, world.moves, [], world.client) == []

    # a quarantine row whose source is still indexed
    fails = index_ops.verify(world.cfg, world.moves, [{"old_sf": "Keep\\k.md"}], world.client)
    assert fails and all("Keep\\k.md" in f for f in fails)

    # an unparseable line that still holds an old stored path
    with open(world.data / "other_chunks.jsonl", "ab") as f:
        f.write(b'{"doc_id": "broken", "x": ' + json.dumps(BOOK).encode("utf-8") + b"\r\n")
    os.utime(world.bm25)
    (fail,) = index_ops.verify(world.cfg, world.moves, [], world.client)
    assert "other_chunks.jsonl" in fail and BOOK in fail

    # the OCR Studio record still naming the old path
    (world.data / "other_chunks.jsonl").write_bytes(b"")
    os.utime(world.bm25)
    (world.data / "ocr_studio" / "stale").mkdir()
    (world.data / "ocr_studio" / "stale" / "doc.json").write_text(
        json.dumps({"source": "Old/Book.pdf"}), encoding="utf-8")
    (fail,) = index_ops.verify(world.cfg, world.moves, [], world.client)
    assert "OCR Studio" in fail and "Old/Book.pdf" in fail


def test_verify_names_a_canvas_row_still_citing_old_sf(world, servers_down):
    _citing_canvas(world)

    fails = index_ops.verify(world.cfg, world.moves, [], world.client)
    for store in ("canvas_chunks.jsonl", ACTIVE):
        for old in (BOOK, NOTE):
            assert f"{store}: canvas Maps/m.canvas node n1 still cites {old}" in fails

    index_ops.apply(world.cfg, world.moves, [], world.client)
    os.utime(world.bm25)
    assert index_ops.verify(world.cfg, world.moves, [], world.client) == []


# ---- main.py relocate -------------------------------------------------------------------

def test_cli_relocate_modes_and_exit_codes(world, servers_down, capsys):
    import main
    cfg_path = world.root / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(world.cfgd), encoding="utf-8")
    _write_map(world.map, world.moves,
               extra=[{**world.moves[0], "kind": "dir"}, {**world.moves[0], "indexed": False}])

    def run(*flags):
        args = main.build_parser().parse_args(
            ["--config", str(cfg_path), "relocate", "--map", str(world.map), *flags])
        try:
            args.func(args)
        except SystemExit as e:
            return e.code
        return 0

    assert run("--dry-run") == 0
    out = capsys.readouterr().out
    assert "Apply would: PROCEED" in out
    assert f"[1] {BOOK}\n -> {BOOK2}\n" in out and "TOTAL  jsonl 4 " in out

    world.moves[1]["approved"] = False
    _write_map(world.map, world.moves)
    assert run("--dry-run") == 1
    out = capsys.readouterr().out
    assert "Apply would: REFUSE" in out and "not approved" in out
    assert run() == 2                                           # apply: a RelocateError
    assert "not approved" in capsys.readouterr().err

    world.moves[1]["approved"] = True
    _write_map(world.map, world.moves)
    assert run() == 0
    assert "NEXT: python rebuild_bm25.py" in capsys.readouterr().out
    assert run("--verify") == 1                                 # the sparse index is older
    assert "FAIL" in capsys.readouterr().out
    os.utime(world.bm25)
    assert run("--verify") == 0
    assert "PASS" in capsys.readouterr().out


def test_cli_relocate_refuses_while_a_port_answers(world, monkeypatch, capsys):
    import main
    monkeypatch.setattr(index_ops, "answering_ports", lambda cfg: [8051, 8501])
    cfg_path = world.root / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(world.cfgd), encoding="utf-8")
    _write_map(world.map, world.moves)
    files = _files(world)
    for flags in ([], ["--dry-run"], ["--verify"]):
        args = main.build_parser().parse_args(
            ["--config", str(cfg_path), "relocate", "--map", str(world.map), *flags])
        with pytest.raises(SystemExit) as e:
            args.func(args)
        assert e.value.code == 2
    assert ":8051, :8501 answers" in capsys.readouterr().err
    assert _files(world) == files
