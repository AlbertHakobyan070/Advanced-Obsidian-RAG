"""canvas_refresh.py: the metadata-only canvas refresh (slice D, T3).

Run:  python -m pytest tests/test_canvas_refresh.py -q

A tmp data folder (canvas rows in canvas_chunks.jsonl and in a console reindex file,
a note row in chunks.jsonl) and a tmp Chroma store (the active collection, an eg2
re-embed, the HyPE questions and the graph twin), with explicit vectors and no model.
The fresh ingest (--new) sits outside the data folder. Nothing here touches data/ or
a live store.
"""
import json
import socket
import sys
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import yaml

import canvas_refresh
from src import index_ops
from src.embeddings.embedder import Embedder
from src.embeddings.registry import EmbeddingSpec
from src.embeddings.sidecar import GRAPH_SUFFIX, make_sidecar, write_sidecar
from src.ingestion.canvas_loader import encode_canvas_edge
from src.utils.chroma_client import persistent_client

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


ACTIVE, EG2, HYPE, TWIN = ("obsidian_vault", "obsidian_vault_eg2_l1024", "hype_questions",
                           "obsidian_vault" + GRAPH_SUFFIX)


def _canvas(doc_id, sf, **meta):
    """A canvas row as indexed before slice D: no canvas_edge_files, and a console retag."""
    return {"doc_id": doc_id, "text": f"[Canvas: {PurePosixPath(sf).stem}]\nNode {doc_id}\n\nConnections:\n  -> z",
            "metadata": {"source_file": sf, "file_type": "canvas", "filename": PurePosixPath(sf).stem,
                         "course_name": "Course", "domain": "ml", "tags": ["retagged"],
                         "canvas_node_id": doc_id, "canvas_neighbors": ["z"],
                         "canvas_edges": [encode_canvas_edge("z", "->", "")],
                         "canvas_file_refs": ["Books/svm.pdf"], "canvas_degree": 1, **meta}}


C1 = _canvas("c1", "Maps/a.canvas", canvas_group="Group 1")
C2 = _canvas("c2", "Maps/a.canvas")
C3 = _canvas("c3", "Maps/b.canvas")
NOTE = {"doc_id": "m1", "text": "a note", "metadata": {"source_file": "Notes\\n.md", "file_type": "note",
                                                       "tags": ["x"]}}


def _fresh():
    """The same three rows from a post-D ingest: loader tags (the retag is not in it),
    canvas_edge_files, arrowhead directions; c1's group box was removed since."""
    rows = json.loads(json.dumps([C1, C2, C3]))
    for r in rows:
        r["metadata"].update(tags=["svm"], canvas_edge_files=[encode_canvas_edge("f1", "Books/svm.pdf", "")])
    del rows[0]["metadata"]["canvas_group"]
    rows[0]["metadata"]["canvas_edges"] = [encode_canvas_edge("z", "--", "")]
    rows[2]["metadata"].update(canvas_edges=[encode_canvas_edge("z", "<>", "")], canvas_edge_files=[
        encode_canvas_edge("f1", "Books/svm.pdf", "#page=7"), encode_canvas_edge("", "Books/svm.pdf", "#page=9")])
    return rows


def _enc(row):
    return json.dumps(row, ensure_ascii=False).encode("utf-8")


def _write_new(path, rows):
    path.write_bytes(b"".join(_enc(r) + b"\n" for r in rows))


def _closed_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def servers_down(monkeypatch):
    """The dev machine may have :8051 up; a run that is meant to proceed says none answers."""
    monkeypatch.setattr(index_ops, "answering_ports", lambda cfg: [])


@pytest.fixture
def world(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "chunks.jsonl").write_bytes(_enc(NOTE) + b"\r\n")
    (data / "canvas_chunks.jsonl").write_bytes(_enc(C1) + b"\n" + _enc(C2) + b"\r\n")   # one LF, one CRLF
    (data / "canvas_edit_x_chunks.jsonl").write_bytes(_enc(C3) + b"\r\n")                 # a console reindex
    (tmp_path / "scratch").mkdir()
    new, fresh = tmp_path / "scratch" / "canvas_new.jsonl", _fresh()
    _write_new(new, fresh)

    chroma = tmp_path / "chroma"
    client = persistent_client(chroma)
    for name, dim, rows, eg2_tags in ((ACTIVE, 3, [NOTE, C1, C2, C3], None), (EG2, 4, [NOTE, C1, C2, C3], "old-tag"),
                                      (TWIN, 3, [C1, C2, C3], None)):
        col = client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
        col.add(ids=[r["doc_id"] for r in rows],
                embeddings=[[1.0 + k] + [0.0] * (dim - 1) for k in range(len(rows))],
                documents=[r["text"] for r in rows],
                # eg2 never received the console retag (manage_api retags the active collection only)
                metadatas=[{**Embedder._clean_meta(r["metadata"]),
                            **({"tags": eg2_tags} if eg2_tags and r is not NOTE else {})} for r in rows])
    client.get_or_create_collection(HYPE, metadata={"hnsw:space": "cosine"}).add(
        ids=["m1::q0"], embeddings=[[1.0, 0.0, 0.0]], documents=["a?"],
        metadatas=[{"doc_id": "m1", "source_file": "Notes\\n.md"}])

    cfgd = {"paths": {"chunks_file": str(data / "chunks.jsonl"), "chroma_dir": str(chroma),
                      "collection_name": ACTIVE},
            "retrieval": {"hype": {"collection": HYPE}},
            "webui": {"port": _closed_port(), "rag_api": f"http://127.0.0.1:{_closed_port()}"}}
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfgd), encoding="utf-8")
    return SimpleNamespace(root=tmp_path, data=data, chroma=chroma, client=client, new=new, fresh=fresh,
                           cfgd=cfgd, cfg_path=cfg_path)


def _run(w, *flags, new=None):
    try:
        canvas_refresh.main(["--config", str(w.cfg_path), "--new", str(new or w.new), *flags])
    except SystemExit as e:
        return e.code
    return 0


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


def _no_store(*a, **kw):
    raise AssertionError("the store was opened")


# ---- the pure parts ---------------------------------------------------------------------

def test_merged_meta_replaces_only_owned_keys():
    base = {"source_file": "Maps/a.canvas", "file_type": "canvas", "tags": "retagged", "domain": "ml",
            "canvas_edges": "z|->|", "canvas_group": "Group 1"}
    new = {"source_file": "Maps/a.canvas", "file_type": "canvas", "tags": "svm", "domain": "general",
           "canvas_edges": "z|--|", "canvas_edge_files": "f1|Books/svm.pdf|"}

    out = canvas_refresh.merged_meta(base, new)

    assert out == {"source_file": "Maps/a.canvas", "file_type": "canvas", "tags": "retagged", "domain": "ml",
                   "canvas_edges": "z|--|", "canvas_edge_files": "f1|Books/svm.pdf|"}
    assert canvas_refresh.merged_meta(out, new) == out         # idempotent: a re-run changes nothing


# ---- refusals: nothing is written ----------------------------------------------------------

def test_only_stored_ids_refuse_apply(world, servers_down, capsys):
    _write_new(world.new, world.fresh[:2])                     # c3's node is gone from its canvas
    files, dump = _files(world), _dump(world)

    assert _run(world) == 0
    assert "only_stored 1 (first: c3)" in capsys.readouterr().out
    assert _run(world, "--apply") == 2
    err = capsys.readouterr().err
    assert err.startswith("ERROR:") and "c3" in err and "orphan" in err
    assert _files(world) == files and _dump(world) == dump


def test_text_change_refuses_apply(world, servers_down, capsys):
    world.fresh[1]["text"] += " edited"
    _write_new(world.new, world.fresh)
    files, dump = _files(world), _dump(world)

    assert _run(world) == 0
    assert "text_changed 1 (first: c2)" in capsys.readouterr().out
    assert _run(world, "--apply") == 2
    assert ("ERROR: 1 canvas row(s) changed text (first: c2): REPLACE those canvases first"
            in capsys.readouterr().err)
    assert _files(world) == files and _dump(world) == dump


def test_new_file_inside_data_dir_is_refused(world, servers_down, monkeypatch, capsys):
    inside = world.data / "canvas_new.jsonl"
    inside.write_bytes(world.new.read_bytes())
    monkeypatch.setattr(canvas_refresh, "persistent_client", _no_store)
    files = _files(world)

    for flags in ([], ["--apply"]):
        assert _run(world, *flags, new=inside) == 2
        assert "sparse union" in capsys.readouterr().err
    assert _files(world) == files


def test_answering_port_refuses_before_opening_the_store(world, monkeypatch, capsys):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(4)          # two runs probe it and nothing accepts: a backlog of 1 times the second out
    port = listener.getsockname()[1]
    try:
        monkeypatch.setattr(index_ops, "STREAMLIT_PORT", _closed_port())
        world.cfgd["webui"] = {"port": port, "rag_api": f"http://127.0.0.1:{_closed_port()}"}
        world.cfg_path.write_text(yaml.safe_dump(world.cfgd), encoding="utf-8")
        monkeypatch.setattr(canvas_refresh, "persistent_client", _no_store)
        for flags in ([], ["--apply"]):
            assert _run(world, *flags) == 2
            err = capsys.readouterr().err
            assert f":{port}" in err and "answers" in err
    finally:
        listener.close()


def test_building_target_refuses_apply(world, servers_down, capsys):
    write_sidecar(world.chroma, EG2, make_sidecar(
        EmbeddingSpec("local", "local", "fake/m"), collection=EG2, role="chunks", status="building",
        dimension=4, count=1, written_by="rebuild-into"))
    files, dump = _files(world), _dump(world)

    assert _run(world) == 0
    (line,) = [x for x in capsys.readouterr().out.splitlines() if x.strip().startswith("building")]
    assert EG2 in line
    assert _run(world, "--apply") == 2
    err = capsys.readouterr().err
    assert EG2 in err and "--rebuild-into" in err
    assert _files(world) == files and _dump(world) == dump


def test_target_missing_a_stored_id_refuses_apply(world, servers_down, capsys):
    world.client.get_collection(EG2).delete(ids=["c2"])        # a delete since eg2 was built
    files, dump = _files(world), _dump(world)

    assert _run(world) == 0
    assert f"missing_in_target {EG2} 1 (first: c2)" in capsys.readouterr().out
    assert _run(world, "--apply") == 2
    err = capsys.readouterr().err
    assert EG2 in err and "index --rebuild-into" in err
    assert _files(world) == files and _dump(world) == dump


def test_diverged_text_refuses_apply(world, servers_down, capsys):
    """The JSONL text differs from the active collection's (the fresh ingest agrees with
    the collection, so no text changed): pre-existing dense/sparse drift, a human call."""
    (world.data / "canvas_edit_x_chunks.jsonl").write_bytes(_enc({**C3, "text": C3["text"] + " drift"}) + b"\r\n")
    files, dump = _files(world), _dump(world)

    assert _run(world) == 0
    out = capsys.readouterr().out
    assert "diverged 1 (first: c3)" in out and "text_changed 0" in out
    assert _run(world, "--apply") == 2
    err = capsys.readouterr().err
    assert "c3" in err and "dense/sparse drift" in err
    assert _files(world) == files and _dump(world) == dump


def test_unindexed_new_rows_refuse_apply_unless_allowed(world, servers_down, capsys):
    _write_new(world.new, [*world.fresh, _canvas("c9", "Maps/b.canvas")])
    files, dump = _files(world), _dump(world)

    assert _run(world, "--apply") == 2
    err = capsys.readouterr().err
    assert "c9" in err and "--allow-unindexed-new" in err                # the message names the way out
    assert _files(world) == files and _dump(world) == dump
    assert _run(world, "--apply", "--allow-unindexed-new") == 0       # the rest is refreshed
    assert all("c9" not in rows for rows in _dump(world).values())    # and the id is never added
    for name in ("chunks.jsonl", "canvas_chunks.jsonl", "canvas_edit_x_chunks.jsonl"):
        assert b'"c9"' not in (world.data / name).read_bytes()
    assert "canvas_edge_files" in _dump(world)[ACTIVE]["c1"][0]


def test_a_jsonl_that_changed_between_read_and_write_raises(world, servers_down, monkeypatch):
    real, skipped = index_ops.rewrite_jsonl, []

    def drop_one_canvas_row(path, fix, **kw):
        if kw.get("tag") != "canvas-refresh":                          # the plan's reads pass through
            return real(path, fix, **kw)

        def fix2(rec):
            if not skipped and (rec.get("metadata") or {}).get("file_type") == "canvas":
                skipped.append(rec["doc_id"])                           # gone between read and write
                return None
            return fix(rec)
        return real(path, fix2, **kw)

    monkeypatch.setattr(index_ops, "rewrite_jsonl", drop_one_canvas_row)
    with pytest.raises(RuntimeError, match="changed between read and write"):
        _run(world, "--apply")
    assert skipped == ["c1"]


def test_a_chunk_file_name_with_glob_characters_is_journalled(world, servers_down):
    """'[v2]' in a JSONL name is a glob character class: the journal names the backup
    rewrite_jsonl wrote, found with the name escaped."""
    odd = world.data / "canvas [v2]_chunks.jsonl"
    (world.data / "canvas_edit_x_chunks.jsonl").rename(odd)
    assert _run(world, "--apply") == 0
    entry = json.loads((world.root / "logs" / "canvas_refresh.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    (bak,) = [p for p in world.data.iterdir() if p.name.startswith(odd.name + ".canvas-refresh-")]
    assert entry["jsonl"][odd.name]["backup"] == bak.name


def test_duplicate_stored_id_refuses_apply(world, servers_down, capsys):
    with open(world.data / "canvas_edit_x_chunks.jsonl", "ab") as f:
        f.write(_enc(C1) + b"\r\n")
    files, dump = _files(world), _dump(world)

    assert _run(world) == 0
    assert "duplicate_stored 1 (first: c1)" in capsys.readouterr().out
    assert _run(world, "--apply") == 2
    assert "c1" in capsys.readouterr().err
    assert _files(world) == files and _dump(world) == dump


# ---- apply ----------------------------------------------------------------------------------

def test_apply_rewrites_jsonl_atomically_with_backup(world, servers_down, capsys):
    original = {n: (world.data / n).read_bytes() for n in
                ("chunks.jsonl", "canvas_chunks.jsonl", "canvas_edit_x_chunks.jsonl")}
    note_mtime = (world.data / "chunks.jsonl").stat().st_mtime_ns

    assert _run(world, "--apply") == 0
    assert "rebuild_bm25.py" in capsys.readouterr().out

    fresh = {r["doc_id"]: r["metadata"] for r in world.fresh}
    for name, rows in (("canvas_chunks.jsonl", [C1, C2]), ("canvas_edit_x_chunks.jsonl", [C3])):
        (bak,) = world.data.glob(name + ".canvas-refresh-*.bak")
        assert bak.read_bytes() == original[name]
        lines = (world.data / name).read_bytes().splitlines(keepends=True)
        assert len(lines) == len(rows)
        for line, row in zip(lines, rows):
            got = json.loads(line)
            assert got["doc_id"] == row["doc_id"] and got["text"] == row["text"]
            assert got["metadata"] == canvas_refresh.merged_meta(row["metadata"], fresh[row["doc_id"]])
            assert got["metadata"]["tags"] == ["retagged"]          # the console's retag survives
    lines = (world.data / "canvas_chunks.jsonl").read_bytes().splitlines(keepends=True)
    assert lines[0].endswith(b"}\n") and lines[1].endswith(b"}\r\n")   # each line keeps its own terminator
    assert (world.data / "chunks.jsonl").read_bytes() == original["chunks.jsonl"]
    assert (world.data / "chunks.jsonl").stat().st_mtime_ns == note_mtime
    assert not list(world.data.glob("chunks.jsonl.*.bak"))
    (entry,) = (world.root / "logs" / "canvas_refresh.jsonl").read_text(encoding="utf-8").splitlines()
    entry = json.loads(entry)
    assert sorted(entry["jsonl"]) == ["canvas_chunks.jsonl", "canvas_edit_x_chunks.jsonl"]
    for name, rec in entry["jsonl"].items():
        (bak,) = world.data.glob(name + ".canvas-refresh-*.bak")
        assert rec["backup"] == bak.name
    assert entry["targets"] == [ACTIVE, EG2]

    # a re-plan finds nothing left to change (the dropped canvas_group is gone from Chroma too)
    assert _run(world) == 0
    assert "meta_changed 0" in capsys.readouterr().out


def test_chroma_update_on_a_tmp_store_matches_merged_meta(world, servers_down, capsys):
    """U1, settled by review (decision 16), observed here on chromadb 1.5.9: update()
    MERGES the keys it is given into the stored metadata (a key it is not given stays),
    and a None value DELETES that key. So an OWNED key the new ingest dropped is sent
    as None and disappears, and every key outside OWNED (a retag) is left as it was."""
    assert _run(world) == 0
    out = capsys.readouterr().out
    assert f"collections {ACTIVE}, {EG2}" in out                  # not the HyPE questions, not the twin
    assert "meta_changed 3" in out and "subpaths 1 (first: c3)" in out and "source_entries 1" in out
    assert f"retags_kept {EG2} 3 (first: c1, c2, c3)" in out and f"retags_kept {ACTIVE} " not in out
    before = _dump(world)

    assert _run(world, "--apply") == 0

    after = _dump(world)
    fresh = {r["doc_id"]: r["metadata"] for r in world.fresh}
    for name in (ACTIVE, EG2):
        for i in ("c1", "c2", "c3"):
            meta, doc = after[name][i]
            assert meta == canvas_refresh.merged_meta(before[name][i][0], Embedder._clean_meta(fresh[i]))
            assert doc == before[name][i][1]
        assert "canvas_group" not in after[name]["c1"][0] and "canvas_edge_files" in after[name]["c1"][0]
        assert after[name]["m1"] == before[name]["m1"]
    assert after[ACTIVE]["c1"][0]["tags"] == "retagged"
    assert after[EG2]["c1"][0]["tags"] == "old-tag"               # kept, never pushed
    assert after[HYPE] == before[HYPE] and after[TWIN] == before[TWIN]

    # merge, not replace: an update naming one key leaves the others
    col = world.client.get_collection(ACTIVE)
    col.update(ids=["m1"], metadatas=[{"probe": 1}])
    assert col.get(ids=["m1"], include=["metadatas"])["metadatas"][0] == {**before[ACTIVE]["m1"][0], "probe": 1}
