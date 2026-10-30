"""The console's delete reaches every store a chunk lives in (slice B, task 4).

Run:  python -m pytest tests/test_delete_all_collections.py -q

A document delete used to leave the chunk in every collection but the active one
(eg2 re-embeds, the HyPE questions, the graph twin), in the HyPE sidecar and in the
parents sidecar, and it rewrote the JSONLs with no backup. Real Chroma stores and
JSONLs under tmp_path; manage_api is pointed at them with monkeypatched globals.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from eval.bench.questions import question_from_dict, save_suite
from src.embeddings.registry import EmbeddingSpec
from src.embeddings.sidecar import make_sidecar, write_sidecar
from src.utils.chroma_client import persistent_client
from src.utils.config_loader import Config

pytest.importorskip("chromadb")

ACTIVE, EG2, HYPE = "obsidian_vault", "obsidian_vault_eg2_l1024", "hype_questions"


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


def _lines(*records):
    return "".join(json.dumps(r) + "\n" for r in records).encode("utf-8")


def _chunk(doc_id, source):
    return {"doc_id": doc_id, "text": "text " + doc_id, "metadata": {"source_file": source}}


@pytest.fixture
def world(management_module, tmp_path, monkeypatch):
    """Three collections and four JSONLs, each holding rows for a.md and b.md, with
    manage_api pointed at them."""
    m = management_module
    data = tmp_path / "data"
    data.mkdir()
    cfg = Config({"paths": {"chunks_file": str(data / "chunks.jsonl"),
                            "chroma_dir": str(tmp_path / "chroma"), "collection_name": ACTIVE},
                  "retrieval": {"parents_file": str(data / "parents_md.jsonl")}}, tmp_path)
    files = {
        "chunks.jsonl": _lines(_chunk("a1", "a.md"), _chunk("a2", "a.md"), _chunk("b1", "b.md")),
        "pdf_chunks.jsonl": _lines(_chunk("p1", "b.md")),
        "hype_questions.jsonl": _lines({"doc_id": "a1", "source_file": "a.md", "questions": ["q?"]},
                                       {"doc_id": "b1", "source_file": "b.md", "questions": ["q?"]}),
        "parents_md.jsonl": _lines({"parent_id": "pa", "source_file": "a.md", "text": "A"},
                                   {"parent_id": "pb", "source_file": "b.md", "text": "B"},
                                   {"parent_id": "pb2", "source_file": "b.md", "text": "B2"}),
    }
    for name, raw in files.items():
        (data / name).write_bytes(raw)

    client = persistent_client(cfg.path("paths.chroma_dir"))

    def fill(name, dim, rows):
        col = client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
        col.add(ids=[i for i, _ in rows], embeddings=[[1.0] + [0.0] * (dim - 1)] * len(rows),
                documents=[i for i, _ in rows], metadatas=[{"source_file": s} for _, s in rows])
        return col

    main = fill(ACTIVE, 3, [("a1", "a.md"), ("a2", "a.md"), ("b1", "b.md")])
    eg2 = fill(EG2, 4, [("a1", "a.md"), ("a2", "a.md"), ("b1", "b.md")])
    hype = fill(HYPE, 3, [("a1::q0", "a.md"), ("a1::q1", "a.md"), ("b1::q0", "b.md")])

    monkeypatch.setattr(m, "CFG", cfg)
    monkeypatch.setattr(m, "COLLECTION", ACTIVE)
    monkeypatch.setattr(m, "DATA_DIR", data)
    monkeypatch.setattr(m, "MANIFEST_CACHE", tmp_path / "manifest.json")
    monkeypatch.setattr(m, "EVAL_SETS_DIR", tmp_path / "sets")    # never the real eval/sets
    return SimpleNamespace(m=m, cfg=cfg, client=client, data=data, files=files,
                           main=main, eg2=eg2, hype=hype, chroma=cfg.path("paths.chroma_dir"))


def _ids(col, source):
    return sorted(col.get(where={"source_file": source}, include=[])["ids"])


def _bak(data, name, tag="delete"):
    (found,) = data.glob(f"{name}.{tag}-*.bak")
    return found


def test_delete_reaches_every_collection(world):
    deleted = world.m._chroma_delete_by_source({"a.md"})

    assert deleted == 2                                            # the ACTIVE count
    for col, kept in ((world.main, ["b1"]), (world.eg2, ["b1"]), (world.hype, ["b1::q0"])):
        assert _ids(col, "a.md") == [], col.name                   # gone from ALL three
        assert _ids(col, "b.md") == kept, col.name                 # the other file is intact


def test_delete_drops_hype_sidecar_rows(world):
    removed = world.m._jsonl_remove_sources({"a.md"}, {"chunks.jsonl"})

    assert removed["chunks.jsonl"] == 2 and removed["hype_questions.jsonl"] == 1
    assert world.data.joinpath("hype_questions.jsonl").read_bytes() == _lines(
        {"doc_id": "b1", "source_file": "b.md", "questions": ["q?"]})
    # every rewrite keeps its original beside it
    assert _bak(world.data, "hype_questions.jsonl").read_bytes() == world.files["hype_questions.jsonl"]
    assert _bak(world.data, "chunks.jsonl").read_bytes() == world.files["chunks.jsonl"]
    assert world.data.joinpath("pdf_chunks.jsonl").read_bytes() == world.files["pdf_chunks.jsonl"]
    assert not list(world.data.glob("pdf_chunks.jsonl.*"))         # a file with no match is not touched


def test_delete_drops_parents_rows(world):
    removed = world.m._jsonl_remove_sources({"a.md"}, {"chunks.jsonl"})

    assert removed["parents_md.jsonl"] == 1
    assert world.data.joinpath("parents_md.jsonl").read_bytes() == _lines(
        {"parent_id": "pb", "source_file": "b.md", "text": "B"},
        {"parent_id": "pb2", "source_file": "b.md", "text": "B2"})
    assert _bak(world.data, "parents_md.jsonl").read_bytes() == world.files["parents_md.jsonl"]


def test_delete_refuses_while_a_collection_is_building(world):
    write_sidecar(world.chroma, EG2, make_sidecar(
        EmbeddingSpec("local", "local", "fake/m"), collection=EG2, role="chunks",
        status="building", dimension=4, count=1, written_by="rebuild-into"))
    client = TestClient(world.m.app, base_url="http://127.0.0.1")

    r = client.post("/api/documents/delete", json={"source_files": ["a.md"], "rebuild": False})

    assert r.status_code == 409
    assert "building" in r.json()["error"] and EG2 in r.json()["error"]
    assert (world.main.count(), world.eg2.count(), world.hype.count()) == (3, 3, 3)
    for name, raw in world.files.items():
        assert world.data.joinpath(name).read_bytes() == raw       # not a byte moved
    assert not list(world.data.glob("*.bak"))


def test_delete_endpoint_end_to_end(world):
    client = TestClient(world.m.app, base_url="http://127.0.0.1")

    r = client.post("/api/documents/delete", json={"source_files": ["a.md"], "rebuild": False})

    body = r.json()
    assert r.status_code == 200 and body["ok"] is True
    assert body["chroma_deleted"] == 2
    assert body["jsonl_removed"] == {"chunks.jsonl": 2, "hype_questions.jsonl": 1,
                                     "parents_md.jsonl": 1}
    assert _ids(world.eg2, "a.md") == [] and _ids(world.hype, "a.md") == []
    assert world.main.count() == 1


def _question(qid, gold_file, status="draft"):
    return question_from_dict({
        "id": qid, "question": f"Question text for {qid}?", "suite": "lexical", "tier": "T1",
        "split": "dev", "answerable": True, "gold": [{"file": gold_file}],
        "nuggets": ["n"], "expect_course": None,
        "provenance": {"author": "draft", "drafted_by": "test", "seed_chunks": ["x"],
                       "status": status, "reviewed_at": None}, "notes": ""}, "test")


def test_delete_names_the_eval_questions_whose_gold_cites_a_deleted_source(world):
    """A warning, not a refusal: the rows go, and the response lists the gold to fix.
    'A.MD' vs 'a.md' proves the match goes through norm_file, not a raw substring;
    a rejected question citing it is not listed (its gold is not going to be fixed)."""
    world.m.EVAL_SETS_DIR.mkdir()
    save_suite(world.m.EVAL_SETS_DIR / "lexical.yaml",
               [_question("lex-0001", "A.MD"), _question("lex-0002", "b.md"),
                _question("lex-0003", "A.MD", status="rejected")])
    client = TestClient(world.m.app, base_url="http://127.0.0.1")

    r = client.post("/api/documents/delete", json={"source_files": ["a.md"], "rebuild": False})

    assert r.status_code == 200 and r.json()["ok"] is True
    assert r.json()["eval_gold"] == ["lex-0001"]
    assert world.main.count() == 1


def test_a_broken_sets_file_fails_the_delete_before_anything_is_deleted(world):
    world.m.EVAL_SETS_DIR.mkdir()
    world.m.EVAL_SETS_DIR.joinpath("lexical.yaml").write_text("{}\n", encoding="utf-8")
    client = TestClient(world.m.app, base_url="http://127.0.0.1")

    r = client.post("/api/documents/delete", json={"source_files": ["a.md"], "rebuild": False})

    assert r.status_code == 500 and "lexical.yaml" in r.json()["error"]
    assert (world.main.count(), world.eg2.count(), world.hype.count()) == (3, 3, 3)
    for name, raw in world.files.items():
        assert world.data.joinpath(name).read_bytes() == raw


def test_delete_spares_the_chunk_files_named_in_keep_jsonls(world):
    """The append refusal's REPLACE recipe (review B2-1): the file waiting to be appended is
    a *_chunks.jsonl like any other, so the delete must be told to leave it alone."""
    client = TestClient(world.m.app, base_url="http://127.0.0.1")
    world.data.joinpath("inbox_new_chunks.jsonl").write_bytes(_lines(_chunk("a9", "a.md")))

    r = client.post("/api/documents/delete", json={"source_files": ["a.md"], "rebuild": False,
                                                   "keep_jsonls": ["INBOX_NEW_CHUNKS.JSONL"]})

    assert r.status_code == 200 and r.json()["jsonl_removed"]["chunks.jsonl"] == 2
    assert "inbox_new_chunks.jsonl" not in r.json()["jsonl_removed"]
    assert world.data.joinpath("inbox_new_chunks.jsonl").read_bytes() == _lines(_chunk("a9", "a.md"))
    assert not list(world.data.glob("inbox_new_chunks.jsonl.*"))


def test_keep_jsonls_naming_no_chunk_file_is_refused_before_anything_is_deleted(world):
    """A typo would silently leave the incoming file unprotected."""
    client = TestClient(world.m.app, base_url="http://127.0.0.1")

    r = client.post("/api/documents/delete", json={"source_files": ["a.md"], "rebuild": False,
                                                   "keep_jsonls": ["inbox_typo_chunks.jsonl"]})

    assert r.status_code == 400 and "inbox_typo_chunks.jsonl" in r.json()["error"]
    assert (world.main.count(), world.eg2.count(), world.hype.count()) == (3, 3, 3)
    for name, raw in world.files.items():
        assert world.data.joinpath(name).read_bytes() == raw


def test_a_failure_in_another_collection_leaves_the_active_one_untouched(world, monkeypatch):
    """The active collection is deleted LAST (review T9): a failure before it leaves the
    served collection whole, and a retry heals."""
    from chromadb.api.models.Collection import Collection
    real = Collection.delete

    def delete(self, *a, **kw):
        if self.name == EG2:
            raise RuntimeError("injected eg2 failure")
        return real(self, *a, **kw)
    monkeypatch.setattr(Collection, "delete", delete)

    with pytest.raises(RuntimeError, match="injected eg2 failure"):
        world.m._chroma_delete_by_source({"a.md"})

    assert world.main.count() == 3 and _ids(world.main, "a.md") == ["a1", "a2"]
    monkeypatch.setattr(Collection, "delete", real)
    assert world.m._chroma_delete_by_source({"a.md"}) == 2           # the retry finishes it
    assert _ids(world.eg2, "a.md") == [] and _ids(world.main, "a.md") == []


def test_a_delete_of_more_than_a_hundred_sources_reaches_them_all(world):
    """Sources are looked up 100 at a time (review T9): B-ops deletes 160+ in one request."""
    names = [f"bulk/{n:03d}.md" for n in range(150)]
    for col, dim in ((world.main, 3), (world.eg2, 4), (world.hype, 3)):
        col.add(ids=[f"{col.name}-{n}" for n in names], embeddings=[[1.0] + [0.0] * (dim - 1)] * 150,
                documents=names, metadatas=[{"source_file": n} for n in names])

    assert world.m._chroma_delete_by_source(set(names)) == 150

    for col in (world.main, world.eg2, world.hype):
        assert col.get(where={"source_file": {"$in": names}}, include=[])["ids"] == [], col.name
        assert col.count() == 3                                      # a.md and b.md untouched


def _delete(world):
    client = TestClient(world.m.app, base_url="http://127.0.0.1")
    return client.post("/api/documents/delete", json={"source_files": ["a.md"], "rebuild": False})


def _assert_chroma_gone_jsonl_kept(world, r):
    """The delete is NOT atomic across stores: Chroma went first, so a chunk file that
    cannot be rewritten leaves the rows in the JSONL (and the next BM25 rebuild would
    resurrect them in the sparse index only). The answer must say that, not "nothing
    changed", and a repeat of the same request must finish the job."""
    body = r.json()
    assert r.status_code == 409 and body["ok"] is False
    assert body["chroma_deleted"] == 2
    assert "ALREADY deleted" in body["error"] and "repeat" in body["error"]
    assert "nothing changed" not in body["error"] and "nothing was changed" not in body["error"]
    assert _ids(world.main, "a.md") == [] and _ids(world.eg2, "a.md") == []
    assert world.data.joinpath("chunks.jsonl").read_bytes() == world.files["chunks.jsonl"]


def test_delete_says_so_when_a_chunk_file_cannot_be_rewritten(world, monkeypatch):
    real = world.m.rewrite_jsonl

    def locked(path, fix, **kw):
        if Path(path).name == "chunks.jsonl":
            raise PermissionError(13, "Access is denied")          # os.replace onto an open file
        return real(path, fix, **kw)
    monkeypatch.setattr(world.m, "rewrite_jsonl", locked)

    _assert_chroma_gone_jsonl_kept(world, _delete(world))

    monkeypatch.setattr(world.m, "rewrite_jsonl", real)            # the lock is gone: a repeat heals
    r = _delete(world)
    assert r.status_code == 200 and r.json()["chroma_deleted"] == 0
    assert r.json()["jsonl_removed"]["chunks.jsonl"] == 2


@pytest.mark.skipif(sys.platform != "win32",
                    reason="only Windows refuses os.replace onto a file another handle holds open")
def test_delete_with_a_chunk_file_really_held_open(world):
    """The real lock: a running index --append streams chunks.jsonl while the user deletes."""
    with open(world.data / "chunks.jsonl", "rb"):
        r = _delete(world)

    _assert_chroma_gone_jsonl_kept(world, r)


# ---- retag: the same reach as delete --------------------------------------------------

def _retag(world):
    client = TestClient(world.m.app, base_url="http://127.0.0.1")
    return client.post("/api/documents/retag", json={"source_files": ["a.md"], "domain": "db",
                                                     "add_tags": ["x"], "rebuild": False})


def test_retag_reaches_every_chunk_collection(world):
    """A console retag used to update the active collection only: the eg2 re-embeds kept
    the old metadata until the next --rebuild-into synced it."""
    r = _retag(world)

    body = r.json()
    assert r.status_code == 200 and body["ok"] is True
    assert body["rows_changed"] == 2 and body["chroma_updated"] == 2       # the ACTIVE count
    assert body["chroma_updated_by_collection"] == {EG2: 2, ACTIVE: 2}     # HyPE is not a chunk store
    assert list(body["chroma_updated_by_collection"])[-1] == ACTIVE        # the active one goes last
    for col in (world.main, world.eg2):
        got = col.get(ids=["a1", "a2", "b1"], include=["metadatas"])
        meta = dict(zip(got["ids"], got["metadatas"]))
        assert meta["a1"]["domain"] == meta["a2"]["domain"] == "db", col.name
        assert meta["a1"]["tags"] == "x", col.name
        assert "domain" not in meta["b1"], col.name                        # the other file is intact
    assert all(m == {"source_file": "a.md"} for m in
               world.hype.get(where={"source_file": "a.md"}, include=["metadatas"])["metadatas"])


def test_retag_keeps_a_bak_of_every_rewritten_jsonl(world):
    _retag(world)

    assert _bak(world.data, "chunks.jsonl", tag="retag").read_bytes() == world.files["chunks.jsonl"]
    rows = [json.loads(l) for l in world.data.joinpath("chunks.jsonl").read_text().splitlines()]
    assert [r["metadata"].get("domain") for r in rows] == ["db", "db", None]
    assert not list(world.data.glob("pdf_chunks.jsonl.*"))         # b.md only: not touched


def test_retag_refuses_while_a_collection_is_building(world):
    write_sidecar(world.chroma, EG2, make_sidecar(
        EmbeddingSpec("local", "local", "fake/m"), collection=EG2, role="chunks",
        status="building", dimension=4, count=1, written_by="rebuild-into"))

    r = _retag(world)

    assert r.status_code == 409 and EG2 in r.json()["error"]
    for name, raw in world.files.items():
        assert world.data.joinpath(name).read_bytes() == raw       # not a byte moved
    assert not list(world.data.glob("*.bak"))
    assert "domain" not in world.main.get(ids=["a1"], include=["metadatas"])["metadatas"][0]
