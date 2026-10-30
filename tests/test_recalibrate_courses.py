"""recalibrate_courses.py changes only labels a machine made (slice B, task 5).

Run:  python -m pytest tests/test_recalibrate_courses.py -q

The old script re-derived every label from the path and overwrote whatever was stored,
hand retags included. It now compares the stored label with what the PRE-slice rules
produced (a frozen copy in the script) and changes a row only when they agree and the
current rule says something else. Real Chroma stores and JSONLs under tmp_path; nothing
here touches data/ or a live store.
"""
import itertools
import json
import socket
import sys
from types import SimpleNamespace

import pytest
from chromadb.api.models.Collection import Collection

import recalibrate_courses
from src import index_ops
from src.embeddings.registry import EmbeddingSpec
from src.embeddings.sidecar import make_sidecar, write_sidecar
from src.ingestion.obsidian_parser import resolve_course_metadata
from src.utils.chroma_client import persistent_client

ACTIVE, EG2, HYPE = "obsidian_vault", "obsidian_vault_eg2_l1024", "hype_questions"
AI_DIR = "00 – AUA_DS\\Current Courses\\AI Engineering\\"
NLP_DIR = "00 – AUA_DS\\Current Courses\\NLP\\"
BOOKS_DIR = "00 – AUA_DS\\Other\\Tech Books\\Data Science (Books)\\"


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


def _label(name):
    m = resolve_course_metadata(name)
    return (m["course_code"], m["course_name"], m["domain"])


INTRO, NLP, RL = _label("Intro to AI"), _label("Natural Language Processing"), _label("Reinforcement Learning")
AIENG, DSB = _label("AI Engineering"), _label("Data Science (Books)")
PROG, MLAI = _label("Programming & Engineering"), _label("Machine Learning & AI")
ORDER_DIR = "Python for Data Science\\Machine Learning\\"     # a heading-pattern parent, an exact-map child
UNK, SKIP = ("unknown", "unknown", "general"), ("__SKIP__", "__SKIP__", "general")
DB = ("Databases & Data Engineering", "Databases & Data Engineering", "db")
NONE3 = (None, None, None)


def _row(doc_id, source, file_type, label, **extra):
    meta = {"source_file": source, "file_type": file_type, **extra}
    if label != NONE3:
        meta.update(course_code=label[0], course_name=label[1], domain=label[2])
    return {"doc_id": doc_id, "text": "text " + doc_id, "metadata": meta}


FILES = {
    "ai_engineering_md_chunks.jsonl": [
        _row("ai_md", AI_DIR + "week1.md", "note", INTRO),
        _row("fm_md", AI_DIR + "x.md", "note", NLP, frontmatter_course="NLP"),
    ],
    "chunks.jsonl": [
        _row("skip_md", "Misc\\todo.md", "note", SKIP),
        _row("daily_md", "Daily\\2026-01-02.md", "daily_note", INTRO),
        _row("nlp_md", NLP_DIR + "x.md", "note", NLP),
        _row("ds_md", BOOKS_DIR + "n.md", "note", NONE3),
        _row("head_md", "Misc\\talk.md", "note", RL),
        _row("bare_md", "Misc\\notes.md", "note", NONE3),
        # review T1: pins the ORDER inside the frozen comparator (heading patterns first, at
        # 27ef688): the parent's ^python pattern made Programming & Engineering
        _row("order_md", ORDER_DIR + "x.md", "note", PROG),
        _row("order_keep", ORDER_DIR + "y.md", "note", MLAI),
    ],
    "tech_chunks.jsonl": [_row("ds_pdf", BOOKS_DIR + "x.pdf", "pdf", UNK)],
    "pdf_chunks.jsonl": [_row("hotel_pdf", "Other\\Hotel\\db.pdf", "pdf", DB)],
}
ALL_IDS = [r["doc_id"] for rows in FILES.values() for r in rows]
CHANGED = {"ai_md": AIENG, "skip_md": UNK, "ds_md": DSB, "ds_pdf": DSB}   # the scoped runs' rows
ORDER = {"order_md": MLAI}                                              # changed by --all-files only


def _triple(meta):
    return (meta.get("course_code"), meta.get("course_name"), meta.get("domain"))


def _line(row):
    return json.dumps(row).encode("utf-8") + b"\r\n"        # the live JSONLs are CRLF


@pytest.fixture
def world(tmp_path):
    """The fixture rows in four CRLF JSONLs, and every id in three collections."""
    data = tmp_path / "data"
    data.mkdir()
    files = {}
    for name, rows in FILES.items():
        files[name] = b"".join(_line(r) for r in rows)
        (data / name).write_bytes(files[name])
    _write_config(tmp_path)

    client = persistent_client(tmp_path / "chroma")
    cols = {}
    for name, dim in ((ACTIVE, 3), (EG2, 4), (HYPE, 3)):
        col = cols[name] = client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
        rows = [r for rs in FILES.values() for r in rs]
        metas = []
        for r in rows:
            m = r["metadata"]
            if name == HYPE:                                     # HyPE metadata has no course
                metas.append({"doc_id": r["doc_id"], "source_file": m["source_file"]})
            else:                                                # tags: a key only Chroma holds
                metas.append({**{k: v for k, v in m.items() if k in ("source_file", "course_code",
                                                                     "course_name", "domain")},
                              "tags": "x"})
        col.add(ids=[r["doc_id"] for r in rows], embeddings=[[1.0] + [0.0] * (dim - 1)] * len(rows),
                documents=[r["doc_id"] for r in rows], metadatas=metas)
    return SimpleNamespace(root=tmp_path, data=data, files=files, cols=cols, report=tmp_path / "r.jsonl")


def _write_config(root, webui_port=8052):
    (root / "config.yaml").write_text(
        "paths:\n  chunks_file: data/chunks.jsonl\n  chroma_dir: chroma\n"
        f"  collection_name: {ACTIVE}\nwebui:\n  port: {webui_port}\n", encoding="utf-8")


@pytest.fixture
def servers_down(monkeypatch):
    """The dev machine may have :8051 up; a write that is meant to proceed says none answers."""
    monkeypatch.setattr(index_ops, "answering_ports", lambda cfg: [])


@pytest.fixture
def fresh_stamps(monkeypatch):
    """rewrite_jsonl raises when one file is rewritten twice inside a second (the .bak
    name is taken). Give each rewrite its own stamp so back-to-back runs can do it."""
    stamps = (f"20260101-0000{n:02d}" for n in itertools.count())
    monkeypatch.setattr(index_ops, "time", SimpleNamespace(strftime=lambda fmt: next(stamps)))


def _run(monkeypatch, world, *argv):
    monkeypatch.setattr(sys, "argv", ["recalibrate_courses.py", "--config", str(world.root / "config.yaml"),
                                      *argv])
    recalibrate_courses.main()


def _report(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _snapshot(data):
    return {p.name: p.read_bytes() for p in data.iterdir()}


def _chroma(col):
    got = col.get(include=["metadatas"])
    return dict(zip(got["ids"], got["metadatas"]))


def _jsonl_triples(data):
    out = {}
    for name in FILES:
        for line in (data / name).read_bytes().splitlines():
            rec = json.loads(line)
            out[rec["doc_id"]] = _triple(rec["metadata"])
    return out


def _boom(*a, **k):
    raise AssertionError("must not be reached")


def test_dry_run_reports_exactly_the_rows_whose_label_changes(world, monkeypatch):
    monkeypatch.setattr(recalibrate_courses, "persistent_client", _boom)   # never opens Chroma
    monkeypatch.setattr(index_ops, "answering_ports", _boom)               # and never probes ports
    before = _snapshot(world.data)

    _run(monkeypatch, world, "--all-files", "--dry-run", "--report", str(world.report))

    rows = _report(world.report)
    change = {r["doc_id"]: r for r in rows if r["kind"] == "change"}
    assert set(change) == {"ai_md", "skip_md", "ds_md", "ds_pdf", "order_md"}
    assert (change["ai_md"]["old"], change["ai_md"]["new"]) == (list(INTRO), list(AIENG))
    assert (change["skip_md"]["old"], change["skip_md"]["new"]) == (list(SKIP), list(UNK))
    assert (change["ds_md"]["old"], change["ds_md"]["new"]) == ([None, None, None], list(DSB))
    assert (change["ds_pdf"]["old"], change["ds_pdf"]["new"]) == (list(UNK), list(DSB))
    assert (change["order_md"]["old"], change["order_md"]["new"]) == (list(PROG), list(MLAI))
    retag = [r for r in rows if r["kind"] == "retag"]
    assert [r["doc_id"] for r in retag] == ["hotel_pdf"]
    assert retag[0]["old"] == list(DB) and retag[0]["new"] is None
    # fm_md, daily_md, nlp_md, head_md, bare_md and order_keep (already the new label) are not reported
    assert len(rows) == 6
    assert change["ai_md"] == {
        "kind": "change", "file": "ai_engineering_md_chunks.jsonl", "doc_id": "ai_md",
        "source_file": AI_DIR + "week1.md", "file_type": "note",
        "key": "00 – AUA_DS/Current Courses/AI Engineering",
        "old": list(INTRO), "new": list(AIENG)}
    assert _snapshot(world.data) == before                  # a dry run writes nothing


def test_scoped_apply_changes_only_reviewed_rows_in_every_chunk_collection(
        world, monkeypatch, servers_down, fresh_stamps):
    before = {name: _chroma(col) for name, col in world.cols.items()}
    r1, r2 = world.root / "r1.jsonl", world.root / "r2.jsonl"
    scoped = ("--all-files", "--include-path", "current courses/ai engineering",
              "--include-path", "data science (books)")

    _run(monkeypatch, world, *scoped, "--report", str(r1))
    _run(monkeypatch, world, "--all-files", "--where-course", "__SKIP__", "--report", str(r2))

    assert {r["doc_id"] for r in _report(r1)} == {"ai_md", "ds_md", "ds_pdf"}   # fm_md is held back
    assert {r["doc_id"] for r in _report(r2)} == {"skip_md"}
    triples = _jsonl_triples(world.data)
    for doc_id, new in CHANGED.items():
        assert triples[doc_id] == new, doc_id
    assert triples["fm_md"] == NLP and triples["hotel_pdf"] == DB and triples["nlp_md"] == NLP
    # a rewritten file keeps its untouched lines byte-for-byte and its CRLF endings
    md = (world.data / "ai_engineering_md_chunks.jsonl").read_bytes()
    assert _line(FILES["ai_engineering_md_chunks.jsonl"][1]) in md
    assert md.count(b"\r\n") == md.count(b"\n") == 2
    assert (world.data / "pdf_chunks.jsonl").read_bytes() == world.files["pdf_chunks.jsonl"]
    baks = sorted(p.name.split(".recal-")[0] for p in world.data.glob("*.recal-*.bak"))
    assert baks == ["ai_engineering_md_chunks.jsonl", "chunks.jsonl", "chunks.jsonl", "tech_chunks.jsonl"]
    (bak,) = world.data.glob("ai_engineering_md_chunks.jsonl.recal-*.bak")
    assert bak.read_bytes() == world.files["ai_engineering_md_chunks.jsonl"]
    # every chunk collection carries the new triple AND keeps the key only Chroma held
    for name in (ACTIVE, EG2):
        now = _chroma(world.cols[name])
        for doc_id in ALL_IDS:
            if doc_id in CHANGED:
                new = CHANGED[doc_id]
                assert {**before[name][doc_id], "course_code": new[0], "course_name": new[1],
                        "domain": new[2]} == now[doc_id], (name, doc_id)
            else:
                assert now[doc_id] == before[name][doc_id], (name, doc_id)
        assert all(m["tags"] == "x" for m in now.values())
    assert _chroma(world.cols[HYPE]) == before[HYPE]       # HyPE metadata has no course

    # the same commands again have nothing left to do
    after = _snapshot(world.data)
    _run(monkeypatch, world, *scoped, "--report", str(r1))
    _run(monkeypatch, world, "--all-files", "--where-course", "__SKIP__", "--report", str(r2))
    assert _report(r1) == [] and _report(r2) == []
    assert _snapshot(world.data) == after


def test_unfiltered_apply_leaves_a_hand_retag_byte_identical(world, monkeypatch, servers_down, fresh_stamps):
    """(R) The DEFAULT file set holds pdf_chunks.jsonl, where the hand retag sits. At
    27ef688 this run re-derived every row's label from its path (the ladder gives
    "unknown" for Other/Hotel/db.pdf) and overwrote the retag without a backup."""
    before = _snapshot(world.data)
    chroma = {name: _chroma(col) for name, col in world.cols.items()}

    _run(monkeypatch, world)                              # no flags: the default set, applied

    assert _snapshot(world.data) == before                # the hand-retagged line, byte for byte
    assert not list(world.data.glob("*.bak"))
    assert {name: _chroma(col) for name, col in world.cols.items()} == chroma


def test_all_files_apply_relabels_machine_made_rows_and_spares_the_hand_retag(
        world, monkeypatch, servers_down, fresh_stamps):
    before = {name: _chroma(world.cols[name])["hotel_pdf"] for name in (ACTIVE, EG2)}

    _run(monkeypatch, world, "--all-files", "--report", str(world.report))

    assert (world.data / "pdf_chunks.jsonl").read_bytes() == world.files["pdf_chunks.jsonl"]
    assert not list(world.data.glob("pdf_chunks.jsonl.*"))
    for name in (ACTIVE, EG2):
        assert _chroma(world.cols[name])["hotel_pdf"] == before[name]
    assert [r["doc_id"] for r in _report(world.report) if r["kind"] == "retag"] == ["hotel_pdf"]
    assert _jsonl_triples(world.data)["ds_pdf"] == DSB      # tech_chunks.jsonl needs --all-files


def test_apply_writes_chroma_before_the_jsonl_and_a_rerun_heals(world, monkeypatch, servers_down, fresh_stamps):
    """EVERY chunk collection is written before the JSONL: the failure is injected in the
    LAST one (review T2). Were the JSONL written between collections, a re-run would read
    every row as already relabelled (stored == new) and never fix that collection."""
    monkeypatch.setattr(recalibrate_courses, "BATCH", 1)
    update = Collection.update

    def flaky(self, *args, **kwargs):
        if self.name == EG2:
            raise RuntimeError("update failed")
        return update(self, *args, **kwargs)

    with monkeypatch.context() as mp:
        mp.setattr(Collection, "update", flaky)
        with pytest.raises(RuntimeError, match="update failed"):
            _run(monkeypatch, world, "--all-files")
    assert (world.data / "chunks.jsonl").read_bytes() == world.files["chunks.jsonl"]   # Chroma failed first
    assert not list(world.data.glob("chunks.jsonl.*"))

    _run(monkeypatch, world, "--all-files")                 # exits 0: the re-run recomputes and heals

    triples = _jsonl_triples(world.data)
    for name in (ACTIVE, EG2):
        now = _chroma(world.cols[name])
        for doc_id in ALL_IDS:
            assert _triple(now[doc_id]) == triples[doc_id], (name, doc_id)
    assert all(triples[d] == new for d, new in {**CHANGED, **ORDER}.items())


def _add_file(world, name, rows):
    """One more CRLF chunk file, its ids in every collection (as the fixture does)."""
    (world.data / name).write_bytes(b"".join(_line(r) for r in rows))
    for cname, col in world.cols.items():
        dim = 4 if cname == EG2 else 3
        col.add(ids=[r["doc_id"] for r in rows], embeddings=[[1.0] + [0.0] * (dim - 1)] * len(rows),
                documents=[r["doc_id"] for r in rows],
                metadatas=[{"doc_id": r["doc_id"], "source_file": r["metadata"]["source_file"]}
                           if cname == HYPE else
                           {k: v for k, v in r["metadata"].items()
                            if k in ("source_file", "course_code", "course_name", "domain")}
                           for r in rows])


def test_a_relocated_row_is_judged_by_the_path_its_label_was_made_for(
        world, monkeypatch, servers_down, fresh_stamps):
    """REGRESSION (review B2-RECAL-AFTER-RELOCATE): relocate rewrites source_file and keeps
    the label, so the comparator judged a moved row's machine-made label against the rules
    for its NEW path, and runbook 6.3 step 2.4 (scoped recal after a relocate) applied
    nothing. --rename-map names the path each label was made for."""
    from src.ingestion.obsidian_parser import detect_course_from_path, detect_md_course_from_path
    moves = [("00 – AUA_DS\\Current Courses\\NLP\\x.pdf", "00 – AUA_DS\\Current Courses\\RL\\x.pdf", "pdf"),
             ("talk.md", "00 – AUA_DS\\Current Courses\\NLP\\talk.md", "note")]
    pdf_old = _triple(detect_course_from_path(moves[0][0].split("\\")))   # what the pdf loader stored
    rows = [_row("mv_pdf", moves[0][0], "pdf", pdf_old), _row("mv_md", moves[1][0], "note", NONE3)]
    rows = [dict(r, metadata=index_ops.relocate_meta(r["metadata"], old, new))
            for r, (old, new, _) in zip(rows, moves)]                      # what relocate leaves
    _add_file(world, "moved_chunks.jsonl", rows)
    rename_map = world.root / "rename_map.jsonl"
    rename_map.write_text("".join(json.dumps(
        {"op": "move", "kind": "file", "indexed": True, "lane": "pdf" if t == "pdf" else "md",
         "old": old.replace("\\", "/"), "new": new.replace("\\", "/"), "old_sf": old, "new_sf": new,
         "n_chunks": 1, "approved": True}) + "\n" for old, new, t in moves), encoding="utf-8")
    want = {"mv_pdf": _triple(detect_course_from_path(moves[0][1].split("\\"))),
            "mv_md": _triple(resolve_course_metadata(detect_md_course_from_path(moves[1][1].split("\\"))))}
    assert want["mv_pdf"] == RL and want["mv_md"] == NLP and pdf_old == NLP
    scope = ("--all-files", "--include-path", "current courses/rl/", "--include-path", "nlp/talk.md")

    _run(monkeypatch, world, *scope, "--dry-run", "--report", str(world.report))
    # judged by the new path, neither label is one a machine made there: both held back
    assert {r["doc_id"]: r["kind"] for r in _report(world.report)} == {"mv_pdf": "retag", "mv_md": "retag"}

    _run(monkeypatch, world, *scope, "--rename-map", str(rename_map), "--report", str(world.report))

    assert {r["doc_id"]: (r["kind"], tuple(r["new"])) for r in _report(world.report)} == {
        d: ("change", t) for d, t in want.items()}
    for line in (world.data / "moved_chunks.jsonl").read_bytes().splitlines():
        rec = json.loads(line)
        assert _triple(rec["metadata"]) == want[rec["doc_id"]]
    for name in (ACTIVE, EG2):
        now = _chroma(world.cols[name])
        assert {d: _triple(now[d]) for d in want} == want, name


def test_a_hand_retag_made_before_a_relocate_stays_a_retag(world, monkeypatch):
    """--rename-map moves the comparator's path only: a label no rule made at the old path
    is still not one a machine made."""
    old, new = "Other\\Hotel\\db.pdf", "00 – AUA_DS\\Current Courses\\RL\\db.pdf"
    row = _row("mv_hotel", old, "pdf", DB)
    _add_file(world, "moved_chunks.jsonl", [dict(row, metadata=index_ops.relocate_meta(row["metadata"], old, new))])
    rename_map = world.root / "rename_map.jsonl"
    rename_map.write_text(json.dumps({"op": "move", "kind": "file", "indexed": True, "lane": "pdf",
                                      "old": old.replace("\\", "/"), "new": new.replace("\\", "/"),
                                      "old_sf": old, "new_sf": new, "n_chunks": 1}) + "\n", encoding="utf-8")

    _run(monkeypatch, world, "--all-files", "--include-path", "rl/db.pdf", "--rename-map", str(rename_map),
         "--dry-run", "--report", str(world.report))

    assert [(r["doc_id"], r["kind"]) for r in _report(world.report)] == [("mv_hotel", "retag")]


def test_apply_refuses_while_a_port_answers(world, monkeypatch, capsys):
    monkeypatch.setattr(recalibrate_courses, "persistent_client", _boom)   # refused before Chroma opens
    before = _snapshot(world.data)
    with socket.socket() as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen()
        port = srv.getsockname()[1]
        _write_config(world.root, webui_port=port)

        with pytest.raises(SystemExit) as exc:
            _run(monkeypatch, world, "--all-files")

    assert exc.value.code == 2
    assert f":{port}" in capsys.readouterr().err
    assert _snapshot(world.data) == before
    assert not list(world.data.glob("*.bak"))


def test_apply_refuses_while_a_collection_is_building(world, monkeypatch, servers_down, capsys):
    """A running re-embed is a second writer that no port shows."""
    write_sidecar(world.root / "chroma", EG2, make_sidecar(
        EmbeddingSpec("local", "local", "fake/m"), collection=EG2, role="chunks",
        status="building", dimension=4, count=1, written_by="rebuild-into"))
    before = _snapshot(world.data)
    chroma = {name: _chroma(col) for name, col in world.cols.items()}

    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, world, "--all-files")

    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "building" in err and EG2 in err
    assert _snapshot(world.data) == before
    assert not list(world.data.glob("*.bak"))
    assert {name: _chroma(col) for name, col in world.cols.items()} == chroma


@pytest.mark.parametrize("name", ["x_chunks.jsonl", "X_CHUNKS.JSONL", "chunks.jsonl", "CHUNKS.JSONL"])
def test_report_may_not_be_a_chunk_file(world, monkeypatch, capsys, name):
    """The report is opened "w" before anything is read, dry run or not: named like a
    chunk file it truncates it. chunks.jsonl (the md corpus) is one, as much as the
    *_chunks.jsonl lanes are."""
    before = _snapshot(world.data)

    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, world, "--all-files", "--dry-run", "--report", str(world.data / name))

    assert exc.value.code == 2
    assert "must not be named *_chunks.jsonl" in capsys.readouterr().err
    assert _snapshot(world.data) == before      # no new file, and chunks.jsonl keeps its bytes


@pytest.mark.parametrize("name", ["parents_md.jsonl", "HYPE_QUESTIONS.JSONL"])
def test_report_may_not_be_a_sidecar(world, monkeypatch, capsys, name):
    """REGRESSION (review B2-6): the report is opened "w" even under --dry-run, and the
    parents and HyPE sidecars are stores a delete and a relocate rewrite."""
    for sidecar in ("parents_md.jsonl", "hype_questions.jsonl"):
        (world.data / sidecar).write_bytes(b'{"doc_id": "x", "source_file": "a.md"}\r\n')
    before = _snapshot(world.data)

    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, world, "--all-files", "--dry-run", "--report", str(world.data / name))

    assert exc.value.code == 2
    assert "sidecar" in capsys.readouterr().err
    assert _snapshot(world.data) == before


@pytest.mark.parametrize("alias", ["chunks.jsonl.", "chunks.jsonl ", "chunks.jsonl::$DATA",
                                   "x_chunks.jsonl ", "./chunks.jsonl"])
def test_report_may_not_be_a_windows_alias_of_a_chunk_file(world, monkeypatch, capsys, alias):
    """NTFS opens these spellings as the plain file; the by-name check cannot see it."""
    before = _snapshot(world.data)

    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, world, "--all-files", "--dry-run", "--report", f"{world.data}/{alias}")

    assert exc.value.code == 2
    assert "--report" in capsys.readouterr().err
    assert _snapshot(world.data) == before


def test_default_run_reads_only_its_four_files(world, monkeypatch):
    _run(monkeypatch, world, "--dry-run", "--report", str(world.report))

    rows = _report(world.report)
    assert {r["file"] for r in rows} == {"pdf_chunks.jsonl"}
    assert [(r["kind"], r["doc_id"]) for r in rows] == [("retag", "hotel_pdf")]
