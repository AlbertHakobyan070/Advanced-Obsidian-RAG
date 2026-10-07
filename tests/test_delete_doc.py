"""delete_doc.py is PREVIEW-ONLY, and it reads the store config.yaml names.

It used to delete: --confirm removed the chunks from Chroma and --rebuild
re-derived the sparse index. The chunk JSONLs still held the rows, and the
rebuild unions every data/*_chunks.jsonl, so the "deleted" chunks came straight
back. The supported delete is the console's POST /api/documents/delete, which
removes the JSONL rows too; this script now previews and, asked to delete,
refuses and prints that call.

It also used to open data/chroma_db and the collection "obsidian_vault" relative
to the current directory. `data` can be a junction to another disk, which made
that path a stale snapshot rather than the live index, so the preview described
a store the query API was not serving.

Nothing here touches a real store. Each test writes its own config.yaml and
builds its Chroma stores under tmp_path, then runs from there — load_config()
looks in the current directory first.
"""
import json
import sys

import pytest

chromadb = pytest.importorskip("chromadb")

ROWS = [("d1", "Doomed_Notes.md"), ("d2", "Doomed_Notes.md"), ("k1", "Keeper.md")]


def _store(path, name):
    """A real on-disk collection holding ROWS. The vectors are explicit, so no
    embedding model is ever loaded."""
    col = chromadb.PersistentClient(path=str(path)).create_collection(
        name, metadata={"hnsw:space": "cosine"})
    col.add(ids=[i for i, _ in ROWS],
            embeddings=[[1.0, 0.0, 0.0] for _ in ROWS],
            documents=[f"text of {i}" for i, _ in ROWS],
            metadatas=[{"filename": f, "source_file": f} for _, f in ROWS])


def _count(path, name):
    return chromadb.PersistentClient(path=str(path)).get_collection(name).count()


@pytest.fixture
def stores(tmp_path, monkeypatch):
    """The configured store, plus a decoy at the OLD hardcoded location holding
    the same rows under the old hardcoded collection name."""
    root = tmp_path.resolve()      # same spelling load_config() resolves to
    (root / "config.yaml").write_text(
        'paths:\n'
        '  chroma_dir: "live_chroma"\n'
        '  collection_name: "configured_collection"\n', encoding="utf-8")
    live, decoy = root / "live_chroma", root / "data" / "chroma_db"
    _store(live, "configured_collection")
    _store(decoy, "obsidian_vault")
    monkeypatch.chdir(root)        # load_config() finds ./config.yaml here
    return live, decoy


def _run(monkeypatch, *argv):
    import delete_doc
    monkeypatch.setattr(sys, "argv", ["delete_doc.py", *argv])
    delete_doc.main()


@pytest.mark.parametrize("flags", [["--confirm"], ["--rebuild"], ["--confirm", "--rebuild"]])
def test_confirm_and_rebuild_are_refused_and_delete_nothing(
        monkeypatch, stores, capsys, flags):
    live, decoy = stores
    # --rebuild used to spawn rebuild_bm25.py: nothing may be started now.
    monkeypatch.setattr("subprocess.run", lambda *a, **k: pytest.fail("a rebuild was started"))
    with pytest.raises(SystemExit) as refused:
        _run(monkeypatch, "doomed", *flags)
    assert refused.value.code not in (0, None)                  # a script must see the refusal
    said = capsys.readouterr().out + str(refused.value.code)
    assert _count(live, "configured_collection") == len(ROWS)   # nothing was deleted...
    assert _count(decoy, "obsidian_vault") == len(ROWS)         # ...anywhere
    assert "POST /api/documents/delete" in said and "Documents tab" in said
    assert "Doomed_Notes.md" in said                            # the example names what matched
    assert "rebuild_bm25" not in said                           # the old "resync" advice is gone


def test_preview_names_the_store_it_reads_and_deletes_nothing(
        monkeypatch, stores, capsys):
    live, _ = stores
    _run(monkeypatch, "doomed")
    out = capsys.readouterr().out
    assert "configured_collection" in out and "live_chroma" in out
    assert "PREVIEW ONLY" in out
    assert _count(live, "configured_collection") == len(ROWS)


def test_preview_points_at_the_console_delete_not_at_the_old_flags(
        monkeypatch, stores, capsys):
    _run(monkeypatch, "doomed")
    out = capsys.readouterr().out
    assert "/api/documents/delete" in out
    assert "--confirm" not in out and "rebuild_bm25" not in out   # both used to be advised


def test_the_curl_example_is_a_console_call_that_survives_a_vault_path():
    import delete_doc
    path = "00 – AUA_DS\\Other\\Doomed.pdf"           # en dash + backslashes, as in a real vault
    cmd = delete_doc._curl_example([path], 8052)
    assert cmd.startswith("curl.exe -s -X POST http://127.0.0.1:8052/api/documents/delete ")
    assert cmd.isascii()                                  # no console code page can mangle it
    shell_body = cmd.split(' -d "', 1)[1][:-1]            # between the quotes, as typed
    body = json.loads(shell_body.replace('\\"', '"'))     # the shell turns \" back into "
    assert body == {"source_files": [path], "rebuild": True}
