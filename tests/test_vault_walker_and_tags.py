"""The shared vault walker, the JSONL reader's malformed-line policy, and the
comma-free tag invariant.

Three small contracts that each replaced a duplicated or accidental one:

  iter_vault_files   one walker for four loaders, pruning as it goes
  iter_jsonl_records one reader, with the malformed-line policy made explicit
  apply_forced_meta  tags can no longer contain the character that separates
                     them once they reach ChromaDB
"""
import json

import pytest

from src.embeddings.embedder import iter_jsonl_records
from src.ingestion.obsidian_parser import apply_forced_meta, iter_vault_files


# ------------------------------------------------------------------- walker

def _tree(root):
    (root / "Course" / "Week1").mkdir(parents=True)
    (root / "Course" / "Week1" / "notes.md").write_text("x", encoding="utf-8")
    (root / "Course" / "code.py").write_text("x", encoding="utf-8")
    (root / ".obsidian").mkdir()
    (root / ".obsidian" / "workspace.md").write_text("x", encoding="utf-8")
    (root / "node_modules" / "pkg").mkdir(parents=True)
    (root / "node_modules" / "pkg" / "index.md").write_text("x", encoding="utf-8")
    return root


def test_walker_prunes_instead_of_filtering(tmp_path):
    root = _tree(tmp_path / "vault")
    found = {p.name for p in iter_vault_files(root, {".md"}, {".obsidian", "node_modules"})}
    assert found == {"notes.md"}


def test_walker_matches_on_suffix_not_substring(tmp_path):
    root = tmp_path / "vault"
    root.mkdir()
    (root / "real.md").write_text("x", encoding="utf-8")
    (root / "not-a-markdown.mdx").write_text("x", encoding="utf-8")

    assert {p.name for p in iter_vault_files(root, {".md"})} == {"real.md"}


def test_walker_never_yields_a_directory_named_like_a_file(tmp_path):
    """This vault contains a real `PSS2_Solutions.sql/` FOLDER. rglob("*")
    returned it and open() then died with EACCES on Windows; os.walk puts it in
    dirnames, so it cannot reach a caller expecting a file."""
    root = tmp_path / "vault"
    (root / "PSS2_Solutions.sql").mkdir(parents=True)
    (root / "PSS2_Solutions.sql" / "actual.sql").write_text("x", encoding="utf-8")

    found = list(iter_vault_files(root, {".sql"}))
    assert [p.name for p in found] == ["actual.sql"]
    assert all(p.is_file() for p in found)


def test_walker_takes_the_loaders_extension_sets_unchanged(tmp_path):
    root = tmp_path / "vault"
    root.mkdir()
    for n in ("a.py", "b.R", "c.ipynb", "d.txt"):
        (root / n).write_text("x", encoding="utf-8")

    # Mixed case on disk and in the set: both sides are lowered.
    found = {p.name for p in iter_vault_files(root, {".py", ".r", ".ipynb"})}
    assert found == {"a.py", "b.R", "c.ipynb"}


# ----------------------------------------------------- malformed-line policy

def _damaged(tmp_path):
    p = tmp_path / "damaged_chunks.jsonl"
    p.write_text(
        '{"doc_id":"a","text":"ok","metadata":{}}\n'
        '{"doc_id":"b","text":"truncated"\n'
        '{"doc_id":"c","text":"fine","metadata":{}}\n',
        encoding="utf-8")
    return p


def test_append_refuses_to_index_a_damaged_file(tmp_path):
    """strict=True is the APPEND policy: half-indexing a damaged file puts rows
    in the dense index that the sparse rebuild will derive differently."""
    with pytest.raises(ValueError) as exc:
        list(iter_jsonl_records(_damaged(tmp_path), strict=True))
    assert "damaged_chunks.jsonl" in str(exc.value)


def test_sparse_rebuild_skips_but_warns(tmp_path, caplog):
    """strict=False is the REBUILD policy: one damaged legacy file must not make
    the whole index unrebuildable — but the skip must be visible, because a
    silent one is how the dense and sparse halves drift apart."""
    with caplog.at_level("WARNING"):
        got = list(iter_jsonl_records(_damaged(tmp_path), strict=False))
    assert [r["doc_id"] for r in got] == ["a", "c"]
    assert "skipped 1 malformed line" in caplog.text


def test_a_clean_file_reads_identically_under_both_policies(tmp_path):
    p = tmp_path / "clean_chunks.jsonl"
    rows = [{"doc_id": f"id{i}", "text": f"t{i}", "metadata": {"n": i}} for i in range(5)]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    assert list(iter_jsonl_records(p, strict=True)) == rows
    assert list(iter_jsonl_records(p, strict=False)) == rows


def test_records_survive_unicode_line_separators(tmp_path):
    """Chunk text legitimately contains U+2028/U+2029. The reader splits on
    b'\\n' ONLY — splitlines() would shred these records."""
    p = tmp_path / "unicode_chunks.jsonl"
    rows = [{"doc_id": "a", "text": "line still same chunk end", "metadata": {}}]
    p.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")

    assert list(iter_jsonl_records(p, strict=True)) == rows


# ---------------------------------------------------------------- tag safety

def test_a_tag_can_never_carry_the_separator_it_is_joined_with(tmp_path):
    """Tags reach ChromaDB as a ", "-joined string, so a comma inside one is
    indistinguishable from the separator between two on read-back — the canvas
    edge-label bug in miniature. The invariant is enforced where tags enter."""
    meta = {}
    apply_forced_meta(meta, None, ["time series, forecasting", "ml"])

    assert all("," not in t for t in meta["tags"])
    assert meta["tags"] == ["time series forecasting", "ml"]


def test_normalising_a_tag_collapses_the_whitespace_it_creates(tmp_path):
    meta = {}
    apply_forced_meta(meta, None, ["a,,b", "  spaced   out  "])
    assert meta["tags"] == ["a b", "spaced out"]


def test_existing_tags_are_normalised_too_and_not_duplicated(tmp_path):
    meta = {"tags": ["ml"]}
    apply_forced_meta(meta, None, ["ml", "nlp"])
    assert meta["tags"] == ["ml", "nlp"]


def test_forced_domain_still_applies_and_text_is_never_touched(tmp_path):
    meta = {"domain": "unknown"}
    apply_forced_meta(meta, "stats", ["hw"])
    assert meta["domain"] == "stats"
    assert meta["tags"] == ["hw"]
    # Metadata only: nothing here may influence a doc_id, which is derived
    # from source_file + text.
    assert "text" not in meta
