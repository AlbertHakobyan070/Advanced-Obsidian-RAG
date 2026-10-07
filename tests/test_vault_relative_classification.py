"""Classification must judge a note by its path INSIDE the vault.

THE BUG THIS PINS. `is_daily_note()` and `ObsidianParser._detect_course_from_path()`
read the ABSOLUTE path, so the folders ABOVE the vault root classified every
note below them: a vault stored under `journal/` turned every note into a daily
note, and one stored under `ML Notes/` labelled every note "Machine Learning &
AI". Discovery had the same flaw and was fixed the same way (see
test_vault_relative_discovery.py).

Each case pairs two halves, because both matter: the poisoned ANCESTOR is
ignored, and the same word INSIDE the vault still counts. Only the first would
let through a "fix" that simply stopped classifying anything.
"""
from pathlib import Path

import pytest

from src.ingestion.obsidian_parser import ObsidianParser, is_daily_note

# Prose that names no course and no date, and whose heading matches no course
# pattern; repeated so its section clears the parser's minimum chunk size.
BODY = ("# Gardening plans\n\n"
        + "Plain prose about the garden, repeated to clear the minimum chunk size. " * 4
        + "\n")


def _note(root: Path, rel: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(BODY, encoding="utf-8")
    return path


def _parse(root: Path, note: Path):
    parser = ObsidianParser(str(root))
    docs = parser.parse_file(note)
    # An empty result would make every "not X" assertion below pass vacuously.
    assert docs, "the fixture note produced no chunks"
    return parser, docs


# ---------------------------------------------------------------- daily notes

def test_a_vault_stored_under_journal_is_not_made_of_daily_notes(tmp_path):
    root = tmp_path / "journal" / "MyVault"
    parser, docs = _parse(root, _note(root, "Reading/note.md"))

    assert {d.metadata["file_type"] for d in docs} == {"note"}
    assert parser.stats["daily_notes"] == 0


@pytest.mark.parametrize("rel", [
    "Reading/2026-05-28.md",        # a dated filename
    "Daily Notes/anything.md",      # a daily folder INSIDE the vault
    "Reading/Journal/anything.md",  # ...at any depth, whatever its case
])
def test_a_real_daily_note_inside_the_vault_still_is_one(tmp_path, rel):
    root = tmp_path / "journal" / "MyVault"
    parser, docs = _parse(root, _note(root, rel))

    assert {d.metadata["file_type"] for d in docs} == {"daily_note"}
    assert parser.stats["daily_notes"] == 1


def test_is_daily_note_reads_only_the_path_below_the_vault_root(tmp_path):
    root = tmp_path / "journal" / "MyVault"

    assert is_daily_note(root / "Reading" / "note.md", root) is False
    assert is_daily_note(root / "Daily" / "note.md", root) is True
    # Without a root the path is judged as given, which is what a caller that
    # already holds a vault-relative path wants.
    assert is_daily_note(Path("Reading/note.md")) is False
    assert is_daily_note(Path("Daily Notes/note.md")) is True


def test_is_daily_note_rejects_a_file_outside_the_vault_root(tmp_path):
    with pytest.raises(ValueError):
        is_daily_note(tmp_path / "elsewhere" / "note.md", tmp_path / "MyVault")


# ------------------------------------------------------------------- courses

def test_a_vault_stored_under_ml_notes_is_not_labelled_machine_learning(tmp_path):
    root = tmp_path / "ML Notes" / "MyVault"
    parser, docs = _parse(root, _note(root, "Reading/note.md"))

    assert all(d.metadata.get("course_name") != "Machine Learning & AI" for d in docs)
    assert parser.stats["course_notes"] == 0 and parser.stats["other_notes"] == 1


def test_a_course_folder_inside_the_vault_still_labels_its_notes(tmp_path):
    root = tmp_path / "ML Notes" / "MyVault"
    parser, docs = _parse(root, _note(root, "Machine Learning/note.md"))

    assert {d.metadata["course_name"] for d in docs} == {"Machine Learning & AI"}
    assert parser.stats["course_notes"] == 1
