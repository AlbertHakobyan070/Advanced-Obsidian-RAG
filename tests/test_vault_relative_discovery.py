"""Discovery must judge a file by its path INSIDE the vault.

THE BUG THIS PINS. Every loader's skip check compared the components of the
ABSOLUTE path, which includes the directories above the vault root. A vault
stored under a folder named `.git`, `node_modules`, `venv` or `_Backups` then
matched the skip set on an ancestor and discovered **nothing at all** — an
empty ingest that reports success, not an error.

Session 24 fixed exactly this in the canvas loader (`iter_canvas_files`). The
same comparison survived in five other discovery paths; these tests close them
and keep them closed.

Each loader gets the same two assertions, because both halves matter:

  1. a vault under a poisoned ancestor still finds its files;
  2. a skipped directory INSIDE the vault is still skipped.

Only the first would let through a "fix" that simply stopped skipping anything.
"""
import collections
from pathlib import Path

import pytest

# Ancestor names that appear in at least one loader's skip set. The vault root
# is created UNDERNEATH one of these.
POISONED = [".git", "node_modules", "_Backups", "venv", ".obsidian"]


def _vault_under(tmp_path: Path, poison: str, name: str, body: str) -> Path:
    """A vault root nested inside a directory named `poison`, holding one file."""
    root = tmp_path / poison / "MyVault"
    (root / "Course").mkdir(parents=True)
    (root / "Course" / name).write_text(body, encoding="utf-8")
    return root


def _with_skipped_child(root: Path, skipped: str, name: str, body: str) -> None:
    """Put a file inside a directory the loader is supposed to skip."""
    d = root / skipped
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(body, encoding="utf-8")


# ------------------------------------------------------------------ markdown

@pytest.mark.parametrize("poison", POISONED)
def test_markdown_discovery_survives_a_poisoned_ancestor(tmp_path, poison):
    from src.ingestion.obsidian_parser import ObsidianParser

    # Both fixtures are over 50 bytes on purpose: discover_files drops files
    # smaller than that, which would make this test pass or fail for the wrong
    # reason in either direction.
    root = _vault_under(tmp_path, poison, "note.md",
                        "# Heading\n\nEnough body text to clear the "
                        "minimum-size filter in discover_files.\n")
    _with_skipped_child(root, ".obsidian", "workspace.md",
                        "# not content\n\nAlso over the minimum size, so its "
                        "absence proves the skip and not the size.\n")

    names = {f.name for f in ObsidianParser(str(root)).discover_files()}

    assert "note.md" in names, (
        f"a vault under {poison!r} discovered nothing — the skip check is "
        f"reading the directories ABOVE the vault root again")
    assert "workspace.md" not in names, "a skipped directory inside the vault leaked in"


# ----------------------------------------------------------------------- pdf

def _pdf_loader(root: Path):
    from src.ingestion.pdf_loader import PDFLoader

    loader = PDFLoader.__new__(PDFLoader)
    loader.vault_path = root
    loader.include_path = None
    loader.include_files = None
    loader.exclude_path = None
    loader.skip_set = set()
    loader.skip_folders = []
    loader.only_book_folders = False
    loader.skip_books = False
    # discover_pdfs increments diagnostic counters by name; a plain dict would
    # KeyError before reaching the assertion under test.
    loader.stats = collections.defaultdict(int)
    return loader


@pytest.mark.parametrize("poison", POISONED)
def test_pdf_discovery_survives_a_poisoned_ancestor(tmp_path, poison):
    root = _vault_under(tmp_path, poison, "paper.pdf", "%PDF-1.4 stub")
    _with_skipped_child(root, "_ingested", "archived.pdf", "%PDF-1.4 stub")

    names = {f.name for f in _pdf_loader(root).discover_pdfs()}

    assert "paper.pdf" in names, f"a vault under {poison!r} discovered no PDFs"
    assert "archived.pdf" not in names, "_ingested archives were rediscovered"


def test_book_detection_ignores_directories_above_the_vault(tmp_path):
    """A vault stored under a folder that tokenises to "books" must not make
    every PDF in the corpus a book — that would make --skip-books ingest
    nothing and --only-books ingest everything."""
    root = tmp_path / "DS Books 2025" / "MyVault"
    (root / "Lectures").mkdir(parents=True)
    ordinary = root / "Lectures" / "week1.pdf"
    ordinary.write_text("%PDF-1.4 stub", encoding="utf-8")
    (root / "Readings").mkdir()
    real_book = root / "Readings" / "text.pdf"
    real_book.write_text("%PDF-1.4 stub", encoding="utf-8")

    loader = _pdf_loader(root)

    assert loader._is_book_path(real_book) is True
    assert loader._is_book_path(ordinary) is False, (
        "an ancestor named 'DS Books 2025' classified an ordinary lecture PDF "
        "as a book")


# ---------------------------------------------------------------------- code

@pytest.mark.parametrize("poison", POISONED)
def test_code_discovery_survives_a_poisoned_ancestor(tmp_path, poison):
    from src.ingestion.code_loader import CodeLoader

    root = _vault_under(tmp_path, poison, "script.py", "x = 1\n")
    _with_skipped_child(root, "__pycache__", "cached.py", "x = 2\n")

    loader = CodeLoader.__new__(CodeLoader)
    loader.vault_path = root
    loader.exts = {".py"}
    loader.include_path = None
    loader.exclude_path = None
    loader.include_files = None
    loader.skip_roots = set()
    loader.stats = {"files_skipped_generated": 0, "files_skipped_root": 0}

    names = {f.name for f in loader.discover_files()}

    assert "script.py" in names, f"a vault under {poison!r} discovered no code"
    assert "cached.py" not in names, "__pycache__ leaked in"


# ----------------------------------------------------------------- notebooks

@pytest.mark.parametrize("poison", POISONED)
def test_notebook_discovery_survives_a_poisoned_ancestor(tmp_path, poison):
    from src.ingestion.ipynb_loader import NotebookLoader

    root = _vault_under(tmp_path, poison, "analysis.ipynb", "{}")
    _with_skipped_child(root, ".ipynb_checkpoints", "analysis-checkpoint.ipynb", "{}")

    loader = NotebookLoader.__new__(NotebookLoader)
    loader.vault_path = root
    loader.exts = {".ipynb"}
    loader.include_path = None
    loader.include_files = None
    loader.skip_roots = set()
    loader.stats = collections.defaultdict(int)

    names = {f.name for f in loader.discover_files()}

    assert "analysis.ipynb" in names, (
        f"a vault under {poison!r} discovered no notebooks")
    assert "analysis-checkpoint.ipynb" not in names, "checkpoints leaked in"
