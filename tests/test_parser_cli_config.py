"""`python -m src.ingestion.obsidian_parser` must follow config.yaml.

THE BUG THIS PINS. The CLI built its parser from its own defaults — chunking
'heading' and the built-in taxonomy — while the config said otherwise, so the
README quickstart's first step ignored `parser.chunking` and the `taxonomy:`
block that every other ingest path honours.

These tests run the REAL entry point in a subprocess, not main() in-process,
because of a trap only the real thing has: run with -m, the file is `__main__`,
and load_config() applies the taxonomy to a SECOND copy of the module that it
imports by name. The copy whose ObsidianParser actually runs never sees it
unless main() applies it itself.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# `none` is the loudest strategy to observe: an oversized section stays ONE
# chunk, where the old default ('heading') splits it into several.
CONFIG = """\
parser:
  chunking: none
taxonomy:
  detect_from_path: true
  folder_map: {"gadgets": "Gadget Engineering"}
  domain_map: {"Gadget Engineering": "eng"}
  code_map: {}
  keywords: {}
"""

# Five paragraphs of ~1.5k characters: well over max_chunk_size (3000), and
# with paragraph breaks so 'heading' packing has somewhere to cut.
OVERSIZED = "# Gadget log\n\n" + "\n\n".join(
    ["Gadgets are discussed here at some length. " * 35] * 5) + "\n"


def _run_cli(tmp_path: Path, *flags: str) -> list[dict]:
    """Run the parser CLI with tmp_path as cwd — so load_config() finds the
    config.yaml written there, not the live one — and return its chunks."""
    (tmp_path / "config.yaml").write_text(CONFIG, encoding="utf-8")
    note = tmp_path / "vault" / "Gadgets" / "note.md"
    note.parent.mkdir(parents=True)
    note.write_text(OVERSIZED, encoding="utf-8")
    out = tmp_path / "out.jsonl"

    proc = subprocess.run(
        [sys.executable, "-m", "src.ingestion.obsidian_parser",
         str(tmp_path / "vault"), "-o", str(out), "--parents-out", "", *flags],
        cwd=tmp_path, capture_output=True, encoding="utf-8", errors="replace",
        env={**os.environ, "PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8"},
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    return [json.loads(line)
            for line in out.read_text(encoding="utf-8").splitlines()]


def test_cli_takes_chunking_and_taxonomy_from_config_yaml(tmp_path):
    chunks = _run_cli(tmp_path)

    assert len(chunks) == 1, (
        f"parser.chunking: none was ignored — the oversized section split "
        f"into {len(chunks)} chunks")
    meta = chunks[0]["metadata"]
    assert "chunk_part" not in meta
    # The taxonomy block reached the copy of the module that actually parsed.
    assert (meta["course_name"], meta["domain"]) == ("Gadget Engineering", "eng")


def test_cli_chunking_flag_still_overrides_config_yaml(tmp_path):
    chunks = _run_cli(tmp_path, "--chunking", "heading")

    assert len(chunks) > 1
    assert all("chunk_part" in c["metadata"] for c in chunks)


# ---- parser.skip_dirs / parser.skip_roots and the scoped-run output guard ----

NOTE = "# Gadget log\n\n" + "Gadgets are discussed here at some length. " * 10 + "\n"
SKIP_CONFIG = CONFIG.replace(
    "  chunking: none\n",
    '  chunking: none\n  skip_dirs: ["venv"]\n  skip_roots: ["Workspace1"]\n')
VAULT_FILES = {"Gadgets/note.md": NOTE, "Gadgets/venv/v.md": NOTE, "Workspace1/a.md": NOTE}


def _run_cli_with(tmp_path: Path, config: str, files: dict, *flags: str,
                  parents_out: str | None = ""):
    """Like _run_cli, but for runs that may fail: returns (proc, chunks or None).

    The output is out.jsonl in tmp_path unless the flags name their own `-o`.
    `parents_out=None` leaves --parents-out off, so the CLI default
    (data/parents_md.jsonl, relative to tmp_path) applies."""
    (tmp_path / "config.yaml").write_text(config, encoding="utf-8")
    for rel, body in files.items():
        f = tmp_path / "vault" / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(body, encoding="utf-8")
    flags = list(flags)
    if "-o" not in flags:
        flags += ["-o", "out.jsonl"]
    out = tmp_path / flags[flags.index("-o") + 1]
    cmd = [sys.executable, "-m", "src.ingestion.obsidian_parser",
           str(tmp_path / "vault"), *flags]
    if parents_out is not None:
        cmd += ["--parents-out", parents_out]
    proc = subprocess.run(
        cmd, cwd=tmp_path, capture_output=True, encoding="utf-8", errors="replace",
        env={**os.environ, "PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8"},
        timeout=120)
    chunks = None
    if proc.returncode == 0:
        chunks = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    return proc, chunks


def _sources(chunks) -> set[str]:
    return {c["metadata"]["source_file"].replace("\\", "/") for c in chunks}


def test_cli_honours_parser_skip_dirs_and_skip_roots(tmp_path):
    proc, chunks = _run_cli_with(tmp_path, SKIP_CONFIG, VAULT_FILES)

    assert proc.returncode == 0, proc.stderr
    assert _sources(chunks) == {"Gadgets/note.md"}


def test_cli_include_path_scopes_a_skipped_root_back_in(tmp_path):
    proc, chunks = _run_cli_with(tmp_path, SKIP_CONFIG, VAULT_FILES,
                                 "--include-path", "workspace1", "-o", "scoped.jsonl")

    assert proc.returncode == 0, proc.stderr
    assert _sources(chunks) == {"Workspace1/a.md"}


def test_cli_explicit_skip_roots_flag_overrides_config(tmp_path):
    proc, chunks = _run_cli_with(tmp_path, SKIP_CONFIG, VAULT_FILES, "--skip-roots", "")

    assert proc.returncode == 0, proc.stderr
    assert "Workspace1/a.md" in _sources(chunks)


def test_cli_scoped_parse_refuses_the_canonical_outputs(tmp_path):
    parents = tmp_path / "data" / "parents_md.jsonl"
    parents.parent.mkdir()
    parents.write_bytes(b'{"parent_id": "p1"}\r\n')
    before = parents.read_bytes()

    # (a) the canonical chunk file, in any case: the old check was an exact string.
    proc, _ = _run_cli_with(tmp_path, SKIP_CONFIG, VAULT_FILES,
                            "--include-path", "gadgets", "-o", "CHUNKS.JSONL")
    assert proc.returncode == 2, proc.stderr

    # (b) its own chunk file, but the default parents sidecar: a partial sidecar
    # would replace the whole-vault one.
    proc, _ = _run_cli_with(tmp_path, SKIP_CONFIG, VAULT_FILES,
                            "--include-path", "gadgets", "-o", "scoped.jsonl",
                            parents_out=None)
    assert proc.returncode == 2, proc.stderr
    assert parents.read_bytes() == before

    # (c) the supported shape: its own chunk file and no sidecar.
    proc, _ = _run_cli_with(tmp_path, SKIP_CONFIG, VAULT_FILES,
                            "--include-path", "gadgets", "-o", "scoped.jsonl")
    assert proc.returncode == 0, proc.stderr
    assert parents.read_bytes() == before


def test_cli_stops_on_a_bad_provenance_instead_of_dropping_the_note(tmp_path):
    """The whole-vault parse used to skip the note, write the rest and exit 0: the note
    left the canonical file while the dense index kept its rows."""
    files = {**VAULT_FILES, "Gadgets/bad.md": "---\nprovenance: Source\n---\n" + NOTE}
    proc, chunks = _run_cli_with(tmp_path, SKIP_CONFIG, files)

    assert proc.returncode != 0 and chunks is None
    assert "bad.md" in proc.stderr and "authored, source, generated" in proc.stderr
    assert not (tmp_path / "out.jsonl").exists()
