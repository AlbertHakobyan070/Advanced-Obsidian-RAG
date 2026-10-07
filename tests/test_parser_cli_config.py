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
