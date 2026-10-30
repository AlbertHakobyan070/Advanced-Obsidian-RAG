"""Provenance resolver (slice D, E5) - src/utils/provenance.py.

The one consumer rule (decision 4): a chunk's own `provenance` wins, else
provenance.by_file_type decides, else authored. No migration: the 161k chunks
indexed without the key resolve through this rule.
"""
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest
import yaml

from src.utils.config_loader import Config
from src.utils.provenance import (
    ProvenanceRule, provenance_map, provenance_of, provenance_rule, stamp_generated, unstamped_generated)


def _map(tmp_path, **prov):
    return provenance_map(Config({"provenance": prov} if prov else {}, tmp_path))


def _rule(tmp_path, canvas=None, **prov) -> ProvenanceRule:
    """No provenance: block without kwargs, else a block holding only the given keys;
    a canvas: block only when `canvas` is given."""
    data = {"provenance": prov} if prov else {}
    if canvas is not None:
        data["canvas"] = canvas
    return provenance_rule(Config(data, tmp_path))


def test_explicit_provenance_wins(tmp_path):
    assert provenance_of({"provenance": "generated", "file_type": "pdf"}, _map(tmp_path)) == "generated"


def test_pdf_is_source_by_the_shipped_default(tmp_path):
    by_type = _map(tmp_path)                      # no provenance: block at all
    assert by_type == {"pdf": "source"}
    assert provenance_of({"file_type": "pdf"}, by_type) == "source"
    assert provenance_of({"file_type": "PDF"}, by_type) == "source"
    assert _map(tmp_path, by_file_type={"PDF": "source"}) == {"pdf": "source"}


def test_unknown_file_type_is_authored(tmp_path):
    assert provenance_of({"file_type": "web_clip"}, _map(tmp_path)) == "authored"


def test_code_without_the_key_is_authored(tmp_path):
    assert provenance_of({"file_type": "code"}, _map(tmp_path)) == "authored"


def test_notebook_without_the_key_is_authored(tmp_path):
    assert provenance_of({"file_type": "ipynb"}, _map(tmp_path)) == "authored"


def test_canvas_without_the_key_is_authored(tmp_path):
    assert provenance_of({"file_type": "canvas"}, _map(tmp_path)) == "authored"


def test_explicit_unknown_value_raises_naming_it(tmp_path):
    with pytest.raises(ValueError, match="'llm'.*authored, source, generated"):
        provenance_of({"provenance": "llm", "file_type": "note"}, _map(tmp_path))


def test_bad_map_raises_naming_the_key(tmp_path):
    with pytest.raises(ValueError, match=r"provenance\.by_file_type.*'primary'"):
        _map(tmp_path, by_file_type={"pdf": "primary"})
    with pytest.raises(ValueError, match=r"provenance\.by_file_type"):
        _map(tmp_path, by_file_type=["pdf"])


# --- the rule object: generated paths, agent-instruction names, canvas markers ---

def test_a_generated_path_beats_a_stored_authored(tmp_path):
    r = _rule(tmp_path, generated_paths=["Courses/AI/_Blueprints — AI/"])
    meta = {"provenance": "authored", "file_type": "note",
            "source_file": "Courses\\AI\\_Blueprints — AI\\00 — START.md"}
    assert provenance_of(meta, r) == "generated"


def test_the_path_rule_matches_whole_segments_only(tmp_path):
    r = _rule(tmp_path, generated_paths=["Courses/AI/Maps"])
    assert provenance_of({"source_file": "Courses/AI/Maps/x.md"}, r) == "generated"
    assert provenance_of({"source_file": "Courses/AI/Maps2/x.md"}, r) == "authored"


def test_mac_path_forms_still_match(tmp_path):          # Review Focus 1
    r = _rule(tmp_path, generated_paths=["00 – AUA_DS/Café/"])
    nfd = unicodedata.normalize("NFD", "00 – aua_ds/CAFÉ/note.md")
    assert provenance_of({"source_file": nfd}, r) == "generated"


def test_agent_instruction_files_and_dirs_are_generated_anywhere(tmp_path):
    r = _rule(tmp_path, agent_instruction_files=["AGENTS.md", "SKILL.md"],
              agent_instruction_dirs=[".claude"])
    for sf in ("a\\b\\AGENTS.md", "x/skills/post/skill.md", "proj/.claude/hooks/guard.py"):
        assert provenance_of({"source_file": sf, "provenance": "authored"}, r) == "generated", sf
    assert provenance_of({"source_file": "a/AGENTS-notes.md"}, r) == "authored"


def test_instruction_file_is_the_name_and_dir_match_only(tmp_path):
    r = _rule(tmp_path, generated_paths=["Maps/"], agent_instruction_files=["AGENTS.md"],
              agent_instruction_dirs=[".claude"])
    assert r.instruction_file("Maps\\AGENTS.md") and r.instruction_file("p/.claude/hooks/g.py")
    assert not r.instruction_file("Maps/a.md") and not r.instruction_file("a/AGENTS-notes.md")


def test_a_canvas_marker_is_generated_without_a_stored_key(tmp_path):
    r = _rule(tmp_path)                                   # canvas.generated_markers defaults to ["(generated)"]
    assert provenance_of({"source_file": "x/Map (Generated).canvas", "file_type": "live_note"}, r) == "generated"
    assert provenance_of({"source_file": "x/Map (generated).md", "file_type": "live_note"}, r) == "authored"
    assert provenance_of({"source_file": "x/Map.canvas", "file_type": "live_note"}, r) == "authored"


def test_missing_keys_take_the_shipped_defaults(tmp_path):
    for r in (_rule(tmp_path), _rule(tmp_path, label_generated=False)):   # no block; a block without the keys
        assert r.generated_paths == ()
        for sf in ("p/AGENTS.md", "p/.claude/hooks/x.py"):
            assert provenance_of({"source_file": sf, "provenance": "authored"}, r) == "generated", sf
        assert provenance_of({"source_file": "p/notes.md"}, r) == "authored"


def test_rule_order_is_stored_generated_then_path_then_stored_then_file_type(tmp_path):
    r = _rule(tmp_path, by_file_type={"pdf": "source"})
    assert provenance_of({"provenance": "authored", "file_type": "pdf"}, r) == "authored"
    assert provenance_of({"file_type": "pdf"}, r) == "source"
    assert provenance_of({}, r) == "authored"


def test_a_plain_mapping_still_works():
    assert provenance_of({"file_type": "pdf"}, {"pdf": "source"}) == "source"


@pytest.mark.parametrize("key,bad", [("generated_paths", "Maps/"), ("generated_paths", [""]),
                                     ("generated_paths", None),
                                     ("agent_instruction_files", "AGENTS.md"),
                                     ("agent_instruction_dirs", [3])])
def test_bad_values_raise_naming_the_key(tmp_path, key, bad):
    with pytest.raises(ValueError, match=f"provenance.{key}"):
        _rule(tmp_path, **{key: bad})


def test_canvas_markers_as_a_string_raise(tmp_path):
    with pytest.raises(ValueError, match="canvas.generated_markers"):
        _rule(tmp_path, canvas={"generated_markers": "(generated)"})


@pytest.mark.parametrize("canvas,expected", [(None, ["(generated)"]),
                                             ({"generated_markers": ["(AI)", " ", "Draft"]}, ["(ai)", "draft"])])
def test_the_canvas_loader_and_the_rule_read_the_same_markers(tmp_path, canvas, expected):
    from src.ingestion.canvas_loader import CanvasLoader
    data = {"parser": {"vault_path": str(tmp_path)}}
    if canvas is not None:
        data["canvas"] = canvas
    cfg = Config(data, tmp_path)
    assert CanvasLoader.from_config(cfg).generated_markers == list(provenance_rule(cfg).generated_markers) == expected


def test_unstamped_generated_counts_rows_whose_stored_key_disagrees(tmp_path):
    r = _rule(tmp_path, generated_paths=["Maps/"], by_file_type={"daily_note": "generated"})
    metas = [{"source_file": "Maps/a.md", "provenance": "authored"},
             {"source_file": "Maps/a.md", "provenance": "generated"},
             {"source_file": "d.md", "file_type": "daily_note"},
             {"source_file": "n.md", "provenance": "authored"}]
    assert unstamped_generated(metas, r) == {"Maps/a.md": 1, "d.md": 1}


def test_stamp_generated_touches_only_what_the_path_rule_names(tmp_path):
    r = _rule(tmp_path, generated_paths=["Maps/"])
    assert stamp_generated({"source_file": "Maps/a.md", "provenance": "authored"}, r)["provenance"] == "generated"
    assert stamp_generated({"source_file": "n.md", "provenance": "authored"}, r)["provenance"] == "authored"
    assert stamp_generated({"source_file": "Maps/a.md", "provenance": "authored"}, None)["provenance"] == "authored"


def test_the_normaliser_agrees_with_the_canvas_one():
    from src.ingestion.canvas_loader import norm_vault_path
    from src.utils.provenance import _norm
    for p in ("A\\B\\c.md", "/a/b/", unicodedata.normalize("NFD", "Café/x")):
        assert _norm(p) == norm_vault_path(p)


def test_the_keys_ship_in_both_configs():
    for name in ("config.yaml", "config.example.yaml"):
        if not (ROOT / name).exists():
            continue
        prov = yaml.safe_load((ROOT / name).read_text(encoding="utf-8"))["provenance"]
        assert isinstance(prov["generated_paths"], list), name
        assert prov["agent_instruction_files"] == ["AGENTS.md", "CLAUDE.md", "GEMINI.md", "SKILL.md"], name
        assert prov["agent_instruction_dirs"] == [".claude"], name
    example = yaml.safe_load((ROOT / "config.example.yaml").read_text(encoding="utf-8"))
    assert example["provenance"]["generated_paths"] == []
