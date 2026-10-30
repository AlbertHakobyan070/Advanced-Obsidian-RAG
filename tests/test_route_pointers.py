"""route-then-cite §5.4: the pointer sidecar (src/retrieval/route.py), its cite grammar,
the route-pointers command and the route_pointers job kind."""
import json
import sys
import time
import unicodedata
import uuid
from pathlib import Path

import pytest
import yaml

from src.retrieval.route import (RoutePointers, build_pointers, canvas_targets, note_targets,
                                 write_pointers)
from src.utils.config_loader import Config
from src.utils.provenance import provenance_rule

ROOT = Path(__file__).resolve().parents[1]
HUYEN = "00 – AUA_DS/Other/Tech Books/AI Engineering - Chip Huyen (2025).pdf"
COURSE = "Course/Prompt Engineering"
DECKS = {"Ch1": "Ch1 - Intro (slides).pdf", "Ch2": "Ch2 - Shots (slides).pdf",
         "Ch3": "Ch3 - Chains (slides).pdf"}


def _t(path, a=None, b=None):
    return {"source_file": path, "page_start": a, "page_end": b}


def _folder(vault: Path, chapters=("Ch1", "Ch2")) -> None:
    """A course folder as the live ones are: one 'ChN - ... (slides).pdf' per chapter plus its md twin."""
    d = vault / COURSE
    d.mkdir(parents=True, exist_ok=True)
    for ch in chapters:
        (d / DECKS[ch]).touch()
        (d / DECKS[ch].replace(".pdf", ".md")).touch()


def _book(vault: Path, rel="Books/b.pdf") -> str:
    (vault / rel).parent.mkdir(parents=True, exist_ok=True)
    (vault / rel).touch()
    return rel


def _deck(ch, a, b):
    return _t(f"{COURSE}/{DECKS[ch]}", a, b)


def _row(doc_id, text, **meta):
    return {"doc_id": doc_id, "text": text, "metadata": meta}


def _world(tmp_path, rows, *, route=True) -> tuple[Path, Config]:
    """data/chunks.jsonl holding `rows`, an empty vault, and the config that names both.
    Maps/ and Notes/ are generated prefixes."""
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "vault").mkdir(exist_ok=True)
    with open(tmp_path / "data" / "chunks.jsonl", "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    d = {"paths": {"chunks_file": "data/chunks.jsonl"},
         "parser": {"vault_path": str(tmp_path / "vault")},
         "provenance": {"generated_paths": ["Maps", "Notes"]}}
    if route:
        d["retrieval"] = {"route": {"pointers_file": "data/route_pointers.json"}}
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(d, allow_unicode=True), encoding="utf-8")
    return cfg_path, Config(d, tmp_path)


def _build(cfg):
    return build_pointers(cfg, provenance_rule(cfg))


# ---------------------------------------------------------------- canvas ----

def test_canvas_targets_keep_full_ranges_and_ignore_the_footer():
    text = ("[Canvas: Huyen map (generated)]\nPrompt caching\n"
            f"Source: {HUYEN} pp. 59-63, 68\n"
            "\n\nConnections:\n→ Other node\nSource: elsewhere.pdf pp. 1-2")
    assert canvas_targets(text) == ([_t(HUYEN, 59, 63), _t(HUYEN, 68, 68)], False)
    # a backwards range is a typo: the line is skipped, never swapped into 3-9, and said so
    assert canvas_targets(f"[Canvas: m]\nSource: {HUYEN} pp. 9-3\nSource: notes/x.md") \
        == ([_t("notes/x.md")], True)


L1, L2 = f"Source: {HUYEN} pp. 59-63", "Source: Lectures/x.md"   # primary: not under Maps/ or Notes/


@pytest.mark.parametrize("part1,part2", [(f"{L1}\n{L2}", ""),     # the brief's: part 1 only
                                         ("", f"{L1}\n{L2}"),     # the loader's: the body's tail
                                         (L1, L2)])               # one line on each part
def test_split_canvas_parts_share_the_node_targets(tmp_path, part1, part2):
    """Part 1 is read first whichever part carries the Source lines, so every part gets the
    node's targets only after all its parts are read."""
    head = "[Canvas: m (generated)]\n"
    _, cfg = _world(tmp_path, [
        _row("p1", f"{head}Long node, part one\n{part1}",
             source_file="Maps/m (generated).canvas", file_type="canvas", canvas_node_id="n1",
             canvas_part=1, canvas_part_count=2),
        _row("p2", f"{head}Long node, part two\n{part2}",
             source_file="Maps/m (generated).canvas", file_type="canvas", canvas_node_id="n1",
             canvas_part=2, canvas_part_count=2),
        _row("other", f"{head}Another node\nSource: {HUYEN} pp. 70-71",
             source_file="Maps/m (generated).canvas", file_type="canvas", canvas_node_id="n2")])
    p = _build(cfg)
    node = [_t(HUYEN, 59, 63), _t("Lectures/x.md")]
    assert p["pointers"] == {"p1": node, "p2": node, "other": [_t(HUYEN, 70, 71)]}
    assert p["unrouted"] == {}


def test_one_node_id_in_two_canvases_keeps_each_files_targets(tmp_path):
    """The live maps share node ids ('book', 'ch01') across canvases: the key is the file too."""
    _, cfg = _world(tmp_path, [
        _row("a", f"[Canvas: a]\nSource: {HUYEN} pp. 59-63", source_file="Maps/a.canvas",
             file_type="canvas", canvas_node_id="ch01"),
        _row("b", "[Canvas: b]\nSource: Books/b.pdf pp. 1-2", source_file="Maps/b.canvas",
             file_type="canvas", canvas_node_id="ch01")])
    assert _build(cfg)["pointers"] == {"a": [_t(HUYEN, 59, 63)], "b": [_t("Books/b.pdf", 1, 2)]}


# ------------------------------------------------------------ note cites ----

@pytest.mark.parametrize("cite,expect", [
    ("(pp. 59-63, 68)",  [(59, 63), (68, 68)]),
    ("(pp. 74,102-104)", [(74, 74), (102, 104)]),
    ("(p.22-23)",        [(22, 23)]),
    ("(p.17, p.27)",     [(17, 17), (27, 27)])])
def test_note_cites_on_a_file_source(tmp_path, cite, expect):
    rel = _book(tmp_path)
    assert note_targets(f"Prompt caching {cite}. Bare p.5 is prose.", {"source": rel}, tmp_path) \
        == ([_t(rel, a, b) for a, b in expect], "")


@pytest.mark.parametrize("cite,expect", [
    ("(Ch2 p.27-32)",          [("Ch2", 27, 32)]),
    ("(Ch2 pp.27-32)",         [("Ch2", 27, 32)]),
    ("(Ch2 p.27-32, 35)",      [("Ch2", 27, 32), ("Ch2", 35, 35)]),
    ("(Ch2 p.3, p.5-7)",       [("Ch2", 3, 3), ("Ch2", 5, 7)]),
    ("(Ch1 pp.26-28; p.30)",   [("Ch1", 26, 28), ("Ch1", 30, 30)]),
    ("(Ch1 p.10; the slide is image-only)", [("Ch1", 10, 10)]),
    ("(Ch2 p.4; Ch2 practice set)",          [("Ch2", 4, 4)])])
def test_note_cites_on_a_folder_source(tmp_path, cite, expect):
    _folder(tmp_path)
    assert note_targets(f"Few-shot {cite}.", {"source": COURSE}, tmp_path) \
        == ([_deck(*e) for e in expect], "")


def test_chapter_cites_resolve_to_the_one_matching_deck(tmp_path):
    deck_dir = tmp_path / "Course/Prompt Engineering"
    deck_dir.mkdir(parents=True)
    (deck_dir / "Ch2 - Shots (slides).pdf").touch(); (deck_dir / "Ch2 - Shots (slides).md").touch()
    targets, why = note_targets("Few-shot (Ch2 p.27-32).", {"source": "Course/Prompt Engineering"}, tmp_path)
    assert targets == [{"source_file": "Course/Prompt Engineering/Ch2 - Shots (slides).pdf",
                        "page_start": 27, "page_end": 32}]


def test_the_deck_match_is_case_blind_files_only_and_delimited(tmp_path):
    """The same on every OS: case-blind (a live course folder spells 'ch1 - ...'), the
    on-disk spelling kept, a folder named like a deck ignored, and Ch1 never matching Ch10."""
    d = tmp_path / COURSE
    d.mkdir(parents=True)
    for name in ("ch2 - x.PDF", "Ch1 - Intro.pdf", "Ch1 - Intro.md", "Ch10 - Late (slides).pdf"):
        (d / name).touch()
    (d / "Ch3 - folder.pdf").mkdir()
    fm = {"source": COURSE}
    assert note_targets("(Ch2 p.3)", fm, tmp_path) == ([_t(f"{COURSE}/ch2 - x.PDF", 3, 3)], "")
    assert note_targets("(Ch1 p.3)", fm, tmp_path) == ([_t(f"{COURSE}/Ch1 - Intro.pdf", 3, 3)], "")
    assert note_targets("(Ch10 p.2)", fm, tmp_path) == ([_t(f"{COURSE}/Ch10 - Late (slides).pdf", 2, 2)], "")
    assert note_targets("(Ch3 p.1)", fm, tmp_path) == ([], "Ch3 matches 0 decks")


@pytest.mark.parametrize("decks,why", [((), "Ch2 matches 0 decks"),
                                       (("Ch2 - a.pdf", "Ch2 - b.pdf"), "Ch2 matches 2 decks")])
def test_a_folder_source_without_exactly_one_deck_is_unrouted(tmp_path, decks, why):
    """Review Focus 3: never a guessed deck."""
    (tmp_path / COURSE).mkdir(parents=True)
    for name in decks:
        (tmp_path / COURSE / name).touch()
    assert note_targets("Few-shot (Ch2 p.27-32).", {"source": COURSE, "pages": "Ch2 p.1-40"},
                        tmp_path) == ([], why)


def test_bare_cite_under_a_folder_uses_the_one_frontmatter_chapter(tmp_path):
    _folder(tmp_path)
    assert note_targets("Zero-shot (pp. 2-4, 11-12).", {"source": COURSE, "pages": "Ch1 p.1-32"},
                        tmp_path) == ([_deck("Ch1", 2, 4), _deck("Ch1", 11, 12)], "")


def test_bare_cite_under_a_folder_with_two_chapters_is_unrouted(tmp_path):
    _folder(tmp_path)
    assert note_targets("Zero-shot (pp. 2-4).", {"source": COURSE, "pages": "Ch1 p.1-32; Ch2 p.1-20"},
                        tmp_path) == ([], "bare pages under a folder source")


def test_chapter_cite_under_a_file_source_is_unrouted(tmp_path):
    rel = _book(tmp_path)
    assert note_targets("Caching (Ch2 p.3-4).", {"source": rel, "pages": "1-29"}, tmp_path) \
        == ([], "chapter pages under a file source")


def test_an_unresolved_cite_does_not_fall_back_to_the_frontmatter(tmp_path):
    _folder(tmp_path)
    fm = {"source": COURSE, "pages": "Ch1 p.1-32"}
    assert note_targets("Chains (Ch3 p.4).", fm, tmp_path) == ([], "Ch3 matches 0 decks")
    # nothing resolves: the first part's reason, not the last's
    assert note_targets("Chains (Ch3 p.4; Ch4 p.1).", fm, tmp_path) == ([], "Ch3 matches 0 decks")
    # a cite with one resolved part keeps it, in the union of every cite's resolved parts
    assert note_targets("Chains (Ch3 p.4; Ch1 p.3) and (Ch2 p.9).", fm, tmp_path) \
        == ([_deck("Ch1", 3, 3), _deck("Ch2", 9, 9)], "")


@pytest.mark.parametrize("text", [
    "Drill (Ch2 practice set), see the (slide p.4) and (the slide is image-only).",
    "Prompt templates (5-7, 24; practice notes 5-8).",   # inline, a page list needs p./pp.
    "As Liu et al. (2025) show.",                         # a year is not a page
    "Intro; p.5"])                                        # bare p.N outside parentheses
def test_a_practice_set_only_group_is_not_a_cite_and_uses_frontmatter(tmp_path, text):
    _folder(tmp_path)
    assert note_targets(text, {"source": COURSE, "pages": "Ch1 p.1-32"}, tmp_path) \
        == ([_deck("Ch1", 1, 32)], "")


def test_a_backwards_range_never_routes(tmp_path):
    """A typo, never swapped: inline the part contributes nothing, in pages: it is malformed."""
    rel = _book(tmp_path)
    fm = {"source": rel, "pages": "1-29"}
    assert note_targets("Typo (pp. 9-3).", fm, tmp_path) == ([_t(rel, 1, 29)], "")    # not a cite
    assert note_targets("Typo (pp. 9-3; p.4).", fm, tmp_path) == ([_t(rel, 4, 4)], "")
    assert note_targets("No cite.", {"source": rel, "pages": "29-1"}, tmp_path) == ([], "malformed pages")


@pytest.mark.parametrize("pages,kind,expect", [
    ("1-29", "file", [(None, 1, 29)]),
    (12, "file", [(None, 12, 12)]),                     # YAML reads pages: 12 as an int
    ("Ch1 pp.1-33; Ch2 pp.1-25", "folder", [("Ch1", 1, 33), ("Ch2", 1, 25)]),
    ("Ch2 p.2-28; Ch3 p.3-34", "folder", [("Ch2", 2, 28), ("Ch3", 3, 34)]),
    ("Ch1 p.21-39 and p.1-40; Ch2 p.3-46 and p.1-48", "folder",
     [("Ch1", 21, 39), ("Ch1", 1, 40), ("Ch2", 3, 46), ("Ch2", 1, 48)]),
    ("none (source has no page numbers)", "file", "no pages"),
    ("see ch.3", "folder", "malformed pages"),
    ("1-29", "folder", "bare pages under a folder source"),
    ("Ch1 p.1-32", "file", "chapter pages under a file source")])
def test_frontmatter_pages_grammar(tmp_path, pages, kind, expect):
    """The p./pp. prefix is optional in pages:, and which values route depends on the source kind."""
    _folder(tmp_path, ("Ch1", "Ch2", "Ch3"))
    rel = _book(tmp_path) if kind == "file" else COURSE
    got = note_targets("A chunk with no cite.", {"source": rel, "pages": pages}, tmp_path)
    if isinstance(expect, str):
        assert got == ([], expect)
    else:
        assert got == ([_t(rel, a, b) if ch is None else _deck(ch, a, b) for ch, a, b in expect], "")


def test_a_note_without_pages_routes_a_file_source_whole_and_leaves_a_folder_unrouted(tmp_path):
    _folder(tmp_path)
    rel = _book(tmp_path)
    assert note_targets("No cite.", {"source": rel}, tmp_path) == ([_t(rel)], "")
    assert note_targets("No cite.", {"source": COURSE, "pages": None}, tmp_path) == ([], "no pages")


@pytest.mark.parametrize("fm,why", [({}, "no source"), ({"source": None, "pages": None}, "no source"),
                                    ({"source": ["a", "b"]}, "no source"),
                                    ({"source": "Nowhere/x.pdf", "pages": "1-2"}, "source not found")])
def test_a_note_with_no_usable_source_is_unrouted(tmp_path, fm, why):
    """Review Focus 3: the live _Blueprints notes have source None, which must not crash."""
    assert note_targets("Text (p.3).", fm, tmp_path) == ([], why)


# ------------------------------------------------------- build_pointers ----

def _note(vault: Path, rel: str, frontmatter: str) -> None:
    (vault / rel).parent.mkdir(parents=True, exist_ok=True)
    (vault / rel).write_text(f"---\n{frontmatter}---\n# Body\n", encoding="utf-8")


class _RecordingVault:
    """The vault root, recording each relative path joined onto it. A WindowsPath reads
    'a\\b' as 'a/b', so only the string handed to the join shows whether a Mac resolves it."""

    def __init__(self, real: Path):
        self.real, self.joined = real, []

    def __truediv__(self, rel):
        self.joined.append(rel)
        return self.real / rel


def test_a_backslash_source_file_resolves_and_bad_yaml_is_unreadable(tmp_path, monkeypatch):
    rows = [_row("ok", "Caching (pp. 59-63).", source_file="Notes\\ch1.md", file_type="note"),
            _row("daily", "Plain.", source_file="Notes\\day.md", file_type="daily_note"),
            _row("bad", "Caching (pp. 59-63).", source_file="Notes\\bad.md", file_type="note"),
            _row("gone", "Caching (pp. 59-63).", source_file="Notes\\gone.md", file_type="note"),
            _row("blue", "Caching (pp. 59-63).", source_file="Notes\\blueprint.md", file_type="note")]
    _, cfg = _world(tmp_path, rows)
    vault = tmp_path / "vault"
    rel = _book(vault)
    _note(vault, "Notes/ch1.md", "source: Books\\b.pdf\npages: 1-29\n")     # a '\' source: too
    _note(vault, "Notes/day.md", f"source: {rel}\npages: 3-4\n")
    _note(vault, "Notes/bad.md", "source: [unclosed\n")
    _note(vault, "Notes/blueprint.md", "title: an agent guide\n")
    rec = _RecordingVault(vault)
    real_path = cfg.path
    monkeypatch.setattr(cfg, "path", lambda key: rec if key == "parser.vault_path" else real_path(key))
    p = _build(cfg)
    assert p["pointers"] == {"ok": [_t(rel, 59, 63)], "daily": [_t(rel, 3, 4)]}
    assert p["unrouted"] == {"bad": "frontmatter unreadable", "gone": "note file not found",
                             "blue": "no source"}
    assert "Notes/ch1.md" in rec.joined and "Books/b.pdf" in rec.joined
    assert not [j for j in rec.joined if "\\" in j]


def test_an_unreadable_note_or_a_list_frontmatter_is_unrouted(tmp_path):
    _, cfg = _world(tmp_path, [
        _row("cp", "Caching (pp. 1-2).", source_file="Notes\\cp1252.md", file_type="note"),
        _row("ls", "Caching (pp. 1-2).", source_file="Notes\\list.md", file_type="note")])
    vault = tmp_path / "vault"
    (vault / "Notes").mkdir()
    (vault / "Notes" / "cp1252.md").write_bytes("---\nsource: café.pdf\n---\n".encode("cp1252"))
    _note(vault, "Notes/list.md", "- a\n- b\n")
    assert _build(cfg)["unrouted"] == {"cp": "note file unreadable", "ls": "frontmatter unreadable"}


def test_a_row_citing_only_generated_files_is_unrouted(tmp_path):
    """A target the rule calls generated (an agent file, a map) can never add a primary page."""
    _, cfg = _world(tmp_path, [
        _row("only", "[Canvas: m]\nSource: proj/agents/AGENTS.md\nSource: Maps/other.md",
             source_file="Maps/m.canvas", file_type="canvas", canvas_node_id="n1"),
        _row("mixed", f"[Canvas: m]\nSource: proj/skills/SKILL.md\nSource: {HUYEN} pp. 59-63",
             source_file="Maps/m.canvas", file_type="canvas", canvas_node_id="n2")])
    p = _build(cfg)
    assert p["pointers"] == {"mixed": [_t(HUYEN, 59, 63)]}
    assert p["unrouted"] == {"only": "cites only generated files"}


def test_the_first_row_of_an_id_wins_as_in_the_sparse_union(tmp_path):
    """An authored 'x' in chunks.jsonl is what the store serves for x, so a generated row
    reusing that id in a lane file is skipped, not routed."""
    _, cfg = _world(tmp_path, [_row("x", "Authored.", source_file="Lectures\\l1.md", file_type="note")])
    with open(tmp_path / "data" / "canvas_chunks.jsonl", "w", encoding="utf-8") as fh:
        fh.write(json.dumps(_row("x", f"[Canvas: m]\nSource: {HUYEN} pp. 1-2", source_file="Maps/m.canvas",
                                 file_type="canvas", canvas_node_id="n1")) + "\n")
    p = _build(cfg)
    assert p["pointers"] == {} and p["unrouted"] == {}


def test_build_pointers_dispatches_on_file_type(tmp_path):
    """A row with no pointer format is never parsed: its vault file does not exist, and
    the reason is the format, not a missing file."""
    _, cfg = _world(tmp_path, [
        _row("py", "def f(): pass  # (pp. 1-2)", source_file="Maps\\tool.py", file_type="py"),
        _row("bare", "No file type (pp. 1-2)", source_file="Maps/x.txt"),
        _row("cv", "[Canvas: m]\nA node with no Source line", source_file="Maps/m.canvas",
             file_type="canvas", canvas_node_id="n9")])
    p = _build(cfg)
    assert p["pointers"] == {}
    assert p["unrouted"] == {"py": "no pointer format for py", "bare": "no pointer format for none",
                             "cv": "no Source line"}


def test_a_node_whose_only_source_line_runs_backwards_says_so(tmp_path):
    """The node has a Source line, with a typo: the reason names the range, as a note's
    frontmatter typo is 'malformed pages'. A good line beside it still routes."""
    _, cfg = _world(tmp_path, [
        _row("bk", f"[Canvas: m]\nSource: {HUYEN} pp. 9-3", source_file="Maps/m.canvas",
             file_type="canvas", canvas_node_id="n1"),
        _row("ok", f"[Canvas: m]\nSource: {HUYEN} pp. 9-3\nSource: {HUYEN} pp. 1-2",
             source_file="Maps/m.canvas", file_type="canvas", canvas_node_id="n2")])
    p = _build(cfg)
    assert p["unrouted"] == {"bk": "backwards page range"}
    assert p["pointers"] == {"ok": [_t(HUYEN, 1, 2)]}


def test_agent_files_are_unrouted_before_any_parser(tmp_path):
    _, cfg = _world(tmp_path, [
        _row("a1", "Do this (pp. 1-2).", source_file="proj\\AGENTS.md", file_type="note"),
        _row("a2", "Do that (pp. 1-2).", source_file="Maps\\Guides\\AGENTS.md", file_type="note"),
        _row("a3", f"Source: {HUYEN} pp. 1-2", source_file="Maps/.claude/m.canvas",
             file_type="canvas", canvas_node_id="n1")])
    assert _build(cfg)["unrouted"] == {"a1": "agent instruction file", "a2": "agent instruction file",
                                       "a3": "agent instruction file"}


def test_build_pointers_skips_primary_rows(tmp_path):
    _, cfg = _world(tmp_path, [
        _row("n", "Authored (pp. 1-2).", source_file="Lectures\\l1.md", file_type="note",
             provenance="authored"),
        _row("b", "Book page", source_file=HUYEN, file_type="pdf", page_start=59, page_end=59),
        _row("c", f"[Canvas: mine]\nSource: {HUYEN} pp. 1-2", source_file="Mine/m.canvas",
             file_type="canvas", canvas_node_id="n1"),
        _row("g", f"[Canvas: m]\nSource: {HUYEN} pp. 1-2", source_file="Mine/m.canvas",
             file_type="canvas", canvas_node_id="n2", provenance="generated")])
    p = _build(cfg)
    assert p["pointers"] == {"g": [_t(HUYEN, 1, 2)]} and p["unrouted"] == {}
    assert p["version"] == 1 and p["built_at"].endswith("Z")


# ------------------------------------------------------ the sidecar file ----

CAFE = "Café/Résumé.pdf"


def test_write_then_load_round_trips_and_a_bad_version_raises(tmp_path, monkeypatch):
    import src.ocr_studio.pages as pages
    assert unicodedata.normalize("NFD", CAFE) != CAFE                  # really decomposes
    nfd = unicodedata.normalize("NFD", CAFE.replace("/", "\\")).upper()
    data = {"version": 1, "built_at": "2026-10-29T00:00:00Z",
            "pointers": {"a": [_t(CAFE, 59, 63)], "b": [_t(nfd, 59, 63), _t("notes/x.md")]},
            "unrouted": {"c": "no Source line"}}
    path = tmp_path / "not yet" / "route_pointers.json"
    writes, real = [], pages.atomic_write
    monkeypatch.setattr(pages, "atomic_write", lambda p, b: writes.append(p) or real(p, b))
    write_pointers(path, data)
    assert writes == [path]                     # temp file + replace: :8051 never reads half a file
    assert json.loads(path.read_text(encoding="utf-8")) == data
    p = RoutePointers.load(path)
    assert p.targets("a") == [_t(CAFE, 59, 63)]
    assert p.targets("c") is None and p.targets("nope") is None      # unrouted and absent alike
    assert len(p) == 2
    # one target spelled NFD, upper case and with '\\' is the same target
    assert p.all_targets() == [_t(CAFE, 59, 63), _t("notes/x.md")]
    assert len(RoutePointers(data)) == 2                               # the dict constructor

    write_pointers(path, {**data, "version": 2})
    with pytest.raises(ValueError, match="version"):
        RoutePointers.load(path)
    with pytest.raises(FileNotFoundError):
        RoutePointers.load(tmp_path / "absent.json")
    path.write_text('{"version": 1, "poin', encoding="utf-8")         # a lane's rows, or a cut-off file
    with pytest.raises(ValueError, match="rebuild it with python main.py route-pointers") as e:
        RoutePointers.load(path)
    assert isinstance(e.value.__cause__, json.JSONDecodeError)


AT = "2026-10-29T00:00:00Z"


@pytest.mark.parametrize("raw", [
    json.dumps({"version": 1}).encode("utf-8"),                                      # no pointers
    json.dumps({"version": 1, "built_at": AT, "pointers": [], "unrouted": {}}).encode("utf-8"),  # a list
    json.dumps({"version": 1, "built_at": AT, "pointers": {"a": [{"source_file": "b.pdf", "page_end": 2}]},
                "unrouted": {}}).encode("utf-8"),                                    # no page_start
    json.dumps({"version": 1, "pointers": {}, "unrouted": {}}).encode("utf-8"),      # no built_at
    json.dumps({"version": 1, "built_at": AT, "pointers": {}, "unrouted": []}).encode("utf-8"),  # a list
    json.dumps({"version": 1, "built_at": AT, "pointers": {"a": "x"},
                "unrouted": {}}).encode("utf-8"),                                    # targets not a list
    json.dumps({"version": 1, "built_at": AT, "pointers": {"a": {}},
                "unrouted": {}}).encode("utf-8"),                # not a list, and nothing in it to reject
    json.dumps({"version": 1, "built_at": AT, "pointers": {"a": ["x"]},
                "unrouted": {}}).encode("utf-8"),                                    # a target not a dict
    json.dumps({"version": 1, "built_at": AT, "pointers": {}, "unrouted": {}}).encode("utf-16")])  # not UTF-8
def test_a_malformed_sidecar_names_the_rebuild(tmp_path, raw):
    """Hand-edited or re-saved (write_pointers is atomic): the boot error names the fix,
    never a bare KeyError, AttributeError or UnicodeDecodeError."""
    path = tmp_path / "route_pointers.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError, match=r"route pointer sidecar .*: rebuild it with python main\.py "
                                         r"route-pointers$"):
        RoutePointers.load(path)


# ----------------------------------------------------------------- config ----

def test_the_route_block_ships_in_both_configs():
    for name in ("config.yaml", "config.example.yaml"):
        path = ROOT / name
        if not path.exists():           # a fresh clone has no config.yaml
            continue
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert cfg["retrieval"]["route"] == {"pointers_file": "data/route_pointers.json",
                                             "router_k": 10, "per_target": 1,
                                             "max_targets": 8, "max_new": 4}, name


# ---------------------------------------------------------------- command ----

def _run(cfg_path, *argv):
    import main          # imported here: it configures logging at import, as test_bench_cli does
    args = main.build_parser().parse_args(["--config", str(cfg_path), "route-pointers", *argv])
    args.func(args)


GENERATED_CANVAS = _row("g1", f"[Canvas: m]\nSource: {HUYEN} pp. 59-63", source_file="Maps/m.canvas",
                        file_type="canvas", canvas_node_id="n1")


def test_route_pointers_without_a_route_block_raises(tmp_path):
    cfg_path, _ = _world(tmp_path, [GENERATED_CANVAS], route=False)
    with pytest.raises(ValueError, match=r"^route-pointers needs retrieval\.route\.pointers_file in "
                                         r"config\.yaml \(the retrieval\.route block\), or --out PATH$"):
        _run(cfg_path)
    assert sorted(p.name for p in (tmp_path / "data").iterdir()) == ["chunks.jsonl"]


def test_out_writes_where_it_is_told(tmp_path, monkeypatch, capsys):
    """Task 16 Step 3: a scratch sidecar from a config with no route block. A relative
    --out is from the cwd, not from the config's folder."""
    proj = tmp_path / "proj"
    proj.mkdir()
    cfg_path, _ = _world(proj, [GENERATED_CANVAS,
                                _row("g2", "[Canvas: m]\nNo source", source_file="Maps/m.canvas",
                                     file_type="canvas", canvas_node_id="n2")], route=False)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    _run(cfg_path, "--out", "scratch/route_pointers.json")
    p = RoutePointers.load(cwd / "scratch" / "route_pointers.json")
    assert p.targets("g1") == [_t(HUYEN, 59, 63)] and p.targets("g2") is None
    assert sorted(x.name for x in (proj / "data").iterdir()) == ["chunks.jsonl"]
    out = capsys.readouterr().out
    assert "1 routed" in out and "1 unrouted" in out and "no Source line" in out


@pytest.mark.parametrize("name", ["x_chunks.jsonl", "chunks.jsonl", "parents_md.jsonl",
                                  "HYPE_QUESTIONS.JSONL"])
def test_route_pointers_refuses_a_chunk_file_or_sidecar_out(tmp_path, name):
    cfg_path, _ = _world(tmp_path, [GENERATED_CANVAS])
    before = (tmp_path / "data" / "chunks.jsonl").read_bytes()
    with pytest.raises(ValueError, match="pointer sidecar"):
        _run(cfg_path, "--out", str(tmp_path / "data" / name))
    assert (tmp_path / "data" / "chunks.jsonl").read_bytes() == before
    assert sorted(p.name for p in (tmp_path / "data").iterdir()) == ["chunks.jsonl"]


@pytest.mark.parametrize("name", ["chunks.jsonl.", "chunks.jsonl ", "chunks.jsonl::$DATA"])
def test_route_pointers_refuses_an_alias_spelling_of_a_chunk_file(tmp_path, name):
    """NTFS opens these as chunks.jsonl itself, and none of them ends in .jsonl."""
    cfg_path, _ = _world(tmp_path, [GENERATED_CANVAS])
    before = (tmp_path / "data" / "chunks.jsonl").read_bytes()
    with pytest.raises(ValueError, match="name its file plainly"):
        _run(cfg_path, "--out", str(tmp_path / "data" / name))
    assert (tmp_path / "data" / "chunks.jsonl").read_bytes() == before
    assert sorted(p.name for p in (tmp_path / "data").iterdir()) == ["chunks.jsonl"]


def test_route_pointers_refuses_a_parents_sidecar_not_named_jsonl(tmp_path):
    cfg_path, _ = _world(tmp_path, [GENERATED_CANVAS])
    d = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    d["retrieval"]["parents_file"] = "data/parents.json"
    cfg_path.write_text(yaml.safe_dump(d, allow_unicode=True), encoding="utf-8")
    parents = tmp_path / "data" / "parents.json"
    parents.write_text("PARENTS", encoding="utf-8")
    with pytest.raises(ValueError, match="pointer sidecar"):
        _run(cfg_path, "--out", str(parents))
    assert parents.read_text(encoding="utf-8") == "PARENTS"


def test_route_pointers_writes_the_configured_pointers_file_by_default(tmp_path, monkeypatch):
    """The console job passes no --out: retrieval.route.pointers_file, from the config's folder."""
    (tmp_path / "proj").mkdir()
    cfg_path, _ = _world(tmp_path / "proj", [GENERATED_CANVAS])
    monkeypatch.chdir(tmp_path)
    _run(cfg_path)
    p = RoutePointers.load(tmp_path / "proj" / "data" / "route_pointers.json")
    assert p.targets("g1") == [_t(HUYEN, 59, 63)]
    assert not (tmp_path / "route_pointers.json").exists()


def test_route_pointers_refuses_a_corpus_with_no_chunk_file(tmp_path):
    """The worktree's config resolves data/ to nothing: that must not write an empty sidecar."""
    cfg_path, _ = _world(tmp_path, [])
    (tmp_path / "data" / "chunks.jsonl").unlink()
    with pytest.raises(ValueError, match="no chunk JSONL"):
        _run(cfg_path, "--out", str(tmp_path / "x.json"))
    assert not (tmp_path / "x.json").exists()


def test_route_pointers_refuses_a_missing_vault(tmp_path):
    """Every note would come out 'note file not found' and the job would still report done."""
    cfg_path, _ = _world(tmp_path, [GENERATED_CANVAS])
    (tmp_path / "vault").rmdir()
    with pytest.raises(ValueError, match="parser.vault_path .* is not a folder"):
        _run(cfg_path, "--out", str(tmp_path / "x.json"))
    assert not (tmp_path / "x.json").exists()


@pytest.mark.parametrize("configured,out", [(None, "data/route_pointers.json"),
                                            ("data/vaults/b/ptrs.json", "elsewhere/PTRS.JSON")])
def test_no_lane_may_write_the_pointer_sidecar(tmp_path, configured, out):
    from src.utils import lane_guard
    cfg = Config({"retrieval": {"route": {"pointers_file": configured}}} if configured else {}, tmp_path)
    for lane in ("pdf", "markdown", "ocrstudio"):
        with pytest.raises(ValueError, match="route pointer sidecar"):
            lane_guard.check_output(cfg, lane, out, scoped=False)


# ------------------------------------------------------- job kind, console ----

def test_route_pointers_job_kind_argv_and_tier(management_module):
    assert management_module._build_argv("route_pointers", {}) == [sys.executable, "main.py", "route-pointers"]
    assert management_module.api_schema()["job_kinds"]["route_pointers"]["permission"] == "mutating"


def test_route_pointers_triggers_the_auto_restart(management_module, monkeypatch, tmp_path):
    """:8051 loads the sidecar at startup (Task 12), so a finished build restarts it
    when webui.auto_restart_rag is on, as graph_embed does."""
    m = management_module
    (tmp_path / "jobs").mkdir()
    monkeypatch.setattr(m, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(m, "_JOBS", {})
    monkeypatch.setattr(m, "_ORDER", [])
    monkeypatch.setattr(m, "load_config",
                        lambda *a, **k: Config({"webui": {"auto_restart_rag": True}}, tmp_path))
    restarts = []
    monkeypatch.setattr(m, "_restart_rag_api",
                        lambda: restarts.append(1) or {"port": 8051, "killed_pid": 1, "new_pid": 2})
    jobs = []
    for kind in ("recalibrate", "route_pointers"):       # a kind that never restarts, then this one
        job = m.Job(id=uuid.uuid4().hex[:10], kind=kind, argv=[sys.executable, "-c", "pass"], params={})
        job.log_file = str(tmp_path / "jobs" / f"{job.id}.log")
        m._JOBS[job.id] = job
        m._ORDER.append(job.id)
        jobs.append(job)
    for job in jobs:
        m._QUEUE.put(job.id)                             # the real worker, one job at a time
    log = Path(jobs[1].log_file)
    deadline = time.time() + 60
    while time.time() < deadline and "auto-restarted" not in (
            log.read_text(encoding="utf-8") if log.exists() else ""):
        time.sleep(0.05)
    assert [j.status for j in jobs] == ["done", "done"]
    assert restarts == [1]                               # route_pointers restarted :8051, recalibrate not
    assert "auto-restarted :8051" in log.read_text(encoding="utf-8")
    assert "auto-restart" not in Path(jobs[0].log_file).read_text(encoding="utf-8")


def test_the_pointer_sidecar_travels_with_the_vault(management_module):
    assert "retrieval.route.pointers_file" in management_module.VAULT_KEYS


def test_a_vault_switch_gives_each_vault_its_own_pointer_sidecar(management_module, monkeypatch, tmp_path):
    """A new vault is scaffolded its own sidecar path, and a vault registered before the key
    existed is backfilled beside its own chunks: never the previous vault's file."""
    m = management_module
    monkeypatch.setattr(m, "ROOT", tmp_path)               # the fixture's example config.yaml
    registry = tmp_path / "data" / ".vault_registry.json"
    monkeypatch.setattr(m, "_VAULT_REGISTRY", registry)
    new, old = tmp_path / "Vault B", tmp_path / "Old Vault"
    new.mkdir()
    old.mkdir()
    registry.parent.mkdir()
    registry.write_text(json.dumps({"vaults": [{"path": str(old), "label": "old", "settings": {
        "parser.vault_path": old.as_posix(), "paths.chunks_file": "data/vaults/old/chunks.jsonl"}}]}),
        encoding="utf-8")

    def pointers_file():
        disk = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
        return disk["retrieval"]["route"]["pointers_file"]

    assert m.vaults_switch(m.VaultSwitchIn(path=str(new)))["scaffolded"] is True
    assert pointers_file() == "data/vaults/vault_b/route_pointers.json"
    assert m.vaults_switch(m.VaultSwitchIn(path=str(old)))["scaffolded"] is False
    assert pointers_file() == "data/vaults/old/route_pointers.json"


def test_the_console_offers_the_job_kind_with_its_hint():
    html = (ROOT / "webui" / "index.html").read_text(encoding="utf-8")
    assert '<option value="route_pointers">route-pointers</option>' in html
    assert ('route_pointers:"Build the map→source pointers that route mode uses (no store write). '
            ':8051 loads them at startup: restart it afterwards (automatic when auto restart is on)."'
            in html)
