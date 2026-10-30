"""Slice B task 3: course labels, the __SKIP__ guard, frontmatter tags and provenance.

THE BUGS THESE PIN (contract 1.2 and 11 #3-#6):
  * a folder named "AI Engineering" was read as the "AI" course, because the md
    lane ran the heading-pattern abbreviations BEFORE the exact folder map;
  * `Data Science (Books)` was a DOMAIN_MAP key with no folder-map key;
  * the "__SKIP__" sentinel was stored as a course label on any note whose first
    section heading was a todo/plan heading;
  * frontmatter `tags:` was dropped unless it was a list, never normalised, and
    CSS colours in the body (#e67e22) became tags;
  * nothing said whether a chunk came from the author's own note or from a source.

Every note section is >= 200 characters (the parser's minimum chunk size) and
none of its headings names a course, so a label can only come from the path or
the frontmatter.
"""
import json
import re
import types
from pathlib import Path

import pytest

from src.ingestion import web_import
from src.ingestion.obsidian_parser import (
    ObsidianParser,
    detect_course_from_path,
    resolve_course_metadata,
)
from src.ingestion.pdf_loader import PDFLoader

BODY = ("# Gardening plans\n\n"
        + "Plain prose about the garden, repeated to clear the minimum chunk size. " * 4
        + "\n")


def _note(root: Path, rel: str, body: str = BODY, fm: str = "") -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(fm + body, encoding="utf-8")
    return path


def _docs(root: Path, rel: str, body: str = BODY, fm: str = ""):
    note = _note(root, rel, body, fm)
    docs = ObsidianParser(str(root)).parse_file(note)
    # An empty result would make every "not X" assertion below pass vacuously.
    assert docs, "the fixture note produced no chunks"
    return docs


def _md_rule():
    # Imported here so a missing function fails the tests that need it, not the module.
    from src.ingestion.obsidian_parser import detect_md_course_from_path
    return detect_md_course_from_path


# ------------------------------------------------------------------- courses

def test_ai_engineering_folder_is_its_own_course(tmp_path):
    docs = _docs(tmp_path, "00 – AUA_DS/Current Courses/AI Engineering/Lesson.md")

    assert {d.metadata["course_name"] for d in docs} == {"AI Engineering"}
    assert {d.metadata["domain"] for d in docs} == {"ml"}
    assert all("Course: AI Engineering" in d.text for d in docs)


def test_tech_books_ai_engineering_bucket_is_the_same_course(tmp_path):
    docs = _docs(tmp_path, "00 – AUA_DS/Other/Tech Books/AI Engineering & LLMs/x.md")

    assert {d.metadata["course_name"] for d in docs} == {"AI Engineering"}


def test_an_exact_folder_beats_a_shallower_abbreviation(tmp_path):
    # Today the parent's `^python\b` heading pattern wins: "Programming & Engineering".
    docs = _docs(tmp_path, "Python for Data Science/Machine Learning/x.md")

    assert {d.metadata["course_name"] for d in docs} == {"Machine Learning & AI"}


def test_python_development_under_a_parent_is_an_intended_change(tmp_path):
    """Pins the order swap's measured reach (plan section 2): the exact child
    folder now beats the parent's abbreviation, so these notes get a new label
    (and, at their next ingest, new ids: that ingest must be a REPLACE)."""
    docs = _docs(tmp_path, "Programming for Data Science/python development/x.md")

    assert {d.metadata["course_name"] for d in docs} == {"Python Development"}


@pytest.mark.parametrize("child", ["AI (A+)", "TSF", "Machine Learning",
                                   "python development", "Homework", "DS 227", None])
def test_prgoramming_rename_is_course_diff_clean(child):
    """The folder rename needs no map key: both spellings already resolve to the
    same course under both rules, for every child and for the parent alone."""
    md_rule = _md_rule()
    base = ["00 – AUA_DS", "01 – Passed"]

    def under(spelling):
        return base + [spelling] + ([child] if child else []) + ["x.md"]

    old, new = under("Prgoramming for Data Science"), under("Programming for Data Science")
    assert md_rule(old) == md_rule(new)
    assert detect_course_from_path(old) == detect_course_from_path(new)


@pytest.mark.parametrize("rel,course", [
    ("00 – AUA_DS/Current Courses/NLP/NLP L7.md", "Natural Language Processing"),
    ("00 – AUA_DS/01 – Passed/Prgoramming for Data Science/AI (A+)/lab.md", "Intro to AI"),
    ("Numerical Analysis/x.md", "Numerical Methods"),
    ("Misc/RL L3 notes.md", "Reinforcement Learning"),     # the filename stem
])
def test_existing_labels_hold(tmp_path, rel, course):
    docs = _docs(tmp_path, rel)

    assert {d.metadata["course_name"] for d in docs} == {course}


def test_frontmatter_course_still_wins(tmp_path):
    docs = _docs(tmp_path, "00 – AUA_DS/Current Courses/NLP/NLP L7.md",
                 fm="---\ncourse: Ethics\n---\n")

    assert {d.metadata["course_name"] for d in docs} == {"Ethics"}


@pytest.mark.parametrize("rel", [
    "00 – AUA_DS/Current Courses/AI Engineering/Lesson.md",
    "00 – AUA_DS/Other/Tech Books/AI Engineering & LLMs/x.md",
    "00 – AUA_DS/Other/Tech Books/Data Science (Books)/n.md",
    "Python for Data Science/Machine Learning/x.md",
    "Programming for Data Science/python development/x.md",
    "00 – AUA_DS/Current Courses/NLP/NLP L7.md",
    "00 – AUA_DS/01 – Passed/Prgoramming for Data Science/AI (A+)/lab.md",
    "Numerical Analysis/x.md",
    "Misc/RL L3 notes.md",
])
def test_detect_md_course_from_path_is_the_parsers_rule(tmp_path, rel):
    """recalibrate_courses.py and the remaster course-diff gate call the module
    function; the parser must be using the very same rule."""
    docs = _docs(tmp_path, rel)

    got = resolve_course_metadata(_md_rule()(Path(rel).parts))["course_name"]
    assert got == docs[0].metadata["course_name"]


def test_data_science_books_folder_is_labelled(tmp_path):
    parts = ["00 – AUA_DS", "Other", "Tech Books", "Data Science (Books)", "b.pdf"]
    assert detect_course_from_path(parts)["course_name"] == "Data Science (Books)"

    loader = PDFLoader.__new__(PDFLoader)       # _detect_course reads only vault_path
    loader.vault_path = tmp_path
    got = loader._detect_course(tmp_path.joinpath(*parts))
    assert got["course_name"] == "Data Science (Books)"
    assert got["domain"] == "ml"


def test_a_skip_heading_is_never_stored_as_a_course(tmp_path):
    section = "Plain prose about the garden, repeated to clear the minimum chunk size. " * 4
    body = f"# Todo\n\n{section}\n\n# RL L3 policy gradients\n\n{section}\n"
    docs = _docs(tmp_path, "Misc/plan.md", body=body)

    assert len(docs) == 2
    assert "__SKIP__" not in json.dumps([d.metadata for d in docs])
    assert all("__SKIP__" not in d.text for d in docs)
    # an uncoursed md chunk carries no course key at all (see
    # test_vault_relative_classification)
    assert docs[0].metadata.get("course_name") in (None, "unknown")
    assert docs[1].metadata["course_name"] == "Reinforcement Learning"


# ---------------------------------------------------------------------- tags

@pytest.mark.parametrize("fm_line,inline,expected", [
    ("tags: ML\n", "", ["ml"]),
    ('tags: "#NLP, Transformers"\n', "", ["nlp", "transformers"]),
    ('tags: ["#Course/NLP", "Kind/Lecture"]\n', "", ["course/nlp", "kind/lecture"]),
    ("tags:\n", "Notes on #topic/rag.", ["topic/rag"]),                 # null: inline only
    ("tags: [ML]\n", "Also #zeta and #alpha.", ["alpha", "ml", "zeta"]),  # sorted union
])
def test_frontmatter_tags_accept_scalar_list_and_hash(tmp_path, fm_line, inline, expected):
    docs = _docs(tmp_path, "Misc/n.md", body=BODY + inline + "\n",
                 fm=f"---\n{fm_line}---\n")

    assert {tuple(d.metadata["tags"]) for d in docs} == {tuple(expected)}


def test_css_hex_colours_are_not_tags(tmp_path):
    line = "color: #e67e22 and #topic/rag and #f8f9fa #ff0000 #f39c12 and #decade\n"
    docs = _docs(tmp_path, "Misc/n.md", body=BODY + line)

    # `decade` is valid hex but has no digit: a real tag must not be eaten
    assert docs[0].metadata["tags"] == ["decade", "topic/rag"]


def test_short_hex_shaped_tags_are_kept(tmp_path):
    """REGRESSION (review B2-6): the 3- and 4-character hex forms ate real tags such as
    #e2e, #b2b, #ec2 and #db2. Every measured colour tag was 6 characters, so only the
    6- and 8-character forms are dropped."""
    line = "Topics: #e2e #b2b #ec2 #db2 #a11y #nlp #fff #e67e22 #ff0000cc\n"
    docs = _docs(tmp_path, "Misc/n.md", body=BODY + line)

    assert docs[0].metadata["tags"] == ["a11y", "b2b", "db2", "e2e", "ec2", "fff", "nlp"]


# ---------------------------------------------------------------- provenance

@pytest.mark.parametrize("rel,fm,expected", [
    ("Misc/a.md", "---\ngenerated: true\n---\n", "generated"),
    ("Misc/a.md", "", "authored"),
    ("Misc/a.md", '---\ngenerated: "true"\n---\n', "authored"),     # strict `is True`
    ("Daily Notes/2026-05-28.md", "", "authored"),
    ("Misc/a.md", "---\nprovenance: source\n---\n", "source"),
    ("Misc/a.md", "---\nprovenance: authored\ngenerated: true\n---\n", "authored"),
])
def test_generated_frontmatter_sets_provenance(tmp_path, rel, fm, expected):
    docs = _docs(tmp_path, rel, fm=fm)

    assert {d.metadata["provenance"] for d in docs} == {expected}


def test_a_bad_provenance_value_is_a_parse_error_naming_the_file(tmp_path):
    bad = _note(tmp_path, "Misc/bad.md", fm="---\nprovenance: Source\n---\n")
    _note(tmp_path, "Misc/good.md")

    with pytest.raises(ValueError) as err:
        ObsidianParser(str(tmp_path)).parse_file(bad)
    assert str(Path("Misc") / "bad.md") in str(err.value)
    assert "authored, source, generated" in str(err.value)


def test_a_bad_provenance_value_stops_the_whole_parse_instead_of_dropping_the_note(tmp_path):
    """parse_all swallows a per-note error and carries on, and both CLIs then write what
    survived and exit 0: the note would vanish from the output while the job went green
    (and a REPLACE would delete its rows and add none). A wrong provenance is the
    author's to fix, so it stops the run, naming EVERY such note at once."""
    _note(tmp_path, "Misc/bad.md", fm="---\nprovenance: Source\n---\n")
    _note(tmp_path, "Misc/worse.md", fm="---\nprovenance: mine\n---\n")
    _note(tmp_path, "Misc/good.md")

    with pytest.raises(ValueError) as err:
        ObsidianParser(str(tmp_path)).parse_all(verbose=False)

    msg = str(err.value)
    assert str(Path("Misc") / "bad.md") in msg and str(Path("Misc") / "worse.md") in msg
    assert "good.md" not in msg and "authored, source, generated" in msg


def test_ingest_md_writes_nothing_when_a_note_has_a_bad_provenance(tmp_path):
    import main
    vault = tmp_path / "vault"
    _note(vault, "scope/bad.md", fm="---\nprovenance: Source\n---\n")
    _note(vault, "scope/good.md")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"parser:\n  vault_path: {json.dumps(str(vault))}\n  chunking: none\n",
                   encoding="utf-8")
    out = tmp_path / "scoped_chunks.jsonl"
    args = main.build_parser().parse_args(
        ["--config", str(cfg), "ingest-md", "--include-path", "scope", "--output", str(out)])

    with pytest.raises(ValueError, match="bad.md"):
        args.func(args)

    assert not out.exists()                      # no output a REPLACE could trust


def test_frontmatter_never_changes_doc_ids(tmp_path):
    plain = _docs(tmp_path / "plain", "Misc/n.md")
    marked = _docs(tmp_path / "marked", "Misc/n.md",
                   fm="---\ntags: ml\ngenerated: true\nprovenance: source\n---\n")

    assert [d.doc_id for d in plain] == [d.doc_id for d in marked]


def test_pdf_chunks_are_marked_source(tmp_path):
    loader = PDFLoader(vault_path=tmp_path, output_file=tmp_path / "u.jsonl",
                       scratch_dir=tmp_path, ocr_enabled=False)
    course = {"course_code": "unknown", "course_name": "unknown", "domain": "general"}

    chunks = loader._chunks_from_md(
        [{"text": "x" * 300, "metadata": {"page_number": 1}}],
        "b.pdf", tmp_path / "b.pdf", course, {})

    assert chunks
    assert all(c.metadata["provenance"] == "source" for c in chunks)
    assert all("provenance" not in c.text for c in chunks)


def test_web_and_office_imports_are_staged_as_source(tmp_path, monkeypatch):
    page = "# Page\n\n" + "body " * 60

    class _MarkItDown:
        def convert(self, path):
            return types.SimpleNamespace(text_content=page)

    monkeypatch.setattr(web_import, "_markitdown", lambda: _MarkItDown())
    monkeypatch.setattr(web_import, "fetch_one",
                        lambda url, backend: (page.encode("utf-8"), "text/html"))
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "a.txt").write_text(page, encoding="utf-8")

    fetched = web_import.fetch_urls(["https://example.org/a"], tmp_path / "conv")
    converted = web_import.convert_files(["a.txt"], inbox, tmp_path / "conv")
    assert fetched[0]["ok"] is True and converted[0]["ok"] is True, (fetched, converted)

    vault = tmp_path / "vault"
    vault.mkdir()
    for name in (fetched[0]["file"], converted[0]["output"]):
        staged = (tmp_path / "conv" / name).read_text(encoding="utf-8")
        assert staged.startswith("---\nprovenance: source\n---\n"), name
        (vault / name).write_text(staged, encoding="utf-8")

    docs = ObsidianParser(str(vault)).parse_all(verbose=False)
    assert docs
    assert {d.metadata["provenance"] for d in docs} == {"source"}
    assert all("provenance" not in d.text for d in docs)


def test_the_console_preview_hides_the_staged_frontmatter():
    """REGRESSION (review B2-7): previewFile fed the raw staged file to marked.parse, which
    has no frontmatter support, so the provenance block rendered as a rule and a
    "provenance: source" heading above the page title. The page needs a browser, so (like
    the other markup tests) the strip regex is lifted out of the page and run here."""
    html = (Path(__file__).resolve().parents[1] / "webui" / "index.html").read_text(encoding="utf-8")
    fn = html[html.index("async function previewFile("):]
    fn = fn[:fn.index("\n}\n")]
    found = re.search(r'text\.replace\(/(.+?)/,""\)', fn)
    assert found and "marked.parse(body)" in fn and "marked.parse(text)" not in fn
    strip = re.compile(found.group(1))
    for nl in ("\n", "\r\n"):
        staged = web_import._SOURCE_FRONTMATTER.replace("\n", nl) + "# Page" + nl + nl + "body"
        assert strip.sub("", staged, count=1).startswith("# Page"), repr(nl)
    plain = "# Page\n\n---\n\nmore\n---\n"                  # a rule further down is not frontmatter
    assert strip.sub("", plain, count=1) == plain
