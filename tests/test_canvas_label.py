"""src/ingestion/canvas_label.py: what "suggest label" shows a model about one edge,
how long nodes are fitted, the prompt, and how the reply is read.

Offline: the vault is a tmp folder and every model call is a fake.
"""
import copy
import json
import re
import string
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from src.ingestion.canvas_label import (
    IMAGE_TYPES, arrow, context_out, edge_context, fit_node, normalise_tag, parse_reply,
    render_prompt,
)
from src.ingestion.obsidian_parser import extract_tags
from src.llm.llm_client import IMAGE_MIMES
from src.prompts.loader import load_prompt

VOCAB = ["is-a", "part-of", "uses", "related-to"]
CANVAS = {
    "nodes": [
        {"id": "a", "type": "text", "text": "# Regression\nFits a line to points.",
         "x": 0, "y": 0, "width": 200, "height": 100},
        {"id": "b", "type": "text", "text": "Linear model", "x": 300, "y": 0,
         "width": 200, "height": 100},
        {"id": "c", "type": "text", "text": "Gradient descent", "x": 0, "y": 300,
         "width": 200, "height": 100},
        {"id": "md", "type": "file", "file": "Notes/lecture.md", "x": 300, "y": 300,
         "width": 200, "height": 100},
        {"id": "g1", "type": "group", "label": "Methods", "x": -50, "y": -50,
         "width": 600, "height": 200},
        {"id": "g2", "type": "group", "label": "Outer, big", "x": -100, "y": -100,
         "width": 2000, "height": 2000},
    ],
    "edges": [
        {"id": "e1", "fromNode": "a", "toNode": "b"},                    # the one to label
        {"id": "e2", "fromNode": "a", "toNode": "c", "label": "uses"},
        {"id": "e3", "fromNode": "md", "toNode": "b", "label": "defines"},
        {"id": "e4", "fromNode": "c", "toNode": "md", "label": "uses"},
        {"id": "e5", "fromNode": "a", "toNode": "md", "label": "explained in"},
    ],
}


def _ctx(data, edge_id, vault, **kw):
    opts = {"read_chars": 20000, "max_items": 12, "images": False,
            "max_image_bytes": 4 << 20, **kw}
    return edge_context(data, edge_id, "Maps/Stats Map.canvas", vault, **opts)


@pytest.fixture
def vault(tmp_path):
    v = tmp_path / "vault"
    (v / "Notes").mkdir(parents=True)
    (v / "Notes" / "lecture.md").write_text(
        "---\ntitle: L1\ntags: [stats]\n---\n# Lecture 1\nSee [[Other|the other note]].\n"
        "<!-- private -->\n", encoding="utf-8")
    return v


def _with_file(path, **node):
    """CANVAS plus one file node f (and edge ef: a -> f) pointing at `path`."""
    data = copy.deepcopy(CANVAS)
    data["nodes"].append({"id": "f", "type": "file", "file": path, "x": 900, "y": 900,
                          "width": 10, "height": 10, **node})
    data["edges"].append({"id": "ef", "fromNode": "a", "toNode": "f"})
    return data


def test_the_arrow_follows_from_end_and_to_end():
    assert arrow({}) == "→"                                            # JSON Canvas defaults
    assert arrow({"fromEnd": "none", "toEnd": "arrow"}) == "→"
    assert arrow({"fromEnd": "arrow"}) == "↔"
    assert arrow({"toEnd": "none"}) == "—"
    assert arrow({"fromEnd": "arrow", "toEnd": "none"}) == "←"


def test_context_has_title_groups_neighbour_labels_and_examples_capped(vault):
    ctx = _ctx(CANVAS, "e1", vault)
    assert (ctx["title"], ctx["direction"], ctx["current_label"]) == ("Stats Map", "→", "")
    assert ctx["groups"] == ["Methods", "Outer, big"]
    assert ctx["from"]["group"] == "Methods"                           # the smallest box
    assert ctx["neighbour_labels"] == {
        "from": ["→ [uses] Gradient descent", "→ [explained in] Notes/lecture.md"],
        "to": ["← [defines] Notes/lecture.md"]}
    assert ctx["examples"] == ["uses", "defines", "explained in"]      # distinct, e1 excluded
    capped = _ctx(CANVAS, "e1", vault, max_items=1)
    assert capped["groups"] == ["Methods"] and capped["examples"] == ["uses"]
    assert capped["neighbour_labels"]["from"] == ["→ [uses] Gradient descent"]


def test_a_label_that_names_no_relation_is_no_example_and_its_edge_shows_the_title_only(vault):
    # Step numbers and a bare #p: on a map labelled that way the model copied "1".
    labels = ["1", "1,2", "10,11", "#p", "#p #q", "Rule", "uses #Stats", "L13",
              "այսինքն, հետևում է, որ"]
    data = {"nodes": [{"id": "a", "type": "text", "text": "A"},
                      {"id": "b", "type": "text", "text": "B"}]
            + [{"id": f"n{i}", "type": "text", "text": f"N{i}"} for i in range(len(labels))],
            "edges": [{"id": "t", "fromNode": "a", "toNode": "b"}]
            + [{"id": f"e{i}", "fromNode": "a", "toNode": f"n{i}", "label": label}
               for i, label in enumerate(labels)]}
    ctx = _ctx(data, "t", vault)
    assert ctx["examples"] == ["Rule", "uses #Stats", "L13", "այսինքն, հետևում է, որ"]
    named = ["→ [Rule] N5", "→ [uses #Stats] N6", "→ [L13] N7", "→ [այսինքն, հետևում է, որ] N8"]
    assert ctx["neighbour_labels"]["from"] == \
        named + ["→ N0", "→ N1", "→ N2", "→ N3", "→ N4"]     # labelled first, then canvas order
    assert _ctx(data, "t", vault, max_items=6)["neighbour_labels"]["from"] == \
        named + ["→ N0", "→ N1"]


def test_a_numbered_map_keeps_its_neighbour_titles_and_no_number_reaches_the_prompt(vault):
    """Statistics Map in miniature: a hub whose edges carry step numbers and #p."""
    names = ["Descriptive", "Statistics", "Data Summary", "Methods", "Lecture notes",
             "Steps to perform", "Inferential"]
    ids = {n: n.lower().replace(" ", "-") for n in names}
    data = {"nodes": [{"id": ids[n], "type": "text", "text": f"# {n}"} for n in names],
            "edges": [{"id": "t", "fromNode": ids["Descriptive"], "toNode": ids["Statistics"]}]
            + [{"id": f"e{i}", "fromNode": ids["Descriptive"], "toNode": ids[n], "label": label}
               for i, (n, label) in enumerate([("Data Summary", "3"), ("Methods", "1,2"),
                                               ("Lecture notes", "10,11"),
                                               ("Steps to perform", "#p")])]
            + [{"id": "e9", "fromNode": ids["Inferential"], "toNode": ids["Statistics"]}]}
    ctx = _ctx(data, "t", vault)
    assert ctx["neighbour_labels"] == {
        "from": ["→ # Data Summary", "→ # Methods", "→ # Lecture notes", "→ # Steps to perform"],
        "to": ["← # Inferential"]}
    assert ctx["examples"] == []
    for side in ("from", "to"):
        fit_node(ctx[side], node_chars=2500, long_nodes="truncate", input_chars=1, complete=None)
    system, user = render_prompt(ctx, VOCAB, None)
    assert "- → # Data Summary" in user and "- ← # Inferential" in user
    assert not re.search(r"\d", user), user                        # no step number anywhere
    for marker in ("[3]", "[1,2]", "[10,11]", "[#p]", "#p"):
        assert marker not in system + user, marker
    assert _ctx(data, "t", vault, max_items=2)["neighbour_labels"]["from"] == \
        ["→ # Data Summary", "→ # Methods"]


def test_context_is_built_from_the_posted_canvas(vault):
    data = copy.deepcopy(CANVAS)
    data["nodes"][0]["text"] = "Edited in the browser, not saved"
    data["edges"].append({"id": "new", "fromNode": "b", "toNode": "a", "label": "unsaved"})
    ctx = _ctx(data, "new", vault)
    assert ctx["to"]["text"] == "Edited in the browser, not saved"
    assert ctx["current_label"] == "unsaved"
    assert not (vault / "Maps").exists()                     # the canvas file was never needed


def test_a_markdown_file_node_is_read_with_frontmatter_stripped(vault):
    ctx = _ctx(CANVAS, "e3", vault)
    md = ctx["from"]
    assert md["text"] == "# Lecture 1\nSee the other note."
    assert md["mode"] == "full" and md["kind"].startswith("markdown note")
    assert md["path"] == "Notes/lecture.md" and md["note"] is None


def test_file_nodes_outside_the_vault_or_missing_are_placeholders_and_never_read(vault, tmp_path):
    outside = tmp_path / "outside.md"
    outside.write_text("SENTINEL-OUTSIDE", encoding="utf-8")
    for path, why in [("../outside.md", "outside the vault"),
                      (str(outside), "outside the vault"),
                      ("00 - AUA_DS/renamed.md", "not found")]:
        ctx = _ctx(_with_file(path), "ef", vault)
        f = ctx["to"]
        assert f["mode"] == "placeholder" and why in f["note"] and why in f["text"]
        assert "SENTINEL" not in json.dumps(ctx, default=str)


def test_an_image_is_attached_only_when_allowed_and_under_the_cap(vault):
    (vault / "img").mkdir()
    (vault / "img" / "a.png").write_bytes(b"\x89PNG-small")
    (vault / "img" / "b.JPG").write_bytes(b"jpeg-bytes")
    off = _ctx(_with_file("img/a.png"), "ef", vault, images=False)
    assert off["to"]["image"] is None and off["to"]["mode"] == "placeholder"
    assert off["images"] == [{"node": "f", "path": "img/a.png", "sent": False,
                              "reason": "the backend is not marked vision: true"}]
    on = _ctx(_with_file("img/a.png"), "ef", vault, images=True)
    assert on["to"]["image"] == (b"\x89PNG-small", "image/png") and on["to"]["mode"] == "image"
    assert on["images"][0]["sent"] is True and on["images"][0]["reason"] is None
    big = _ctx(_with_file("img/a.png"), "ef", vault, images=True, max_image_bytes=3)
    assert big["to"]["image"] is None and "cap" in big["images"][0]["reason"]
    assert _ctx(_with_file("img/b.JPG"), "ef", vault, images=True)["to"]["image"][1] == "image/jpeg"
    assert set(IMAGE_TYPES.values()) <= set(IMAGE_MIMES)
    out = context_out(on)                                              # no bytes in the reply
    assert "image" not in out["to"] and "PNG-small" not in json.dumps(out)


def test_a_file_that_cannot_be_read_is_a_placeholder_not_an_error(vault, monkeypatch):
    (vault / "img").mkdir()
    (vault / "img" / "a.png").write_bytes(b"\x89PNG-small")
    (vault / "Notes" / "sub.canvas").write_text('{"nodes": []}', encoding="utf-8")

    def denied(p, *args, **kwargs):
        raise PermissionError(13, "Permission denied", str(p))
    monkeypatch.setattr("src.ingestion.canvas_label._read_text", denied)   # markdown and code
    monkeypatch.setattr(Path, "read_bytes", denied)                        # an image
    monkeypatch.setattr(Path, "read_text", denied)                         # a sub-canvas
    for path in ("Notes/lecture.md", "img/a.png", "Notes/sub.canvas"):
        ctx = _ctx(_with_file(path), "ef", vault, images=True)
        f = ctx["to"]
        assert (f["mode"], f["image"]) == ("placeholder", None), path
        assert "could not be read: PermissionError" in f["note"], path
        assert "could not be read" in f["text"], path
    png = _ctx(_with_file("img/a.png"), "ef", vault, images=True)["images"]
    assert len(png) == 1 and png[0]["sent"] is False and "PermissionError" in png[0]["reason"]


def test_link_pdf_and_sub_canvas_are_described_not_fetched(vault):
    (vault / "Notes" / "book.pdf").write_bytes(b"%PDF-SENTINEL")
    (vault / "Notes" / "sub.canvas").write_text(json.dumps({"nodes": [
        {"id": "s1", "type": "text", "text": "\nSub one\nmore"},
        {"id": "s2", "type": "text", "text": "Sub two"},
        {"id": "s3", "type": "file", "file": "x.md"}]}), encoding="utf-8")
    (vault / "Notes" / "broken.canvas").write_text("{not json", encoding="utf-8")
    data = copy.deepcopy(CANVAS)
    data["nodes"].append({"id": "l", "type": "link", "url": "https://example.org/x"})
    data["edges"].append({"id": "el", "fromNode": "a", "toNode": "l"})
    link = _ctx(data, "el", vault)["to"]
    assert "https://example.org/x" in link["text"] and "not fetched" in link["text"]
    assert link["mode"] == "placeholder"
    pdf = _ctx(_with_file("Notes/book.pdf"), "ef", vault)
    assert pdf["to"]["mode"] == "placeholder" and "SENTINEL" not in json.dumps(pdf, default=str)
    sub = _ctx(_with_file("Notes/sub.canvas"), "ef", vault)["to"]
    assert sub["text"] == "A canvas whose text nodes include: Sub one · Sub two"
    broken = _ctx(_with_file("Notes/broken.canvas"), "ef", vault)["to"]
    assert broken["mode"] == "placeholder" and "unreadable canvas" in broken["note"]
    group = _ctx({"nodes": CANVAS["nodes"], "edges": [{"id": "eg", "fromNode": "g1",
                                                      "toNode": "c"}]}, "eg", vault)["from"]
    assert group["text"] == 'A group box "Methods" holding: # Regression · Linear model'
    assert group["group"] == "Outer, big"                    # its own box is not its group


@pytest.mark.parametrize("data, edge_id, msg", [
    (CANVAS, "nope", "no edge 'nope'"),
    ({"nodes": CANVAS["nodes"], "edges": [{"id": "x", "fromNode": "a", "toNode": "zz"}]},
     "x", "toNode 'zz'"),
    ({"nodes": CANVAS["nodes"], "edges": [{"id": "x", "fromNode": ["a"], "toNode": "b"}]},
     "x", "fromNode"),
    ({"nodes": {}, "edges": []}, "e1", "must be"),
    ({"nodes": []}, "e1", "must be"),
])
def test_an_unknown_edge_or_dangling_endpoint_raises(vault, data, edge_id, msg):
    with pytest.raises(ValueError, match=re.escape(msg)):
        _ctx(data, edge_id, vault)


def _node(text, **kw):
    return {"id": "n", "kind": "text card", "title": "T", "text": text, "mode": "full",
            "note": None, **kw}


def test_long_content_is_truncated_or_summarised_and_says_which(vault):
    calls = []

    def complete(system, user):
        calls.append((system, user))
        return NS(text="<think>long thought</think> the gist", model="m-sum")
    short = fit_node(_node("short"), node_chars=100, long_nodes="summarize", input_chars=1000,
                     complete=complete)
    assert (short["sent"], short["mode"], calls) == ("short", "full", [])
    cut = fit_node(_node("x" * 5000), node_chars=100, long_nodes="truncate", input_chars=1000,
                   complete=complete)
    assert (len(cut["sent"]), cut["mode"], calls) == (100, "truncated", [])
    s = fit_node(_node("y" * 5000), node_chars=100, long_nodes="summarize", input_chars=1000,
                 complete=complete)
    assert (s["sent"], s["mode"], s["summary_model"]) == ("the gist", "summarized", "m-sum")
    assert len(calls) == 1 and "y" * 1000 in calls[0][1] and "y" * 1001 not in calls[0][1]
    assert "at most 100 characters" in calls[0][0] and "first 1000 characters" in s["note"]
    for empty in ("", "  ", "<think>only thinking</think>"):
        with pytest.raises(ValueError, match="came back empty"):
            fit_node(_node("z" * 500), node_chars=100, long_nodes="summarize", input_chars=1000,
                     complete=lambda sy, us, t=empty: NS(text=t, model="m"))
    with pytest.raises(ValueError, match="long_nodes"):
        fit_node(_node("z"), node_chars=100, long_nodes="drop", input_chars=1000, complete=None)
    # A file read stops at read_chars + 1 characters, and says more remained.
    (vault / "Notes" / "big.py").write_text("print(1)\n" * 50, encoding="utf-8")
    f = _ctx(_with_file("Notes/big.py"), "ef", vault, read_chars=20)["to"]
    assert len(f["text"]) == 21 and "longer than 20 characters" in f["note"]
    fit_node(f, node_chars=20, long_nodes="truncate", input_chars=20, complete=None)
    assert (len(f["sent"]), f["mode"]) == (20, "truncated")


def test_the_templates_render_with_every_placeholder(vault):
    label, summary = load_prompt("canvas_label"), load_prompt("canvas_node_summary")
    assert label["version"] == 1 and summary["version"] == 1

    def fields(t):
        return {f for _, f, _, _ in string.Formatter().parse(t) if f}
    assert fields(label["user"]) == {
        "canvas", "vocabulary", "groups", "examples", "direction", "from_kind", "from_group",
        "from_text", "to_kind", "to_group", "to_text", "from_neighbours", "to_neighbours",
        "current", "guidance"}
    assert fields(label["system"]) == set()
    assert fields(summary["system"]) == {"max_chars"}
    assert fields(summary["user"]) == {"kind", "title", "text"}
    ctx = _ctx(CANVAS, "e3", vault)
    for side in ("from", "to"):
        fit_node(ctx[side], node_chars=2500, long_nodes="truncate", input_chars=1, complete=None)
    system, user = render_prompt(ctx, VOCAB, "prefer verbs")
    assert '{"label": "...", "relation": "..."' in system               # the braces survive
    # The wording label quality rests on: a relationship, never a number or a tag,
    # and the map's labels are a style guide only.
    flat = " ".join(system.split())
    for rule in ("a short phrase of 1 to 5 words that names the relationship, read along "
                 "the arrow", "show its language, casing and register only, not what to write",
                 "A label is never a bare number, a step marker or a #tag.",
                 'It must hold read along the arrow the same way: "FROM <relation> TO" for '
                 'FROM → TO, "TO <relation> FROM" for FROM ← TO.',
                 "If only the reverse holds, for example when TO is an example of FROM, use "
                 'related-to and say why in "why".'):
        assert rule in flat, rule
    assert "Labels already on the map (their language, casing and register only):" in user
    assert "Other edges at FROM:" in user and "Other edges at TO:" in user
    for part in ("Map: Stats Map", "is-a, part-of, uses, related-to", "Methods | Outer, big",
                 "uses | explained in", "FROM → TO", 'markdown note "Notes/lecture.md"',
                 "# Lecture 1", "Linear model", "- ← [uses] Gradient descent",
                 "- ← [explained in] # Regression", "Its current label: defines",
                 "Guidance from the map's owner: prefer verbs"):
        assert part in user, part
    assert "[defines]" not in user                       # the edge's own label is not a neighbour


ANSWER = ('{"label": "part of", "relation": "part-of", "tags": ["stats"], '
          '"alternatives": ["belongs to"], "confidence": "medium", "why": "A is in B."}')


def test_parse_takes_the_last_json_object_with_a_label():
    reply = ('<think>first {x} then {"label": "a thought"}</think>\n'
             'The schema is {"label": "...", "relation": "...", "tags": ["..."]}.\n'
             'So: ' + ANSWER + "\nDone.")
    got = parse_reply(reply, VOCAB, 12)
    assert got == {"label": "part of", "relation": "part-of", "tags": ["stats"],
                   "alternatives": ["belongs to"], "confidence": "medium", "why": "A is in B.",
                   "warnings": []}
    # <think> blocks are dropped first, even one that comes after the answer.
    assert parse_reply(ANSWER + '<think>{"label": "late thought"}</think>', VOCAB, 12)[
        "label"] == "part of"


def test_parse_keeps_a_label_whose_relation_is_unknown_and_warns():
    got = parse_reply('{"label": "powers", "relation": "drives", "confidence": "sure"}', VOCAB, 12)
    assert (got["label"], got["relation"], got["confidence"]) == ("powers", None, None)
    assert any("drives" in w for w in got["warnings"])
    assert any("sure" in w for w in got["warnings"])
    for spelled in ("Part Of", "PART_OF", "part-of", " Part-Of"):
        assert parse_reply(json.dumps({"label": "x", "relation": spelled}), VOCAB, 12)[
            "relation"] == "part-of"


@pytest.mark.parametrize("reply", [
    "Let me think about {A} and {B}. The arrow runs from A to B, so the label has to read",
    '{"label": ""}', '{"relation": "part-of"}', '{"label": 3}', "",
    '<think>still going {"label": "inside the thought"', "no json at all",
])
def test_parse_rejects_a_reply_with_no_label(reply):
    with pytest.raises(ValueError, match='no JSON object with a non-empty "label"'):
        parse_reply(reply, VOCAB, 12)


def test_tags_are_normalised_to_what_the_loader_extracts():
    raw = ["#Stats", "Machine Learning", "#3d", "Վիճակագրություն", "stats", "time/series", 7]
    got = parse_reply(json.dumps({"label": "x", "tags": raw}), VOCAB, 12)
    assert got["tags"] == ["stats", "machine-learning", "time/series"]
    for t in got["tags"]:
        assert extract_tags("x #" + t) == [t]
    dropped = [w for w in got["warnings"] if w.startswith("tag ")]
    assert len(dropped) == 3 and any("3d" in w for w in dropped)
    assert any("Վիճակագրություն" in w for w in dropped)
    assert normalise_tag("#p") == "p" and normalise_tag("  ") is None
    assert parse_reply(json.dumps({"label": "x", "tags": ["a", "b", "c"]}), VOCAB, 2)["tags"] == \
        ["a", "b"]
