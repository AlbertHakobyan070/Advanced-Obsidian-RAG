import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.ingestion.canvas_edit import merge_canvas
from src.ingestion.canvas_loader import SOURCE_LINE_RE, decode_canvas_edges, encode_canvas_edge
from src.knowledge_map.assemble import (
    BODY_MAX, apply_caps, build_map, canvas_name, merge_nodes, node_body, render_canvas)
from src.knowledge_map.graph import Edge, Node, edge_id, node_id
from src.knowledge_map.scout import Scout, Section
from src.knowledge_map.templates import Template

LAYERS = ("prerequisite-of", "leads-to", "precedes")
PDF = "Books/Designing Data Apps.pdf"
MARKERS = ("(generated)",)


def mk(title, importance=3, section="sec-1", pages=(151,), kind="concept"):
    q = f"verbatim quote about {title} spanning enough words here"
    return Node(node_id(kind, title), kind, title, f"{title} is a thing. It matters.", section, "u1",
                importance, [q], [(q, pages)], "verified")


def edge(a, b, type="contrasts-with", status="stated"):
    return Edge(edge_id(a.id, b.id, type), a.id, b.id, type, "an edge quote of several words", (152,), status)


def scout(n_sections=2):
    secs = [Section(f"sec-{i}", f"Section {i}", 140 + 10 * i, 149 + 10 * i) for i in range(1, n_sections + 1)]
    return Scout(Path("x.pdf"), PDF, "abc", "Designing Data Apps", "Chapter 3: Data Models?", 140, 140 + 10 * n_sections,
                 {}, secs, [], {}, {}, [], True)


TEMPLATE = Template("concept", "Concepts and how they relate.", ("concept",), ("contrasts-with",), LAYERS, "g")


def test_merge_duplicate_titles():
    a, b = mk("Document Model", 2), mk("document  model", 5, section="sec-2")
    b.quotes = ["second quote with enough words in it", "third quote with enough words in it", "fourth quote words here again", "fifth"]
    out = merge_nodes([a, b, mk("Other")])
    assert [n.title for n in out] == ["Document Model", "Other"]
    assert out[0].importance == 5 and out[0].section == "sec-1"
    assert len(out[0].quotes) == 3 and out[0].quotes[0] == a.quotes[0]


def test_node_cap_drops_lowest_importance():
    nodes = [mk("A", 5), mk("B", 1), mk("C", 3), mk("D", 1, section="sec-2")]
    e = [edge(nodes[0], nodes[1]), edge(nodes[0], nodes[2])]
    kept, edges, rejects = apply_caps(nodes, e, SimpleNamespace(node_cap=2, fanout_max=8))
    assert [n.title for n in kept] == ["A", "C"]
    assert [r["title"] for r in rejects] == ["B", "D"] and {r["reason"] for r in rejects} == {"cap"}
    assert edges == [e[1]]


def test_fanout_prefers_stated():
    hub = mk("Hub", 5)
    others = [mk(f"N{i}", 1 + i) for i in range(4)]
    stated = [edge(hub, others[0]), edge(hub, others[1])]
    inferred = [edge(hub, others[2], status="inferred"), edge(hub, others[3], status="inferred")]
    kept, edges, rejects = apply_caps([hub, *others], inferred + stated, SimpleNamespace(node_cap=40, fanout_max=3))
    assert {e.id for e in edges} == {stated[0].id, stated[1].id, inferred[1].id}   # higher-importance inferred wins
    assert [r["id"] for r in rejects] == [inferred[0].id] and rejects[0]["reason"] == "fanout"


def test_structural_edges_no_orphans():
    a, b = mk("A"), mk("B", section="sec-2")
    m = build_map(scout(), TEMPLATE, "because", [a, b], [edge(a, b)], MARKERS)
    linked = {e["source"] for e in m["edges"]} | {e["target"] for e in m["edges"]}
    assert {n["id"] for n in m["nodes"]} <= linked
    part_of = {e["source"]: e["target"] for e in m["edges"] if e["type"] == "part-of"}
    assert part_of[a.id] == "sec-1" and part_of[b.id] == "sec-2" and part_of["sec-1"] == "chapter"
    assert {e["status"] for e in m["edges"] if e["type"] == "part-of"} == {"structural"}
    assert m["nodes"][0]["id"] == "chapter" and m["nodes"][0]["source_pages"] == [[140, 160]]
    n = next(n for n in m["nodes"] if n["id"] == a.id)
    assert n["source_pages"] == [[151, 151]] and n["quotes"][0]["pages"] == [151]
    assert m["canvas"].endswith("(generated).canvas") and m["source"]["file"] == PDF
    json.dumps(m)


def test_build_map_rejects_unlocated_node_and_dangling_edge():
    a = mk("A")
    a.located = []
    with pytest.raises(ValueError, match="no located quote"):
        build_map(scout(), TEMPLATE, "r", [a], [], MARKERS)
    b = mk("B")
    with pytest.raises(ValueError, match="not in the graph"):
        build_map(scout(), TEMPLATE, "r", [b], [edge(b, mk("Ghost"))], MARKERS)


def test_source_line_parses():
    m = build_map(scout(), TEMPLATE, "r", [mk("A", pages=(151, 152, 153))], [], MARKERS)
    node = next(n for n in m["nodes"] if n["id"].startswith("concept-"))
    node["source_pages"] = [[151, 151], [152, 153]]
    last = node_body(node, PDF).splitlines()[-1]
    mm = SOURCE_LINE_RE.fullmatch(last)
    assert mm and mm.group("path") == PDF and mm.group("pages") == "151, 152-153"


def test_body_limit_keeps_source_line():
    node = {"id": "concept-x", "title": "T" * 200, "summary": "Sentence one is here. " * 200,
            "quotes": [{"text": "word " * 25, "pages": [1]} for _ in range(3)], "source_pages": [[1, 2]]}
    body = node_body(node, PDF)
    assert len(body) <= BODY_MAX
    assert len(body.splitlines()[0]) <= 80
    assert SOURCE_LINE_RE.fullmatch(body.splitlines()[-1])
    assert body.count("> ") == 3 and body.splitlines()[1].count(".") <= 3


def test_body_drops_later_quotes_last():
    node = {"id": "concept-x", "title": "T", "summary": "Short.",
            "quotes": [{"text": "word " * 29, "pages": [1]} for _ in range(3)], "source_pages": [[1, 1]]}
    node["quotes"] = [{"text": f"q{i} " + "word " * 29, "pages": [1]} for i in range(3)]
    assert node_body(node, PDF).count("> ") == 3          # fits: nothing dropped
    node["quotes"] = [{"text": f"q{i} " + "w" * 700, "pages": [1]} for i in range(3)]
    body = node_body(node, PDF)
    assert len(body) <= BODY_MAX and body.count("> ") == 2 and "q0 " in body and "q2 " not in body
    assert body.splitlines()[-1].startswith("Source: ")


def test_canvas_name_is_windows_safe():
    name = canvas_name('Designing <Data> "Apps"', "Chapter 3: Data Models?", MARKERS)
    assert not set('<>:"/\\|?*') & set(name)
    assert name == "Designing -Data- -Apps- - Chapter 3- Data Models- (generated).canvas"
    long = canvas_name("B" * 300, "C" * 300, MARKERS)
    assert len(long) <= 150 and long.endswith(" (generated).canvas")


def test_canvas_name_checks_the_configured_markers():
    """M1: the check uses canvas.generated_markers, so a marker list that would not mark the
    name generated (the canvas lane would treat the map as authored) is refused."""
    with pytest.raises(ValueError, match="canvas.generated_markers"):
        canvas_name("Book", "Ch", ("(ai-made)",))
    assert canvas_name("Book", "Ch", ("(ai-made)", "(generated)")).endswith(" (generated).canvas")


def _big_map():
    nodes, edges = [], []
    for s in range(1, 5):
        sec = [mk(f"S{s} N{i}", importance=1 + i % 5, section=f"sec-{s}") for i in range(7)]
        nodes += sec
        edges += [edge(sec[i], sec[i + 1], "leads-to") for i in range(6)]
        edges.append(edge(sec[6], sec[0], "leads-to"))                   # a cycle
    edges.append(edge(nodes[0], nodes[8], "contrasts-with", "inferred"))   # across sections
    return build_map(scout(4), TEMPLATE, "r", nodes, edges, MARKERS)


def test_nodes_inside_groups_no_overlap():
    c = render_canvas(_big_map(), LAYERS)
    groups = {n["id"]: n for n in c["nodes"] if n["type"] == "group"}
    assert len(groups) == 4
    texts = [n for n in c["nodes"] if n["type"] == "text" and n["id"] != "chapter"]
    assert len(texts) == 28
    sec_of = {n["id"]: n["section"] for n in _big_map()["nodes"] if "section" in n}

    def box(n):
        return n["x"], n["y"], n["x"] + n["width"], n["y"] + n["height"]

    def overlap(a, b):
        return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]

    for n in texts:
        g = box(groups["group-" + sec_of[n["id"]]])
        b = box(n)
        assert g[0] <= b[0] and g[1] <= b[1] and b[2] <= g[2] and b[3] <= g[3]
    boxes = [box(n) for n in c["nodes"] if n["type"] != "group"]
    boxes_g = [box(g) for g in groups.values()]
    for i, a in enumerate(boxes):
        assert not any(overlap(a, b) for b in boxes[i + 1:])
        assert not any(overlap(a, g) for g in boxes_g if not (g[0] <= a[0] and a[2] <= g[2] and g[1] <= a[1] and a[3] <= g[3]))
    for i, a in enumerate(boxes_g):
        assert not any(overlap(a, b) for b in boxes_g[i + 1:])
    assert sorted({g["x"] for g in groups.values()}) == [0, 1 * (groups["group-sec-1"]["width"] + 40), 2 * (groups["group-sec-1"]["width"] + 40)]


def test_layers_follow_layer_edges():
    a, b, c = mk("A", 1), mk("B", 5), mk("C", 3)
    m = build_map(scout(1), TEMPLATE, "r", [a, b, c], [edge(a, b, "leads-to"), edge(b, c, "leads-to")], MARKERS)
    ys = {n["id"]: n["y"] for n in render_canvas(m, LAYERS)["nodes"] if n["type"] == "text"}
    assert ys[a.id] < ys[b.id] < ys[c.id]


def test_layout_deterministic():
    m = _big_map()
    assert json.dumps(render_canvas(m, LAYERS)) == json.dumps(render_canvas(json.loads(json.dumps(m)), LAYERS))


def test_inferred_label_survives_edge_escape():
    enc = encode_canvas_edge("n", "->", "leads-to (inferred)")
    assert decode_canvas_edges([enc]) == [{"neighbor": "n", "direction": "->", "label": "leads-to (inferred)"}]
    labels = {e["label"] for e in render_canvas(_big_map(), LAYERS)["edges"]}
    assert "contrasts-with (inferred)" in labels and "leads-to" in labels


def test_canvas_validates():
    c = render_canvas(_big_map(), LAYERS)
    assert merge_canvas(c, c) == c
    types = [n["type"] for n in c["nodes"]]
    assert types.count("file") == 1 and types.count("group") == 4
    f = next(n for n in c["nodes"] if n["type"] == "file")
    assert f["file"] == PDF and f["subpath"] == "#page=140"
    ids = {n["id"] for n in c["nodes"]}
    assert all(e["fromNode"] in ids and e["toNode"] in ids for e in c["edges"])
    chapter = next(n for n in c["nodes"] if n["id"] == "chapter")
    assert SOURCE_LINE_RE.fullmatch(chapter["text"].splitlines()[-1])


def test_merge_cross_kind_same_title_is_one_node():
    a, b = mk("Document Model", 2, kind="concept"), mk("Document-Model", 4, kind="term", section="sec-2")
    out = merge_nodes([a, b])
    assert len(out) == 1 and (out[0].id, out[0].kind, out[0].section) == (a.id, "concept", "sec-1")
    assert out[0].importance == 4 and len(out[0].quotes) == 2 and len(out[0].located) == 2
    assert len({n.id for n in merge_nodes([a, b, mk("Other")])}) == 2


def test_caps_cut_node_edges_go_quietly_but_unknown_endpoint_raises():
    a, b = mk("A", 5), mk("B", 1)
    kept, edges, rejects = apply_caps([a, b], [edge(a, b)], SimpleNamespace(node_cap=1, fanout_max=8))
    assert [n.id for n in kept] == [a.id] and edges == [] and [r["reason"] for r in rejects] == ["cap"]
    ghost = mk("Ghost")
    bad = edge(a, ghost)
    with pytest.raises(ValueError, match=f"{bad.id}.*{ghost.id}"):
        apply_caps([a, b], [bad], SimpleNamespace(node_cap=40, fanout_max=8))


def test_cap_tiebreak_uses_section_number_not_input_order():
    late, early = mk("Late", 3, section="sec-10"), mk("Early", 3, section="sec-2")
    kept, _, rejects = apply_caps([late, early], [], SimpleNamespace(node_cap=1, fanout_max=8))
    assert [n.title for n in kept] == ["Early"] and rejects[0]["title"] == "Late"


def test_build_map_duplicate_id_raises_naming_both_titles():
    a, b = mk("Document Model"), mk("Document Model")
    b.title = "Document model (again)"
    with pytest.raises(ValueError, match=r"Document Model.*Document model \(again\)"):
        build_map(scout(), TEMPLATE, "r", [a, b], [], MARKERS)
