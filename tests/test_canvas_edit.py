"""The Graph tab's canvas editing: the merge, the writer, and the save fences.

Saving is the one console feature that writes INTO the vault, so these pin what
must never happen: a field the editor does not edit being lost, a file changed
on disk being overwritten, a write without a backup outside the vault, a write
to anything but an existing .canvas, and a half-written file.
"""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from src.ingestion.canvas_edit import dump_canvas, merge_canvas
from src.utils.config_loader import Config

CANVAS = {
    "nodes": [
        {"id": "a1", "type": "text", "text": "Alpha", "x": 0, "y": 0, "width": 250,
         "height": 60, "color": "4", "styleAttributes": {"shape": "pill"}},
        {"id": "f1", "type": "file", "file": "Notes/x.md", "subpath": "#h", "x": 300,
         "y": 0, "width": 400, "height": 400},
        {"id": "g1", "type": "group", "label": "Box", "x": -50, "y": -50, "width": 900,
         "height": 600, "background": "img.png", "backgroundStyle": "cover"},
        {"id": "l1", "type": "link", "url": "https://example.org", "x": 0, "y": 500,
         "width": 200, "height": 100},
    ],
    "edges": [
        {"id": "e1", "fromNode": "a1", "fromSide": "right", "toNode": "f1", "toSide": "left",
         "toEnd": "arrow", "color": "#ff0000", "label": "explains"},
        {"id": "e2", "fromNode": "a1", "toNode": "l1"},
    ],
    "metadata": {"version": "1.0-1.0", "frontmatter": {}},
}


def _edit(**changes):
    data = copy.deepcopy(CANVAS)
    for fn in changes.values():
        fn(data)
    return data


# ---- merge --------------------------------------------------------------------

def test_an_untouched_canvas_merges_to_itself():
    assert merge_canvas(CANVAS, copy.deepcopy(CANVAS)) == CANVAS


def test_only_editable_fields_change_everything_else_is_the_files():
    new = copy.deepcopy(CANVAS)
    new["nodes"][0].update(text="Alpha 2", x=10.4, color="1")    # colour: not editable
    del new["nodes"][0]["styleAttributes"]                          # a client dropping a key
    new["nodes"][1].pop("subpath")
    new["edges"][0].update(label="describes", color="#000000")
    out = merge_canvas(CANVAS, new)
    a1, f1 = out["nodes"][0], out["nodes"][1]
    assert a1["text"] == "Alpha 2" and a1["x"] == 10
    assert a1["color"] == "4" and a1["styleAttributes"] == {"shape": "pill"}
    assert f1["subpath"] == "#h" and out["nodes"][2]["backgroundStyle"] == "cover"
    assert out["edges"][0] == {**CANVAS["edges"][0], "label": "describes"}
    assert out["metadata"] == CANVAS["metadata"]
    assert list(a1) == list(CANVAS["nodes"][0])                     # key order kept


def test_add_and_delete_nodes_and_edges():
    new = copy.deepcopy(CANVAS)
    new["nodes"] = [n for n in new["nodes"] if n["id"] != "l1"]
    new["edges"] = [new["edges"][0],
                    {"id": "e9", "fromNode": "a1", "toNode": "n1", "label": "next",
                     "toEnd": "arrow"}]
    new["nodes"].append({"id": "n1", "type": "text", "text": "New", "x": 5, "y": 6,
                         "width": 250, "height": 60, "junk": 1})
    out = merge_canvas(CANVAS, new)
    assert [n["id"] for n in out["nodes"]] == ["a1", "f1", "g1", "n1"]
    assert out["nodes"][-1] == {"id": "n1", "type": "text", "text": "New", "x": 5, "y": 6,
                                "width": 250, "height": 60}
    assert out["edges"][1] == {"id": "e9", "fromNode": "a1", "toNode": "n1",
                               "toEnd": "arrow", "label": "next"}


def test_clearing_a_label_removes_the_key():
    new = copy.deepcopy(CANVAS)
    new["edges"][0]["label"] = ""
    assert "label" not in merge_canvas(CANVAS, new)["edges"][0]


@pytest.mark.parametrize("break_it, msg", [
    (lambda d: d["nodes"].pop(3), "points at node"),                 # edge e2 dangles
    (lambda d: d["nodes"][1].update(text="x"), "only text nodes"),
    (lambda d: d["edges"][0].update(toNode="l1"), "reconnecting"),
    (lambda d: d["nodes"].append({"id": "a1", "type": "text", "text": "",
                                  "x": 0, "y": 0, "width": 1, "height": 1}), "duplicate"),
    (lambda d: d["nodes"].append({"id": "z", "type": "file", "file": "a.md",
                                  "x": 0, "y": 0, "width": 1, "height": 1}), "only adds text"),
    (lambda d: d["edges"].append({"id": "e7", "fromNode": "a1", "toNode": "f1",
                                  "toSide": "middle"}), "toSide"),
])
def test_bad_edits_are_refused(break_it, msg):
    new = copy.deepcopy(CANVAS)
    break_it(new)
    with pytest.raises(ValueError, match=msg):
        merge_canvas(CANVAS, new)


# ---- writer -------------------------------------------------------------------

def _classic(data, nl="\n"):
    def c(v):
        return json.dumps(v, ensure_ascii=False, separators=(",", ":"))
    parts = []
    for k, v in data.items():
        if isinstance(v, list) and v:
            parts.append(f"\t{c(k)}:[{nl}" + ("," + nl).join("\t\t" + c(x) for x in v)
                         + f"{nl}\t]")
        else:
            parts.append(f"\t{c(k)}:{c(v)}")
    return "{" + nl + ("," + nl).join(parts) + nl + "}"


@pytest.mark.parametrize("text", [
    _classic(CANVAS),                                             # older Obsidian
    _classic(CANVAS, "\r\n"),
    json.dumps(CANVAS, ensure_ascii=False, indent=4),             # newer, no final newline
    json.dumps(CANVAS, ensure_ascii=False, indent="\t") + "\n",
    json.dumps(CANVAS, ensure_ascii=False, indent="\t", separators=(",", ":")),  # Obsidian 1.x
    json.dumps(CANVAS, ensure_ascii=False, separators=(",", ":")),
    # Advanced Canvas: 2-space pretty print, empty [] / {} broken over two lines
    '{\n  "nodes": [\n    {\n      "id": "n1",\n      "type": "text",\n      "text": "Ուրարտու",\n'
    '      "styleAttributes": {\n      },\n      "x": 0,\n      "y": 0,\n      "width": 200,\n'
    '      "height": 60\n    }\n  ],\n  "edges": [\n  ],\n  "metadata": {\n  }\n}',
])
def test_the_writer_reproduces_the_files_own_layout(text):
    assert dump_canvas(json.loads(text), text) == text


# What Obsidian itself writes: tab indent, no space after ':', most objects
# expanded, some one-line, small leaf objects inline.
MIXED = (
    '{\n\t"nodes":[\n'
    '\t\t{"id":"a1","type":"text","text":"Alpha","x":0,"y":0,"width":250,"height":60},\n'
    '\t\t{\n\t\t\t"id":"b1",\n\t\t\t"type":"text",\n\t\t\t"text":"Beta",\n'
    '\t\t\t"styleAttributes":{"shape":"pill"},\n\t\t\t"x":300,\n\t\t\t"y":0,\n'
    '\t\t\t"width":250,\n\t\t\t"height":60\n\t\t}\n\t],\n'
    '\t"edges":[\n'
    '\t\t{\n\t\t\t"id":"e1",\n\t\t\t"fromNode":"a1",\n\t\t\t"toNode":"b1",\n'
    '\t\t\t"label":"leads to"\n\t\t},\n'
    '\t\t{"id":"e2","fromNode":"b1","toNode":"a1"}\n\t],\n'
    '\t"metadata":{"version":"1.0-1.0"}\n}')


def _changed_lines(before, after):
    import difflib
    return [ln for ln in difflib.unified_diff(before.splitlines(), after.splitlines(),
                                              lineterm="", n=0)
            if ln[:1] in "+-" and not ln.startswith(("+++", "---"))]


def test_an_edit_rewrites_only_what_changed():
    data = json.loads(MIXED)
    edit = copy.deepcopy(data)
    edit["nodes"][1]["text"] = "Beta, revised"
    out = dump_canvas(merge_canvas(data, edit), MIXED)
    assert _changed_lines(MIXED, out) == ['-\t\t\t"text":"Beta",', '+\t\t\t"text":"Beta, revised",']
    edit["nodes"][0]["x"] = 40                        # a one-line node stays one line
    out = dump_canvas(merge_canvas(data, edit), MIXED)
    assert '\t\t{"id":"a1","type":"text","text":"Alpha","x":40,"y":0,' in out


def test_new_and_dropped_members_keep_the_objects_layout():
    data = json.loads(MIXED)
    edit = copy.deepcopy(data)
    edit["edges"][0]["label"] = ""                     # clearing drops the key and its comma
    edit["edges"][1]["label"] = "back"                 # a new key joins the one-line object
    out = dump_canvas(merge_canvas(data, edit), MIXED)
    assert '\t\t\t"toNode":"b1"\n\t\t},' in out and '"leads to"' not in out
    assert '{"id":"e2","fromNode":"b1","toNode":"a1","label":"back"}' in out
    assert json.loads(out) == merge_canvas(data, edit)


def test_a_new_node_and_edge_follow_the_last_one_in_their_list():
    data = json.loads(MIXED)
    edit = copy.deepcopy(data)
    edit["nodes"].append({"id": "c1", "type": "text", "text": "Gamma", "x": 0, "y": 200,
                          "width": 250, "height": 60})
    edit["edges"].append({"id": "e3", "fromNode": "c1", "toNode": "a1"})
    out = dump_canvas(merge_canvas(data, edit), MIXED)
    assert '\t\t},\n\t\t{\n\t\t\t"id":"c1",\n\t\t\t"type":"text",' in out    # expanded, like b1
    assert '{"id":"e2","fromNode":"b1","toNode":"a1"},\n\t\t{"id":"e3",' in out  # one-line, like e2
    assert out.endswith('\t"metadata":{"version":"1.0-1.0"}\n}')


def test_a_moved_node_with_float_geometry_is_rounded_but_an_untouched_one_is_not():
    data = {"nodes": [{"id": "a", "type": "text", "text": "x", "x": 12.5, "y": 3.25,
                       "width": 100, "height": 50}]}
    assert merge_canvas(data, copy.deepcopy(data)) == data
    moved = copy.deepcopy(data)
    moved["nodes"][0]["x"] = 52.5
    assert merge_canvas(data, moved)["nodes"][0]["x"] == 52 and \
        merge_canvas(data, moved)["nodes"][0]["y"] == 3.25


def test_non_ascii_labels_are_written_as_is():
    data = copy.deepcopy(CANVAS)
    data["edges"][0]["label"] = "այսինքն, հետևում է, որ"
    assert "այսինքն" in dump_canvas(data, _classic(CANVAS))


# ---- the save endpoint ----------------------------------------------------------

@pytest.fixture
def vault(management_module, monkeypatch, tmp_path):
    m = management_module
    v = tmp_path / "vault"
    (v / "Maps").mkdir(parents=True)
    (v / "Maps" / "m.canvas").write_text(_classic(CANVAS), encoding="utf-8")
    (v / "Maps" / "note.md").write_text("x", encoding="utf-8")
    monkeypatch.setattr(m, "_vault_root", lambda: v)
    monkeypatch.setattr(m, "CFG", Config({"canvas": {"backup_dir": str(tmp_path / "bk")},
                                          "parser": {"vault_path": str(v)}}, tmp_path))
    monkeypatch.setattr(m, "_indexed_as", lambda rel: {})
    return m, v, tmp_path / "bk"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_save_merges_backs_up_outside_the_vault_and_replaces_atomically(vault):
    m, v, bk = vault
    f = v / "Maps" / "m.canvas"
    before = f.read_bytes()
    loaded = m.canvas_file("Maps/m.canvas")
    data = loaded["data"]
    data["nodes"][0]["text"] = "Alpha edited"
    r = m.canvas_save(m.CanvasSaveIn(path="Maps/m.canvas", base_sha256=loaded["sha256"],
                                     data=data))
    assert r["ok"] and r["sha256"] == _sha(f)
    on_disk = json.loads(f.read_text(encoding="utf-8"))
    assert on_disk["nodes"][0]["text"] == "Alpha edited"
    assert on_disk["nodes"][0]["styleAttributes"] == {"shape": "pill"}
    assert on_disk["metadata"] == CANVAS["metadata"]
    (backup,) = bk.iterdir()
    assert backup.read_bytes() == before and v not in backup.parents
    assert sorted(p.name for p in (v / "Maps").iterdir()) == ["m.canvas", "note.md"]


def test_a_file_changed_since_loading_is_never_overwritten(vault):
    m, v, bk = vault
    f = v / "Maps" / "m.canvas"
    loaded = m.canvas_file("Maps/m.canvas")
    f.write_text(f.read_text(encoding="utf-8").replace("Alpha", "Obsidian edit"),
                 encoding="utf-8")
    theirs = f.read_bytes()
    r = m.canvas_save(m.CanvasSaveIn(path="Maps/m.canvas", base_sha256=loaded["sha256"],
                                     data=loaded["data"]))
    assert r.status_code == 409 and f.read_bytes() == theirs
    assert not bk.exists()


def test_an_unchanged_save_writes_nothing(vault):
    m, v, bk = vault
    loaded = m.canvas_file("Maps/m.canvas")
    r = m.canvas_save(m.CanvasSaveIn(path="Maps/m.canvas", base_sha256=loaded["sha256"],
                                     data=loaded["data"]))
    assert r["unchanged"] is True and not bk.exists()


@pytest.mark.parametrize("path, code", [("Maps/note.md", 400), ("../outside.canvas", 400),
                                        ("Maps/missing.canvas", 404)])
def test_only_existing_canvases_inside_the_vault(vault, path, code):
    m, _, _ = vault
    assert m.canvas_file(path).status_code == code
    r = m.canvas_save(m.CanvasSaveIn(path=path, base_sha256="0" * 64, data={"nodes": []}))
    assert r.status_code == code


def test_a_backup_dir_inside_the_vault_is_refused(vault, monkeypatch, tmp_path):
    m, v, _ = vault
    monkeypatch.setattr(m, "CFG", Config({"canvas": {"backup_dir": str(v / "bk")},
                                          "parser": {"vault_path": str(v)}}, tmp_path))
    loaded = m.canvas_file("Maps/m.canvas")
    loaded["data"]["nodes"][0]["text"] = "changed"
    r = m.canvas_save(m.CanvasSaveIn(path="Maps/m.canvas", base_sha256=loaded["sha256"],
                                     data=loaded["data"]))
    assert r.status_code == 400 and "inside the vault" in json.loads(r.body)["error"]


# ---- re-index -------------------------------------------------------------------

def test_reindex_removes_old_rows_then_queues_a_scoped_run(vault, monkeypatch):
    m, v, _ = vault
    monkeypatch.setattr(m, "_indexed_as", lambda rel: {
        "Maps/m.canvas": {"chunks": 2, "jsonls": ["canvas_chunks.jsonl"]}})
    calls = []
    monkeypatch.setattr(m, "_chroma_delete_by_source", lambda s: calls.append(("chroma", s)) or 2)
    monkeypatch.setattr(m, "_jsonl_remove_sources",
                        lambda s, n: calls.append(("jsonl", n)) or {"canvas_chunks.jsonl": 2})
    monkeypatch.setattr(m, "build_manifest", lambda force=False: {})

    class J:
        def __init__(self, kind, prm):
            calls.append(("job", kind, prm))

        def public(self):
            return {}
    monkeypatch.setattr(m, "enqueue", J)
    r = m.canvas_reindex(m.CanvasPathIn(path="Maps/m.canvas"))
    assert r["ok"] and [c[0] for c in calls] == ["chroma", "jsonl", "job", "job"]
    kind, prm = calls[2][1], calls[2][2]
    assert kind == "ingest_canvas" and prm["include_path"] == "Maps/m.canvas"
    assert Path(prm["output"]).name.startswith("canvas_edit_m_")
    # and the lane's own guard accepts it (not the canonical canvas_chunks.jsonl)
    assert "--output" in m._build_argv("ingest_canvas", prm)


def test_reindex_refuses_a_filter_that_matches_more_than_one_canvas(vault):
    m, v, _ = vault
    (v / "Old").mkdir()
    (v / "Old" / "xMaps").mkdir()
    (v / "Old" / "xMaps" / "m.canvas").write_text(_classic(CANVAS), encoding="utf-8")
    r = m.canvas_reindex(m.CanvasPathIn(path="Maps/m.canvas"))    # "maps/m.canvas" is in both
    assert r.status_code == 400 and "2 file(s)" in json.loads(r.body)["error"]


# ---- suggest a label --------------------------------------------------------------

def test_suggest_label_returns_a_vocabulary_label_and_writes_nothing(vault, monkeypatch):
    m, v, _ = vault
    before = (v / "Maps" / "m.canvas").read_bytes()

    class Fake:
        model = "fake"

        def complete(self, system, user, temperature=None, max_tokens=None):
            assert "Alpha" in user and "related-to" in user
            return type("R", (), {"text": '{"relation": "part-of", "label": "part of"}'})()
    monkeypatch.setattr("src.llm.llm_client.LLMClient.from_config", lambda cfg: Fake())
    monkeypatch.setattr(m, "load_config", lambda *a, **k: Config({}, Path(".")))
    body = m.CanvasLabelIn(path="Maps/m.canvas", from_text="Alpha", to_text="Beta")
    assert m.canvas_suggest_label(body)["label"] == "part of"
    assert (v / "Maps" / "m.canvas").read_bytes() == before

    Fake.complete = lambda self, *a, **k: type("R", (), {"text": "no idea"})()
    assert m.canvas_suggest_label(body).status_code == 502
