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

from src.llm.llm_client import LLMResponse

# What a reasoning model sends before its answer: prose with braces in it.
REASONING = ("Let me think about {A} and {B}. The arrow runs from A to B, so the label "
             "has to read from A towards B. A is one section of what B covers, which is "
             "a part-whole relation rather than an example. ")


class Reasoner:
    """A model that reasons aloud before its JSON answer, and is cut off mid-thought
    when the call leaves it too few tokens: the reply behind the owner's 502s."""
    model, vision, backend = "auto", False, "reasoner"

    def __init__(self, answer='{"relation": "Part-Of", "label": "part of", "tags": ["#Stats"], '
                              '"alternatives": ["belongs to"], "confidence": "high", '
                              '"why": "A is one section of B."}'):
        self.answer, self.calls = answer, []

    def complete(self, system, user, **kw):
        self.calls.append({"system": system, "user": user, **kw})
        if (kw.get("max_tokens") or 0) < 400:
            return LLMResponse(text=REASONING[:kw["max_tokens"]], model="vendor/reasoner-1",
                               provider="openai")
        return LLMResponse(text=REASONING + self.answer, model="vendor/reasoner-1",
                           provider="openai", usage={"input_tokens": 50, "output_tokens": 20})


def _call(fn, *args):
    """(status, body) whether the endpoint returned a dict (200) or a JSONResponse."""
    r = fn(*args)
    return (200, r) if isinstance(r, dict) else (r.status_code, json.loads(r.body))


def test_suggest_label_no_longer_502s_on_a_model_that_reasons_first(vault, monkeypatch):
    m, v, _ = vault
    monkeypatch.setattr("src.llm.llm_client.LLMClient.from_config",
                        lambda cfg, role="generation": Reasoner())
    monkeypatch.setattr(m, "load_config", lambda *a, **k: Config({}, Path(".")))
    # Both the pre-v2 fields and the v2 ones: extra fields are ignored either way.
    body = m.CanvasLabelIn(path="Maps/m.canvas", from_text="Alpha", to_text="Notes/x.md",
                           examples=["explains"], edge_id="e1", data=copy.deepcopy(CANVAS))
    status, out = _call(m.canvas_suggest_label, body)
    assert status == 200, out
    assert (out["label"], out["relation"], out["tags"], out["model"]) == \
        ("part of", "part-of", ["stats"], "vendor/reasoner-1")


def _body(m, edge_id="e1", data=None, **kw):
    return m.CanvasLabelIn(path="Maps/m.canvas", edge_id=edge_id,
                           data=copy.deepcopy(CANVAS) if data is None else data, **kw)


def _with(monkeypatch, m, client=None, cfg=None):
    if client is not None:
        monkeypatch.setattr("src.llm.llm_client.LLMClient.from_config",
                            lambda cfg, role="generation": client)
    monkeypatch.setattr(m, "load_config", lambda *a, **k: Config(cfg or {}, Path(".")))
    return client


def test_suggest_label_reads_the_posted_canvas_and_writes_nothing(vault, monkeypatch):
    m, v, bk = vault
    f = v / "Maps" / "m.canvas"
    before = f.read_bytes()
    r = _with(monkeypatch, m, Reasoner())
    data = copy.deepcopy(CANVAS)                            # an edit the file does not have
    data["nodes"].append({"id": "n9", "type": "text", "text": "Unsaved node text", "x": 0,
                          "y": 900, "width": 10, "height": 10})
    data["edges"].append({"id": "e9", "fromNode": "a1", "toNode": "n9"})
    status, out = _call(m.canvas_suggest_label, _body(m, "e9", data))
    assert status == 200, out
    assert out["context"]["to"]["sent"] == "Unsaved node text"
    assert "Unsaved node text" in r.calls[-1]["user"]
    assert f.read_bytes() == before and not bk.exists()
    assert sorted(p.name for p in (v / "Maps").iterdir()) == ["m.canvas", "note.md"]


def test_suggest_label_names_the_routed_model_backend_and_what_was_sent(vault, monkeypatch):
    m, _, _ = vault
    r = _with(monkeypatch, m, Reasoner())
    status, out = _call(m.canvas_suggest_label, _body(m))
    assert status == 200, out
    assert (out["backend"], out["protocol"], out["model"]) == ("reasoner", "openai",
                                                               "vendor/reasoner-1")
    assert out["usage"] == {"input_tokens": 50, "output_tokens": 20, "calls": 1}
    assert (out["prompt_version"], out["label_tags_indexed"]) == (1, False)
    assert (out["alternatives"], out["confidence"], out["why"], out["warnings"]) == \
        (["belongs to"], "high", "A is one section of B.", [])
    assert out["note"].startswith("nothing was written")
    ctx = out["context"]
    assert (ctx["title"], ctx["direction"], ctx["current_label"]) == ("m", "→", "explains")
    assert ctx["groups"] == ["Box"] and ctx["from"]["group"] == ctx["to"]["group"] == "Box"
    assert ctx["from"]["sent"] == "Alpha" and ctx["from"]["mode"] == "full"
    assert ctx["to"]["mode"] == "placeholder" and "not found" in ctx["to"]["note"]
    assert ctx["to"]["sent"] in r.calls[-1]["user"] and "Alpha" in r.calls[-1]["user"]
    assert (r.calls[-1]["max_tokens"], r.calls[-1]["temperature"]) == (4000, 0.0)
    assert "images" not in r.calls[-1] and "image" not in ctx["to"]


class Broken(Reasoner):
    def complete(self, system, user, **kw):
        raise ConnectionError("connection refused")


def test_suggest_label_errors_are_typed(vault, monkeypatch, tmp_path):
    m, _, _ = vault

    def call(body, client=None, cfg=None):
        _with(monkeypatch, m, client or Reasoner(), cfg)
        return _call(m.canvas_suggest_label, body)
    status, out = call(_body(m, "nope"))
    assert status == 400 and "no edge 'nope'" in out["error"]
    assert call(_body(m, data={"nodes": []}))[0] == 400
    assert call(m.CanvasLabelIn(path="Maps/missing.canvas", edge_id="e1", data=CANVAS))[0] == 404
    status, out = call(_body(m, provider="typo"))
    assert (status, out["stage"]) == (400, "client") and "provider = 'typo'" in out["error"]
    status, out = call(_body(m), cfg={"canvas": {"augment": {"provider": "typo2"}}})
    assert status == 400 and "canvas.augment.provider = 'typo2'" in out["error"]
    reg = {"needkey": {"kind": "openai", "base_url": "http://127.0.0.1:9/v1", "model": "x",
                       "api_key_env": "TEST_SUGGEST_LABEL_KEY"},
           "badvision": {"kind": "openai", "base_url": "http://127.0.0.1:9/v1", "model": "x",
                         "vision": "yes"}}
    monkeypatch.delenv("TEST_SUGGEST_LABEL_KEY", raising=False)
    for name, says in (("needkey", "TEST_SUGGEST_LABEL_KEY"), ("badvision", "vision")):
        status, out = call(_body(m, provider=name), cfg={"providers": reg})
        assert (status, out["stage"]) == (400, "client") and says in out["error"]
    status, out = call(_body(m), cfg={"canvas": {"augment": {"long_nodes": "drop"}}})
    assert status == 400 and "long_nodes" in out["error"]

    status, out = call(_body(m), client=Broken())
    assert (status, out["stage"], out["backend"], out["model"]) == (502, "label", "reasoner", "auto")
    assert "connection refused" in out["error"]
    status, out = call(_body(m), client=Reasoner(answer='{"relation": "part-of"}'))
    assert (status, out["stage"], out["model"]) == (502, "parse", "vendor/reasoner-1")
    assert out["reply"] == (REASONING + '{"relation": "part-of"}')[:2000]
    assert out["error"] == 'the model\'s reply has no JSON object with a non-empty "label"'
    empty = Reasoner()
    empty.complete = lambda system, user, **kw: LLMResponse(text="", model="m", provider="openai")
    status, out = call(_body(m), client=empty, cfg={"canvas": {"augment": {"node_chars": 3}}})
    assert (status, out["stage"]) == (502, "summary:from") and "empty" in out["error"]


PNG = b"\x89PNG\r\n\x1a\nsmall"


def test_suggest_label_sends_images_only_to_a_vision_backend(vault, monkeypatch):
    m, v, _ = vault
    (v / "Maps" / "pic.png").write_bytes(PNG)
    data = copy.deepcopy(CANVAS)
    data["nodes"].append({"id": "p1", "type": "file", "file": "Maps/pic.png", "x": 0,
                          "y": 900, "width": 10, "height": 10})
    data["edges"].append({"id": "ep", "fromNode": "a1", "toNode": "p1"})
    sent = []

    def wire(self, system, user, temp, maxtok, images=None):   # the SDK call, nothing else
        sent.append(images)
        return LLMResponse(text='{"label": "shows"}', model=self.model, provider="openai")
    monkeypatch.setattr("src.llm.llm_client.LLMClient._complete_openai_compatible", wire)
    reg = {name: {"kind": "openai", "base_url": "http://127.0.0.1:9/v1", "model": name, **extra}
           for name, extra in (("blind", {}), ("eyes", {"vision": True}))}
    _with(monkeypatch, m, cfg={"providers": reg})
    status, out = _call(m.canvas_suggest_label, _body(m, "ep", data, provider="blind"))
    assert status == 200 and sent[-1] is None and out["context"]["to"]["mode"] == "placeholder"
    assert out["context"]["images"] == [{"node": "p1", "path": "Maps/pic.png", "sent": False,
                                         "reason": "the backend is not marked vision: true"}]
    status, out = _call(m.canvas_suggest_label, _body(m, "ep", data, provider="eyes"))
    assert status == 200 and sent[-1] == [(PNG, "image/png")]
    assert out["context"]["images"][0]["sent"] is True and out["context"]["to"]["mode"] == "image"


def test_backend_precedence_is_request_then_canvas_augment_then_generation(vault, monkeypatch):
    m, _, _ = vault

    def complete(self, system, user, **kw):
        return LLMResponse(text='{"label": "x"}', model=self.model, provider=self.provider)
    monkeypatch.setattr("src.llm.llm_client.LLMClient.complete", complete)
    reg = {name: {"kind": "openai", "base_url": "http://127.0.0.1:9/v1", "model": f"{name}-own"}
           for name in ("gen", "aug", "req")}

    def run(augment, **req):
        _with(monkeypatch, m, cfg={"providers": reg, "canvas": {"augment": augment},
                                   "generation": {"provider": "gen", "model": "gen-role"}})
        status, out = _call(m.canvas_suggest_label, _body(m, **req))
        assert status == 200, out
        return out["backend"], out["model"]
    assert run({}) == ("gen", "gen-role")                  # exactly the generation client
    assert run({"model": "pinned"}) == ("gen", "pinned")
    assert run({"provider": "aug"}) == ("aug", "aug-own")
    assert run({"provider": "aug", "model": "pinned"}) == ("aug", "pinned")
    assert run({"provider": "aug"}, provider="req") == ("req", "req-own")
    assert run({"provider": "aug", "model": "pinned"}, provider="req") == ("req", "pinned")
    assert run({"provider": "aug", "model": "pinned"}, provider="req", model="m2") == ("req", "m2")


def test_suggest_options_reports_the_default_and_each_vision_flag(management_module, monkeypatch):
    m = management_module
    shipped = m.canvas_suggest_options()                   # the config a fresh clone ships
    assert shipped["ok"] and shipped["default"]["source"] == "generation"
    assert shipped["providers"] and not any(p["vision"] for p in shipped["providers"])
    assert (shipped["long_nodes"], shipped["node_chars"], shipped["label_tags"]) == \
        ("summarize", 2500, False)
    assert shipped["vocabulary"] == m._DEFAULT_RELATIONS
    reg = {"proxy": {"kind": "openai", "model": "auto"},
           "eyes": {"kind": "openai", "model": "vl", "vision": True}}

    def options(augment):
        _with(monkeypatch, m, cfg={"providers": reg, "canvas": {"augment": augment,
                                                                "label_tags": True},
                                   "generation": {"provider": "proxy", "model": "auto"}})
        return m.canvas_suggest_options()
    out = options({"long_nodes": "truncate"})
    assert out["default"] == {"backend": "proxy", "model": "auto", "source": "generation"}
    assert {p["name"]: p["vision"] for p in out["providers"]} == {"proxy": False, "eyes": True}
    assert (out["long_nodes"], out["label_tags"]) == ("truncate", True)
    assert options({"provider": "eyes"})["default"] == \
        {"backend": "eyes", "model": "vl", "source": "canvas.augment"}
    assert options({"model": "pinned"})["default"] == \
        {"backend": "proxy", "model": "pinned", "source": "canvas.augment"}


PICKER_REGISTRY = {"proxy": {"kind": "openai", "base_url": "http://127.0.0.1:9/v1", "model": "auto"},
                   "eyes": {"kind": "openai", "base_url": "http://127.0.0.1:9/v1", "model": "vl",
                            "vision": True}}


@pytest.mark.parametrize("cfg, shown", [
    # the generation role: LLMClient.from_config takes the role's model, then the entry's
    ({"generation": {"provider": "proxy", "model": "openai/gpt-oss-120b"}},
     ("proxy", "openai/gpt-oss-120b", "generation")),
    ({"generation": {"provider": "proxy"}}, ("proxy", "auto", "generation")),
    ({"generation": {"provider": "proxy", "model": "role"}, "canvas": {"augment": {"model": ""}}},
     ("proxy", "role", "generation")),                                   # blank = unset
    # canvas.augment: from_provider_override takes the pinned model, then the entry's own
    ({"generation": {"provider": "proxy", "model": "role"}, "canvas": {"augment": {"provider": "eyes"}}},
     ("eyes", "vl", "canvas.augment")),
    ({"generation": {"provider": "proxy", "model": "role"},
      "canvas": {"augment": {"provider": "eyes", "model": "pinned"}}}, ("eyes", "pinned", "canvas.augment")),
    ({"generation": {"provider": "proxy", "model": "role"}, "canvas": {"augment": {"model": "pinned"}}},
     ("proxy", "pinned", "canvas.augment")),
    # the reserved legacy names keep their own rules
    ({"generation": {"provider": "local", "local": {"model": "gemma"}}}, ("local", "gemma", "generation")),
    ({"generation": {"provider": "local"}}, ("local", "local-model", "generation")),
    ({"generation": {"provider": "openai"}}, ("openai", "gpt-4o-mini", "generation")),
    ({"generation": {"provider": "anthropic", "model": "claude-x"}}, ("anthropic", "claude-x", "generation")),
    ({"generation": {"provider": "proxy", "model": "role"}, "canvas": {"augment": {"provider": "anthropic"}}},
     ("anthropic", "role", "canvas.augment")),
])
def test_the_default_suggest_options_shows_is_the_model_the_default_call_sends(
        vault, monkeypatch, cfg, shown):
    m, _, _ = vault
    # Legacy clients need a key set. Not key-shaped: the packager refuses to ship
    # anything that looks like one (deploy/package/common.ps1 Assert-NoSecrets).
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-key")
    _with(monkeypatch, m, cfg={"providers": PICKER_REGISTRY, **cfg})
    default = m.canvas_suggest_options()["default"]
    assert (default["backend"], default["model"], default["source"]) == shown
    sent = []

    def complete(self, system, user, **kw):                         # the real client, no wire
        sent.append((self.backend, self.model))
        return LLMResponse(text='{"label": "x"}', model=self.model, provider=self.provider)
    monkeypatch.setattr("src.llm.llm_client.LLMClient.complete", complete)
    status, out = _call(m.canvas_suggest_label, _body(m))
    assert status == 200, out
    assert sent == [(default["backend"], default["model"])]


def test_suggest_label_treats_an_unreadable_file_node_as_a_placeholder(vault, monkeypatch):
    m, _, _ = vault

    def denied(p, read_chars):
        raise PermissionError(13, "Permission denied", str(p))
    monkeypatch.setattr("src.ingestion.canvas_label._read_text", denied)
    _with(monkeypatch, m, Reasoner())
    data = copy.deepcopy(CANVAS)
    data["nodes"].append({"id": "n9", "type": "file", "file": "Maps/note.md", "x": 0, "y": 900,
                          "width": 10, "height": 10})
    data["edges"].append({"id": "e9", "fromNode": "a1", "toNode": "n9"})
    status, out = _call(m.canvas_suggest_label, _body(m, "e9", data))
    assert status == 200, out                                        # was a 500
    to = out["context"]["to"]
    assert to["mode"] == "placeholder" and "could not be read: PermissionError" in to["note"]


def test_suggest_label_settings_are_editable_with_request_time_values(management_module,
                                                                       monkeypatch, tmp_path):
    from src.utils.config_loader import load_config
    m = management_module
    real_persist = m._persist_section_keys
    monkeypatch.setattr(m, "_torch_devices", lambda: ({}, ["auto", "cpu"]))
    names, _ = m._provider_choices(load_config())
    editable = m.settings()["editable"]
    assert editable["canvas.augment.provider"]["values"] == ["null"] + names
    assert editable["canvas.augment.provider"]["value"] == "null"   # untouched null: no pending change
    assert editable["canvas.augment.model"]["allow_empty"] is True
    assert editable["canvas.augment.model"]["value"] == ""            # blank, not None
    assert editable["canvas.label_tags"]["values"] == ["true", "false"]
    written = []
    monkeypatch.setattr(m, "_persist_section_keys",
                        lambda path, changes: written.append(dict(changes)) or list(changes))
    changes = {"canvas.augment.provider": "freellmapi", "canvas.augment.model": "",
               "canvas.label_tags": "true"}
    out = m.settings_update(m.SettingsIn(changes=changes))
    assert out["ok"] and "none (read per request)" in out["note"]
    assert written == [{**changes, "canvas.augment.model": "null"}]   # blank = unset = null
    assert m.settings_update(m.SettingsIn(changes={"canvas.augment.provider": "null"}))["ok"]
    for key, bad in (("canvas.augment.provider", "typo"), ("canvas.label_tags", "maybe")):
        r = m.settings_update(m.SettingsIn(changes={key: bad}))
        assert r.status_code == 400 and key in json.loads(r.body)["error"]
    # The real writer resolves the keys below the vocabulary's two-line flow list.
    probe = tmp_path / "probe.yaml"
    probe.write_bytes((Path(__file__).resolve().parents[1] / "config.example.yaml").read_bytes())
    real_persist(probe, {**changes, "canvas.augment.model": "MiniMax-M3"})
    back = load_config(probe)
    assert (back.get("canvas.augment.provider"), back.get("canvas.augment.model"),
            back.get("canvas.label_tags")) == ("freellmapi", "MiniMax-M3", True)
    assert back.get("canvas.augment.vocabulary") == m._DEFAULT_RELATIONS


def test_saving_the_blank_suggest_model_back_writes_no_change(management_module, monkeypatch,
                                                               tmp_path):
    """GET /api/settings shows a null canvas.augment.model as blank; posting that
    blank back must leave config.yaml as it was. The fixture's config.yaml (a
    copy of the shipped example in tmp_path) is the file both read and written."""
    from src.utils.config_loader import load_config
    m = management_module
    monkeypatch.setattr(m, "_torch_devices", lambda: ({}, ["auto", "cpu"]))
    monkeypatch.setattr(m, "ROOT", tmp_path)
    cfg_file = tmp_path / "config.yaml"
    before = cfg_file.read_text(encoding="utf-8")
    assert load_config(cfg_file).get("canvas.augment.model") is None
    shown = m.settings()["editable"]["canvas.augment.model"]["value"]
    assert shown == ""
    assert m.settings_update(m.SettingsIn(changes={"canvas.augment.model": shown}))["ok"]
    assert cfg_file.read_text(encoding="utf-8") == before
    m.settings_update(m.SettingsIn(changes={"canvas.augment.model": "pinned-model"}))
    assert load_config(cfg_file).get("canvas.augment.model") == "pinned-model"
    m.settings_update(m.SettingsIn(changes={"canvas.augment.model": "  "}))     # cleared again
    assert cfg_file.read_text(encoding="utf-8") == before
