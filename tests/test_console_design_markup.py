"""The console's shared UI parts (webui/index.html): themed dialogs instead of the native ones,
token-only CSS for the design-system block, one icon sprite, and the theme designer's text model.

Grep-style for the markup, like the other tests that pin the console's HTML: the page cannot run
without a browser. The designer's text model is DOM-free on purpose and runs under node (skipped
when node is absent), because it is what rewrites a user's CSS box token by token.
"""
import json
import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "webui" / "index.html").read_text(encoding="utf-8")
NODE = shutil.which("node")
COLOUR_LITERAL = r"#[0-9a-fA-F]{3,8}\b|\brgba?\(|\bhsla?\("


def test_no_native_confirm_prompt_or_alert_is_left():
    # They ignore the theme and block the page; uiConfirm / uiPrompt replace them. A method
    # call on some other object (x.confirm) is not a native dialog, window.confirm is.
    native = re.findall(r"(?<![\w$.])(?:window\.)?(?:confirm|prompt|alert)\s*\(", HTML)
    assert native == [], native
    # the unsaved-edits guard stays native: only the browser can hold a page that is closing
    assert 'window.addEventListener("beforeunload",e=>{if(state.gx.dirty||state.os.dirty)' in HTML


def test_the_dialogs_are_their_own_top_layer_elements_and_keep_keys_to_themselves():
    assert "function uiConfirm(message,o)" in HTML and "function uiPrompt(message,o)" in HTML
    helper = HTML[HTML.index("function uiDialog("):HTML.index("function uiConfirm(")]
    assert 'createElement("dialog")' in helper and "showModal()" in helper
    assert "#modal" not in helper                       # authFetch's prompt may open over a preview
    assert 'addEventListener("keydown",e=>e.stopPropagation())' in helper   # no shortcut acts behind it
    # Cancel and Esc both close without "ok": false for a confirm, null for a prompt
    assert 'resolve(d.returnValue==="ok"?(inp?inp.value:true):(inp?null:false))' in helper
    assert ".ui-dlg::backdrop{background:var(--scrim)}" in HTML


def test_the_design_system_css_uses_theme_tokens_only():
    css = HTML[HTML.index("/* ---- design system:"):HTML.index("/* ---- /design system ---- */")]
    bodies = re.findall(r"\{([^{}]*)\}", css)
    assert len(bodies) > 40, "the design-system block is missing or truncated"
    for body in bodies:
        assert not re.search(COLOUR_LITERAL, body), body


def test_every_icon_the_page_names_is_in_the_sprite():
    symbols = set(re.findall(r'<symbol id="i-([\w-]+)" viewBox="0 0 24 24">', HTML))
    named = set(re.findall(r'href="#i-([\w-]+)"', HTML)) - {"name"}          # "#i-name" is the doc example
    named |= set(re.findall(r'\bicon\("([\w-]+)"', HTML))
    named |= {x for pair in re.findall(r'icon\([^)]*?\?"([\w-]+)":"([\w-]+)"', HTML) for x in pair}
    named |= set(re.findall(r':"([\w-]+)"', re.search(r"const KIND_ICON=\{([^}]*)\}", HTML).group(1)))
    assert symbols, "the icon sprite is missing"
    assert named <= symbols, f"no <symbol> for {sorted(named - symbols)}"
    # the sprite carries geometry only: svg.i sets fill and stroke, so icons follow the text colour
    sprite = HTML[HTML.index('<svg class="sprite"'):HTML.index("</svg>", HTML.index('<svg class="sprite"'))]
    assert not re.search(r"\b(?:fill|stroke)=", sprite)


def test_theme_vars_carry_the_radii_and_no_derived_token():
    names = re.findall(r'"([\w-]+)"', re.search(r"const THEME_VARS=\[([^\]]*)\]", HTML).group(1))
    assert {"r-sm", "r", "r-lg", "r-pill"} <= set(names)          # Export and the designer see them
    assert not {"scrim", "eyebrow-size", "eyebrow-track"} & set(names)   # an export would freeze them
    root = re.search(r"^:root\{(.*?)^\}", HTML, re.S | re.M).group(1)
    for v in names:                                               # Export never writes an empty value
        assert re.search(rf"--{re.escape(v)}:", root), v


def _pure_block() -> str:
    start = HTML.index("/* thm-bulk:pure-begin")
    return HTML[start:HTML.index("/* thm-bulk:pure-end */", start)]


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_the_designer_text_model_edits_one_token_and_leaves_the_rest_alone():
    scenario = textwrap.dedent("""
        const localStorage = {getItem: () => null, setItem() {}, removeItem() {}, length: 0, key: () => null};
        const p = new Function("localStorage", __BLOCK__ +
          "; return {themeVarsParse, themeVarsSet, themeVarsSeed, cssColorParts, cssColorWithHex};")(localStorage);
        const out = {};
        out.parse = p.themeVarsParse('[data-theme="x"]{\\n  --bg: #101010;\\n  /* mine */ --amber:#E8A33D; --bg:#111;\\n  color:red; --bad: a}b;\\n}');
        const box = "{\\n  --bg: #000;\\n  --ink: #fff;\\n}";
        out.replace = p.themeVarsSet(box, "bg", "#123");
        out.add = p.themeVarsSet(box, "amber", "#E8A33D");
        out.remove = p.themeVarsSet(box, "bg", "");
        out.prefix = [p.themeVarsSet("--r-sm: 4px; --r: 6px;", "r", "9px"),
                      p.themeVarsSet("--ink-dim: #aaa;", "ink", "#fff")];
        out.last_wins = p.themeVarsSet("--bg:#1;--bg:#2;", "bg", "#3");
        out.bare = [p.themeVarsSet("", "bg", "#1;}x"), p.themeVarsSet("--x: 1;", "bg", "")];
        out.round_trip = p.themeVarsParse(p.themeVarsSet(box, "r-lg", "10px"));
        out.colours = ["#abc", "#AABBCC80", "rgba(58,51,42,.5)", "rgb(1 2 3 / 50%)", "color-mix(in srgb,red 5%,blue)",
                       "rgb(300,0,0)", ""].map(v => p.cssColorParts(v));
        out.with_hex = [p.cssColorWithHex("rgba(58,51,42,.5)", "#ff0000"), p.cssColorWithHex("#000", "#ff0000"),
                        p.cssColorWithHex("bogus", "#ff0000")];
        const live = {bg: "#F2ECDF", panel: "#FBF8F1", amber: "#B26E10", "r-sm": "4px"};
        out.seed = p.themeVarsSeed("", live);
        out.seed_edit = p.themeVarsParse(p.themeVarsSet(p.themeVarsSeed(" \\n", live), "amber", "#123456"));
        out.seed_keeps = [p.themeVarsSeed("--bg: #000;", live), p.themeVarsSeed("/* mine */", {bg: "#1"})];
        console.log(JSON.stringify(out));
    """).replace("__BLOCK__", json.dumps(_pure_block()))
    r = subprocess.run([NODE, "-e", scenario], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    o = json.loads(r.stdout)
    assert o["parse"] == {"bg": "#111", "amber": "#E8A33D"}     # last wins, comments and non-tokens ignored
    assert o["replace"] == "{\n  --bg: #123;\n  --ink: #fff;\n}"
    assert o["add"] == "{\n  --bg: #000;\n  --ink: #fff;\n  --amber: #E8A33D;\n}"
    assert o["remove"] == "{\n  --ink: #fff;\n}"
    assert o["prefix"] == ["--r-sm: 4px; --r: 9px;", "--ink-dim: #aaa;\n--ink: #fff;"]   # --r is not --r-sm
    assert o["last_wins"] == "--bg:#1;--bg: #3;"                # the declaration in force is the one edited
    assert o["bare"] == ["--bg: #1x;", "--x: 1;"]               # no ; { } can break out of the box
    assert o["round_trip"] == {"bg": "#000", "ink": "#fff", "r-lg": "10px"}
    a, b, c, d, mix, bad, empty = o["colours"]
    assert a == {"hex": "#aabbcc", "a": 1} and b["hex"] == "#aabbcc" and abs(b["a"] - 128 / 255) < 1e-9
    assert c == {"hex": "#3a332a", "a": 0.5} and d == {"hex": "#010203", "a": 0.5}
    assert mix is None and bad is None and empty is None        # the well is disabled; the text still edits
    assert o["with_hex"] == ["rgba(255,0,0,0.5)", "#ff0000", "#ff0000"]   # a tint stays a tint
    # an empty box plus one row edit: every token of the theme in use, and the edited value
    assert o["seed"] == "{\n  --bg: #F2ECDF;\n  --panel: #FBF8F1;\n  --amber: #B26E10;\n  --r-sm: 4px;\n}"
    assert o["seed_edit"] == {"bg": "#F2ECDF", "panel": "#FBF8F1", "amber": "#123456", "r-sm": "4px"}
    assert o["seed_keeps"] == ["--bg: #000;", "/* mine */\n--bg: #1;"]   # the user's text is never replaced


def test_export_and_the_first_row_edit_start_from_the_same_seed():
    # Export writes thmSeed(""); a row edit seeds only a box that holds no tokens, so the rows
    # never show values the box does not hold, and a box with the user's tokens is left alone
    export = HTML[HTML.index('$("#thm-export").onclick'):][:120]
    assert 'thmSeed("")' in export
    rows = HTML[HTML.index('$("#thm-vars").addEventListener("input"'):][:260]
    assert "if(!Object.keys(themeVarsParse(ta.value)).length)ta.value=thmSeed(ta.value);" in rows
    seed = HTML[HTML.index("function thmSeed(text){"):][:320]
    assert '$("#thm-mode").value=cur.mode' in seed and "themeVarsSeed(text," in seed


def _fn(name):
    i = HTML.index(f"function {name}(")
    return HTML[i:HTML.index("\nfunction ", i + 1)]


def test_generated_graph_nodes_render_unticked_and_disabled_with_a_note():
    body = _fn("renderGraphTree")
    assert 'n.provenance==="generated"' in body
    assert "disabled" in body and "a map, not evidence" in body


def test_disabled_nodes_are_never_selected_even_after_select_all():
    assert ":checked:not(:disabled)" in _fn("graphSelectedIds")
    i = HTML.index('$("#qg-all").onclick')
    assert "disabled" in HTML[i:HTML.index("\n", i)]


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_only_the_generated_node_renders_disabled_and_unticked():
    # /graph/expand still returns generated nodes (a map); /answer 400s on their ids
    esc = re.search(r"^const esc=.*$", HTML, re.M).group(0)
    scenario = esc + "\n" + _fn("renderGraphTree") + textwrap.dedent("""
        const data = {seed_ids: ["a"], nodes: [
            {id: "a", depth: 0, provenance: "authored", text: "seed"},
            {id: "g", depth: 1, provenance: "generated", text: "map"},
            {id: "g2", depth: 1, provenance: "generated", text: "routed map", routed_ids: ["p1", "p2"]},
            {id: "g3", depth: 1, provenance: "generated", text: "unrouted map", routed_ids: []},
            {id: "s", depth: 1, text: "no provenance field"}],
          tree: [{parent: "a", child: "g", direction: "->"}, {parent: "a", child: "s", direction: "->"},
                 {parent: "a", child: "g2", direction: "->"}, {parent: "a", child: "g3", direction: "->"}]};
        console.log(JSON.stringify(renderGraphTree(data)));
    """)
    r = subprocess.run([NODE, "-e", scenario], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    tags = re.findall(r"<input\b[^>]*>", json.loads(r.stdout))
    box = {re.search(r'data-id="([^"]*)"', t).group(1): t for t in tags}
    assert sorted(box) == ["a", "g", "g2", "g3", "s"]
    s1_title = 'title="Generated: a map, not evidence. Its Source pages are what get cited"'
    for k in ("g", "g3"):                                 # no sidecar, or no targets: as in S1
        assert re.search(r"\sdisabled\b", box[k]) and not re.search(r"\schecked\b", box[k])
        assert s1_title in box[k] and "data-routed" not in box[k]
    # a map with routed pages can be ticked, as those pages; it starts unticked (opt-in)
    assert 'data-routed="p1,p2"' in box["g2"]
    assert "disabled" not in box["g2"] and not re.search(r"\schecked\b", box["g2"])
    assert ('title="Generated: a map, not evidence. Ticking adds the first chunks of each span this '
            'map cites"') in box["g2"]
    for k in ("a", "s"):                                  # every other node is exactly as before
        assert re.search(r"\schecked\b", box[k]) and "disabled" not in box[k]
        assert 'title="Include this node in the answer"' in box[k] and "data-routed" not in box[k]


def test_only_the_answer_call_sends_a_ticked_map_as_its_routed_pages():
    # The count and "Copy raw nodes" stay per node (graphSelectedIds); /answer gets the pages.
    i = HTML.index("async function runGraphAnswer(")
    answer = HTML[i:HTML.index("\n}\n", i)]
    assert "dataset.routed" in answer and 'split(",")' in answer
    assert "dataset.routed" not in _fn("graphSelectedIds")
    assert "routed_ids.length" in _fn("renderGraphTree")


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_answer_gets_a_ticked_maps_pages_deduped_and_never_its_id():
    i = HTML.index("async function runGraphAnswer(")
    scenario = _fn("graphSelectedIds") + textwrap.dedent("""
        const state = {queryBusy: false, lastQ: null};
        const els = {"#q-text": {value: "why?"}, "#q-maxtok": {value: ""}};
        const $ = s => els[s] || {};
        const box = (id, checked, disabled, routed) =>
            ({checked, disabled, dataset: routed ? {id, routed} : {id}});
        const boxes = [box("n1", true, false), box("g2", true, false, "p1,p2"), box("p1", true, false),
                       box("g4", false, false, "p9"),      // a routed map left unticked (opt-in)
                       box("n7", false, false),            // a node the user unticked
                       box("g", true, true)];              // disabled, ticked anyway: the filter keeps it out
        // querySelectorAll's answer for the two state filters the console uses
        const $$ = s => boxes.filter(b => (!s.includes(":checked") || b.checked)
                                          && (!s.includes(":not(:disabled)") || !b.disabled));
        let body = null;
        const ragCall = async (path, opts) => { body = JSON.parse(opts.body); return {}; };
        const toast = m => { throw new Error(m); };
        const esc = String, groundableIds = () => ({ok: []});
        const setGraphBusy = () => {}, renderGraphFork = () => {}, renderAnswerBody = () => {};
    """) + HTML[i:HTML.index("\n}\n", i) + 2] + '\nrunGraphAnswer(false).then(() => console.log(JSON.stringify(body.docs)));'
    r = subprocess.run([NODE, "-e", scenario], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout) == ["n1", "p1", "p2"]
