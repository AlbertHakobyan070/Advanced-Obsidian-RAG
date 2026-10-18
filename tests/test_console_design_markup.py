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
