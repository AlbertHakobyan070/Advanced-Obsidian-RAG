"""The console's Eval tab (webui/index.html): in the nav, wired to the four review
endpoints and to the query API's search, with its keyboard shortcuts guarded.

Grep-style on purpose, like the other tests that pin the console's markup: the page
cannot run without a browser. Its behaviour was checked live against a spare-port
console with DOM reads, and the API side has its own tests (test_eval_review_api.py).
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "webui" / "index.html").read_text(encoding="utf-8")


def _eval_script() -> str:
    """The Eval tab's JS: from its banner comment to the boot section."""
    start = HTML.index("/* ---------- eval review queue")
    return HTML[start:HTML.index("/* ---------- boot ---------- */", start)]


def test_the_nav_has_an_eval_tab_that_opens_its_section():
    assert '<button data-tab="eval">Eval</button>' in HTML
    assert '<section id="tab-eval">' in HTML
    # the shared tab handler is what shows the section; it must also load the queue
    assert 'if(b.dataset.tab==="eval")evOpen();' in HTML


def test_the_tab_wires_the_four_review_endpoints_and_the_search():
    js = _eval_script()
    assert 'evApi("/api/eval/progress")' in js                     # GET progress
    assert '"/api/eval/questions?"' in js                           # GET list, filters in the query string
    assert '"/api/eval/questions/"+encodeURIComponent(id)' in js    # GET detail and POST review
    assert 'method:"POST"' in js
    for action in ("verify", "reject", "edit"):
        assert f'evAct("{action}"' in js
    # Check retrieval reuses the Query tab's call, with HyDE off and the top five
    assert 'ragCall("/search"' in js
    assert "EV_CHECK={hyde:false,top_k:5," in js
    # nothing the tab calls may swallow an error: an empty catch block is a failure shown to nobody
    assert not re.search(r"catch\(\w*\)\{\s*\}", js)


def test_shortcuts_ignore_typing_modifiers_and_key_repeat():
    js = _eval_script()
    handler = js[js.index('document.addEventListener("keydown"'):]
    assert 'closest("input,textarea,select")' in handler    # a letter in a field is text
    assert "e.ctrlKey||e.metaKey||e.altKey" in handler       # Ctrl+R must still reload the page
    assert "e.repeat" in handler                             # a held key must not walk the queue
    assert "state.ev.editing" in handler                     # the edit form owns the keyboard
    for key in ("v:", "r:", "e:", "j:", "k:"):
        assert key in handler


def test_the_edit_form_round_trips_what_it_does_not_render():
    # A gold entry may carry `alternatives` (the same passage in other files), and a later
    # schema may add more fields. The POSTed record must start from the STORED one, so none
    # of that is dropped: a kept gold row is a copy of its stored entry, never rebuilt.
    js = _eval_script()
    read = js[js.index("function evReadForm()"):js.index("async function evSave()")]
    assert "JSON.parse(JSON.stringify(stored))" in read             # the record starts as the stored one
    assert "JSON.parse(JSON.stringify(orig))" in read               # ...and so does each kept gold entry
    assert "stored.gold[+row.dataset.gi]" in read                   # a form row finds its entry by stored index
    assert "g={file" not in read.replace(" ", "")                   # no gold entry is rebuilt from the form alone
    assert 'data-gi="${i}"' in js                                   # rows rendered from the record carry that index
    assert "edited(" in read                                        # only a changed field is overwritten
    # alternatives are shown read-only, in the view and in the form
    assert js.count("evAltHtml(") >= 3 and "alternatives" in js
    # ...and a kept entry whose locator was edited drops them: they name the old place
    assert "if(orig&&g.alternatives&&(edited(file)||edited(pg)||edited(h)))delete g.alternatives" in read


def test_the_position_bar_follows_the_list():
    # "n of m" and Prev/Next live in the open question's action bar. A filter or reload changes
    # the list under that pane without re-rendering it, so every list render must re-sync the bar
    # (a stale "2 of 2" with Prev enabled was a real bug, caught by a live check).
    js = _eval_script()
    assert "function evSyncBar()" in js
    assert "evSyncBar();" in js[js.index("function evRenderList()"):js.index("async function evLoadList()")]
    assert "evSyncBar();" in js[js.index("function evRenderDetail()"):js.index("/* -- actions -- */")]


def test_the_tab_css_uses_theme_tokens_only():
    # A colour literal that suits one theme is a bug in the others.
    css = HTML[HTML.index("/* ---- Eval review queue (Eval tab)"):HTML.index("</style>")]
    bodies = re.findall(r"\{([^{}]*)\}", css)       # declaration bodies only, never selectors
    assert bodies, "the Eval tab's CSS block is missing"
    for body in bodies:
        assert not re.search(r"#[0-9a-fA-F]{3,8}\b|\brgba?\(|\bhsla?\(", body), body
