"""Bulk theme export / import in the Settings theme panel (webui/index.html).

Two kinds of check. The markup ones are grep-style, like the other tests that pin the
console's HTML. The logic ones run the page's own DOM-free block (between the
`thm-bulk:pure-*` markers) under node with a fake localStorage, because that block is what
decides whether a stored preset can ever be lost: back up an unreadable value once, never
overwrite on import, validate a file all-or-nothing. They are skipped when node is absent;
the live DOM behaviour was checked against a spare-port console.
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
needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")


def _pure_block() -> str:
    start = HTML.index("/* thm-bulk:pure-begin")
    return HTML[start:HTML.index("/* thm-bulk:pure-end */", start)]


def _run_node(scenario_js: str) -> dict:
    """Run `scenario_js` against a fresh fake localStorage; it fills `out`, which comes back as JSON.
    `load()` is one page load: the block's own state starts fresh each time, the storage does not."""
    harness = textwrap.dedent("""
        const store = new Map();
        const localStorage = {
          getItem: k => (store.has(k) ? store.get(k) : null),
          setItem: (k, v) => { store.set(k, String(v)); },
          removeItem: k => { store.delete(k); },
          get length() { return store.size; },
          key: i => [...store.keys()][i] ?? null,
        };
        const load = () => new Function("localStorage", __BLOCK__ +
          "; return {customThemes, saveCustomThemes, parseThemesFile, mergeThemes, themeEntryProblem, fault: () => themeFault};")(localStorage);
        const backups = () => [...store.keys()].filter(k => k.startsWith("ledger-custom-themes.corrupt-"));
        const out = {};
        __SCENARIO__
        console.log(JSON.stringify(out));
    """).replace("__BLOCK__", json.dumps(_pure_block())).replace("__SCENARIO__", scenario_js)
    r = subprocess.run([NODE, "-e", harness], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


# ---- markup -------------------------------------------------------------------------

def test_the_theme_panel_has_the_bulk_controls_and_a_warning_slot():
    for el in ('id="thm-export-all"', 'id="thm-import-all"', 'id="thm-import-file"',
               'id="thm-import-text"', 'id="thm-import-paste"', 'id="thm-warn"'):
        assert el in HTML, el
    assert ">Export all themes<" in HTML and ">Import themes…<" in HTML
    assert 'accept=".json,application/json"' in HTML
    assert 'id="thm-warn" role="alert"' in HTML          # a live region, so the warning is announced


def test_single_theme_export_and_save_are_untouched():
    assert ">Export current → CSS<" in HTML and ">Save as preset<" in HTML
    for line in ('toast("Active theme exported to the editor (and clipboard)");',
                 'if(!/--[\\w-]+\\s*:/.test(body)){toast("No CSS variables found',
                 't[key]={label:name,mode:$("#thm-mode").value==="light"?"light":"dark",css:body};'):
        assert line in HTML, line


def test_the_export_file_format_and_name():
    assert 'THEME_FILE_FORMAT="noetrix-themes"' in HTML and "THEME_FILE_VERSION=1" in HTML
    assert "a.download=`noetrix-themes-${ymd}.json`" in HTML
    assert "{format:THEME_FILE_FORMAT,version:THEME_FILE_VERSION,exported_at:d.toISOString(),themes:t}" in HTML


def test_custom_themes_no_longer_swallows_a_parse_error():
    fn = re.search(r"function customThemes\(\)\{.*?\n\}\n", HTML, re.S).group(0)
    assert "catch(e){return{}}" not in fn.replace(" ", "")      # the old swallow-and-return-{}
    assert "backupUnreadableThemes(raw)" in fn
    assert 'P="ledger-custom-themes.corrupt-"' in HTML


def test_the_pure_block_stays_free_of_the_dom():
    # tests/ run it under node: a document / $() / toast() reference would make it untestable
    assert not re.search(r"\bdocument\b|\$\(|toast\(|\besc\(", _pure_block())


def test_the_new_theme_css_uses_theme_tokens_only():
    rules = re.findall(r"^\.thm-msg[^{]*\{([^}]*)\}", HTML, re.M)
    assert rules, "the .thm-msg rules are missing"
    for body in rules:
        assert not re.search(r"#[0-9a-fA-F]{3,8}\b|\brgba?\(|\bhsla?\(", body), body


# ---- logic, run under node ----------------------------------------------------------

@needs_node
def test_an_unreadable_value_is_backed_up_once_then_replaced_safely():
    o = _run_node("""
        const BAD = '{"a":{"label":"x","mode":"dark","css":"--bg:#000;"';          // truncated
        store.set("ledger-custom-themes", BAD);
        let p = load();
        out.first = p.customThemes();
        out.fault = p.fault();
        out.backup_is_the_raw_value = backups().map(k => store.get(k) === BAD);
        p.customThemes(); p.customThemes();                                         // the many calls of one render
        out.after_calls = backups().length;
        p = load(); p.customThemes();                                               // a page reload
        out.after_reload = backups().length;
        out.raw_left_in_place = store.get("ledger-custom-themes") === BAD;          // reading never overwrites it
        store.set("ledger-custom-themes", "{a different bad value");
        p = load(); p.customThemes();
        out.two_distinct_values = backups().length;
        p.saveCustomThemes({k: {label: "K", mode: "dark", css: "--bg:#111;"}});     // the next save replaces the bad value...
        out.after_save = Object.keys(p.customThemes());
        out.fault_after_save = p.fault();
        out.backups_survive = backups().length;                                      // ...and the copies stay
    """)
    assert o["first"] == {}
    assert o["fault"]["key"].startswith("ledger-custom-themes.corrupt-") and "not valid JSON" in o["fault"]["why"]
    assert o["fault"]["partial"] is False
    assert o["backup_is_the_raw_value"] == [True]
    assert o["after_calls"] == 1 and o["after_reload"] == 1           # ONCE, however often it is read
    assert o["raw_left_in_place"] is True
    assert o["two_distinct_values"] == 2                              # one copy per distinct bad value
    assert o["after_save"] == ["k"] and o["fault_after_save"] is None
    assert o["backups_survive"] == 2


@needs_node
def test_a_failed_backup_is_reported_and_the_value_is_left_alone():
    # e.g. a full storage quota: no copy could be made, so the warning must say that, and nothing is lost
    o = _run_node("""
        const BAD = "{not json";
        store.set("ledger-custom-themes", BAD);
        const realSet = localStorage.setItem;
        localStorage.setItem = (k, v) => { if (k.startsWith("ledger-custom-themes.corrupt-")) throw new Error("QuotaExceededError"); realSet(k, v); };
        const p = load();
        out.themes = p.customThemes();
        out.fault = p.fault();
        out.backups = backups().length;
        out.raw_left_in_place = store.get("ledger-custom-themes") === BAD;
    """)
    assert o["themes"] == {}
    assert o["fault"]["key"] is None and "QuotaExceededError" in o["fault"]["error"]
    assert o["backups"] == 0 and o["raw_left_in_place"] is True


@needs_node
def test_a_parseable_but_unusable_value_is_treated_the_same_way():
    o = _run_node("""
        out.cases = {};
        for (const raw of ["null", "[]", '"abc"', "5", "true"]) {
          store.set("ledger-custom-themes", raw);
          const p = load();
          const t = p.customThemes();
          out.cases[raw] = {empty: Object.keys(t).length === 0, why: p.fault().why,
                            backed_up: backups().some(k => store.get(k) === raw)};
        }
        // entries without a css string would throw in themeList(): they are left out, the rest survive
        const mixed = '{"ok":{"label":"OK","mode":"dark","css":"--bg:#111;"},"bad":{"label":"x"},"worse":5}';
        store.set("ledger-custom-themes", mixed);
        const p = load();
        out.mixed = {keys: Object.keys(p.customThemes()), fault: p.fault(), backed_up: backups().some(k => store.get(k) === mixed)};
        // healthy and empty storage raise nothing
        store.clear();
        let q = load();
        out.empty = {t: q.customThemes(), fault: q.fault(), backups: backups().length};
        store.set("ledger-custom-themes", '{"a":{"label":"A","mode":"light","css":"--bg:#fff;"}}');
        q = load();
        out.healthy = {keys: Object.keys(q.customThemes()), fault: q.fault(), backups: backups().length};
    """)
    for raw, case in o["cases"].items():
        assert case["empty"] and case["backed_up"] and "not a {key: preset} map" in case["why"], raw
    assert o["mixed"]["keys"] == ["ok"] and o["mixed"]["fault"]["partial"] is True and o["mixed"]["backed_up"]
    assert "2 presets have no usable css" in o["mixed"]["fault"]["why"]
    assert o["empty"] == {"t": {}, "fault": None, "backups": 0}
    assert o["healthy"] == {"keys": ["a"], "fault": None, "backups": 0}


@needs_node
def test_the_wrapper_and_the_bare_map_both_parse_to_the_same_presets():
    o = _run_node("""
        const p = load();
        const themes = {"my-dark": {label: "  My dark ", mode: "dark", css: " --bg:#101010; --amber:#E8A33D; "},
                        "my-light": {label: "My light", mode: "light", css: "--bg:#f0f0f0;"}};
        const wrapped = JSON.stringify({format: "noetrix-themes", version: 1, exported_at: "2026-10-07T00:00:00.000Z", themes});
        out.wrapped = p.parseThemesFile(wrapped);
        out.bare = p.parseThemesFile(JSON.stringify(themes));
        // a preset that happens to be called "format" is a preset, not the wrapper
        out.format_named = Object.keys(p.parseThemesFile(JSON.stringify({format: {label: "F", mode: "dark", css: "--x:1;"}})));
    """)
    assert o["wrapped"] == o["bare"]
    assert o["wrapped"]["my-dark"] == {"label": "My dark", "mode": "dark", "css": "--bg:#101010; --amber:#E8A33D;"}
    assert o["format_named"] == ["format"]


@needs_node
def test_a_bad_file_is_refused_whole_with_a_reason():
    o = _run_node("""
        const p = load();
        const ok = '{"label":"A","mode":"dark","css":"--x:1;"}';
        const bad = {
          not_json: "{oops",
          array: "[]",
          scalar: "5",
          other_format: '{"format":"other","version":1,"themes":{}}',
          future_version: '{"format":"noetrix-themes","version":2,"themes":{}}',
          no_themes: '{"format":"noetrix-themes","version":1}',
          themes_array: '{"format":"noetrix-themes","version":1,"themes":[]}',
          no_label: '{"a":{"mode":"dark","css":"--x:1;"}}',
          blank_label: '{"a":{"label":"  ","mode":"dark","css":"--x:1;"}}',
          bad_mode: '{"a":{"label":"A","mode":"blue","css":"--x:1;"}}',
          no_mode: '{"a":{"label":"A","css":"--x:1;"}}',
          css_not_string: '{"a":{"label":"A","mode":"dark","css":5}}',
          css_brace: '{"a":{"label":"A","mode":"dark","css":"--x:1; } body{display:none"}}',
          css_no_var: '{"a":{"label":"A","mode":"dark","css":"color:red;"}}',
          entry_not_object: '{"a":"--x:1;"}',
          proto_key: '{"__proto__":' + ok + '}',
          quote_key: '{"a\\\\"b":' + ok + '}',
          bracket_key: '{"a]b":' + ok + '}',
          one_bad_among_good: '{"good":' + ok + ',"bad":{"label":"B","mode":"dark"}}',
        };
        out.msgs = {};
        for (const [name, text] of Object.entries(bad)) {
          try { p.parseThemesFile(text); out.msgs[name] = null; } catch (e) { out.msgs[name] = e.message; }
        }
    """)
    expect = {
        "not_json": "not valid JSON", "array": "Expected a JSON object", "scalar": "Expected a JSON object",
        "other_format": 'Unknown format "other"', "future_version": "Unsupported version 2",
        "no_themes": 'no "themes" map', "themes_array": 'no "themes" map',
        "no_label": "label must be", "blank_label": "label must be", "bad_mode": 'mode must be "dark" or "light"',
        "no_mode": 'mode must be "dark" or "light"', "css_not_string": "css must be a string",
        "css_brace": "no { or }", "css_no_var": "no --variable", "entry_not_object": "must be an object",
        "proto_key": "is not allowed", "quote_key": "is not allowed", "bracket_key": "is not allowed",
        "one_bad_among_good": 'Theme "bad"',
    }
    for name, fragment in expect.items():
        assert o["msgs"][name] and fragment in o["msgs"][name], (name, o["msgs"][name])


@needs_node
def test_merging_never_overwrites_and_importing_twice_changes_nothing():
    o = _run_node("""
        const p = load();
        const E = {a: {label: "A", mode: "dark", css: "--bg:#111;"}, b: {label: "B", mode: "light", css: "--bg:#eee;"}};
        const before = JSON.stringify(E);
        const file = p.parseThemesFile(JSON.stringify({format: "noetrix-themes", version: 1, themes: {
          a: {label: "A", mode: "dark", css: "--bg:#111;"},       // identical            -> skipped
          b: {label: "B", mode: "light", css: "--bg:#fff;"},      // same key, other css -> b-2
          c: {label: "C", mode: "dark", css: "--bg:#222;"},       // new                  -> imported
        }}));
        const r1 = p.mergeThemes(E, file);
        out.r1 = {imported: r1.imported, renamed: r1.renamed, skipped: r1.skipped, keys: Object.keys(r1.merged),
                  input_untouched: JSON.stringify(E) === before, b_kept: r1.merged.b.css, b2: r1.merged["b-2"].css};
        const r2 = p.mergeThemes(r1.merged, file);                // the same file again
        out.r2 = {imported: r2.imported, renamed: r2.renamed, skipped: r2.skipped,
                  unchanged: JSON.stringify(r2.merged) === JSON.stringify(r1.merged)};
        const third = p.parseThemesFile(JSON.stringify({b: {label: "B", mode: "light", css: "--bg:#aaa;"}}));
        const r3 = p.mergeThemes(r2.merged, third);               // a third variant of b
        out.r3 = {renamed: r3.renamed, keys: Object.keys(r3.merged)};
        const dup = p.mergeThemes(r3.merged, p.parseThemesFile(JSON.stringify({"b-2": {label: "B", mode: "light", css: "--bg:#fff;"}})));
        out.r4 = {skipped: dup.skipped, renamed: dup.renamed};    // already there under that very name: identical
        // the label and the mode count as content, not just the css
        const lab = p.mergeThemes(E, p.parseThemesFile(JSON.stringify({a: {label: "A renamed", mode: "dark", css: "--bg:#111;"}})));
        const mod = p.mergeThemes(E, p.parseThemesFile(JSON.stringify({a: {label: "A", mode: "light", css: "--bg:#111;"}})));
        out.r5 = {label: lab.renamed, mode: mod.renamed};
        // names that exist on every object are ordinary keys
        const c = p.mergeThemes({}, p.parseThemesFile(JSON.stringify({constructor: {label: "Ctor", mode: "dark", css: "--bg:#000;"}})));
        out.r6 = {imported: c.imported, own: Object.prototype.hasOwnProperty.call(c.merged, "constructor"), label: c.merged.constructor.label};
        // a stored entry with an odd mode is read as dark, as themeList() reads it
        const odd = p.mergeThemes({x: {label: "X", mode: "sepia", css: "--bg:#333;"}}, p.parseThemesFile(JSON.stringify({x: {label: "X", mode: "dark", css: "--bg:#333;"}})));
        out.r7 = {skipped: odd.skipped};
    """)
    assert o["r1"] == {"imported": ["c"], "renamed": [["b", "b-2"]], "skipped": ["a"],
                       "keys": ["a", "b", "b-2", "c"], "input_untouched": True, "b_kept": "--bg:#eee;", "b2": "--bg:#fff;"}
    assert o["r2"] == {"imported": [], "renamed": [], "skipped": ["a", "b", "c"], "unchanged": True}
    assert o["r3"] == {"renamed": [["b", "b-3"]], "keys": ["a", "b", "b-2", "c", "b-3"]}
    assert o["r4"] == {"skipped": ["b-2"], "renamed": []}
    assert o["r5"] == {"label": [["a", "a-2"]], "mode": [["a", "a-2"]]}
    assert o["r6"] == {"imported": ["constructor"], "own": True, "label": "Ctor"}
    assert o["r7"] == {"skipped": ["x"]}
