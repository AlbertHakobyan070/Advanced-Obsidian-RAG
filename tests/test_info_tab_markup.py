"""The console's Info tab (webui/index.html): two interactive pipeline diagrams whose
stages show their description, their config keys and whether they are on.

Grep-style on purpose, like the other tests that pin the console's markup: the page cannot run
without a browser. Its behaviour was checked live against a spare-port console (DOM reads and
getBoundingClientRect); these tests keep the three things that drift silently from drifting:

  1. THE STAGE TABLE (the JSON in #dg-stages) against the SVG: a stage with no markup is
     never shown, markup with no row opens an empty card.
  2. THE CONFIG KEYS the table names against PIPELINE_STATE_KEYS (what /api/pipeline-state
     will report) and against the shipped config: a key in only one place shows "not in
     config.yaml" for a setting that exists, or a dot judged from a key that does not.
  3. THE SHIPPED DEFAULTS the table claims ("ships on" / "ships off") against the config.
"""
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "webui" / "index.html").read_text(encoding="utf-8")
SHIPPED = yaml.safe_load((ROOT / "config.example.yaml").read_text(encoding="utf-8-sig"))

KINDS = {"core", "optional", "auto", "exp"}
MISSING = object()
# Keys the Info tab names that config.example.yaml (the fresh-clone config) does not carry: the HyPE
# lane and the code loader run on their code defaults there. Listed, not skipped, so the test fails
# the day one is added to the example and this list stops being true.
EXAMPLE_LACKS = {
    "retrieval.hype.enabled", "retrieval.hype.collection", "retrieval.hype.top_k",
    "code.output_file", "code.extensions", "code.skip_roots", "code.max_file_bytes",
}


def _info() -> str:
    """The whole Info tab: from its section to the end of that section."""
    start = HTML.index('<section id="tab-info">')
    return HTML[start:HTML.index("</section>", start)]


def _diagram(name: str) -> str:
    """One diagram's wrapper div (dg-query or dg-ingest)."""
    info = _info()
    start = info.index(f'id="dg-{name}"')
    other = info.index('id="dg-ingest"') if name == "query" else info.index('<script type="application/json"')
    return info[start:other]


def _stages() -> dict:
    blob = re.search(r'<script type="application/json" id="dg-stages">(.*?)</script>', HTML, re.S)
    assert blob, "the stage table (#dg-stages) is missing"
    return json.loads(blob.group(1))


def _shipped(key: str):
    node = SHIPPED
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return MISSING
        node = node[part]
    return node


def _keys_the_page_names() -> set[str]:
    """Every config key the Info tab reads: stage keys, the keys its on/off rules judge from,
    and the [data-cfg] values in the knobs table."""
    names = set()
    for row in _stages().values():
        names.update(row["keys"])
        rule = row.get("on", {})
        for k in ("flag", "mode"):
            if k in rule:
                names.add(rule[k])
        names.update(rule.get("any", []))
        if "pick" in rule:
            names.add(rule["pick"]["key"])
            if "req" in rule["pick"]:
                names.add(rule["pick"]["req"])
    names.update(re.findall(r'data-cfg="([^"]+)"', _info()))
    return names


# ------------------------------------------------------------------ markup <-> table

def test_every_stage_in_the_markup_has_a_table_row_and_the_reverse():
    table = _stages()
    for name, prefix in (("query", "q-"), ("ingest", "i-")):
        ids = re.findall(r'<g class="stg" data-stage="([^"]+)"', _diagram(name))
        assert ids, f"the {name} diagram has no stages"
        assert len(ids) == len(set(ids)), f"a stage id repeats in the {name} diagram"
        assert all(i.startswith(prefix) for i in ids), ids      # the card picks its diagram from the prefix
        assert set(ids) == {k for k in table if k.startswith(prefix)}, name


def test_every_table_row_can_be_shown_to_a_reader():
    for sid, row in _stages().items():
        assert row["n"] and row["w"], f"{sid} has no name or description"
        assert row["k"] in KINDS, f"{sid} kind {row['k']!r}"
        assert isinstance(row["keys"], list)
        # a stage needs some way to say whether it is on: a rule, or the experimental registry
        assert ("on" in row) != ("exp" in row), f"{sid} needs exactly one of 'on' / 'exp'"
        assert (row["k"] == "exp") == ("exp" in row), f"{sid}: kind exp <-> registry id"
        if "on" in row:
            assert set(row["on"]) & {"always", "auto", "flag", "any", "mode", "pick"}, sid


def test_every_stage_is_a_labelled_keyboard_reachable_button():
    for g in re.findall(r'<g class="stg"[^>]*>', _info()):
        assert 'tabindex="0"' in g and 'role="button"' in g, g
        assert 'aria-pressed="false"' in g and 'aria-label="' in g, g
    # each stage needs one rect the highlight and the dot can attach to
    for block in re.findall(r'<g class="stg".*?</g>', _info(), re.S):
        assert re.search(r'class="[^"]*\bbx\b', block), block[:80]


def test_the_script_covers_hover_focus_click_enter_space_and_escape():
    js = HTML[HTML.index("/* ---------- info: interactive pipeline diagrams"):HTML.index("dgApply();   // idle cards")]
    for event in ('"mouseover"', '"mouseout"', '"focusin"', '"focusout"', '"click"', '"keydown"'):
        assert f"addEventListener({event}" in js, event
    assert '"Enter"' in js and '" "' in js and '"Escape"' in js
    assert "e.preventDefault()" in js                       # Space must not scroll the page
    assert 'g.setAttribute("aria-pressed"' in js            # the pin is announced
    # nothing here may swallow an error: the only failure path shows what failed
    assert not re.search(r"catch\(\w*\)\{\s*\}", js)
    assert "Could not read the live config" in js and "b.onclick=dgLoad" in js
    assert 'if(b.dataset.tab==="info")dgLoad();' in HTML  # fresh read each time the tab opens


# ------------------------------------------------------------------ config keys

def test_the_keys_the_page_names_are_exactly_the_keys_the_endpoint_reports(management_module):
    wanted = _keys_the_page_names()
    reported = set(management_module.PIPELINE_STATE_KEYS)
    assert wanted <= reported, f"the page reads keys the endpoint will not report: {sorted(wanted - reported)}"
    assert reported <= wanted, f"the endpoint reports keys no stage shows: {sorted(reported - wanted)}"


def test_every_key_exists_in_the_shipped_config():
    gone = {k for k in _keys_the_page_names() if _shipped(k) is MISSING}
    assert gone == EXAMPLE_LACKS, (
        f"absent from config.example.yaml: {sorted(gone)}; the test expects {sorted(EXAMPLE_LACKS)}")


def test_no_secret_is_in_the_allowlist(management_module):
    for key in management_module.PIPELINE_STATE_KEYS:
        assert not re.search(r"(^|[._])(api_key\w*|secret\w*|password\w*|token)$", key), key


def test_the_shipped_defaults_the_table_claims_are_the_shipped_defaults():
    checked = 0
    for sid, row in _stages().items():
        claim = row.get("d")
        if not claim:
            continue
        rule = row["on"]
        keys = [rule["flag"]] if "flag" in rule else rule["any"]
        if any(_shipped(k) is MISSING for k in keys):
            continue                    # the example carries no value; the code default applies
        actual = any(bool(_shipped(k)) for k in keys)
        assert actual == (claim == "on"), f"{sid} claims it ships {claim}, config.example.yaml says {actual}"
        checked += 1
    assert checked >= 4      # the optional stages are the point of the claim


def test_experimental_stages_mirror_the_registry(management_module):
    registry = {f["id"] for f in management_module.EXPERIMENTAL_FEATURES}
    on_page = {row["exp"] for row in _stages().values() if "exp" in row}
    assert on_page == registry, "every experimental feature has a stage, and no stage invents one"
    for row in _stages().values():
        if "exp" in row:
            assert row["k"] == "exp"


# ------------------------------------------------------------------ the endpoint

def test_endpoint_reports_values_and_the_registry_off_the_disk_config(management_module):
    out = management_module.pipeline_state()
    assert set(out["unset"]) == EXAMPLE_LACKS     # the console under test runs on config.example.yaml
    assert set(out["values"]) | set(out["unset"]) == set(management_module.PIPELINE_STATE_KEYS)
    assert out["values"]["retrieval.relevance_gate.enabled"] is False
    assert {f["id"] for f in out["experimental"]} == {f["id"] for f in management_module.EXPERIMENTAL_FEATURES}
    assert all(f["on"] is False for f in out["experimental"])      # experimental ships off


def test_endpoint_tells_a_missing_key_from_a_null_one(management_module, monkeypatch, tmp_path):
    from src.utils.config_loader import Config
    cfg = Config({"retrieval": {"use_hyde": True, "relevance_gate": {"threshold": None}}}, tmp_path)
    monkeypatch.setattr("src.utils.config_loader.load_config", lambda *a, **k: cfg)
    out = management_module.pipeline_state()
    assert out["values"]["retrieval.use_hyde"] is True
    assert out["values"]["retrieval.relevance_gate.threshold"] is None     # YAML null is a value
    assert "retrieval.relevance_gate.threshold" not in out["unset"]
    assert "retrieval.hype.enabled" in out["unset"]                        # absent is not null
    assert "retrieval.hype.enabled" not in out["values"]


def test_the_endpoint_is_in_the_capability_map(management_module):
    entry = management_module.api_schema()["endpoints"]["GET /api/pipeline-state"]
    assert entry["permission"] == "read"


# ------------------------------------------------------------------ content policy

def test_the_rrf_explainer_is_gone_and_fusion_carries_no_math():
    info = _info()
    assert "The fusion math" not in info and "Reciprocal Rank" not in info
    assert "RRF" not in info, "the Info tab names the stage 'fusion' without the RRF formula"
    assert "renderInfoMath" not in HTML and "infoRendered" not in HTML      # the KaTeX hook went with it
    assert not re.search(r"\\\(|\$\$|score \+=|1 / \(", info)                # no formula markup left
    assert "q-fusion" in info


def test_the_info_tab_states_no_counts_that_change():
    # Counts of lanes, modes or stages go stale the moment one is added; describe the structure.
    prose = re.sub(r'<script type="application/json".*?</script>', "", _info(), flags=re.S)
    prose = re.sub(r"<svg.*?</svg>", "", prose, flags=re.S) + " ".join(
        row["w"] for row in _stages().values())
    words = r"(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)"
    assert not re.search(rf"\b{words}\s+(?:\w+\s+)?(?:lanes|loaders|modes|stages|surfaces|tabs|presets|scorers)\b",
                         prose, re.I), re.search(
        rf"\b{words}\s+(?:\w+\s+)?(?:lanes|loaders|modes|stages|surfaces|tabs|presets|scorers)\b", prose, re.I)


def test_the_workspace_map_has_one_card_per_nav_tab_in_nav_order():
    nav = HTML[HTML.index('<nav id="nav">'):HTML.index("</nav>")]
    tabs = re.findall(r'<button data-tab="[^"]+"(?: class="on")?>([^<]+)</button>', nav)
    info = _info()
    grid = info[info.index("Workspace map"):info.index("The Query knobs")]
    assert re.findall(r'<div class="infocard">.*?<h3>([^<]+)</h3>', grid) == tabs


def test_defaults_in_the_knobs_table_are_read_live_not_typed():
    table = _info()[_info().index("The Query knobs"):]
    assert not re.search(r"Default \d", table), "a default typed into the table goes stale"
    for key in ("retrieval.rerank_top_k", "retrieval.dense_top_k", "retrieval.sparse_top_k",
                "generation.max_tokens", "retrieval.hype.enabled"):
        assert f'data-cfg="{key}"' in table, key


# ------------------------------------------------------------------ themes

def test_the_diagrams_and_their_css_use_theme_tokens_only():
    # A colour literal that suits one theme is a bug in the others.
    css = HTML[HTML.index("/* Interactive pipeline diagrams"):HTML.index("@media(prefers-reduced-motion:reduce){.stg")]
    bodies = re.findall(r"\{([^{}]*)\}", css)
    assert bodies, "the diagram CSS block is missing"
    literal = r"#[0-9a-fA-F]{3,8}\b|\brgba?\(|\bhsla?\("
    for body in bodies:
        assert not re.search(literal, body), body
    # the SVG: attributes and style="" too (the pre-existing wash uses fill="var(--amber)")
    svg_text = " ".join(re.findall(r"<svg.*?</svg>", _info(), re.S))
    assert not re.search(rf'(?:fill|stroke|color|stop-color)\s*[=:]\s*"?(?:{literal})', svg_text), \
        re.search(rf'(?:fill|stroke|color|stop-color)\s*[=:]\s*"?(?:{literal})', svg_text)
    # the highlight is not hover-only: focus and the pinned state get it too
    assert ".stg:focus-visible .bx" in css and ".stg.sel .bx" in css
    # filled vs hollow, not green vs grey alone
    assert ".dg-led.off{fill:var(--panel)" in css


def test_scope_lanes_are_drawn_as_optional():
    # The scope lanes are a toggle (retrieval.scope_lanes), so they must not read as always-on.
    row = _stages()["q-scope"]
    assert row["k"] == "optional" and row["on"] == {"flag": "retrieval.scope_lanes"}
    block = re.search(r'<g class="stg" data-stage="q-scope".*?</g>', _info(), re.S).group(0)
    assert "optional" in block and 'class="core' not in block
