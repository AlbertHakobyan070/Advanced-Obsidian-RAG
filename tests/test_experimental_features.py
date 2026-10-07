"""The experimental-feature gate.

A feature behind a flag has three surfaces that can drift apart, and all three
are silent when they do:

  1. THE FLAG. Every registry row must name a real editable setting, so the
     switch goes through the ordinary config writer, validator and
     restart-reporting instead of a bespoke path nobody tested.
  2. THE SHIPPED DEFAULT. An experimental feature that ships ON is not
     experimental. This is the assertion that actually enforces the policy.
  3. THE UI GATE. The console hides each feature's controls by matching
     `data-exp="<id>"` against the registry. A renamed id, a dropped `hidden`
     attribute or a missing [hidden] CSS rule all leave an unproven panel
     sitting in the default Query tab — visibly fine, wrong by policy.

These are cheap, offline assertions precisely because the failures are not
loud: nothing errors, the feature simply stops being experimental.
"""
from pathlib import Path
from types import SimpleNamespace

import serve_api

ROOT = Path(__file__).resolve().parents[1]

# The fields the console's Experimental panel renders. A row missing one of
# these renders a card with a blank explanation, which is worse than no card:
# the reader concludes the feature has no caveat.
REQUIRED_FIELDS = ("id", "key", "label", "what", "why_off", "unaffected",
                   "surfaces")

# Registry rows whose flag changes API / pipeline behaviour only. The console has
# no controls of theirs to hide (their card in Settings > Experimental features
# is rendered from the registry itself), so there is no `data-exp` gate to find.
# Adding one to the markup for such a row fails the test below on purpose: the
# row then belongs out of this set.
NO_CONSOLE_CONTROLS = {"laya_rerank"}


def test_every_experimental_flag_is_a_real_editable_setting(management_module):
    """The registry must not invent config keys the writer cannot persist."""
    manage_api = management_module
    assert manage_api.EXPERIMENTAL_FEATURES, "registry is empty"
    for feat in manage_api.EXPERIMENTAL_FEATURES:
        key = feat["key"]
        spec = manage_api.EDITABLE_SETTINGS.get(key)
        assert spec, f"{key} is not in EDITABLE_SETTINGS — POST /api/settings "                      f"would reject the switch with 'unknown setting'"
        # Boolean-valued: the panel renders a checkbox and posts "true"/"false".
        assert spec["kind"] == "enum", f"{key} must be an enum flag"
        assert spec["values"] == ["true", "false"],             f"{key} values are {spec['values']}, not a true/false flag"


def test_every_experimental_row_can_be_explained_to_a_reader(management_module):
    for feat in management_module.EXPERIMENTAL_FEATURES:
        for field in REQUIRED_FIELDS:
            assert feat.get(field), f"{feat.get('id')!r} is missing {field!r}"
        assert feat["surfaces"], f"{feat['id']!r} lists no surfaces"


def test_experimental_features_ship_switched_off(shipped_cfg, management_module):
    """The policy itself. An experimental feature default-on is a contradiction."""
    for feat in management_module.EXPERIMENTAL_FEATURES:
        assert shipped_cfg.get(feat["key"]) is False, (
            f"{feat['key']} ships ON — an experimental feature must default to "
            f"false, so a fresh install gets the measured workflow only")


def test_settings_endpoint_reports_the_registry_with_live_values(
        management_module):
    payload = management_module.settings()
    rows = {f["id"]: f for f in payload["experimental"]}
    assert rows, "GET /api/settings dropped the experimental block"
    for feat in management_module.EXPERIMENTAL_FEATURES:
        row = rows[feat["id"]]
        assert row["on"] is False          # matches the shipped config
        # The key stays in `editable` too: the console filters it out of the
        # generic grid, and the writer needs it whitelisted either way.
        assert feat["key"] in payload["editable"]


def test_console_gates_every_feature_it_reports(management_module):
    """`data-exp` ids in the console must match the registry, both ways.

    An id in the markup that the registry no longer reports is a gate that can
    never be opened; a registry row with no gate is a flag that hides nothing
    (unless it has no console controls at all: NO_CONSOLE_CONTROLS).
    """
    import re

    html = (ROOT / "webui" / "index.html").read_text(encoding="utf-8")
    gated = set(re.findall(r'data-exp="([^"]+)"', html))
    declared = ({f["id"] for f in management_module.EXPERIMENTAL_FEATURES}
                - NO_CONSOLE_CONTROLS)
    assert gated == declared, f"markup gates {gated}, registry declares {declared}"

    # The gate is the `hidden` attribute plus the rule that makes it stick.
    # .q-compare sets display on this element, so a plain [hidden] without
    # !important would lose to it and the panel would stay visible.
    assert "[hidden]{display:none!important}" in html
    graph_panel = html[html.index('id="q-graph-panel"'):][:200]
    assert "hidden" in graph_panel,         "the graph panel must ship hidden, so it never flashes before the gate"

    # Hidden by default, revealed by the gate — not the other way round.
    assert "applyExperimentalGates" in html


def test_experimental_toggle_stays_out_of_the_save_changes_scan():
    """The toggle must not carry `data-key`.

    Settings' Save button collects edits with $$("#tab-settings [data-key]")
    and reads el.value. For a checkbox that value is the literal string "on",
    with no entry in settingsOrig to compare against — so a data-key on the
    experimental toggle posts an unknown setting and 400s the WHOLE save,
    taking every unrelated edit with it. The toggle owns its own POST and uses
    data-exp-key instead.
    """
    html = (ROOT / "webui" / "index.html").read_text(encoding="utf-8")
    toggle = html[html.index('class="exp-toggle"'):][:240]
    assert "data-exp-key=" in toggle
    assert "data-key=" not in toggle


def test_schema_marks_graph_endpoints_experimental_and_reports_the_switch(
        monkeypatch):
    """An agent must be able to plan around a disabled feature, not discover it
    through a 503."""
    monkeypatch.setitem(
        serve_api._STATE, "rag",
        SimpleNamespace(presets={},
                        graph=SimpleNamespace(file_type="canvas",
                                              enabled=False)),
    )
    contract = serve_api.schema()

    assert contract["endpoints"]["POST /graph/expand"]["experimental"] is True
    assert contract["endpoints"]["POST /graph/expand"]["available"] is False
    assert contract["endpoints"]["POST /answer"]["experimental"] is True

    block = contract["experimental"]["features"]["graph_rag"]
    assert block["config_key"] == "graph.enabled"
    assert block["enabled"] is False
    assert "POST /graph/expand" in block["endpoints"]
    # The reason must travel with the flag: "off" without "unmeasured" reads as
    # "broken".
    assert "golden set" in block["unmeasured"]
    assert "/query" in block["when_off"]


def test_schema_reports_the_feature_on_when_the_flag_is_on(monkeypatch):
    monkeypatch.setitem(
        serve_api._STATE, "rag",
        SimpleNamespace(presets={},
                        graph=SimpleNamespace(file_type="canvas",
                                              enabled=True)),
    )
    contract = serve_api.schema()
    assert contract["experimental"]["features"]["graph_rag"]["enabled"] is True
    assert contract["endpoints"]["POST /graph/expand"]["available"] is True


def test_schema_lists_every_registry_row_with_its_config_key(
        management_module, monkeypatch):
    """/schema's experimental block mirrors the console's registry. A feature added
    there and not here is one an agent cannot plan around (laya_rerank was)."""
    monkeypatch.setitem(serve_api._STATE, "rag", SimpleNamespace(presets={}))
    listed = serve_api.schema()["experimental"]["features"]
    registry = {f["id"]: f["key"] for f in management_module.EXPERIMENTAL_FEATURES}
    assert {name: block["config_key"] for name, block in listed.items()} == registry


def test_schema_reports_laya_on_only_when_the_reranker_holds_a_scorer(monkeypatch):
    for scorer, expected in ((object(), True), (None, False)):
        monkeypatch.setitem(
            serve_api._STATE, "rag",
            SimpleNamespace(presets={}, reranker=SimpleNamespace(laya_scorer=scorer)))
        block = serve_api.schema()["experimental"]["features"]["laya_rerank"]
        assert block["enabled"] is expected
        assert "POST /search" in block["endpoints"]
        assert "retrieval.laya.enabled" in block["when_off"]      # the way back on travels with it


def test_disabled_graph_503_names_both_ways_to_turn_it_on(monkeypatch):
    """The error is the only thing an API-only caller sees, so it carries both
    the console path and the config key."""
    import pytest
    from fastapi import HTTPException

    monkeypatch.setitem(
        serve_api._STATE, "rag",
        SimpleNamespace(graph=SimpleNamespace(enabled=False)),
    )
    with pytest.raises(HTTPException) as exc:
        serve_api._graph()
    detail = str(exc.value.detail)
    assert exc.value.status_code == 503
    assert "Experimental" in detail
    assert "graph.enabled: true" in detail
