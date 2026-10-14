"""The product version is set once (`VERSION`) and every place that states it
agrees, so a bump cannot forget one of them.

The contracts worth pinning:

  1. VERSION, CHANGELOG.md's newest entry and the `Noetrix version:` line in
     both skills name the same version; the changelog descends and every entry
     says something.
  2. Both APIs report it: the FastAPI apps, /health, /api/schema.
  3. /api/releases tells, per part of the install, whether it is `current`,
     `behind` (an older version, or a running service started from an older
     commit), `missing` or `not running` — never a guess, and the failure to
     check one part never hides the others.
"""
import re
from pathlib import Path

import pytest
import requests

from src.utils import release
from src.utils.config_loader import Config

ROOT = Path(__file__).resolve().parents[1]
V = release.PRODUCT_VERSION


# ------------------------------------------------------------ one version

def test_version_file_holds_a_semantic_version():
    assert re.fullmatch(r"\d+\.\d+\.\d+", (ROOT / "VERSION").read_text(encoding="utf-8").strip())


def test_the_newest_changelog_entry_is_the_version():
    entries = release.parse_changelog((ROOT / "CHANGELOG.md").read_text(encoding="utf-8"))
    assert entries and entries[0]["version"] == V


def test_changelog_versions_and_dates_descend_and_every_entry_says_something():
    entries = release.parse_changelog((ROOT / "CHANGELOG.md").read_text(encoding="utf-8"))
    versions = [tuple(map(int, e["version"].split("."))) for e in entries]
    assert versions == sorted(versions, reverse=True) and len(set(versions)) == len(versions)
    dates = [e["date"] for e in entries]
    assert dates == sorted(dates, reverse=True)
    for e in entries:
        assert any(e["sections"].values()), f"{e['version']} has no items"


@pytest.mark.parametrize("name", ["SKILL.md", "SKILL_RAGOPS.md"])
def test_each_skill_states_the_version(name):
    if not (ROOT / name).is_file():
        pytest.skip(f"{name} is not in the public release; the skills ship from their own repo")
    assert release.skill_version(ROOT / name) == V


def test_a_missing_or_malformed_version_file_raises(tmp_path):
    with pytest.raises(RuntimeError, match="missing"):
        release.read_version(tmp_path / "VERSION")
    (tmp_path / "VERSION").write_text("v1\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="MAJOR.MINOR.PATCH"):
        release.read_version(tmp_path / "VERSION")


def test_parse_changelog_folds_continuations_and_skips_the_intro():
    text = ("# Changelog\nintro line\n\n## [1.2.0] - 2026-01-02\n\nA preamble.\n\n"
            "### Added\n- one\n  more\n- two\n### Fixed\n- three\n"
            "## [1.1.0] - 2026-01-01\n### Added\n- x\n")
    e = release.parse_changelog(text)
    assert [x["version"] for x in e] == ["1.2.0", "1.1.0"]
    assert e[0]["preamble"] == "A preamble."
    assert e[0]["sections"] == {"Added": ["one more", "two"], "Fixed": ["three"]}


# ------------------------------------------------------------ the APIs say it

def test_both_apis_report_the_version(management_module):
    import serve_api
    assert serve_api.app.version == V
    assert serve_api.health()["version"] == V
    assert management_module.app.version == V
    schema = management_module.api_schema()
    assert schema["version"] == V
    assert schema["endpoints"]["GET /api/releases"]["permission"] == "read"


# ------------------------------------------------------------ /api/releases

class _Health:
    def __init__(self, body):
        self._body = body

    def json(self):
        return self._body


@pytest.fixture
def rel(management_module, tmp_path, monkeypatch):
    m = management_module
    skills = tmp_path / "skills"
    for name, ver in (("personal-rag", V), ("rag-ops", "0.1.0")):
        (skills / name).mkdir(parents=True)
        (skills / name / "SKILL.md").write_text(f"# T\n\nNoetrix version: {ver}\n", encoding="utf-8")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    cfg = Config({"webui": {"release_check": {
        "skill_dirs": [str(skills), str(tmp_path / "nowhere")],
        "bundle_app_dir": str(bundle)}}}, tmp_path)
    monkeypatch.setattr(m, "load_config", lambda *a, **k: cfg)
    head = release.git_head(release.ROOT)
    state = {"health": {"version": V, "commit": head and head["commit"]}}

    def fake_get(url, timeout=None):
        if isinstance(state["health"], Exception):
            raise state["health"]
        return _Health(state["health"])
    monkeypatch.setattr(requests, "get", fake_get)
    return m, state, bundle


def _by_name(out):
    return {c["name"]: c for c in out["components"]}


def test_releases_reports_each_part_against_the_checkout(rel):
    m, state, bundle = rel
    out = m.releases()
    c = _by_name(out)
    assert out["version"] == V
    assert c["query API"]["state"] == "current"
    skills = [x for x in out["components"] if x["name"].startswith("skill ")]
    states = sorted((x["name"].split()[1], x["state"]) for x in skills)
    assert states == [("personal-rag", "current"), ("personal-rag", "missing"),
                      ("rag-ops", "behind"), ("rag-ops", "missing")]
    # A bundle folder with no VERSION file predates versioning.
    assert c["bundle"]["state"] == "behind" and "predates" in c["bundle"]["note"]
    (bundle / "VERSION").write_text(V + "\n", encoding="utf-8")
    assert _by_name(m.releases())["bundle"]["state"] == "current"
    assert out["notes"][0]["version"] == V


def test_a_query_api_on_another_version_or_commit_is_behind(rel):
    m, state, _ = rel
    state["health"] = {"version": "0.1.0", "commit": None}
    assert _by_name(m.releases())["query API"]["state"] == "behind"
    state["health"] = {"version": V, "commit": "0000000"}
    assert _by_name(m.releases())["query API"]["state"] == "behind"
    state["health"] = {"ready": True}                    # predates version reporting
    q = _by_name(m.releases())["query API"]
    assert q["state"] == "behind" and "predates" in q["note"]


def test_an_unreachable_query_api_is_not_running_and_hides_nothing_else(rel):
    m, state, _ = rel
    state["health"] = requests.ConnectionError("refused")
    out = m.releases()
    assert _by_name(out)["query API"]["state"] == "not running"
    assert len(out["components"]) == 7       # console, query API, 4 skills, bundle


def test_a_malformed_bundle_version_is_reported_not_raised(rel):
    m, _, bundle = rel
    (bundle / "VERSION").write_text("latest\n", encoding="utf-8")
    b = _by_name(m.releases())["bundle"]
    assert b["state"] == "invalid" and "MAJOR.MINOR.PATCH" in b["note"]
