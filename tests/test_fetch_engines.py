"""Settings > Fetch engines: GET /api/fetch-engines, the dry-run check, and the
engine_update job kind. PyPI, uv/pip and package metadata are all faked: nothing
here touches the network or the venv."""
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src import fetch_engines as fe
from src.mcp_bridge.catalog import console_tiers

VERSIONS = {"crawl4ai": "0.9.4", "scrapling": "0.4.15", "crawlee": "1.10.3",
            "playwright": "1.63.0", "patchright": "1.62.0", "numpy": "2.1.0"}


@pytest.fixture
def client(management_module, monkeypatch):
    monkeypatch.delenv("RAG_API_TOKEN", raising=False)
    # A loopback Host: the request guard refuses TestClient's default "testserver".
    return TestClient(management_module.app, base_url="http://127.0.0.1")


@pytest.fixture
def fake_versions(monkeypatch):
    def version(name):
        if name not in VERSIONS:
            raise fe.metadata.PackageNotFoundError(name)
        return VERSIONS[name]
    monkeypatch.setattr(fe.metadata, "version", version)


def test_pins_are_read_from_requirements_with_extras_stripped(tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text("# c\nscrapling[fetchers]==0.4.15\ncrawlee[playwright,beautifulsoup]==1.10.3  # x\n"
                   "Foo_Bar==1.0\nloose>=2\n", encoding="utf-8")
    assert fe.pins(req) == {"scrapling": "0.4.15", "crawlee": "1.10.3", "foo-bar": "1.0"}


def test_every_engine_is_pinned_in_the_real_requirements():
    assert set(fe.ENGINES) <= set(fe.pins())


def test_get_shape(client, fake_versions, monkeypatch):
    monkeypatch.setattr(fe, "pins", lambda path=fe.REQUIREMENTS: {
        "crawl4ai": "0.9.4", "scrapling": "0.4.15", "crawlee": "1.10.3",
        "playwright": "1.63.0", "patchright": "1.63.0", "markitdown": "0.1.8"})
    called = []
    monkeypatch.setattr(fe, "latest_version", lambda n: called.append(n))
    body = client.get("/api/fetch-engines").json()
    rows = {r["name"]: r for r in body["engines"]}
    assert list(rows) == list(fe.ENGINES) and not called           # GET never asks PyPI
    assert rows["scrapling"]["spec"] == "scrapling[fetchers]"
    assert rows["crawl4ai"] == {**rows["crawl4ai"], "installed": "0.9.4", "pin": "0.9.4", "status": []}
    assert rows["patchright"]["status"] == ["differs_from_pin"]
    assert rows["markitdown"]["installed"] is None and rows["markitdown"]["status"] == ["not_installed"]
    assert rows["playwright"]["post_install"] == ["python -m playwright install chromium"]
    assert rows["markitdown"]["post_install"] == []
    env = body["environment"]
    assert env["python"] == sys.executable and isinstance(env["venv"], bool)
    assert set(env) >= {"container", "installer", "note"}


def test_container_is_detected_and_carries_the_rebuild_note(monkeypatch):
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
    env = fe.environment()
    assert env["container"] is True and "image is rebuilt" in env["note"]
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST")
    monkeypatch.setattr(fe.Path, "exists", lambda self: False)
    assert fe.environment() == {**fe.environment(), "container": False, "note": None}


UV_OUT = """Resolved 40 packages in 812ms
Would download 2 packages
Would uninstall 2 packages
Would install 2 packages
 - crawl4ai==0.9.4
 + crawl4ai==0.9.6
 - numpy==2.1.0
 + numpy==2.3.1
"""
PIP_OUT = """Requirement already satisfied: requests in x (2.32.5)
Would install crawl4ai-0.9.6 numpy-2.3.1 typing_extensions-4.15.0
"""


def test_dry_run_parsing_uv_and_pip(fake_versions):
    assert fe.parse_dry_run(UV_OUT) == [
        {"package": "crawl4ai", "from": "0.9.4", "to": "0.9.6"},
        {"package": "numpy", "from": "2.1.0", "to": "2.3.1"}]
    assert fe.parse_dry_run(PIP_OUT) == [
        {"package": "crawl4ai", "from": "0.9.4", "to": "0.9.6"},
        {"package": "numpy", "from": "2.1.0", "to": "2.3.1"},
        {"package": "typing_extensions", "from": None, "to": "4.15.0"}]
    assert fe.parse_dry_run("Requirement already satisfied: crawl4ai\n") == []


@pytest.mark.parametrize("uv", ["C:/bin/uv.exe", None])
def test_install_argv_uv_and_pip(monkeypatch, uv):
    monkeypatch.setattr(fe.shutil, "which", lambda name: uv)
    argv = fe.install_argv("scrapling", dry_run=True)
    if uv:
        assert argv == [uv, "pip", "install", "--dry-run", "--python", sys.executable,
                        "--upgrade-package", "scrapling", "scrapling[fetchers]"]
    else:
        assert argv == [sys.executable, "-m", "pip", "install", "--dry-run", "-U", "scrapling[fetchers]"]
    assert "--dry-run" not in fe.install_argv("scrapling", dry_run=False)


def _fake_run(out, rc=0, seen=None):
    def run(argv, **kw):
        if seen is not None:
            seen.append(argv)
        return SimpleNamespace(returncode=rc, stdout="", stderr=out)
    return run


def test_check_reports_latest_and_changes(client, fake_versions, monkeypatch):
    monkeypatch.setattr(fe.shutil, "which", lambda name: "uv")
    monkeypatch.setattr(fe, "latest_version", lambda n: "0.9.6")
    seen = []
    monkeypatch.setattr(fe.subprocess, "run", _fake_run(UV_OUT, seen=seen))
    r = client.post("/api/fetch-engines/crawl4ai/check").json()
    assert r["ok"] is True and r["error"] is None
    assert (r["installed"], r["pin"], r["latest"]) == ("0.9.4", "0.9.4", "0.9.6")
    assert r["status"] == ["update_available"]
    assert [c["package"] for c in r["changes"]] == ["crawl4ai", "numpy"]
    assert "--dry-run" in seen[0] and "--dry-run" in r["command"] and "+ numpy==2.3.1" in r["raw_tail"]


def test_check_failures_are_reported_not_swallowed(fake_versions, monkeypatch):
    monkeypatch.setattr(fe.shutil, "which", lambda name: None)

    def boom(n):
        raise ConnectionError("pypi.org unreachable")
    monkeypatch.setattr(fe, "latest_version", boom)
    monkeypatch.setattr(fe.subprocess, "run",
                        _fake_run("ERROR: no such option: --dry-run", rc=2))
    r = fe.check("crawl4ai")
    assert r["ok"] is False and r["status"][0] == "check_failed"
    assert "pypi.org unreachable" in r["error"] and "older than 22.2" in r["error"]

    def slow(argv, **kw):
        raise subprocess.TimeoutExpired(argv, 1)
    monkeypatch.setattr(fe, "latest_version", lambda n: "0.9.4")
    monkeypatch.setattr(fe.subprocess, "run", slow)
    r = fe.check("crawl4ai")
    assert r["ok"] is False and "timed out" in r["error"] and r["latest"] == "0.9.4"


def test_check_refuses_an_unknown_name(client):
    r = client.post("/api/fetch-engines/numpy/check")
    assert r.status_code == 400 and "unknown fetch engine" in r.json()["error"]


def test_engine_update_argv_and_refusal(management_module, monkeypatch):
    m = management_module
    assert m._build_argv("engine_update", {"name": "crawlee"}) == \
        [sys.executable, "-m", "src.fetch_engines", "crawlee"]
    for prm in ({}, {"name": "numpy"}, {"name": "crawl4ai==0.1"}, {"name": "-r requirements.txt"}):
        with pytest.raises(ValueError, match="engine_update name must be one of"):
            m._build_argv("engine_update", prm)
    monkeypatch.setattr(m, "_JOBS", {})
    monkeypatch.setattr(m, "_ORDER", [])
    r = m.jobs_create(m.JobIn(kind="engine_update", params={"name": "torch"}))
    assert r.status_code == 400 and "engine_update name" in json.loads(r.body)["error"]


def test_update_commands_per_engine(monkeypatch):
    monkeypatch.setattr(fe.shutil, "which", lambda name: None)
    py = sys.executable
    pw, pr = [py, "-m", "playwright", "install", "chromium"], [py, "-m", "patchright", "install", "chromium"]
    pip = lambda spec: [py, "-m", "pip", "install", "-U", spec]
    assert fe.update_commands("crawl4ai") == [pip("crawl4ai"), pw, pr]
    assert fe.update_commands("scrapling") == [pip("scrapling[fetchers]"), pw, pr]
    assert fe.update_commands("crawlee") == [pip("crawlee[playwright,beautifulsoup]"), pw]
    assert fe.update_commands("playwright") == [pip("playwright"), pw]
    assert fe.update_commands("patchright") == [pip("patchright"), pr]
    assert fe.update_commands("markitdown") == [pip("markitdown[pdf,docx,pptx,xlsx]")]
    with pytest.raises(ValueError):
        fe.update_commands("numpy")


def test_main_runs_each_command_and_stops_on_failure(monkeypatch, fake_versions):
    monkeypatch.setattr(fe.shutil, "which", lambda name: None)
    ran = []

    def run(cmd, check):
        ran.append(cmd)
        if "patchright" in cmd:
            raise subprocess.CalledProcessError(1, cmd)
    monkeypatch.setattr(fe.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        fe.main(["crawl4ai"])
    assert len(ran) == 3 and ran[1][2] == "playwright"


def test_schema_lists_the_kind_and_both_routes(management_module):
    schema = management_module.api_schema()
    kind = schema["job_kinds"]["engine_update"]
    assert kind["permission"] == "mutating"
    assert set(kind["params"]["name"].split("|")) == set(fe.ENGINES)
    assert schema["endpoints"]["GET /api/fetch-engines"]["permission"] == "read"
    assert schema["endpoints"]["POST /api/fetch-engines/{name}/check"]["permission"] == "read"
    assert console_tiers(schema)[2]["engine_update"] == "mutating"


def test_dry_run_parsing_strips_uvs_colour_codes(fake_versions):
    """Real uv output (captured from this PC's venv): coloured even into a pipe."""
    out = ("\x1b[2mResolved \x1b[1m23 packages\x1b[0m\x1b[0m\n"
           " \x1b[31m-\x1b[39m \x1b[1mscrapling\x1b[0m\x1b[2m==0.4.15\x1b[0m\n"
           " \x1b[32m+\x1b[39m \x1b[1mscrapling\x1b[0m\x1b[2m==0.5\x1b[0m")
    assert fe.parse_dry_run(out) == [{"package": "scrapling", "from": "0.4.15", "to": "0.5"}]
