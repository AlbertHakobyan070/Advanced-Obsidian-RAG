"""mcp_server.py as a process: stdio hygiene and HTTP mode.

Over stdio, stdout IS the protocol channel: one stray byte (a print, a library
banner, an unflushed buffer at exit) corrupts the session. The SDK's client
logs and skips a bad line mid-session and silently drains whatever arrives at
exit, so neither client can see a leak; a tee between client and server keeps a
copy of every byte instead. Both service URLs point at a closed port, so no
server needs to be running.
"""
import json
import os
import sys
from pathlib import Path

import anyio
import pytest
import yaml
from mcp import Client, ClientSession, StdioServerParameters, stdio_client

ROOT = Path(__file__).resolve().parents[1]

# Runs the server, forwards stdin to it, and copies its stdout both to our
# stdout (the client) and to a capture file until the server's stdout closes.
TEE = r'''
import subprocess, sys, threading
server, capture = sys.argv[1], sys.argv[2]
p = subprocess.Popen([sys.executable, server], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
def pump():
    for chunk in iter(lambda: sys.stdin.buffer.read1(65536), b""):
        p.stdin.write(chunk); p.stdin.flush()
    p.stdin.close()
threading.Thread(target=pump, daemon=True).start()
with open(capture, "wb") as f:
    for chunk in iter(lambda: p.stdout.read1(65536), b""):
        f.write(chunk); f.flush()
        sys.stdout.buffer.write(chunk); sys.stdout.buffer.flush()
open(capture + ".exit", "w").write(str(p.wait()))
'''


@pytest.fixture
def offline_dir(tmp_path):
    """A cwd whose config.yaml points both services at a closed port and does
    not wait for them."""
    cfg = yaml.safe_load((ROOT / "config.example.yaml").read_text(encoding="utf-8-sig"))
    cfg["webui"]["rag_api"] = "http://127.0.0.1:9"
    cfg["webui"]["port"] = 9
    cfg["mcp"]["startup_retry_s"] = 0
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    (tmp_path / "tee.py").write_text(TEE, encoding="utf-8")
    return tmp_path


async def _modern(params):
    async with Client(params) as c:
        tools = (await c.list_tools()).tools
        health = await c.call_tool("noetrix_health", {})
    return [t.name for t in tools], health


async def _legacy(params):
    async with stdio_client(params) as (read, write), ClientSession(read, write) as s:
        await s.initialize()
        tools = (await s.list_tools()).tools
        health = await s.call_tool("noetrix_health", {})
    return [t.name for t in tools], health


@pytest.mark.parametrize("session", [_modern, _legacy], ids=["Client", "stdio_client+ClientSession"])
def test_stdout_carries_only_protocol_bytes_during_the_session_and_at_exit(offline_dir, session):
    capture = offline_dir / "stdout.bin"
    params = StdioServerParameters(command=sys.executable,
                                   args=[str(offline_dir / "tee.py"), str(ROOT / "mcp_server.py"), str(capture)],
                                   cwd=str(offline_dir), env=dict(os.environ))
    names, health = anyio.run(session, params)

    assert names == ["noetrix_health"]                    # both services are down
    report = json.loads(health.content[0].text)
    assert not health.is_error and set(report["services_missing"]) == {"query", "console"}
    assert (offline_dir / "stdout.bin.exit").is_file()    # the tee saw the server exit: nothing is missing
    lines = [ln for ln in capture.read_bytes().split(b"\n") if ln.strip()]
    assert lines
    for ln in lines:
        assert json.loads(ln)["jsonrpc"] == "2.0", ln
    assert (offline_dir / "logs" / "mcp_server.log").read_text(encoding="utf-8").strip()


def test_rag_bat_has_a_silent_mcp_branch_and_stays_crlf():
    bat = ROOT / "rag.bat"
    if not bat.is_file():
        pytest.skip("no rag.bat in this checkout")
    raw = bat.read_bytes()
    assert raw.count(b"\n") == raw.count(b"\r\n")             # cmd.exe misparses LF-only lines
    text = raw.decode("ascii")
    branch = text[text.index('if /i "%~1"=="mcp" ('):]
    branch = branch[:branch.index("\r\n)\r\n")]
    assert '"%PYEXE%" mcp_server.py' in branch
    assert "echo" not in branch.lower()                        # stdout is the protocol channel


def test_http_mode_answers_initialize_behind_the_guard(monkeypatch):
    import mcp_server
    from starlette.testclient import TestClient

    from src.mcp_bridge.server import build_server
    from src.utils.config_loader import Config

    monkeypatch.setenv("RAG_API_TOKEN", "s3cret")
    cfg = Config({"mcp": {"port": 8053, "startup_retry_s": 0,
                          "timeouts_s": {"default": 30, "query": 180, "compare": 240, "job_log_margin": 15},
                          "search_defaults": {"include_text": 700, "max_sources": 5}},
                  "webui": {"port": 9, "rag_api": "http://127.0.0.1:9"}}, ROOT)
    app = mcp_server.build_http_app(build_server(mcp_server.services_from_config(cfg), cfg), cfg, "127.0.0.1")
    init = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-11-25", "capabilities": {},
                       "clientInfo": {"name": "test", "version": "0"}}}
    headers = {"Accept": "application/json, text/event-stream", "Authorization": "Bearer s3cret"}
    with TestClient(app, base_url="http://127.0.0.1:8053") as c:
        ok = c.post("/mcp", json=init, headers=headers)
        assert ok.status_code == 200 and '"name":"noetrix"' in ok.text
        assert c.post("/mcp", json=init, headers={**headers, "Host": "evil.example:8053"}).status_code == 400
        no_token = {k: v for k, v in headers.items() if k != "Authorization"}
        assert c.post("/mcp", json=init, headers=no_token).status_code == 401


def test_a_config_without_an_mcp_block_stops_the_server_naming_it():
    """Review Focus 3. An install whose config.yaml predates the mcp: block (an older
    checkout, a container config generated before this change) must stop at once with a
    message that names the block, not fail later inside a tool call. The guard's half is
    pinned in Task 1: guard_settings with no security: block keeps loopback only."""
    import mcp_server
    from src.utils.config_loader import Config
    with pytest.raises(SystemExit, match="mcp:"):
        mcp_server.services_from_config(Config({}, ROOT))
