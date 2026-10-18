"""The MCP server's building blocks (src/mcp_bridge).

Every schema here is captured from the real apps (app.openapi(),
serve_api.schema(), manage_api.api_schema()), never written by hand, and every
HTTP call goes to the real apps in-process through Starlette's TestClient. No
network and no running servers.
"""
from pathlib import Path

import pytest

from src.utils.config_loader import load_config

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- Task 5: config + SDK

@pytest.mark.parametrize("name", ["config.example.yaml", "config.yaml"])
def test_the_mcp_config_block_loads(name):
    if not (ROOT / name).is_file():
        pytest.skip(f"no {name} in this checkout")
    mcp = load_config(ROOT / name).get("mcp")
    assert set(mcp) == {"port", "startup_retry_s", "timeouts_s", "search_defaults"}
    assert set(mcp["timeouts_s"]) == {"default", "query", "compare", "job_log_margin"}
    assert set(mcp["search_defaults"]) == {"include_text", "max_sources"}
    numbers = [mcp["port"], mcp["startup_retry_s"], *mcp["timeouts_s"].values(),
               *mcp["search_defaults"].values()]
    assert all(isinstance(v, (int, float)) for v in numbers)


def test_the_sdk_is_installed_and_our_package_does_not_shadow_it():
    import mcp
    from mcp.server.lowlevel import Server  # noqa: F401

    import src.mcp_bridge  # noqa: F401
    assert not Path(mcp.__file__).resolve().is_relative_to(ROOT)


def test_the_first_tools_list_waits_less_than_the_clients_startup_budgets():
    """Review Focus 1. A service that is down when an agent connects makes the first
    tools/list wait up to mcp.startup_retry_s. Codex gives a server 10 s to start by
    default and Claude Code 30 s, so the wait must stay under 10 s. The down service's
    tools appear on a later tools/list."""
    mcp = load_config(ROOT / "config.example.yaml").get("mcp")
    assert mcp["startup_retry_s"] < 10


# ---------------------------------------------------------------- Task 6: catalog
import json
from types import SimpleNamespace

from jsonschema import Draft202012Validator

from src.mcp_bridge import catalog as C


@pytest.fixture
def query_docs(monkeypatch):
    """:8051's /openapi.json and GET /schema. The schema reads the warm pipeline
    only for its live preset names."""
    import serve_api
    monkeypatch.setitem(serve_api._STATE, "rag", SimpleNamespace(presets={"code": {"rerank_top_k": 11}}))
    return serve_api.app.openapi(), serve_api.schema()


@pytest.fixture
def console_docs(management_module):
    return management_module.app.openapi(), management_module.api_schema()


@pytest.fixture
def routes(query_docs, console_docs):
    q_tiers, q_purposes = C.query_tiers(query_docs[1])
    c_tiers, c_purposes, _ = C.console_tiers(console_docs[1])
    return {**C.build_routes(query_docs[0], q_tiers, q_purposes, "query"),
            **C.build_routes(console_docs[0], c_tiers, c_purposes, "console")}


def _all_keys(*openapis):
    return {f"{m.upper()} {p}" for doc in openapis for p, ops in doc["paths"].items() for m in ops}


def test_every_input_schema_stands_alone(routes):
    assert len(routes) > 60
    for key, r in routes.items():
        assert "$ref" not in json.dumps(r.input_schema), key
        Draft202012Validator.check_schema(r.input_schema)
        assert r.input_schema["type"] == "object" and r.input_schema["additionalProperties"] is False, key


def test_nested_models_are_inlined(routes):
    branch = routes["POST /compare"].input_schema["properties"]["branches"]["items"]
    assert {"id", "rerank", "provider"} <= set(branch["properties"])
    group = routes["POST /api/ingest_custom"].input_schema["properties"]["groups"]["items"]
    assert group["required"] == ["kind", "files"]


def test_tiers_and_purposes_come_from_the_capability_maps(routes, console_docs):
    assert routes["POST /config"].tier == "mutating"
    assert routes["POST /search"].tier == "read"
    assert routes["POST /search"].purpose.startswith("retrieval only")
    assert routes["GET /api/jobs/{jid}"].tier == "read"            # the map writes {id}
    assert routes["GET /api/service/log"].tier == "read"           # the map writes ?lines=
    assert routes["POST /api/documents/delete"].tier == "destructive"
    assert {r.service for k, r in routes.items() if k.startswith("GET /api/")} == {"console"}
    kinds = C.console_tiers(console_docs[1])[2]
    assert kinds["index_rebuild"] == "destructive"
    assert kinds["index_append"] == "mutating"
    assert "read" not in kinds.values()                   # no job kind is read-tier


def test_path_query_and_body_are_split_like_the_route(routes):
    log = routes["GET /api/jobs/{jid}/log"]
    assert (log.path_params, log.query_params, log.body) == (("jid",), ("offset", "tail", "wait"), False)
    assert log.input_schema["required"] == ["jid"]
    assert log.input_schema["properties"]["tail"]["maximum"] == 500
    search = routes["POST /search"]
    assert (search.path_params, search.query_params, search.body) == ((), (), True)
    assert search.input_schema["required"] == ["q"]
    review = routes["POST /api/eval/questions/{qid}"]
    assert review.path_params == ("qid",) and review.body
    assert "qid" in review.input_schema["required"] and "action" in review.input_schema["properties"]


def test_unexposed_routes_are_left_out_and_each_names_a_real_route(routes, query_docs, console_docs):
    every = _all_keys(query_docs[0], console_docs[0])
    assert set(C.NOT_EXPOSED) <= every
    assert not set(C.NOT_EXPOSED) & set(routes)
    for doc in (query_docs[0], console_docs[0]):
        for path, ops in doc["paths"].items():
            for method, op in ops.items():
                body = op.get("requestBody", {}).get("content", {})
                if body and "application/json" not in body:
                    assert f"{method.upper()} {path}" in C.NOT_EXPOSED


def test_a_route_with_no_tier_is_left_out_and_logged(query_docs, caplog):
    tiers, purposes = C.query_tiers(query_docs[1])
    del tiers["GET /stats"]
    routes = C.build_routes(query_docs[0], tiers, purposes, "query")
    assert "GET /stats" not in routes and "GET /health" in routes
    assert "GET /stats has no tier" in caplog.text


def test_a_query_api_without_a_permissions_map_is_refused(query_docs):
    doc = dict(query_docs[1])
    del doc["permissions"]
    with pytest.raises(ValueError, match="restart :8051"):
        C.query_tiers(doc)


# ---------------------------------------------------------------- Task 7: client
from urllib.parse import urlsplit

import requests
from fastapi.testclient import TestClient

from src.mcp_bridge import client as K

QUERY_URL, CONSOLE_URL = "http://127.0.0.1:8051", "http://127.0.0.1:8052"
TIMEOUTS = {"default": 30, "query": 180, "compare": 240, "job_log_margin": 15}


@pytest.fixture
def services():
    return K.Services(QUERY_URL, CONSOLE_URL, dict(TIMEOUTS))


@pytest.fixture
def apps(monkeypatch, management_module):
    """`requests.request` answered by the real apps in-process, picked by port;
    any other port is a refused connection. management_module also puts a
    config.yaml in the cwd, which the query API's guard reads on first use."""
    import serve_api
    monkeypatch.delenv("RAG_API_TOKEN", raising=False)
    clients = {8051: TestClient(serve_api.app, base_url=QUERY_URL),
               8052: TestClient(management_module.app, base_url=CONSOLE_URL)}
    sent = []

    def fake(method, url, *, params=None, json=None, headers=None, timeout=None):
        sent.append({"method": method, "url": url, "params": params, "json": json,
                     "headers": headers, "timeout": timeout})
        port = urlsplit(url).port
        if port not in clients:
            raise requests.ConnectionError(f"connection refused: {url}")
        return clients[port].request(method, url, params=params, json=json, headers=headers)

    monkeypatch.setattr(K.requests, "request", fake)
    return SimpleNamespace(query=serve_api, console=management_module, sent=sent)


def _fake_rag(generate=None, search_error=None):
    doc = SimpleNamespace(id="chunk-1", text="retrieved evidence", metadata={}, debug={},
                          rerank_score=1.0, score=0.0, source_label="")
    llm = SimpleNamespace(backend="minimax", provider="anthropic", model="M")

    def search(q, **kw):
        if search_error:
            raise search_error
        return [doc], {"rerank_mode": "lexical"}
    return SimpleNamespace(presets={}, search=search, generator=SimpleNamespace(llm=llm, generate=generate))


def test_search_200_with_error_is_an_error(apps, routes, services, monkeypatch):
    monkeypatch.setitem(apps.query._STATE, "rag", _fake_rag(search_error=ValueError("unknown preset 'nope'")))
    out = K.call_route(routes["POST /search"], {"q": "x"}, services, timeout=5)
    assert out.is_error and "unknown preset 'nope'" in out.text


def test_query_confidence_error_is_an_error_that_keeps_the_sources(apps, routes, services, monkeypatch):
    def generate(*a, **kw):
        raise RuntimeError("rate limit reached")
    monkeypatch.setitem(apps.query._STATE, "rag", _fake_rag(generate=generate))
    out = K.call_route(routes["POST /query"], {"q": "x", "include_text": 100}, services, timeout=5)
    body = json.loads(out.text)
    assert out.is_error and body["confidence"] == "ERROR"
    assert body["sources"][0]["id"] == "chunk-1"


def test_low_confidence_is_an_answer_not_an_error(apps, routes, services, monkeypatch):
    answer = SimpleNamespace(text="Nothing relevant was found.", confidence="LOW", citations=[],
                             sources=[], usage=None, timings={}, retrieval=None)
    monkeypatch.setitem(apps.query._STATE, "rag", _fake_rag(generate=lambda *a, **kw: answer))
    out = K.call_route(routes["POST /query"], {"q": "x"}, services, timeout=5)
    assert not out.is_error and json.loads(out.text)["confidence"] == "LOW"


def test_422_is_an_error_with_the_field_list(apps, routes, services):
    out = K.call_route(routes["POST /search"], {}, services, timeout=5)
    assert out.is_error and "HTTP 422" in out.text and "body.q: Field required" in out.text


def test_503_is_an_error_with_the_reason(apps, routes, services, monkeypatch):
    monkeypatch.delitem(apps.query._STATE, "rag")
    out = K.call_route(routes["POST /search"], {"q": "x"}, services, timeout=5)
    assert out.is_error and "HTTP 503" in out.text and "still loading" in out.text


def test_a_guard_refusal_is_an_error_naming_the_setting(apps, routes):
    elsewhere = K.Services("http://evil.example:8051", CONSOLE_URL, dict(TIMEOUTS))
    out = K.call_route(routes["GET /health"], {}, elsewhere, timeout=5)
    assert out.is_error and "HTTP 400" in out.text and "security.allowed_hosts" in out.text


def test_a_console_refusal_is_an_error_carrying_its_message(apps, routes, services):
    out = K.call_route(routes["POST /api/jobs"], {"kind": "no_such_kind", "params": {}}, services, timeout=5)
    assert out.is_error and "HTTP 400" in out.text and "no_such_kind" in out.text


def test_refused_and_timed_out_connections_name_the_service_and_port(routes, services, monkeypatch):
    def refused(*a, **kw):
        raise requests.ConnectionError("refused")
    monkeypatch.setattr(K.requests, "request", refused)
    out = K.call_route(routes["GET /stats"], {}, services, timeout=5)
    assert out.is_error and "the query API at http://127.0.0.1:8051 is not reachable" in out.text

    def slow(*a, **kw):
        raise requests.Timeout("slow")
    monkeypatch.setattr(K.requests, "request", slow)
    out = K.call_route(routes["GET /api/overview"], {}, services, timeout=7)
    assert out.is_error and "http://127.0.0.1:8052 did not answer GET /api/overview within 7 s" in out.text


def test_a_call_puts_each_param_where_the_route_wants_it_and_sends_the_token(routes, services, monkeypatch):
    sent = []

    def record(method, url, **kw):
        sent.append((method, url, kw))
        return SimpleNamespace(status_code=200, json=lambda: {"ok": True}, text="")
    monkeypatch.setattr(K.requests, "request", record)
    monkeypatch.setenv("RAG_API_TOKEN", "s3cret")
    out = K.call_route(routes["GET /api/jobs/{jid}/log"], {"jid": "a b", "tail": 5}, services, timeout=9)
    method, url, kw = sent[0]
    assert (method, url) == ("GET", CONSOLE_URL + "/api/jobs/a%20b/log")
    assert kw["params"] == {"tail": 5} and kw["json"] is None and kw["timeout"] == 9
    assert kw["headers"] == {"Authorization": "Bearer s3cret"}
    assert not out.is_error
    K.call_route(routes["POST /api/eval/questions/{qid}"], {"qid": "q1", "action": "verify"}, services, timeout=9)
    assert sent[1][1].endswith("/api/eval/questions/q1") and sent[1][2]["json"] == {"action": "verify"}


def test_each_route_gets_its_configured_timeout(routes):
    assert K.timeout_for(routes["GET /api/jobs/{jid}/log"], {"wait": 40}, TIMEOUTS) == 55
    assert K.timeout_for(routes["GET /api/jobs/{jid}/log"], {}, TIMEOUTS) == 15
    assert K.timeout_for(routes["POST /query"], {}, TIMEOUTS) == 180
    assert K.timeout_for(routes["POST /compare"], {}, TIMEOUTS) == 240
    assert K.timeout_for(routes["GET /stats"], {}, TIMEOUTS) == 30


def test_fetch_catalog_reads_both_services(apps, services, query_docs):
    cat = K.fetch_catalog(services, retry_s=0)
    assert cat.missing == {}
    assert "POST /search" in cat.routes and "POST /api/ingest_custom" in cat.routes
    assert cat.job_kind_tiers["index_rebuild"] == "destructive"
    # retry_s 0 (the refresh a tool call makes): one round, each request at the default timeout
    assert {s["timeout"] for s in apps.sent} == {TIMEOUTS["default"]}


def test_fetch_catalog_records_a_service_that_is_down(apps):
    down = K.Services(QUERY_URL, "http://127.0.0.1:9", dict(TIMEOUTS))
    cat = K.fetch_catalog(down, retry_s=0)
    assert "console" in cat.missing and "http://127.0.0.1:9" in cat.missing["console"]
    assert not any(r.service == "console" for r in cat.routes.values())


def test_fetch_catalog_retries_a_503_until_it_clears(apps, services, monkeypatch):
    monkeypatch.setattr(K, "RETRY_INTERVAL_S", 0)
    monkeypatch.delitem(apps.query._STATE, "rag", raising=False)
    calls = []
    inner = K.requests.request

    def heal_after_first_schema(method, url, **kw):
        resp = inner(method, url, **kw)
        if url.endswith("/schema") and not calls:
            calls.append(resp.status_code)
            apps.query._STATE["rag"] = SimpleNamespace(presets={})
        return resp
    monkeypatch.setattr(K.requests, "request", heal_after_first_schema)
    try:
        cat = K.fetch_catalog(services, retry_s=30)
    finally:
        apps.query._STATE.pop("rag", None)
    assert calls == [503] and cat.missing == {} and "POST /search" in cat.routes


def test_fetch_catalog_stops_at_its_deadline_against_a_refused_port():
    """Review Focus 1, as behaviour. A refused connect to 127.0.0.1 costs about 2 s on
    Windows (the OS retries the SYN), so with the full default timeout per request a down
    service held the first tools/list well past retry_s. Every catalog request is now
    capped by what is left of retry_s, and no round starts that cannot finish before it.
    The bound is retry_s plus one refused connect at its slowest here."""
    import socket
    import time
    with socket.socket() as s:                      # a port nothing listens on
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    down = K.Services(f"http://127.0.0.1:{port}", f"http://127.0.0.1:{port}", dict(TIMEOUTS))
    retry_s = 1.5
    t0 = time.monotonic()
    cat = K.fetch_catalog(down, retry_s=retry_s)
    elapsed = time.monotonic() - t0
    assert set(cat.missing) == {"query", "console"}
    assert elapsed < retry_s + 2.0, f"{elapsed:.2f} s"


def test_a_malformed_openapi_document_drops_only_that_service(apps, services, monkeypatch):
    monkeypatch.setitem(apps.query._STATE, "rag", _fake_rag())
    inner = K.requests.request

    def broken_console_openapi(method, url, **kw):
        if url == CONSOLE_URL + "/openapi.json":
            return SimpleNamespace(status_code=200, json=lambda: {"openapi": "3.1.0"}, text="")
        return inner(method, url, **kw)
    monkeypatch.setattr(K.requests, "request", broken_console_openapi)
    cat = K.fetch_catalog(services, retry_s=0)
    assert "POST /search" in cat.routes
    assert not any(r.service == "console" for r in cat.routes.values())
    assert "KeyError: 'paths'" in cat.missing["console"]


# ---------------------------------------------------------------- Task 8: tools + server
import anyio
import mcp_types as types
from mcp import Client

from src.mcp_bridge import server as S
from src.mcp_bridge import tools as T
from src.utils.config_loader import Config


def _cfg():
    return Config({"mcp": {"port": 8053, "startup_retry_s": 0, "timeouts_s": dict(TIMEOUTS),
                           "search_defaults": {"include_text": 700, "max_sources": 5}},
                   "webui": {"port": 8052}}, ROOT)


def _with_client(server, fn):
    async def go():
        async with Client(server) as c:
            return await fn(c)
    return anyio.run(go)


def _list(server) -> dict[str, types.Tool]:
    async def fn(c):
        return {t.name: t for t in (await c.list_tools()).tools}
    return _with_client(server, fn)


def _call(server, name, args):
    async def fn(c):
        return await c.call_tool(name, args)
    return _with_client(server, fn)


@pytest.fixture
def live(apps, services, monkeypatch):
    """A server over both real apps, with a warm-pipeline stand-in on :8051."""
    monkeypatch.setitem(apps.query._STATE, "rag", _fake_rag())
    return SimpleNamespace(server=S.build_server(services, _cfg()), apps=apps)


def test_tools_carry_their_tier_annotations_and_the_prompt_flag(live):
    tools = _list(live.server)
    named = {t.name for t in T.NAMED} - {"noetrix_engines", "noetrix_set_engine"}   # those arrive with Part 3
    assert named | set(T.DISPATCHERS) <= set(tools)
    for t in tools.values():
        a = t.annotations
        if a.read_only_hint:
            assert t.meta is None, t.name
        else:
            assert t.meta == T.PROMPT_META, t.name
    assert tools["noetrix_search"].annotations.open_world_hint is True
    assert tools["noetrix_get_chunk"].annotations.open_world_hint is False
    assert tools["noetrix_ingest"].annotations.read_only_hint is False
    assert tools["noetrix_ingest"].annotations.destructive_hint is False
    assert tools["noetrix_remove"].annotations.destructive_hint is True
    assert {n for n, t in tools.items() if t.meta} >= {"noetrix_ingest", "noetrix_change", "noetrix_remove"}
    assert tools["noetrix_search"].input_schema["required"] == ["q"]
    assert tools["noetrix_search"].description.startswith("retrieval only")


def test_dispatchers_list_their_own_tiers_ops(live):
    tools = _list(live.server)
    read, change, remove = (tools[n].description for n in ("noetrix_read", "noetrix_change", "noetrix_remove"))
    assert "GET /api/schema" in read and "POST /api/jobs (kind" not in read   # no read-tier job kind
    assert "POST /api/settings" in change and "POST /api/documents/delete" not in change
    assert "POST /api/documents/delete" in remove and "POST /api/jobs (kind: index_rebuild)" in remove


@pytest.mark.parametrize("name, args, expected", [
    ("noetrix_read", {"op": "POST /api/settings", "params": {"changes": {"retrieval.rerank_top_k": 8}}},
     "is mutating, not read: call noetrix_change instead"),
    ("noetrix_change", {"op": "POST /api/jobs", "params": {"kind": "index_rebuild"}},
     "with kind 'index_rebuild' is destructive, not mutating: call noetrix_remove instead"),
    ("noetrix_change", {"op": "POST /api/jobs", "params": {"kind": "no_such_kind"}},
     "job kind 'no_such_kind' has no tier"),
    ("noetrix_read", {"op": "GET /api/import/file", "params": {"name": "a.pdf"}},
     "is not exposed over MCP"),
    ("noetrix_read", {"op": "GET /nowhere"}, "is not an exposed route"),
    ("noetrix_search", {"q": 5}, "q: 5 is not of type 'string'"),
    ("noetrix_search", {"q": "x", "bogus": 1}, "Additional properties are not allowed"),
    ("noetrix_read", {"op": "GET /api/jobs/{id}/log", "params": {"jid": "j", "tail": 9999}},
     "tail: 9999 is greater than the maximum of 500"),
])
def test_refusals_happen_before_any_http_call(live, name, args, expected):
    _list(live.server)
    live.apps.sent.clear()
    result = _call(live.server, name, args)
    assert result.is_error and expected in result.content[0].text
    assert live.apps.sent == []


def test_an_untiered_route_is_refused(query_docs):
    tiers, purposes = C.query_tiers(query_docs[1])
    del tiers["GET /stats"]
    cat = K.Catalog(routes=C.build_routes(query_docs[0], tiers, purposes, "query"))
    out = T.resolve(cat, "noetrix_read", {"op": "GET /stats"}, {})
    assert out.is_error and "has no tier" in out.text


def test_search_defaults_fill_only_the_fields_left_unset(live):
    _list(live.server)
    result = _call(live.server, "noetrix_search", {"q": "x", "max_sources": 2})
    assert not result.is_error
    assert live.apps.sent[-1]["json"] == {"q": "x", "include_text": 700, "max_sources": 2}


def test_noetrix_job_reads_the_status_or_the_log(live, tmp_path, monkeypatch):
    m = live.apps.console
    monkeypatch.setattr(m, "JOBS_DIR", tmp_path)
    monkeypatch.setattr(m, "_JOBS", {})
    monkeypatch.setattr(m, "_ORDER", [])
    job = m.Job(id="j1", kind="index_append", argv=["py", "main.py"], params={}, status="done")
    job.log_file = str(tmp_path / "j1.log")
    (tmp_path / "j1.log").write_text("line one\nline two\n", encoding="utf-8")
    m._JOBS["j1"] = job
    _list(live.server)
    status = _call(live.server, "noetrix_job", {"jid": "j1"})
    assert json.loads(status.content[0].text)["status"] == "done"
    assert live.apps.sent[-1]["url"].endswith("/api/jobs/j1")
    log = _call(live.server, "noetrix_job", {"jid": "j1", "tail": 1, "wait": 2})
    assert json.loads(log.content[0].text)["data"] == "line two"
    assert live.apps.sent[-1]["url"].endswith("/api/jobs/j1/log") and live.apps.sent[-1]["timeout"] == 17


def test_an_exception_in_a_handler_becomes_an_error_result(live, monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("boom")
    monkeypatch.setattr(S, "resolve", boom)
    result = _call(live.server, "noetrix_stats", {})
    assert result.is_error and "RuntimeError: boom" in result.content[0].text


def test_a_missing_service_is_picked_up_on_the_next_tools_list(live):
    live.apps.query._STATE.pop("rag")                 # :8051 answers 503
    first = _list(live.server)
    assert "noetrix_search" not in first and "noetrix_documents" in first
    report = json.loads(_call(live.server, "noetrix_health", {}).content[0].text)
    assert "HTTP 503" in report["services_missing"]["query"]
    assert "noetrix_search" in report["tools"]["not_loaded"]
    live.apps.query._STATE["rag"] = _fake_rag()
    assert "noetrix_search" in _list(live.server)


def test_a_tool_call_retries_a_missing_service_without_a_second_list(live):
    """Claude Code and Codex list the tools once per connection. A service that was down
    at that list must still be reachable by a later call, and noetrix_health must not
    keep reporting it missing."""
    live.apps.query._STATE.pop("rag")                 # :8051 answers 503
    assert "noetrix_search" not in _list(live.server)
    live.apps.query._STATE["rag"] = _fake_rag()
    for name, args in (("noetrix_search", {"q": "x"}),
                       ("noetrix_read", {"op": "POST /search", "params": {"q": "x"}})):
        result = _call(live.server, name, args)
        assert not result.is_error, result.content[0].text
    report = json.loads(_call(live.server, "noetrix_health", {}).content[0].text)
    assert report["services_missing"] == {}


def test_the_dispatcher_ops_line_resolves_as_written(live):
    """The dispatcher description lists `POST /api/jobs (kind: ...)`; an agent that
    pastes that line as op must reach the route."""
    assert C.norm_key("POST /api/jobs (kind: ingest_md | index_append)") == "POST /api/jobs"
    result = _call(live.server, "noetrix_change",
                   {"op": "POST /api/jobs (kind: index_rebuild)", "params": {"kind": "index_rebuild"}})
    assert result.is_error and "is destructive, not mutating" in result.content[0].text


def test_the_schema_lookup_serves_the_cached_schemas(live):
    assert live.server.get_tool_input_schema("noetrix_search") is None      # nothing listed yet
    tools = _list(live.server)
    assert live.server.get_tool_input_schema("noetrix_search") == tools["noetrix_search"].input_schema
    assert live.server.get_tool_input_schema("nope") is None


def test_noetrix_health_reports_both_services_and_the_tools(live):
    _list(live.server)
    report = json.loads(_call(live.server, "noetrix_health", {}).content[0].text)
    assert report["query_api"]["state"] in {"loading", "ready", "failed"}
    assert report["console"]["reachable"] is True and report["console"]["version"]
    assert "noetrix_search" in report["tools"]["loaded"]
    assert set(report["tools"]["not_loaded"]) <= {"noetrix_engines", "noetrix_set_engine"}


def test_the_console_maps_param_spelling_is_refused_naming_the_real_one(live):
    """Review Focus 4. /api/schema writes GET /api/jobs/{id}, but the route's path
    parameter is jid. An agent that copies the map's spelling must get an error that
    names jid, before any HTTP call."""
    result = _call(live.server, "noetrix_read", {"op": "GET /api/jobs/{id}", "params": {"id": "abc"}})
    assert result.is_error and "jid" in result.content[0].text
