"""HTTP to the two services, and every answer mapped to an MCP result.

Blocking `requests` calls; the async handlers run them in a worker thread.
Every request carries the bearer token when RAG_API_TOKEN is set.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from urllib.parse import quote

import requests

from src.mcp_bridge.catalog import Route, build_routes, console_tiers, query_tiers
from src.utils.http_guard import auth_headers

log = logging.getLogger(__name__)

LABEL = {"query": "the query API", "console": "the console"}
RETRY_INTERVAL_S = 1.0
JOB_LOG = "GET /api/jobs/{jid}/log"
# The routes that may call the generation model (HyDE and decomposition on
# /search, generation on /query, /answer and query-mode /compare).
_TIMEOUT_KEY = {"POST /search": "query", "POST /query": "query", "POST /answer": "query",
                "POST /compare": "compare"}


@dataclass
class Services:
    query_url: str        # cfg webui.rag_api
    console_url: str      # http://127.0.0.1:{webui.port}
    timeouts: dict        # cfg mcp.timeouts_s

    def url(self, service: str) -> str:
        return (self.query_url if service == "query" else self.console_url).rstrip("/")


@dataclass
class ToolOutcome:
    is_error: bool
    text: str             # JSON text of the result, or the error message


@dataclass
class Catalog:
    routes: dict[str, Route] = field(default_factory=dict)
    job_kind_tiers: dict[str, str] = field(default_factory=dict)
    missing: dict[str, str] = field(default_factory=dict)   # service -> why it is missing


def _text(body) -> str:
    return body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)


def _detail(body) -> str:
    """The readable part of an error body: the guard's and the console's
    `error`, FastAPI's `detail`, or the raw text."""
    if isinstance(body, dict):
        for k in ("error", "detail"):
            if body.get(k):
                return _text(body[k])
    return _text(body)


def map_response(route: Route, status: int, body) -> ToolOutcome:
    """Spec §5: no failure may read as "no results"."""
    where = f"{route.key} on {LABEL[route.service]}"
    if status == 422 and isinstance(body, dict) and isinstance(body.get("detail"), list):
        fields = "; ".join(f"{'.'.join(str(p) for p in e.get('loc', []))}: {e.get('msg')}"
                           for e in body["detail"])
        return ToolOutcome(True, f"{where} rejected the arguments (HTTP 422): {fields}")
    if status == 503:
        return ToolOutcome(True, f"{where} is not ready (HTTP 503): {_detail(body)}")
    if status >= 400:
        return ToolOutcome(True, f"{where} refused (HTTP {status}): {_detail(body)}")
    if isinstance(body, dict):
        if body.get("confidence") == "ERROR":       # retrieval worked, generation failed
            return ToolOutcome(True, _text(body))
        if isinstance(body.get("error"), str) and body["error"]:
            return ToolOutcome(True, f"{where} answered with an error: {body['error']}")
    return ToolOutcome(False, _text(body))


def timeout_for(route: Route, params: dict, timeouts: dict) -> float:
    if route.key == JOB_LOG:                         # a long poll: its wait plus a margin
        return float(params.get("wait", 0)) + float(timeouts["job_log_margin"])
    return float(timeouts[_TIMEOUT_KEY.get(route.key, "default")])


def call_route(route: Route, params: dict, services: Services, *, timeout: float) -> ToolOutcome:
    base = services.url(route.service)
    path = route.path
    for name in route.path_params:
        path = path.replace("{" + name + "}", quote(str(params[name]), safe=""))
    query = {k: params[k] for k in route.query_params if k in params}
    body = ({k: v for k, v in params.items() if k not in route.path_params and k not in route.query_params}
            if route.body else None)
    try:
        resp = requests.request(route.method, base + path, params=query or None, json=body,
                                headers=auth_headers(), timeout=timeout)
    except requests.Timeout:
        return ToolOutcome(True, f"{LABEL[route.service]} at {base} did not answer {route.key} "
                                 f"within {timeout:g} s; nothing was retried")
    except requests.ConnectionError as e:
        return ToolOutcome(True, f"{LABEL[route.service]} at {base} is not reachable "
                                 f"({type(e).__name__}): is it running? noetrix_health shows both services")
    try:
        payload = resp.json()
    except ValueError:
        payload = resp.text
    return map_response(route, resp.status_code, payload)


def probe(services: Services, service: str, path: str) -> dict:
    """One GET for noetrix_health: the JSON body, or why there is none."""
    url = services.url(service) + path
    try:
        resp = requests.request("GET", url, headers=auth_headers(), timeout=float(services.timeouts["default"]))
        body = resp.json()
    except (requests.RequestException, ValueError) as e:
        return {"reachable": False, "url": url, "error": f"{type(e).__name__}: {e}"}
    if resp.status_code != 200:
        return {"reachable": True, "url": url, "status": resp.status_code, "error": _detail(body)}
    return body


class _ServiceDown(Exception):
    def __init__(self, reason: str, retry: bool):
        super().__init__(reason)
        self.retry = retry


def _get_json(services: Services, service: str, path: str, deadline: float | None):
    timeout = float(services.timeouts["default"])
    if deadline is not None:                     # never wait past the catalog's deadline
        timeout = min(timeout, max(0.1, deadline - time.monotonic()))
    try:
        resp = requests.request("GET", services.url(service) + path, headers=auth_headers(),
                                timeout=timeout)
    except requests.RequestException as e:
        raise _ServiceDown(f"GET {path}: {type(e).__name__}: {e}", retry=True) from e
    if resp.status_code != 200:
        # 503 = the pipeline is not built (yet); anything else will not fix itself.
        raise _ServiceDown(f"GET {path} answered HTTP {resp.status_code}: {resp.text[:300]}",
                           retry=resp.status_code == 503)
    try:
        return resp.json()
    except ValueError as e:
        raise _ServiceDown(f"GET {path} did not answer JSON", retry=False) from e


def _fetch_service(services: Services, service: str, deadline: float | None):
    openapi = _get_json(services, service, "/openapi.json", deadline)
    try:
        if service == "query":
            tiers, purposes = query_tiers(_get_json(services, service, "/schema", deadline))
            kinds = {}
        else:
            tiers, purposes, kinds = console_tiers(_get_json(services, service, "/api/schema", deadline))
        return build_routes(openapi, tiers, purposes, service), kinds
    except _ServiceDown:
        raise
    except Exception as e:                       # a malformed document drops this service only
        raise _ServiceDown(f"its documents could not be read: {type(e).__name__}: {e}",
                           retry=False) from e


def fetch_catalog(services: Services, *, retry_s: float) -> Catalog:
    """Both services' routes. A service that is down or answers 503 is retried
    until `retry_s` has passed, then left out with the reason in `missing`.
    Each request is capped by what is left of `retry_s`, and no round starts
    that cannot finish before it. retry_s 0 (the refresh a tool call makes) is
    one round at the default timeout: a tight cap there could drop a live service."""
    deadline = time.monotonic() + retry_s
    cap = deadline if retry_s > 0 else None
    catalog = Catalog()
    pending = ["query", "console"]
    while True:
        retry = []
        for service in pending:
            try:
                routes, kinds = _fetch_service(services, service, cap)
            except _ServiceDown as e:
                catalog.missing[service] = f"{LABEL[service]} at {services.url(service)}: {e}"
                if e.retry:
                    retry.append(service)
                continue
            catalog.missing.pop(service, None)
            catalog.routes.update(routes)
            catalog.job_kind_tiers.update(kinds)
        pending = retry
        if not pending or time.monotonic() + RETRY_INTERVAL_S >= deadline:
            return catalog
        time.sleep(RETRY_INTERVAL_S)
        if time.monotonic() >= deadline:
            return catalog
