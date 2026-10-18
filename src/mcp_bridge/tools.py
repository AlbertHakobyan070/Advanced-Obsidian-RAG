"""The tool surface: thirteen named tools and three tier dispatchers.

A tool's tier decides its MCP annotations, and every mutating or destructive
tool also carries PROMPT_META, because Claude Code ignores the annotations
when deciding whether to ask the user.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import mcp_types as types
from jsonschema import Draft202012Validator

from src.mcp_bridge.catalog import NOT_EXPOSED, TIERS, Route, norm_key
from src.mcp_bridge.client import JOB_LOG, Catalog, Services, ToolOutcome, probe


@dataclass(frozen=True)
class NamedTool:
    name: str
    route_key: str        # "" for noetrix_health, which has no route of its own
    open_world: bool      # the call may reach the generation model
    guidance: str


HEALTH = "noetrix_health"
SEARCH = "noetrix_search"
JOB = "noetrix_job"
JOB_GET = "GET /api/jobs/{jid}"
JOBS_POST = "POST /api/jobs"
_LOG_PARAMS = {"offset", "tail", "wait"}

NAMED: tuple[NamedTool, ...] = (
    NamedTool(HEALTH, "", False,
              "Call it first when a tool is missing or a call cannot connect."),
    NamedTool(SEARCH, "POST /search", True,
              "Default to this one: reranked chunks with labels, text and stable ids, no generated "
              "answer. Escalate to noetrix_query only for a cited write-up."),
    NamedTool("noetrix_query", "POST /query", True,
              "A generated, cited answer: it spends a generation call, so use it for a deliverable, "
              "not to explore. confidence ERROR comes back as an error that keeps the sources."),
    NamedTool("noetrix_compare", "POST /compare", True,
              "Take ready-made branches from noetrix_compare_options."),
    NamedTool("noetrix_compare_options", "GET /compare/options", False,
              "Pick one dimension and post 2+ of its branches to noetrix_compare."),
    NamedTool("noetrix_get_chunk", "GET /chunks/{chunk_id}", False,
              "Pull the full text only for the one or two results you build on."),
    NamedTool("noetrix_stats", "GET /stats", False,
              "Read `latency` to price a call before making it."),
    NamedTool("noetrix_documents", "GET /api/documents", False,
              "Filter by filename substring (q), domain, course or tag."),
    NamedTool("noetrix_vault_search", "GET /api/vault/search", False,
              "Filename search over the whole vault, indexed or not."),
    NamedTool("noetrix_ingest", "POST /api/ingest_custom", False,
              "Returns job ids: follow them with noetrix_job."),
    NamedTool(JOB, JOB_LOG, False,
              "With jid alone: the job's status, progress and ETA. Add offset, tail or wait to read "
              "its log as well; wait (up to 60 s) returns as soon as the log grows past offset or "
              "the status changes. Follow a job call by call."),
    NamedTool("noetrix_engines", "GET /api/engines", False,
              "Check here before noetrix_set_engine."),
    NamedTool("noetrix_set_engine", "POST /api/engines/set", False,
              "It never restarts anything: make the call its next_op names, separately."),
)
_NAMED_BY_NAME = {t.name: t for t in NAMED}

DISPATCHERS = {"noetrix_read": "read", "noetrix_change": "mutating", "noetrix_remove": "destructive"}
PROMPT_META = {"anthropic/requiresUserInteraction": True}

_EMPTY = {"type": "object", "properties": {}, "additionalProperties": False}
_DISPATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "op": {"type": "string", "description": 'a route key as the capability maps write it, '
                                                'e.g. "GET /api/jobs"'},
        "params": {"type": "object", "description": "the route's path, query and body fields, "
                                                    "in one object"},
    },
    "required": ["op"],
    "additionalProperties": False,
}
_NOT_EXPOSED_NORM = {norm_key(k): why for k, why in NOT_EXPOSED.items()}


def _tool(name: str, description: str, schema: dict, tier: str, open_world: bool) -> types.Tool:
    if tier == "read":
        annotations = types.ToolAnnotations(read_only_hint=True, open_world_hint=open_world)
    else:
        annotations = types.ToolAnnotations(read_only_hint=False, destructive_hint=tier == "destructive",
                                            open_world_hint=open_world)
    return types.Tool(name=name, description=description, input_schema=schema, annotations=annotations,
                      meta=None if tier == "read" else dict(PROMPT_META))


def dispatch_ops(catalog: Catalog, tier: str) -> list[str]:
    ops = [k for k, r in catalog.routes.items() if r.tier == tier and k != JOBS_POST]
    kinds = sorted(k for k, t in catalog.job_kind_tiers.items() if t == tier)
    if JOBS_POST in catalog.routes and kinds:
        ops.append(f"{JOBS_POST} (kind: {' | '.join(kinds)})")
    return ops


def _dispatcher_description(tier: str, ops: list[str]) -> str:
    return (f"Call any {tier} route of the Noetrix services. op is a route key exactly as the "
            f"capability maps write it; params holds its path, query and body fields in one object, "
            f"checked against the route's schema before anything is sent. A route of another tier is "
            f"refused with the dispatcher to use instead. The full map with every purpose: noetrix_read "
            f'with op "GET /api/schema" (console) or "GET /schema" (query API).\n\nOps:\n'
            + "\n".join(ops))


def build_tool_list(catalog: Catalog) -> list[types.Tool]:
    """Every tool the catalog can back. A named tool whose route is absent is
    omitted, and so is a dispatcher with no ops; noetrix_health says why."""
    tools = [_tool(HEALTH, "Both services' readiness, which tools loaded, and why any did not. "
                   + NAMED[0].guidance, _EMPTY, "read", False)]
    for t in NAMED[1:]:
        route = catalog.routes.get(t.route_key)
        if route is not None:
            tools.append(_tool(t.name, f"{route.purpose}\n\n{t.guidance}", route.input_schema,
                               route.tier, t.open_world))
    for name, tier in DISPATCHERS.items():
        ops = dispatch_ops(catalog, tier)
        if ops:
            # read reaches /query and /compare, mutating reaches suggest_label
            # and the OCR warm-up: both can call a model. Removal never does.
            tools.append(_tool(name, _dispatcher_description(tier, ops), _DISPATCH_SCHEMA, tier,
                               open_world=tier != "destructive"))
    return tools


def validate(schema: dict, value) -> str | None:
    """Every way `value` fails `schema`, or None. The SDK does not validate."""
    errors = sorted(Draft202012Validator(schema).iter_errors(value), key=lambda e: list(e.absolute_path))
    return "; ".join(f"{'.'.join(str(p) for p in e.absolute_path) or '(arguments)'}: {e.message}"
                     for e in errors) or None


def _find(catalog: Catalog, op: str) -> Route | None:
    n = norm_key(op)
    return next((r for k, r in catalog.routes.items() if norm_key(k) == n), None)


def _dispatch(catalog: Catalog, tier: str, op: str, params: dict) -> tuple[Route, dict] | ToolOutcome:
    route = _find(catalog, op)
    if route is None:
        why = _NOT_EXPOSED_NORM.get(norm_key(op))
        if why:
            return ToolOutcome(True, f"{op!r} is not exposed over MCP: {why}")
        return ToolOutcome(True, f"{op!r} is not an exposed route: it does not exist, or it has no tier "
                                 'in its service\'s capability map. noetrix_read with op "GET /api/schema" '
                                 'or "GET /schema" lists the routes')
    if route.key != JOBS_POST and route.tier != tier:
        return _wrong_tier(route.key, route.tier, tier)
    err = validate(route.input_schema, params)
    if err:
        return ToolOutcome(True, f"{route.key}: invalid params: {err}")
    if route.key == JOBS_POST:                       # its tier is the job kind's
        kind = params["kind"]
        effective = catalog.job_kind_tiers.get(kind)
        if effective not in TIERS:
            return ToolOutcome(True, f"job kind {kind!r} has no tier in GET /api/schema's job_kinds, so "
                                     f"it cannot be queued over MCP. Kinds: {', '.join(sorted(catalog.job_kind_tiers))}")
        if effective != tier:
            return _wrong_tier(f"{JOBS_POST} with kind {kind!r}", effective, tier)
    return route, params


def _wrong_tier(what: str, actual: str, tier: str) -> ToolOutcome:
    right = next(n for n, t in DISPATCHERS.items() if t == actual)
    return ToolOutcome(True, f"{what} is {actual}, not {tier}: call {right} instead")


def resolve(catalog: Catalog, name: str, args: dict, search_defaults: dict) -> tuple[Route, dict] | ToolOutcome:
    """The route and params a call goes to, or the refusal. `args` has already
    passed the tool's own schema."""
    if name in DISPATCHERS:
        return _dispatch(catalog, DISPATCHERS[name], args["op"], args.get("params") or {})
    key = _NAMED_BY_NAME[name].route_key
    params = dict(args)
    if name == SEARCH:
        params = {**search_defaults, **params}       # fill only what the agent left unset
    if name == JOB and not params.keys() & _LOG_PARAMS:
        key = JOB_GET
    route = catalog.routes.get(key)
    if route is None:
        return ToolOutcome(True, f"{key} is not loaded; noetrix_health says why")
    return route, params


def health(catalog: Catalog, services: Services, loaded: list[str]) -> ToolOutcome:
    not_loaded = {}
    for t in NAMED[1:]:
        if t.name not in loaded:
            service = "console" if t.route_key.split(" ", 1)[1].startswith("/api/") else "query"
            not_loaded[t.name] = catalog.missing.get(service) or f"{t.route_key} is not a route of {services.url(service)}"
    for name, tier in DISPATCHERS.items():
        if name not in loaded:
            not_loaded[name] = f"no {tier} routes are loaded"
    report = {
        "query_api": probe(services, "query", "/health"),
        "console": probe(services, "console", "/api/schema"),
        "tools": {"loaded": loaded, "not_loaded": not_loaded},
        "services_missing": catalog.missing,
    }
    console = report["console"]
    if "endpoints" in console:              # the whole map is too much for a health line
        report["console"] = {"reachable": True, "service": console.get("service"),
                             "version": console.get("version")}
    return ToolOutcome(False, json.dumps(report, ensure_ascii=False))
