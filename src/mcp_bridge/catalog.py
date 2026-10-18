"""The routes the MCP server can call, built from the services' own documents.

Parameter schemas come from each app's /openapi.json (the real request models);
tiers and purposes come from the capability maps (:8051 GET /schema's
`permissions` and `endpoints`, :8052 GET /api/schema's `endpoints` and
`job_kinds`). Nothing here is hand-written per route except NOT_EXPOSED.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

log = logging.getLogger(__name__)

TIERS = ("read", "mutating", "destructive")

# Routes that are tiered but cannot work over MCP. Each needs its own design
# before it is exposed: passing file bytes over MCP is an open question.
NOT_EXPOSED: dict[str, str] = {
    "GET /": "the console's HTML page, not an API",
    "GET /api/ocrstudio/page.png": "returns a PNG image",
    "GET /api/ocrstudio/mask.png": "returns a PNG image",
    "GET /api/import/file": "returns the bytes of a staged file",
    "POST /api/ocrstudio/export": "returns a zip archive",
    "POST /api/upload": "takes a multipart file upload",
    "POST /api/ocrstudio/upload": "takes a multipart file upload",
}

_REF_PREFIX = "#/components/schemas/"
_PATH_PARAM = re.compile(r"\{[^}]*\}")


@dataclass(frozen=True)
class Route:
    service: str          # "query" | "console"
    method: str
    path: str             # "/api/jobs/{jid}"
    tier: str             # "read" | "mutating" | "destructive"
    purpose: str
    input_schema: dict    # self-contained JSON Schema of {path params + query params + body}
    path_params: tuple[str, ...]
    query_params: tuple[str, ...]
    body: bool

    @property
    def key(self) -> str:
        return f"{self.method} {self.path}"


def norm_key(key: str) -> str:
    """'GET /api/jobs/{id}/log?tail=' -> 'GET /api/jobs/{}/log'.

    The capability maps name path parameters differently from the routes
    ({id} vs {jid}) and some keys carry a query hint, and the dispatcher
    descriptions add ' (kind: ...)', so keys are compared in this form everywhere."""
    return _PATH_PARAM.sub("{}", key.split("?", 1)[0].split(" (", 1)[0].strip())


def inline_refs(node, components: dict):
    """A copy of `node` with every '#/components/schemas/X' reference replaced
    by X itself, so the schema stands alone as MCP requires. (No request model
    here is recursive; one that were would end in a RecursionError.)"""
    if isinstance(node, list):
        return [inline_refs(v, components) for v in node]
    if not isinstance(node, dict):
        return node
    ref = node.get("$ref")
    if ref is None:
        return {k: inline_refs(v, components) for k, v in node.items()}
    if not ref.startswith(_REF_PREFIX):
        raise ValueError(f"cannot inline $ref {ref!r}: only {_REF_PREFIX}* is supported")
    merged = {**components[ref[len(_REF_PREFIX):]], **{k: v for k, v in node.items() if k != "$ref"}}
    return inline_refs(merged, components)


def _purpose(entry) -> str:
    return entry if isinstance(entry, str) else str(entry.get("purpose", ""))


def query_tiers(schema_doc: dict) -> tuple[dict[str, str], dict[str, str]]:
    """(tiers, purposes) from :8051's GET /schema, keyed by norm_key."""
    perms = schema_doc.get("permissions")
    if not isinstance(perms, dict):
        raise ValueError("the query API's GET /schema has no `permissions` map: it predates "
                         "this version, restart :8051")
    tiers = {norm_key(k): v for k, v in perms.items()}
    purposes = {norm_key(k): _purpose(v) for k, v in schema_doc.get("endpoints", {}).items()}
    return tiers, purposes


def console_tiers(api_schema_doc: dict) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """(tiers, purposes, job_kind_tiers) from :8052's GET /api/schema."""
    tiers, purposes = {}, {}
    for key, entry in api_schema_doc.get("endpoints", {}).items():
        tiers[norm_key(key)] = entry.get("permission")
        purposes[norm_key(key)] = _purpose(entry)
    kinds = {kind: entry.get("permission") for kind, entry in api_schema_doc.get("job_kinds", {}).items()}
    return tiers, purposes, kinds


def _route(service: str, method: str, path: str, op: dict, tier: str, purpose: str,
           components: dict) -> Route:
    key = f"{method} {path}"
    props: dict = {}
    required: list[str] = []
    path_params: list[str] = []
    query_params: list[str] = []
    for p in op.get("parameters", []):
        if p["in"] not in ("path", "query"):
            raise ValueError(f"{key}: a {p['in']} parameter ({p['name']!r}) cannot be sent over MCP")
        (path_params if p["in"] == "path" else query_params).append(p["name"])
        schema = dict(p["schema"])
        if p.get("description"):
            schema["description"] = p["description"]
        props[p["name"]] = inline_refs(schema, components)
        if p.get("required"):
            required.append(p["name"])
    body = "requestBody" in op
    if body:
        content = op["requestBody"]["content"]
        if "application/json" not in content:
            raise ValueError(f"{key}: a {next(iter(content))} body cannot be sent over MCP; "
                             "list the route in NOT_EXPOSED")
        schema = inline_refs(content["application/json"]["schema"], components)
        if schema.get("type") != "object" or "properties" not in schema:
            raise ValueError(f"{key}: its body is not an object with properties")
        clash = set(schema["properties"]) & set(props)
        if clash:
            raise ValueError(f"{key}: body field(s) {sorted(clash)} collide with path/query parameters")
        props.update(schema["properties"])
        required += schema.get("required", [])
    input_schema: dict = {"type": "object", "properties": props, "additionalProperties": False}
    if required:
        input_schema["required"] = required
    return Route(service, method, path, tier, purpose, input_schema,
                 tuple(path_params), tuple(query_params), body)


def build_routes(openapi: dict, tiers: dict, purposes: dict, service: str) -> dict[str, Route]:
    """Every tiered, exposed route of one service, keyed by Route.key."""
    components = openapi.get("components", {}).get("schemas", {})
    routes: dict[str, Route] = {}
    for path, ops in openapi["paths"].items():
        for method, op in ops.items():
            key = f"{method.upper()} {path}"
            if key in NOT_EXPOSED:
                continue
            tier = tiers.get(norm_key(key))
            if tier not in TIERS:
                log.warning("%s %s has no tier in its capability map, so it is not exposed over MCP",
                            service, key)
                continue
            routes[key] = _route(service, method.upper(), path, op, tier,
                                 purposes.get(norm_key(key)) or op.get("summary", ""), components)
    return routes
