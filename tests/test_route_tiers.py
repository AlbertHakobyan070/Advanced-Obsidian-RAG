"""Every route of both services carries a permission tier, so a client that gates
calls by tier (the MCP server, the skills) never meets one without.

:8051 keeps its tiers in GET /schema's top-level `permissions` map, keyed exactly
as its routes are declared. :8052 keeps one in each GET /api/schema `endpoints`
entry, whose keys name path parameters their own way ({id} for {jid}) and may add
a query hint ("?lines="). A route in neither map must be in UNLISTED_CONSOLE with
the reason, or this fails; so does a tier naming a route that no longer exists.
"""
import re
from types import SimpleNamespace

from fastapi.routing import APIRoute

import serve_api

UNLISTED_CONSOLE = {"GET /": "the console page itself"}
TIERS = {"read", "mutating", "destructive"}


def _routes(app) -> set[str]:
    """'METHOD /path' for every route the app declares; FastAPI's own /docs and
    /openapi.json are not APIRoutes and stay out."""
    return {f"{method} {r.path}" for r in app.routes if isinstance(r, APIRoute) for method in r.methods}


def _shape(key: str) -> str:
    """'GET /api/jobs/{id}/log?tail=' and 'GET /api/jobs/{jid}/log' are one route."""
    return re.sub(r"\{[^}]*\}", "{}", key.split("?", 1)[0])


def test_every_console_route_is_tiered_or_listed_with_a_reason(management_module):
    routes = _routes(management_module.app)
    endpoints = management_module.api_schema()["endpoints"]
    tiered = {_shape(key): entry["permission"] for key, entry in endpoints.items()}
    assert set(tiered.values()) <= TIERS
    assert {r for r in routes if _shape(r) not in tiered} == set(UNLISTED_CONSOLE)
    assert {k for k in endpoints if _shape(k) not in {_shape(r) for r in routes}} == set()


def test_the_query_api_tiers_every_route_and_only_post_config_mutates(monkeypatch):
    monkeypatch.setitem(serve_api._STATE, "rag", SimpleNamespace(presets={}))
    permissions = serve_api.schema()["permissions"]
    assert set(permissions) == _routes(serve_api.app)
    assert set(permissions.values()) <= TIERS
    assert {key for key, tier in permissions.items() if tier != "read"} == {"POST /config"}
    assert permissions["POST /config"] == "mutating"
