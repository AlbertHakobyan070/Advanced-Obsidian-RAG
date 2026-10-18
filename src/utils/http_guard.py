"""
http_guard.py — the request guard in front of Noetrix's HTTP services.

Installed on serve_api (:8051), manage_api (:8052) and the MCP server's HTTP mode.
CORS stops a foreign web page READING a response; it does not stop the page SENDING
one, and DNS rebinding gets past it entirely. One pure-ASGI middleware runs three
checks, in this order:

  1. Host, every request: a loopback name or one listed in security.allowed_hosts,
     else 400. A rebound request is given away only by its Host header.
  2. Origin, on POST/PUT/PATCH/DELETE that carry one: the console's own origin, the
     service's own, or one listed in security.allowed_origins, else 403. No Origin
     (curl, agents, server-to-server) passes.
  3. Token, only when RAG_API_TOKEN is set: `Authorization: Bearer <token>`, else
     401. Exempt: GET /health (probes), any OPTIONS (a preflight cannot carry it),
     and whatever the service adds (the console adds its page shell, GET /).

Every refusal is JSON naming the setting to change.
"""
from __future__ import annotations

import hmac
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from urllib.parse import urlsplit

from starlette.responses import JSONResponse

LOOPBACK_HOSTS: frozenset[str] = frozenset({"localhost", "127.0.0.1", "::1"})
TOKEN_ENV = "RAG_API_TOKEN"
_STATE_CHANGING = frozenset({"POST", "PUT", "PATCH", "DELETE"})


@dataclass(frozen=True)
class GuardSettings:
    allowed_hosts: frozenset[str]    # LOOPBACK_HOSTS | lower-cased security.allowed_hosts
    allowed_origins: frozenset[str]  # pinned local origins | security.allowed_origins
    token: str | None                # None when RAG_API_TOKEN is unset or blank


def read_token() -> str | None:
    """RAG_API_TOKEN from the environment, stripped. Unset or blank means no token."""
    return os.environ.get(TOKEN_ENV, "").strip() or None


def auth_headers() -> dict[str, str]:
    """The header a client of either service sends: empty when no token is set."""
    token = read_token()
    return {"Authorization": f"Bearer {token}"} if token else {}


def _names(cfg, key: str) -> list[str]:
    value = cfg.get(key) or []           # a config without the security block: no extras
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError(f"{key} in config.yaml must be a list of strings, got {value!r}")
    return value


def guard_settings(cfg, *, own_port: int) -> GuardSettings:
    """The guard's settings from a loaded config. The pinned origins are the console's
    (webui.port) and this service's own (own_port), each as localhost and 127.0.0.1."""
    ports = {int(own_port), int(cfg.get("webui.port", 8052))}
    pinned = {f"http://{host}:{port}" for host in ("localhost", "127.0.0.1") for port in ports}
    extra_origins = {o.rstrip("/").lower() for o in _names(cfg, "security.allowed_origins")}
    extra_hosts = {h.lower() for h in _names(cfg, "security.allowed_hosts")}
    return GuardSettings(allowed_hosts=LOOPBACK_HOSTS | extra_hosts,
                         allowed_origins=frozenset(pinned | extra_origins),
                         token=read_token())


def _hostname(raw: str) -> str:
    """'127.0.0.1:8051' -> '127.0.0.1', '[::1]:8051' -> '::1', lower-cased; '' if unparsable."""
    try:
        return urlsplit("//" + raw).hostname or ""
    except ValueError:                   # e.g. an unclosed '[::1'
        return ""


def _refusal(scope, s: GuardSettings, open_routes: frozenset) -> tuple[int, str, str] | None:
    headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
    raw_host = headers.get("host", "")
    host = _hostname(raw_host)
    if host not in s.allowed_hosts:
        return (400, "host", f"Host {host or raw_host!r} is not allowed; add it to "
                             "security.allowed_hosts in config.yaml")

    method = scope["method"]
    origin = headers.get("origin")
    if method in _STATE_CHANGING and origin is not None:
        own = f"{scope['scheme']}://{raw_host}".lower()   # a page this server served itself
        if origin.rstrip("/").lower() not in s.allowed_origins and origin.lower() != own:
            return (403, "origin", f"Origin {origin!r} may not send {method} requests here; "
                                   "add it to security.allowed_origins in config.yaml")

    if s.token is None or method == "OPTIONS" or (method, scope["path"]) in open_routes:
        return None
    scheme, _, given = headers.get("authorization", "").partition(" ")
    # latin-1 restores the header's raw bytes, so a non-ASCII value compares instead of raising.
    if scheme.lower() != "bearer" or not hmac.compare_digest(
            given.strip().encode("latin-1"), s.token.encode("utf-8")):
        return (401, "token", f"missing or wrong bearer token: send 'Authorization: Bearer "
                              f"<token>' with the value of {TOKEN_ENV} from the server's .env")
    return None


class _Guard:
    def __init__(self, app, *, owner, settings: Callable[[], GuardSettings], open_routes: frozenset):
        self.app, self.owner, self.settings, self.open_routes = app, owner, settings, open_routes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":      # lifespan; neither service has websocket routes
            await self.app(scope, receive, send)
            return
        state = self.owner.state
        if state.guard_settings is None:
            try:
                state.guard_settings = self.settings()
            except ValueError as e:      # a malformed security.* value: name it, not a bare 500
                await JSONResponse({"error": f"{type(e).__name__}: {e}", "guard": "config"},
                                   500)(scope, receive, send)
                return
        refusal = _refusal(scope, state.guard_settings, self.open_routes)
        if refusal is None:
            await self.app(scope, receive, send)
            return
        status, guard, error = refusal
        headers = {"WWW-Authenticate": "Bearer"} if status == 401 else None
        await JSONResponse({"error": error, "guard": guard}, status, headers)(scope, receive, send)


def install_guard(app, settings: Callable[[], GuardSettings], *,
                  exempt: Iterable[tuple[str, str]] = ()) -> None:
    """Put the guard in front of `app`'s routes.

    Call it BEFORE the app adds its CORSMiddleware. Starlette runs the middleware
    added last outermost, so CORS then wraps the guard and decorates its refusals:
    the console page reads :8051's 401 across ports, and without CORS headers the
    browser would hide it.

    `settings` is called on the first request, not here (serve_api loads its config
    only in its lifespan). The result stays in app.state.guard_settings; setting that
    back to None makes the next request read the settings again.
    `exempt` adds (METHOD, path) pairs that need no token, beside GET /health.
    """
    open_routes = frozenset({("GET", "/health"), *((m.upper(), p) for m, p in exempt)})
    app.state.guard_settings = None
    app.add_middleware(_Guard, owner=app, settings=settings, open_routes=open_routes)
