"""The SDK's low-level Server, wired to the catalog and the tools."""
from __future__ import annotations

import logging
import threading

import anyio
import mcp_types as types
from mcp.server.lowlevel import Server

from src.mcp_bridge.client import Catalog, Services, ToolOutcome, call_route, fetch_catalog, timeout_for
from src.mcp_bridge.tools import HEALTH, build_tool_list, health, resolve, validate
from src.utils.branding import SERVICE_SLUG
from src.utils.release import PRODUCT_VERSION

log = logging.getLogger(__name__)


class _State:
    """The catalog and the tools built from it, shared by every session."""

    def __init__(self, services: Services, cfg):
        self.services = services
        self.retry_s = float(cfg.get("mcp.startup_retry_s"))
        self.search_defaults = dict(cfg.get("mcp.search_defaults"))
        self.catalog: Catalog | None = None
        self.tools: dict[str, types.Tool] = {}
        self._lock = threading.Lock()

    def refresh(self) -> list[types.Tool]:
        """The first call waits up to mcp.startup_retry_s for the services;
        while one is still missing, every later call tries it once more."""
        with self._lock:
            if self.catalog is None or self.catalog.missing:
                self.catalog = fetch_catalog(self.services,
                                             retry_s=self.retry_s if self.catalog is None else 0)
                for service, why in self.catalog.missing.items():
                    log.warning("%s tools not loaded: %s", service, why)
                self.tools = {t.name: t for t in build_tool_list(self.catalog)}
            return list(self.tools.values())

    def schema_for(self, name: str):
        tool = self.tools.get(name)
        return tool.input_schema if tool else None

    def call(self, name: str, args: dict) -> ToolOutcome:
        # Clients list tools once per connection: a service missing then is retried here.
        if self.catalog is None or self.catalog.missing:
            self.refresh()
        with self._lock:                          # the pair refresh() assigned together
            catalog, tools = self.catalog, self.tools
        if name == HEALTH:
            return health(catalog, self.services, list(tools))
        tool = tools.get(name)
        if tool is None:
            return ToolOutcome(True, f"no tool named {name!r} is loaded; noetrix_health says which "
                                     "tools loaded and why any did not")
        err = validate(tool.input_schema, args)
        if err:
            return ToolOutcome(True, f"{name}: invalid arguments: {err}")
        target = resolve(catalog, name, args, self.search_defaults)
        if isinstance(target, ToolOutcome):
            return target
        route, params = target
        return call_route(route, params, self.services,
                          timeout=timeout_for(route, params, self.services.timeouts))


def build_server(services: Services, cfg) -> Server:
    state = _State(services, cfg)

    async def list_tools(ctx, params) -> types.ListToolsResult:
        return types.ListToolsResult(tools=await anyio.to_thread.run_sync(state.refresh))

    async def call_tool(ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
        # Everything is caught: a raised exception would reach the agent as a
        # protocol error instead of a result it can read.
        try:
            outcome = await anyio.to_thread.run_sync(state.call, params.name, params.arguments or {})
        except Exception as e:                        # noqa: BLE001 — reported as the result
            log.exception("tool %s failed", params.name)
            outcome = ToolOutcome(True, f"{params.name} failed inside the MCP server: {type(e).__name__}: {e}")
        return types.CallToolResult(content=[types.TextContent(type="text", text=outcome.text)],
                                    is_error=outcome.is_error)

    # get_tool_input_schema: over streamable HTTP the SDK would otherwise re-run
    # list_tools (and so re-fetch both services) on every call with arguments.
    return Server(SERVICE_SLUG, version=PRODUCT_VERSION, on_list_tools=list_tools,
                  on_call_tool=call_tool, get_tool_input_schema=state.schema_for)
