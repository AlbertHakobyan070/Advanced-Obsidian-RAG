"""Noetrix MCP server: the query API (:8051) and the console (:8052) as MCP tools.

    python mcp_server.py                  stdio, started by the agent (`rag mcp` on Windows)
    python mcp_server.py --http           streamable HTTP at /mcp on mcp.port
    python mcp_server.py --http --host 0.0.0.0   inside the container

A plain HTTP client of both services, exactly as the skills are: it never
imports the pipeline, so a crash here cannot take the APIs down. Tools are
built from the services' own /openapi.json and capability maps; docs/mcp.md
lists them and how to connect each client.

Over stdio, stdout IS the protocol channel. Logs go to logs/mcp_server.log and
stderr, never stdout, and nothing here may print().
"""
from __future__ import annotations

import argparse

import anyio

from src.mcp_bridge.client import Services
from src.mcp_bridge.server import build_server
from src.utils.config_loader import load_config
from src.utils.logger import configure_logging, get_logger

log = get_logger("mcp_server")


def services_from_config(cfg) -> Services:
    if cfg.get("mcp") is None:
        raise SystemExit("config.yaml has no `mcp:` block; copy it from config.example.yaml")
    return Services(query_url=cfg.get("webui.rag_api", "http://127.0.0.1:8051"),
                    console_url=f"http://127.0.0.1:{cfg.get('webui.port', 8052)}",
                    timeouts=dict(cfg.get("mcp.timeouts_s")))


def build_http_app(server, cfg, host: str):
    """The SDK's streamable-HTTP app behind the request guard. The SDK's own
    DNS-rebinding check is off on purpose: it switches itself on only for a
    loopback bind, so native and container runs would differ, and the guard
    covers the same ground everywhere."""
    from mcp.server.transport_security import TransportSecuritySettings
    from src.utils.http_guard import guard_settings, install_guard

    app = server.streamable_http_app(
        host=host, transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False))
    install_guard(app, lambda: guard_settings(cfg, own_port=int(cfg.get("mcp.port"))))
    return app


async def _serve_stdio(server) -> None:
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Noetrix MCP server (stdio unless --http)")
    parser.add_argument("--http", action="store_true", help="serve streamable HTTP at /mcp on mcp.port")
    parser.add_argument("--host", default="127.0.0.1", help="bind address for --http (0.0.0.0 in a container)")
    args = parser.parse_args(argv)

    cfg = load_config()
    configure_logging(cfg.get("logging.level", "INFO"), cfg.project_root / "logs" / "mcp_server.log",
                      console=True)                 # console = stderr
    server = build_server(services_from_config(cfg), cfg)
    if args.http:
        import uvicorn

        port = int(cfg.get("mcp.port"))
        log.info("MCP server: streamable HTTP on http://%s:%d/mcp", args.host, port)
        # log_config=None: uvicorn's own config sends access lines to stdout.
        uvicorn.run(build_http_app(server, cfg, args.host), host=args.host, port=port, log_config=None)
    else:
        log.info("MCP server: stdio")
        anyio.run(_serve_stdio, server)


if __name__ == "__main__":
    main()
