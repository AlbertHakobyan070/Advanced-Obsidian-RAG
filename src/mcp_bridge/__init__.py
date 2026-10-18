"""The MCP server's internals (mcp_server.py at the repo root is the entry point).

Named mcp_bridge, never `mcp`: a package called `mcp` under src/ would shadow
the official SDK for anything that puts src/ on the path.

catalog  builds tool routes from the two services' /openapi.json and their
         capability maps (tiers, purposes, job kinds)
client   calls those routes over HTTP and maps each answer to an MCP result
tools    the named tools, the three tier dispatchers, argument validation
server   wires all of it into the SDK's low-level Server
"""
