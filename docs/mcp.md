# MCP server

`mcp_server.py` exposes both services to agents as
[Model Context Protocol](https://modelcontextprotocol.io) tools, so Claude Code,
Claude Desktop, Codex and other MCP clients can drive Noetrix without a skill
that teaches raw HTTP.

It is a process of its own and a plain HTTP client of the query API (`:8051`)
and the console API (`:8052`), exactly as the skills are. It never loads the
pipeline: a crash cannot take either service down, and while `:8051` reloads it
can still tell an agent that the query API is restarting.

## Start it

| Where | Command | Transport |
|---|---|---|
| Windows checkout | `rag mcp` | stdio |
| Any checkout | `python mcp_server.py` from the repo root, project environment active | stdio |
| Any checkout, over HTTP | `python mcp_server.py --http` | streamable HTTP at `http://127.0.0.1:<mcp.port>/mcp` |
| Docker | started by the `rag` container; host port `NOETRIX_MCP_PORT` (default `8053`) | streamable HTTP at `/mcp` |
| Docker, no port | `docker exec -i <rag container> python mcp_server.py` | stdio |

Start `:8051` and `:8052` first. A service that does not answer is left out,
along with its tools, and `noetrix_health` says why. While one is missing,
every `tools/list` and every tool call tries it again, so a call to its routes
works as soon as it is up, without restarting the MCP server. A client that
lists the tools only once per connection, as Claude Code and Codex do, sees the
new tools after it lists again or reconnects.

Over stdio, stdout *is* the protocol channel. The server logs to
`logs/mcp_server.log` and to stderr, never to stdout, and `rag mcp` prints
nothing of its own.

## Tools

Named tools cover the frequent work. Their input schemas come from each route's
own request model, so arguments are validated before any HTTP call.

| Tool | Calls | Tier |
|---|---|---|
| `noetrix_health` | `/health` on `:8051` and `/api/schema` on the console (it has no `/health`), plus which tools loaded and why any did not | read |
| `noetrix_search` | `POST /search`, the primary agent call, with every per-call engine knob | read, open-world |
| `noetrix_query` | `POST /query`: a cited answer; per-call provider and model | read, open-world |
| `noetrix_compare` | `POST /compare`: one question down a tree of engine branches | read, open-world |
| `noetrix_compare_options` | `GET /compare/options`: ready-to-post branches, the real caps, unavailable backends | read |
| `noetrix_get_chunk` | `GET /chunks/{chunk_id}` | read |
| `noetrix_stats` | `GET /stats` | read |
| `noetrix_documents` | `GET /api/documents`: is a file indexed, and with what metadata | read |
| `noetrix_vault_search` | `GET /api/vault/search` | read |
| `noetrix_ingest` | `POST /api/ingest_custom`: ingest named inbox files | mutating |
| `noetrix_job` | `GET /api/jobs/{jid}` and its log, with `offset`, `tail` and `wait` | read |
| `noetrix_engines` | `GET /api/engines`: every engine per component, and what a switch costs | read |
| `noetrix_set_engine` | `POST /api/engines/set`: a checked, persistent engine switch | mutating |

"Open-world" means the call may reach the generation provider: HyDE and
decomposition on `/search`, the answer on `/query` and on query-mode `/compare`.

Everything else goes through three **dispatchers**, one per tier:
`noetrix_read`, `noetrix_change` and `noetrix_remove`. Each takes `op`, a route
key written exactly as the capability maps write it (`"POST /api/settings"`),
and `params`. A dispatcher refuses an `op` of another tier and names the right
one. `POST /api/jobs` takes the tier of its job kind, so an `index_rebuild`
goes only through `noetrix_remove`. The full list of ops is one call away:
`noetrix_read` with `op: "GET /api/schema"`, and `GET /schema` for `:8051`.

Not exposed: the console page `GET /`, the binary routes (`/api/ocrstudio/page.png`,
`/api/ocrstudio/mask.png`, `/api/import/file`, `POST /api/ocrstudio/export`) and the
multipart uploads `POST /api/upload` and `POST /api/ocrstudio/upload`. Use the
console or plain HTTP for those.

### Tiers, annotations and prompts

Each tool carries MCP annotations from its tier: `read` sets `readOnlyHint`,
`destructive` sets `destructiveHint`, `mutating` sets both hints to false. Clients such as
the local agent approve by annotation. Claude Code does not, so every mutating and
destructive tool (`noetrix_ingest`, `noetrix_set_engine`, `noetrix_change`,
`noetrix_remove`) also sets `_meta["anthropic/requiresUserInteraction"]`, which
makes Claude Code ask before each call.

### Experimenting with engines

From cheapest to most persistent:

1. **Per call.** The engine knobs on `noetrix_search` and `noetrix_query`
   (reranker, provider and model, preset, lanes and weights, HyDE,
   decomposition, the gate). Nothing shared changes.
2. **A tree.** `noetrix_compare` after `noetrix_compare_options`. Different
   questions are just parallel `noetrix_search` calls.
3. **Warm defaults for every client.** `noetrix_change` with `POST /config`.
4. **A persistent switch.** `noetrix_engines` lists each component's options:
   `available`, `reason`, `swappable` and `switch`, the cheapest way to use it
   (`per_call`, `per_job`, `restart`, `build` or `reembed`).
   `noetrix_set_engine` checks the switch first and writes it only if the check
   passes; it never restarts anything. Its `next_op` is the restart to run,
   `POST /api/service/restart`, when `:8051` must restart, and `null` when only
   the console (`:8052`) must, which no tool can restart: read `takes_effect`,
   and `warnings` says when to restart the console by hand.

### Long jobs

`noetrix_job` returns a job's status, progress and ETA. With `wait` (up to a
minute) it long-polls the log and returns as soon as the log grows past
`offset` or the status changes. An agent follows an ingest call by call; the
server never loops on its own.

### Errors

A failure is always a tool result with `isError: true` and a message, never an
empty result: a rejected knob, a `confidence: "ERROR"` answer (its sources are
kept), a `422` with FastAPI's field list, a service that is loading or failed,
a guard refusal naming its setting, a timeout naming the service and port. A
LOW-confidence answer from the relevance gate is not an error: it is the answer.
Nothing is retried on another engine or route.

## The token

Both services, and the HTTP transport, check `Host` and `Origin` on every
request. A bearer token is optional: set `RAG_API_TOKEN` in `.env` (blank means
unset) and restart the services. The MCP server reads it from the environment
and sends it on every call; an HTTP client must send it to the MCP server too.
See [Security](api.md#security).

A known limit: the console keeps the token in the browser's local storage, and
the note content it renders is not sanitised, so a crafted note could read it.
Use a token with a vault you trust. The MCP server and other non-browser
callers keep it in their own environment and are unaffected.

## Connect a client

The paths below are placeholders: use your checkout's real location.

### Claude Code

stdio, Windows. A `.bat` needs `cmd /c`, and from PowerShell call `claude.cmd`
rather than `claude`, because the PowerShell shim drops the `--`:

```powershell
claude.cmd mcp add noetrix --scope user -- cmd /c "C:\path\to\rag_project\rag.bat" mcp
claude.cmd mcp list
```

Write `rag.bat` with its path, absolute as here or `.\rag.bat` from the
checkout: when Windows has `NoDefaultCurrentDirectoryInExePath` set, `cmd /c`
does not find a bare `rag.bat` in the current directory.

stdio, macOS or Linux:

```bash
claude mcp add noetrix --scope user -- /path/to/venv/bin/python /path/to/rag_project/mcp_server.py
```

HTTP (the Docker container, or `--http`):

```bash
claude mcp add --transport http noetrix http://127.0.0.1:8053/mcp \
  --header "Authorization: Bearer $RAG_API_TOKEN"
```

`--header` stores the expanded token in Claude Code's own config file. Omit it
when no token is set.

Claude Code cuts a tool call off after a default of about a minute, and
`/query` or `/compare` can take longer on a CPU-only machine. Raise it with the
`MCP_TOOL_TIMEOUT` environment variable (milliseconds), or with a larger
`timeout` (milliseconds) on the server's entry. `MCP_TIMEOUT` is the separate
startup limit.

### Claude Desktop

stdio only: Claude Desktop's own connectors run in Anthropic's cloud and cannot
reach `127.0.0.1`. In `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "noetrix": {
      "command": "cmd",
      "args": ["/c", "C:\\path\\to\\rag_project\\rag.bat", "mcp"]
    }
  }
}
```

Claude Desktop is reported to mishandle spaces in arguments. If the server does
not start and the path has a space in it, point the entry at a path without
one. For the HTTP endpoint, a stdio-to-HTTP bridge such as `mcp-remote` works;
follow its own documentation for passing the `Authorization` header.

### Codex

In `~/.codex/config.toml`, stdio:

```toml
[mcp_servers.noetrix]
command = "cmd"
args = ["/c", "C:\\path\\to\\rag_project\\rag.bat", "mcp"]
tool_timeout_sec = 300
```

or HTTP, with the token read from the environment variable named here:

```toml
[mcp_servers.noetrix]
url = "http://127.0.0.1:8053/mcp"
bearer_token_env_var = "RAG_API_TOKEN"
tool_timeout_sec = 300
```

Codex's own per-tool default is about a minute; `tool_timeout_sec` raises it.

### the local agent

the local agent reads `mcp_servers:` from the active profile's `config.yaml`. stdio:

```yaml
mcp_servers:
  noetrix:
    command: cmd
    args: ["/c", "C:\\path\\to\\rag_project\\rag.bat", "mcp"]
```

HTTP, with the header written into the YAML:

```yaml
mcp_servers:
  noetrix:
    url: http://127.0.0.1:8053/mcp
    headers:
      Authorization: "Bearer <your RAG_API_TOKEN>"
```

the local agent approves by annotation: on an untrusted server, every tool without
`readOnlyHint` needs approval.

## Configuration

The `mcp:` block of `config.yaml`:

| Key | Meaning |
|---|---|
| `port` | The HTTP port (`--http` only). |
| `startup_retry_s` | How long the first `tools/list` waits for both services: one that refuses connections, or `:8051`'s `/schema` answering `503` while the pipeline loads. Each request is capped by what is left of the wait. |
| `timeouts_s` | The server's own HTTP timeout per call to the services: `default`, `query`, `compare`, and `job_log_margin`, added to a job-log call's `wait`. |
| `search_defaults` | `include_text` and `max_sources` for `noetrix_search`, filled in only when the agent left them unset. |

The service URLs come from the existing keys (`webui.rag_api`, `webui.port`);
nothing is duplicated. Client-side timeouts are the client's own: set them as
shown above.
