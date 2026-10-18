#!/bin/sh
# Run BOTH APIs in one container so the console's server-side health probe and
# the browser both reach the query API at 127.0.0.1:8051.
#
# serve_api runs under a SUPERVISOR LOOP rather than as a plain background job.
# That is what makes the console's "restart :8051" button work here: the
# console SIGTERMs the pid in $RAG_SERVE_PID_FILE and this loop brings it back.
# Without it, nothing saved in Settings or the vault switcher could ever be
# applied without restarting the whole container — which is how the Mac build
# ended up feeling read-only.
#
# Two things the first version got wrong, both of which made the restart button
# look broken rather than slow:
#
#   1. uvicorn exits 0 on a graceful SIGTERM, not 143. The "was this an
#      intentional restart?" test therefore never matched, so every restart
#      took the 5s crash penalty and the console — which only waited 10s for a
#      new pid — could give up before the relaunch had happened.
#   2. serve_api's output went only to the container's stdout. When the
#      pipeline failed to build (a reranker that cannot download, a bad model
#      id) the operator had no way to SEE that from the console: it just spun.
#      The log now also lands in a file the console tails over HTTP.
#
# The two servers also stay in ONE container (and so one Kubernetes pod) because
# the restart handshake is a signal to a pid in a shared process namespace.
# Splitting them would need the container runtime's socket, a far bigger grant
# than the feature is worth.
set -e

PID_FILE="${RAG_SERVE_PID_FILE:-/tmp/serve_api.pid}"
GEN_FILE="${PID_FILE}.generation"
RAG_PORT="${RAG_PORT:-8051}"
CONSOLE_PORT="${RAG_CONSOLE_PORT:-8052}"
LOG_DIR="${RAG_LOG_DIR:-/app/logs}"
LOG_FILE="${LOG_DIR}/serve_api_${RAG_PORT}.out.log"

# The image runs as a non-root user. A named volume that an OLDER (root) image
# already populated keeps its root ownership, and the failure that follows is an
# obscure PermissionError deep inside a model download or a job. Say what is
# wrong and how to fix it now. Only a warning: a read-only data mount is a
# legitimate query-only setup.
#   (bind mounts take their permissions from the host; /app/data and /rag_data
#    on a Linux host need `chown -R 10001:10001` there. Docker Desktop on a Mac
#    or Windows maps ownership for you.)
for d in "$LOG_DIR" "${HF_HOME:-/hf_cache}" /app/data /rag_data; do
  if [ -d "$d" ] && [ ! -w "$d" ]; then
    echo "[rag] WARNING: $d is not writable by uid $(id -u). If it is a docker volume" >&2
    echo "[rag]          created by an older image, fix it once with:" >&2
    echo "[rag]   docker compose up -d volume-perms     (or: chown -R $(id -u):$(id -g) <path>)" >&2
  fi
done

mkdir -p "$LOG_DIR"
: > "$LOG_FILE"

echo "[rag] starting query API on :${RAG_PORT} (loads the index once; first query"
echo "[rag] also downloads the embed/rerank models into the hf-cache volume) ..."
echo "[rag] query-API log: ${LOG_FILE}"

# Mirror the query API's log into `docker logs` too. A pipe would make $! the
# pid of the pipe's LAST element, and the console has to signal uvicorn itself
# — so the launch below redirects to the file and this follower does the
# mirroring. -F survives the per-launch truncation.
tail -n +1 -F "$LOG_FILE" 2>/dev/null &

supervise_serve_api() {
  generation=0
  while true; do
    generation=$((generation + 1))
    # Truncate per launch: the console tails this to explain a failed start,
    # and a file holding every previous boot buries the one that matters.
    : > "$LOG_FILE"
    echo "[rag] --- serve_api launch #${generation} at $(date -u +%FT%TZ) ---" \
      >> "$LOG_FILE"
    python -m uvicorn serve_api:app --host 0.0.0.0 --port "${RAG_PORT}" \
      >> "$LOG_FILE" 2>&1 &
    child=$!
    echo "$child" > "$PID_FILE"
    # The generation counter is the console's proof that a relaunch actually
    # happened. Comparing pids cannot prove it (the kernel may hand back the
    # same number); a counter that only ever increases can.
    echo "$generation" > "$GEN_FILE"
    if wait "$child"; then
      status=0
    else
      status=$?
    fi
    # 143 = killed by SIGTERM; 0 = uvicorn caught SIGTERM and shut down
    # cleanly, which is the normal outcome of the console's restart. Both are
    # intentional, so relaunch immediately. Anything else is a real crash:
    # pause first, or a config error becomes a hot restart loop that pins a
    # CPU core and floods the log.
    if [ "$status" -eq 143 ] || [ "$status" -eq 0 ]; then
      echo "[rag] :${RAG_PORT} stopped (status ${status}) — relaunching ..."
    else
      echo "[rag] :${RAG_PORT} exited with status ${status} — relaunching in 5s."
      sleep 5
    fi
  done
}

supervise_serve_api &

# The MCP server (mcp_server.py --http, on mcp.port) for agents. Optional, so
# no supervisor loop: a crash shows up as "connection refused" on its port, with
# the reason in ${LOG_DIR}/mcp_server.log, and healthcheck.sh ignores it on
# purpose. Add a loop like serve_api's if it ever crashes in practice.
echo "[rag] starting MCP server (streamable HTTP at /mcp) ..."
python mcp_server.py --http --host 0.0.0.0 &

echo "[rag] starting management console on :${CONSOLE_PORT} ..."
# Foreground: its exit stops the container, same as before.
exec python -m uvicorn manage_api:app --host 0.0.0.0 --port "${CONSOLE_PORT}"
