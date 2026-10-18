#!/bin/sh
# One probe for compose, the image HEALTHCHECK and Kubernetes (exec probes).
#
#   healthcheck.sh          READY: the query API reports "ready":true.
#   healthcheck.sh live     ALIVE: both servers answer HTTP. Says nothing about
#                           the index; a pipeline that failed to build, or is
#                           still loading, is alive.
#
# Why not just `curl -f /health`: that endpoint returns 200 while the index is
# loading and 200 again when the pipeline failed to build (the body then says
# "state":"failed"). A plain status check would call both healthy.
#
# The console probe is /docs: FastAPI's built-in page, a ~1 KB static string, no
# disk, no index. (/ serves the 360 KB UI and /api/overview walks the corpus.)
# /health needs no token, but /docs does when RAG_API_TOKEN is set, so the console
# probe sends it then.
set -eu

API="http://127.0.0.1:${RAG_PORT:-8051}"
CONSOLE="http://127.0.0.1:${RAG_CONSOLE_PORT:-8052}"

case "${1:-ready}" in
  ready)
    curl -fsS --max-time 4 "$API/health" | grep -q '"ready":true'
    ;;
  live)
    curl -fsS --max-time 4 -o /dev/null "$API/health"
    if [ -n "${RAG_API_TOKEN:-}" ]; then
      curl -fsS --max-time 4 -o /dev/null -H "Authorization: Bearer ${RAG_API_TOKEN}" "$CONSOLE/docs"
    else
      curl -fsS --max-time 4 -o /dev/null "$CONSOLE/docs"
    fi
    ;;
  *)
    echo "usage: healthcheck.sh [ready|live]" >&2
    exit 2
    ;;
esac
