#!/usr/bin/env python3
"""Derive the container's config.yaml from the project's config.example.yaml.

WHY THIS EXISTS. A container config kept by hand drifts: by 2026-09-27 the two
hand-kept copies were missing 26 keys the project had added since -- the entire
`graph:` and `canvas:` blocks and all eight `retrieval.lane_weights`. A container
shipped with one of them starts fine and then silently behaves like an older
build, which is worse than failing.

So no deployment carries a copy of the config. It is generated from
config.example.yaml -- the file the repo already keeps current -- and mounted:
compose bind-mounts it at /app/config.yaml, Kubernetes turns it into a ConfigMap.
Only what a container must change is rewritten:

  * paths that live on a volume rather than beside the code,
  * service URLs, because `localhost` inside the rag container is the rag
    container, not the host and not a sibling service,
  * the academic taxonomy defaults, which encode one particular vault.

Every rewrite is anchored on an exact string and asserts how many times it must
match, so a renamed key breaks the BUILD -- loudly, on the machine that has the
repo in front of it -- instead of shipping a config pointing nowhere. And after
rewriting, any loopback URL that is still there fails the run too: that is the
failure the anchors cannot see (a NEW local URL has no anchor), and it is how the
EmbeddingGemma sidecar URLs shipped unreachable.

Usage (from the repo root):
    python deploy/docker/make_docker_config.py config.example.yaml deploy/docker/config.docker.yaml

The author's own-machine variant lives in deploy/package/make_personal_config.py.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# (old, new, expected_hits). Ordered, applied as plain string replacements.
# ---------------------------------------------------------------------------
REWRITES: list[tuple[str, str, int]] = [
    # -- the vault ----------------------------------------------------------
    # Any string here may reference an env var (config_loader expands ${NAME}),
    # and docker-compose bind-mounts $RAG_HOST_HOME at this same absolute path
    # inside the container. One value in .env therefore points both sides at
    # the same real folder, so what the console browses is what Finder shows.
    # (Kubernetes sets RAG_HOST_HOME=/vault and mounts a volume there.)
    #
    # It starts at HOME, which is not a vault: pick the real one once in the
    # console (Settings -> Vaults) and it is remembered per vault.
    ('  vault_path: "path/to/your/documents"',
     '  vault_path: "${RAG_HOST_HOME}"',
     1),

    # -- index locations ----------------------------------------------------
    # chunks_file stays under data/: ./data is mounted at /app/data, so the
    # JSONLs (the source of truth) stay visible on the host. The derived
    # indexes go to the separate /rag_data mount.
    ('  chroma_dir: "data/chroma_db"',
     '  chroma_dir: "/rag_data/chroma_db"',
     1),
    ('  bm25_index: "data/bm25_index.pkl"',
     '  bm25_index: "/rag_data/bm25_index.pkl"',
     1),

    # -- taxonomy -----------------------------------------------------------
    # The built-in folder maps encode one academic vault's folder names and
    # course codes. Left on, they mislabel a vault that does not share that
    # convention -- and the mislabel is stored in every chunk's metadata. Off
    # means chunks start "unknown" and the console's retag tools assign real
    # labels. The comment block above this key explains turning it back on
    # with your own maps.
    ('  label: course                   # e.g. course | project | client | topic',
     '  label: topic                    # e.g. course | project | client | topic',
     1),
    ('  label_plural: courses',
     '  label_plural: topics',
     1),
    ('  detect_from_path: true',
     '  detect_from_path: false         # cold start: no folder convention assumed',
     1),

    # -- the console's view of the query API --------------------------------
    # Same container, so loopback stays right -- but the PORT is whatever compose
    # publishes (NOETRIX_API_PORT, passed in as RAG_PORT), because the console's
    # PAGE calls this URL from the browser. Hard-coded, a remapped host port left
    # the Query tab talking to whatever else owns :8051. Unset (Kubernetes), it is
    # 8051.
    ('  rag_api: "http://127.0.0.1:8051"           # the query endpoint the console talks to',
     '  rag_api: "http://127.0.0.1:${RAG_PORT:-8051}"   # the query endpoint the console talks to',
     1),

    # -- sibling containers -------------------------------------------------
    # localhost inside the rag container IS the rag container. The generation
    # proxy is a service name on the compose network (and a Service in k8s).
    ('    base_url: "http://localhost:3001/v1"',
     '    base_url: "http://freellmapi:3001/v1"',
     1),
    ('  base_url: "http://localhost:3001/v1"   # legacy providers only',
     '  base_url: "http://freellmapi:3001/v1"  # legacy providers only',
     1),
    # PaddleOCR sidecar (compose --profile ocr / ocr-gpu).
    ('    base_url: "http://127.0.0.1:8103"',
     '    base_url: "http://paddleocr:8103"',
     1),
    # The external reranker: the compose service publishes 8102, not 8101.
    ('    base_url: "http://127.0.0.1:8101/v1"',
     '    base_url: "http://rerank:8102/v1"',
     1),
    # EmbeddingGemma 2 sidecars (compose --profile embed): one service per
    # max_seq_length cap, because the client refuses a server whose cap differs
    # from the registry entry's.
    ('base_url: "http://127.0.0.1:8104/v1"',
     'base_url: "http://embed:8104/v1"',
     1),
    ('base_url: "http://127.0.0.1:8105/v1"',
     'base_url: "http://embed-l512:8105/v1"',
     1),

    # -- servers you run on the host itself ---------------------------------
    # host.docker.internal is declared in docker-compose.yml's extra_hosts, so
    # it resolves on a Linux host too, not only Docker Desktop.
    # Two: the Ollama generation backend, and the commented ollama_nomic
    # embedding example, which needs the same host once uncommented in here.
    ('    base_url: "http://localhost:11434/v1"',
     '    base_url: "http://host.docker.internal:11434/v1"',
     2),
    ('    base_url: "http://localhost:1234/v1"',
     '    base_url: "http://host.docker.internal:1234/v1"',
     1),
    # llama.cpp appears twice: a provider entry and generation.local.
    ('    base_url: "http://localhost:5001/v1"',
     '    base_url: "http://host.docker.internal:5001/v1"',
     2),
    # Obsidian's Omnisearch HTTP server -- Obsidian runs on the host.
    ('    base_url: "http://127.0.0.1:51361"',
     '    base_url: "http://host.docker.internal:51361"',
     1),
    # A document VLM is minutes per page on container CPU, so it is always a
    # server on the host.
    ('    base_url: "http://127.0.0.1:8100/v1"',
     '    base_url: "http://host.docker.internal:8100/v1"',
     1),
]

# config.example.yaml has no `code:` section (the live config.yaml does), so
# every code.* read falls back to its in-code default -- including
# `skip_roots: null`, which means "the built-in agent-project roots". Those are
# one developer's folder names and have no business being skipped in someone
# else's vault. State the block explicitly instead of inheriting that.
CODE_SECTION = '''
# ---------------------------------------------------------------------------
# Source-code ingestion (`ingest-code`). Added by make_docker_config.py:
# config.example.yaml omits this section, so without it every value below comes
# from code_loader's own defaults -- and `skip_roots: null` would mean "skip
# the built-in agent-project folder names", which are not yours.
# ---------------------------------------------------------------------------
code:
  # vault_path falls back to notebooks -> pdf -> parser vault_path if unset
  output_file: "data/code_chunks.jsonl"
  # extensions: null = every ext in the built-in language map; set a list to
  # restrict, e.g. [".js", ".ts", ".sql"]. Add new ones via language_map below.
  extensions: null
  # language_map: extra/override ext -> fence-label pairs merged over the
  # built-in map (DEFAULT_LANGUAGE_MAP in code_loader.py). e.g. {".zig": "zig"}
  language_map: {}
  min_chunk_size: 200
  max_chunk_size: 3000
  overlap_size: 150
  skip_scripts: ["armenian"]
  max_file_bytes: 1500000                        # skip files bigger than this (~1.5MB)
  # Folders to leave out of the code lane unless --include-path names them
  # (vendored dependencies, build output, someone else's checkout).
  # [] = skip nothing. null would mean "the built-in defaults" instead.
  skip_roots: []
'''

HEADER = """\
# ===========================================================================
#  GENERATED -- do not hand-edit this copy in the deployment.
#
#  Built by make_docker_config.py from the project's config.example.yaml, so
#  it carries every key the current build has. Only container-specific values
#  were rewritten: index paths onto the /rag_data volume, service URLs onto
#  compose service names or host.docker.internal, and the academic taxonomy
#  defaults switched off for a vault nobody has labelled yet.
#
#  EDITING IT ON THE MACHINE THAT RUNS IT IS FINE -- that is what it is for.
#  Under compose the console's Settings tab writes here and so can you; under
#  Kubernetes it is a ConfigMap, so change it by regenerating. The warning is
#  only about editing a copy inside a source tree, where the next build
#  overwrites it.
# ===========================================================================
"""

# Final values asserted after rewriting, so a no-op substitution cannot ship.
EXPECTED = {
    "parser.vault_path": "${RAG_HOST_HOME}",
    "paths.chroma_dir": "/rag_data/chroma_db",
    "paths.bm25_index": "/rag_data/bm25_index.pkl",
    "paths.chunks_file": "data/chunks.jsonl",
    "taxonomy.detect_from_path": False,
    "webui.rag_api": "http://127.0.0.1:${RAG_PORT:-8051}",   # same container: host stays loopback
    "code.skip_roots": [],
    "graph.enabled": False,
    "embedding.providers.eg2.base_url": "http://embed:8104/v1",
    "embedding.providers.eg2_l512.base_url": "http://embed-l512:8105/v1",
}

# The one loopback URL that is correct inside the container: the console and the
# query API share it.
LOOPBACK_ALLOWED = ("rag_api:",)


def dig(tree: dict, dotted: str):
    node = tree
    for part in dotted.split("."):
        node = node[part]
    return node


def flat(tree: dict, prefix: str = "") -> dict:
    out = {}
    for key, value in (tree or {}).items():
        if isinstance(value, dict) and value:
            out.update(flat(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out


def apply_rewrites(text: str, rewrites: list[tuple[str, str, int]], source: str) -> str:
    for old, new, expected in rewrites:
        hits = text.count(old)
        if hits != expected:
            raise SystemExit(
                f"{source}: expected {expected} occurrence(s) of\n"
                f"    {old!r}\n"
                f"  but found {hits}. The source config changed -- update the "
                f"rewrite list rather than ship a config with the wrong value."
            )
        text = text.replace(old, new)
    return text


def reject_host_references(text: str) -> None:
    """No absolute host path and no loopback URL may survive the rewrite: those
    are the values that make the container read the wrong filesystem or call
    itself. Comments are exempt (they explain, they do not configure)."""
    code_lines = [(n, ln) for n, ln in enumerate(text.splitlines(), 1)
                  if not ln.lstrip().startswith("#")]
    paths = [ln for _, ln in code_lines
             if "A:" in ln or "G:\\" in ln or "/Users/" in ln or re.search(r"\b[A-Za-z]:/", ln)]
    if paths:
        raise SystemExit("generated config carries a host path:\n  " + "\n  ".join(paths))
    # The text before an inline ` #` comment is the value; the comment is not.
    loops = [f"line {n}: {ln.strip()}" for n, ln in code_lines
             if re.search(r"127\.0\.0\.1|localhost", ln.split(" #")[0])
             and not any(ok in ln for ok in LOOPBACK_ALLOWED)]
    if loops:
        raise SystemExit(
            "generated config still points at the container itself (loopback). "
            "Add a rewrite for each:\n  " + "\n  ".join(loops))


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    src, dst = Path(argv[1]), Path(argv[2])
    text = src.read_text(encoding="utf-8")
    text = apply_rewrites(text, REWRITES, src.name)
    text = HEADER + text.lstrip("\ufeff") + CODE_SECTION

    # Parse what we are about to ship. A config that does not load is the one
    # failure mode a generator must never pass on.
    import yaml
    tree = yaml.safe_load(text)
    for dotted, want in EXPECTED.items():
        got = dig(tree, dotted)
        if got != want:
            raise SystemExit(f"generated config: {dotted} is {got!r}, expected {want!r}")
    reject_host_references(text)

    # LF on every build machine: a Windows build would otherwise write CRLF into a
    # file that is mounted into a Linux container and edited there (the archive is
    # then not the same bytes whichever OS built it).
    with open(dst, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print(f"generated {dst.name} from {src.name} "
          f"({len(REWRITES)} rewrites, {len(tree)} top-level sections)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
