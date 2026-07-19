"""
serve_api.py — Warm HTTP endpoint for the RAG, for agents and bots.

Loads the pipeline ONCE at startup and keeps it warm, so every request is a
sub-second-to-proxy-latency HTTP call instead of a 350MB cold reload. Returns
compact JSON (not HTML) so an agent can read the answer directly with one curl,
without browser automation / page snapshots / vision tokens.

Run (inside your venv, from project root):
    python -m uvicorn serve_api:app --host 127.0.0.1 --port 8051

Endpoints (agent-facing; GET /schema returns this list machine-readably):
    GET  /health      -> {"ready": true}
    GET  /schema      -> endpoint + knob discovery for agents
    GET  /stats       -> corpus size, per-domain / per-file-type breakdown
    GET  /omnisearch  -> raw LIVE-vault results (Obsidian Omnisearch passthrough)
    POST /search      -> retrieval ONLY (chunks + labels + text; no LLM needed —
                         works even when the generation proxy is down)
    POST /query       -> full RAG (retrieve + grounded, cited generation)
    GET/POST /config  -> read / live-update retrieval defaults

Query (cmd.exe — escaped quotes; PowerShell needs Invoke-RestMethod instead):
    curl.exe -s -X POST http://127.0.0.1:8051/query ^
         -H "Content-Type: application/json" ^
         -d "{\"q\": \"Where in my coursework did I use conjugate priors?\"}"

Per-query knobs (all optional, never restart anything):
    {"q": "...", "top_k": 10}            # override rerank_top_k for this call
    {"q": "...", "preset": "code"}       # named bundle from retrieval.presets
    {"q": "...", "hyde": false}          # force HyDE off (or on) for this call
    {"q": "...", "hype": true}           # HyPE question-matching lane on/off
    {"q": "...", "omnisearch": true}     # add the live-vault lane (Obsidian open)
    {"q": "...", "retrieve_only": true}  # skip generation, return the chunks
    {"q": "...", "include_text": 800}    # attach up to N chars of each source
    {"q": "...", "max_tokens": 800}      # cap the ANSWER length (output tokens)

Defaults (live-update the warm pipeline; persist=true also writes config.yaml):
    curl.exe -s -X POST http://127.0.0.1:8051/config ^
         -H "Content-Type: application/json" ^
         -d "{\"rerank_top_k\": 10, \"persist\": true}"

Port 8051 by default — pick any free port (8000 often collides with Jupyter)."""
from __future__ import annotations

from collections import Counter
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.generation.generator import Generator
from src.pipeline import RAGPipeline
from src.utils.config_loader import load_config, persist_config_values

# Holds the warm pipeline; built once in the lifespan handler below.
_STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the heavy pipeline a single time at server startup.
    cfg = load_config()
    _STATE["rag"] = RAGPipeline.from_config(cfg)
    _STATE["cfg_path"] = cfg.project_root / "config.yaml"
    _STATE["ready"] = True
    yield
    _STATE.clear()


app = FastAPI(title="Personal RAG API", version="0.3.0", lifespan=lifespan)

# The management web UI (manage_api.py, default :8052) calls this API from the
# browser; same-machine, different port = CORS. Localhost-only origins.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryIn(BaseModel):
    q: str
    # Per-call override of rerank_top_k (how many reranked chunks reach the
    # LLM). Omit to use the config default / preset value.
    top_k: int | None = Field(default=None, ge=1, le=50)
    # Per-lane candidate-pool sizes fed INTO RRF fusion (before rerank). Bigger
    # = a wider net for that lane. dense = vector/semantic lane, sparse = BM25
    # keyword lane. Omit to use the preset / config default (20 each). These
    # govern what the reranker gets to choose from — the higher-leverage knobs.
    dense_top_k: int | None = Field(default=None, ge=1, le=200)
    sparse_top_k: int | None = Field(default=None, ge=1, le=200)
    # Named bundle from retrieval.presets: code | concept | synthesis.
    # Unset = auto (code preset kicks in on code-intent queries).
    preset: str | None = None
    # Force HyDE on/off for this call (beats the preset value). Unset = default.
    hyde: bool | None = None
    # Add/skip the live-vault Omnisearch lane for this call. Unset = the
    # configured default (retrieval.omnisearch.enabled). Needs Obsidian open.
    omnisearch: bool | None = None
    # E2 small-to-big, per call (beats preset, beats config; unset = default):
    # parent_context swaps note chunks for their full section; neighbor_context
    # appends adjacent-page PDF chunks as marked supplementary sources.
    parent_context: bool | None = None
    neighbor_context: bool | None = None
    # HyPE lane (query→hypothetical-question matching; needs build_hype.py
    # to have populated the question collection — fails soft otherwise).
    hype: bool | None = None
    # Rerank method for this call: cross_encoder (semantic, the default) |
    # lexical (query-term coverage, model-free) | none (fused order as-is).
    rerank: str | None = None
    # true = skip generation entirely; the response carries the reranked chunks
    # (with labels + text) and confidence "RETRIEVE_ONLY". No LLM involved, so
    # this works even when the generation proxy is down — the calling agent can
    # reason over the chunks itself.
    retrieve_only: bool = False
    # Attach up to N characters of each source's text to the response (0 = off
    # for /query with generation; retrieve_only defaults to 1200 if unset).
    include_text: int | None = Field(default=None, ge=0, le=6000)
    # Cap how many sources come back in the JSON. Omit to show every chunk
    # that reached the generator (i.e. it follows the effective top_k).
    max_sources: int | None = Field(default=None, ge=1)
    # Cap the LLM's ANSWER length for this call (output tokens). Unset = the
    # config default (generation.max_tokens, 2500). No effect on retrieval or
    # retrieve_only calls — it only bounds generation.
    max_tokens: int | None = Field(default=None, ge=64, le=8192)


class SearchIn(BaseModel):
    q: str
    top_k: int | None = Field(default=None, ge=1, le=50)
    dense_top_k: int | None = Field(default=None, ge=1, le=200)
    sparse_top_k: int | None = Field(default=None, ge=1, le=200)
    preset: str | None = None
    hyde: bool | None = None
    omnisearch: bool | None = None
    parent_context: bool | None = None
    neighbor_context: bool | None = None
    hype: bool | None = None
    # Rerank method for this call: cross_encoder | lexical | none.
    # Unset = the configured retrieval.rerank_mode (cross_encoder).
    rerank: str | None = None
    include_text: int = Field(default=1200, ge=0, le=6000)
    max_sources: int | None = Field(default=None, ge=1)


class CitationOut(BaseModel):
    n: int
    label: str


class SourceOut(BaseModel):
    n: int
    label: str
    cited: bool
    live: bool = False            # came from the live-vault (Omnisearch) lane
    score: float | None = None    # fused / rerank score, for agent triage
    text: str | None = None       # present when include_text > 0


class QueryOut(BaseModel):
    answer: str
    confidence: str
    citations: list[CitationOut]
    sources: list[SourceOut]
    # Echo of what actually ran: preset, effective top_k, hyde_used, etc.
    retrieval: dict[str, Any] = {}


class ConfigIn(BaseModel):
    rerank_top_k: int | None = Field(default=None, ge=1, le=50)
    dense_top_k: int | None = Field(default=None, ge=1, le=200)
    sparse_top_k: int | None = Field(default=None, ge=1, le=200)
    use_hyde: bool | None = None
    # Toggle the live-vault lane default. LIVE-ONLY (not persisted to yaml —
    # flip retrieval.omnisearch.enabled in config.yaml by hand to make it stick).
    use_omnisearch: bool | None = None
    # true = also rewrite config.yaml (comment-preserving) so the new values
    # survive a restart. false = warm-pipeline only, reverts on restart.
    persist: bool = True


def _rag() -> RAGPipeline:
    return _STATE["rag"]


def _label_for(doc) -> str:
    return Generator._source_label(doc.metadata, getattr(doc, "source_label", "") or "source")


def _doc_score(doc) -> float | None:
    s = getattr(doc, "rerank_score", None)
    if s is None:
        s = getattr(doc, "score", None)
    try:
        return round(float(s), 4) if s is not None else None
    except (TypeError, ValueError):
        return None


def _sources_out(
    docs, citations, include_text: int, cap: int | None
) -> list[SourceOut]:
    cited_nums = {c.number for c in citations}
    label_by_n = {c.number: c.source_label for c in citations}
    out: list[SourceOut] = []
    for i, doc in enumerate(docs[: cap or len(docs)], start=1):
        text = None
        if include_text:
            t = (doc.text or "").strip()
            text = t if len(t) <= include_text else t[: include_text - 1].rstrip() + "…"
        out.append(
            SourceOut(
                n=i,
                label=label_by_n.get(i, _label_for(doc)),
                cited=(i in cited_nums),
                live=bool(doc.metadata.get("live")),
                score=_doc_score(doc),
                text=text,
            )
        )
    return out


def _current_config() -> dict[str, Any]:
    rag = _rag()
    omni = rag.retriever.omnisearch
    return {
        "rerank_top_k": rag.rerank_top_k,
        "dense_top_k": rag.retriever.dense_top_k,
        "sparse_top_k": rag.retriever.sparse_top_k,
        "use_hyde": rag.hyde.enabled,
        "omnisearch": {
            "configured": omni is not None,
            "enabled": bool(omni.enabled) if omni else False,
            "base_url": omni.base_url if omni else None,
        },
        "presets": rag.presets,
    }


@app.get("/health")
def health() -> dict:
    return {"ready": _STATE.get("ready", False)}


@app.get("/config")
def get_config() -> dict:
    return _current_config()


@app.post("/config")
def set_config(body: ConfigIn) -> dict:
    """Change retrieval defaults on the WARM pipeline — no restart needed."""
    rag = _rag()
    if body.rerank_top_k is not None:
        rag.rerank_top_k = body.rerank_top_k
    if body.dense_top_k is not None:
        rag.retriever.dense_top_k = body.dense_top_k
    if body.sparse_top_k is not None:
        rag.retriever.sparse_top_k = body.sparse_top_k
    if body.use_hyde is not None:
        rag.hyde.enabled = body.use_hyde
    if body.use_omnisearch is not None and rag.retriever.omnisearch is not None:
        rag.retriever.omnisearch.enabled = body.use_omnisearch

    persisted: list[str] = []
    if body.persist:
        try:
            persisted = persist_config_values(
                _STATE["cfg_path"],
                {
                    "rerank_top_k": body.rerank_top_k,
                    "dense_top_k": body.dense_top_k,
                    "sparse_top_k": body.sparse_top_k,
                    "use_hyde": body.use_hyde,
                },
            )
        except ValueError as e:
            return {"ok": False, "error": str(e), "effective": _current_config()}

    return {"ok": True, "persisted": persisted, "effective": _current_config()}


@app.post("/query", response_model=QueryOut)
def query(body: QueryIn) -> QueryOut:
    rag = _rag()

    # ---- retrieval-only fast path: no LLM, works with the proxy down ----
    if body.retrieve_only:
        try:
            docs, info = rag.search(
                body.q, preset=body.preset, top_k=body.top_k,
                dense_top_k=body.dense_top_k, sparse_top_k=body.sparse_top_k,
                hyde=body.hyde, omnisearch=body.omnisearch,
                parent_context=body.parent_context,
                neighbor_context=body.neighbor_context,
                hype=body.hype, rerank=body.rerank,
            )
        except (KeyError, ValueError) as e:
            return QueryOut(answer=f"Bad request: {e.args[0]}",
                            confidence="ERROR", citations=[], sources=[])
        inc = 1200 if body.include_text is None else body.include_text
        return QueryOut(
            answer="",
            confidence="RETRIEVE_ONLY",
            citations=[],
            sources=_sources_out(docs, [], inc, body.max_sources),
            retrieval=info,
        )

    # ---- full path: retrieve + grounded generation ----
    try:
        ans = rag.query(body.q, preset=body.preset, top_k=body.top_k,
                        dense_top_k=body.dense_top_k, sparse_top_k=body.sparse_top_k,
                        hyde=body.hyde, omnisearch=body.omnisearch,
                        parent_context=body.parent_context,
                        neighbor_context=body.neighbor_context,
                        hype=body.hype, rerank=body.rerank,
                        max_tokens=body.max_tokens)
    except (KeyError, ValueError) as e:
        # Unknown preset name — tell the caller what IS available.
        return QueryOut(
            answer=f"Bad request: {e.args[0]}",
            confidence="ERROR",
            citations=[],
            sources=[],
        )
    except Exception as e:
        # A down FreeLLMAPI proxy (or any generation failure) surfaces here.
        # Return a readable payload the agent can relay instead of a raw 500.
        return QueryOut(
            answer=(
                "Generation backend unreachable — is FreeLLMAPI running on :3001? "
                f"({type(e).__name__}: {e}) "
                "Tip: POST /search (or retrieve_only=true) still works without it."
            ),
            confidence="ERROR",
            citations=[],
            sources=[],
        )

    citations = [CitationOut(n=c.number, label=c.source_label) for c in ans.citations]
    return QueryOut(
        answer=ans.text,
        confidence=ans.confidence,
        citations=citations,
        sources=_sources_out(ans.sources, ans.citations,
                             body.include_text or 0, body.max_sources),
        retrieval=ans.retrieval or {},
    )


@app.post("/search")
def search(body: SearchIn) -> dict:
    """
    Retrieval ONLY — hybrid search + rerank, no generation. The response is
    the reranked chunks with human labels and (by default) their text, so an
    agent with its own strong model can ground itself on the author's materials
    and reason over them directly. Zero dependency on the generation proxy.
    """
    rag = _rag()
    try:
        docs, info = rag.search(
            body.q, preset=body.preset, top_k=body.top_k,
            dense_top_k=body.dense_top_k, sparse_top_k=body.sparse_top_k,
            hyde=body.hyde, omnisearch=body.omnisearch,
            parent_context=body.parent_context,
            neighbor_context=body.neighbor_context,
            hype=body.hype, rerank=body.rerank,
        )
    except (KeyError, ValueError) as e:
        return {"error": e.args[0], "results": [], "retrieval": {}}
    results = _sources_out(docs, [], body.include_text, body.max_sources)
    return {
        "results": [r.model_dump() for r in results],
        "retrieval": info,
    }


@app.get("/omnisearch")
def omnisearch(q: str, k: int = 8) -> dict:
    """
    Raw LIVE-vault lookup via Obsidian's Omnisearch plugin (no index, no LLM).
    Useful for "did I write anything about X recently" — including notes not
    yet ingested. Requires Obsidian open with Omnisearch's HTTP server on.
    """
    omni = _rag().retriever.omnisearch
    if omni is None:
        return {"error": "Omnisearch is not configured — add the "
                         "retrieval.omnisearch block to config.yaml.",
                "results": []}
    rows = omni.search(q, top_k=max(1, min(k, 50)))
    return {
        "results": [
            {
                "path": r.get("path"),
                "basename": r.get("basename"),
                "score": r.get("score"),
                "excerpt": r.get("excerpt"),
            }
            for r in rows
        ],
        "reachable": bool(rows) or not omni._warned_down,
    }


@app.get("/stats")
def stats() -> dict:
    """
    Corpus overview computed from the warm BM25 payload's metadata (already in
    memory once the first query has run). Cached after first build. Gives an
    agent/UI the domain map without a 168K-row ChromaDB scan.
    """
    if "stats" in _STATE:
        return _STATE["stats"]
    rag = _rag()
    try:
        payload = rag.retriever._get_bm25()
        metas = payload.get("metadatas", [])
    except Exception as e:
        return {"error": f"BM25 payload unavailable ({type(e).__name__}: {e})"}

    by_domain: Counter = Counter()
    by_type: Counter = Counter()
    by_course: Counter = Counter()
    for m in metas:
        m = m or {}
        by_domain[str(m.get("domain") or "unknown")] += 1
        by_type[str(m.get("file_type") or "?")] += 1
        c = str(m.get("course_name") or m.get("course") or "unknown")
        by_course[c] += 1

    out = {
        "chunks": len(metas),
        "by_domain": dict(by_domain.most_common()),
        "by_file_type": dict(by_type.most_common()),
        "top_courses": dict(by_course.most_common(20)),
        "config": _current_config(),
    }
    _STATE["stats"] = out
    return out


@app.get("/schema")
def schema() -> dict:
    """
    Machine-readable capability map so an agent can discover the knobs without
    reading the source. Presets/scopes reflect the live config.
    """
    rag = _rag()
    return {
        "service": "personal-rag",
        "version": app.version,
        "endpoints": {
            "GET /health": "readiness probe -> {ready: bool}",
            "GET /stats": "corpus size + per-domain/file-type breakdown",
            "GET /omnisearch?q=&k=": "raw live-vault results (Obsidian must be open)",
            "POST /search": {
                "purpose": "retrieval only — chunks + labels + text; no LLM needed",
                "body": {"q": "str", "top_k": "1-50? (reranked chunks to the LLM)",
                         "dense_top_k": "1-200? (vector lane pool into RRF)",
                         "sparse_top_k": "1-200? (BM25 lane pool into RRF)",
                         "preset": "str?",
                         "hyde": "bool?", "omnisearch": "bool?",
                         "parent_context": "bool? (E2: swap note chunks for full sections)",
                         "neighbor_context": "bool? (E2: add adjacent-page PDF context)",
                         "hype": "bool? (HyPE question-matching lane; needs build_hype.py)",
                         "rerank": "cross_encoder|lexical|none? (rerank method; "
                                   "unset = config retrieval.rerank_mode)",
                         "include_text": "0-6000 chars (default 1200)",
                         "max_sources": "int?"},
            },
            "POST /query": {
                "purpose": "full RAG: retrieve + grounded, cited answer",
                "body": {"q": "str", "top_k": "1-50?",
                         "dense_top_k": "1-200?", "sparse_top_k": "1-200?",
                         "preset": "str?",
                         "hyde": "bool?", "omnisearch": "bool?",
                         "parent_context": "bool?", "neighbor_context": "bool?",
                         "hype": "bool? (HyPE question-matching lane)",
                         "rerank": "cross_encoder|lexical|none?",
                         "retrieve_only": "bool (skip generation)",
                         "include_text": "0-6000 chars?", "max_sources": "int?",
                         "max_tokens": "64-8192? (cap the answer's output "
                                       "tokens; default generation.max_tokens)"},
                "confidence_values": ["HIGH", "MEDIUM", "LOW", "UNKNOWN",
                                      "RETRIEVE_ONLY", "ERROR"],
            },
            "GET/POST /config": "read / live-update retrieval defaults "
                                "(persist=true also rewrites config.yaml)",
        },
        "presets": {name: dict(vals) for name, vals in (rag.presets or {}).items()},
        "auto_preset": "the 'code' preset self-applies on code-intent queries",
        "scope_routing": "queries naming a domain ('my statistics homework') or "
                         "content type ('in the tech books') get reserved "
                         "retrieval lanes automatically — just name them in q",
        "example_cmd": (
            'curl.exe -s -X POST http://127.0.0.1:8051/search -H '
            '"Content-Type: application/json" -d '
            '"{\\"q\\": \\"how did I tune mBERT in my capstone\\", '
            '\\"top_k\\": 8}"'
        ),
    }
