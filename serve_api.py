"""
serve_api.py — Warm HTTP endpoint for the RAG, for agents and bots.

Loads the pipeline ONCE at startup and keeps it warm, so every request is a
sub-second-to-proxy-latency HTTP call instead of a 350MB cold reload. Returns
compact JSON (not HTML) so an agent can read the answer directly with one curl,
without browser automation / page snapshots / vision tokens.

Run (inside your venv, from project root):
    python -m uvicorn serve_api:app --host 127.0.0.1 --port 8051

Endpoints (agent-facing; GET /schema returns this list machine-readably):
    GET  /health       -> {"ready": true}
    GET  /schema       -> endpoint + knob discovery for agents
    GET  /config       -> live retrieval/generation defaults and provenance
    GET  /providers    -> configured generation backends and key readiness
    GET  /stats        -> corpus size, per-domain / per-file-type breakdown
    GET  /history      -> recent calls and effective retrieval settings
    GET  /chunks/{id}  -> fetch one stable evidence id when lookup is available
    GET  /omnisearch   -> raw LIVE-vault results (Obsidian passthrough)
    POST /search       -> retrieval ONLY (chunks + labels + text; no LLM needed)
    POST /query        -> full RAG (retrieve + grounded, cited generation)
    POST /compare      -> bounded query tree across retrieval/provider branches
    POST /config       -> live-update retrieval defaults

Query (cmd.exe — escaped quotes; PowerShell needs Invoke-RestMethod instead):
    curl.exe -s -X POST http://127.0.0.1:8051/query ^
         -H "Content-Type: application/json" ^
         -d "{\"q\": \"Where in my coursework did I use conjugate priors?\"}"

Per-query knobs (all optional, never restart anything):
    {"q": "...", "top_k": 10}            # override rerank_top_k for this call
    {"q": "...", "preset": "code"}       # named bundle from retrieval.presets
    {"q": "...", "auto_preset": false}   # config-only comparison baseline
    {"q": "...", "hyde": false}          # force HyDE off (or on) for this call
    {"q": "...", "hype": true}           # HyPE question-matching lane on/off
    {"q": "...", "omnisearch": true}     # add the live-vault lane (Obsidian open)
    {"q": "...", "rerank": "lexical"}    # per-call reranking method
    {"q": "...", "provider": "name"}      # configured generation backend
    {"q": "...", "model": "model-id"}     # optional model on that backend
    {"q": "...", "retrieve_only": true}  # skip generation, return the chunks
    {"q": "...", "include_text": 800}    # attach up to N chars of each source
    {"q": "...", "max_tokens": 800}      # cap the ANSWER length (output tokens)

Sources carry stable evidence ids plus their indexed origin ids and lookup
availability. Successful /query and generated /compare
branches also report generation backend/protocol/model/usage provenance; the
retrieval echo reports the effective preset, reranker model and max length.

Defaults (live-update the warm pipeline; persist=true also writes config.yaml):
    curl.exe -s -X POST http://127.0.0.1:8051/config ^
         -H "Content-Type: application/json" ^
         -d "{\"rerank_top_k\": 10, \"persist\": true}"

Port 8051 by default — pick any free port (8000 often collides with Jupyter)."""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter, deque
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.generation.generator import Generator
from src.llm.llm_client import LLMClient
from src.pipeline import RAGPipeline
from src.retrieval.reranker import RERANK_MODES, RerankerExecutionError
from src.utils.config_loader import load_config, persist_config_values

# Holds the warm pipeline; built once in the lifespan handler below.
_STATE: dict = {}

# Rolling call history (in-memory, newest first via GET /history). Lets an
# agent see which knob combinations were tried recently — presets, pool
# sizes, expansion toggles — and what each run actually did (the retrieval
# echo), so it can tune hyperparameters instead of guessing blind.
_HISTORY: deque = deque(maxlen=50)


def _record_history(endpoint: str, body, retrieval: dict | None,
                    confidence: str, n_sources: int, t0: float) -> None:
    knobs = {k: v for k, v in body.model_dump().items()
             if v is not None and k not in ("q", "include_text", "max_sources")}
    _HISTORY.appendleft({
        "at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "endpoint": endpoint,
        "q": body.q,
        "knobs": knobs,
        "retrieval": retrieval or {},
        "confidence": confidence,
        "sources": n_sources,
        "ms": int((time.time() - t0) * 1000),
    })


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the heavy pipeline a single time at server startup.
    cfg = load_config()
    _STATE["rag"] = RAGPipeline.from_config(cfg)
    _STATE["cfg"] = cfg
    _STATE["cfg_path"] = cfg.project_root / "config.yaml"
    _STATE["generators"] = {}
    _STATE["ready"] = True
    yield
    _STATE.clear()


app = FastAPI(title="Personal RAG API", version="0.4.0", lifespan=lifespan)

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
    # Named bundle from retrieval.presets. GET /schema returns the live names.
    # Unset = auto (code preset kicks in on code-intent queries when available).
    preset: str | None = None
    # False suppresses the implicit code preset and gives comparison calls an
    # explicit config-only baseline.
    auto_preset: bool = True
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
    # Rerank method for this call: cross_encoder (local semantic model) | http
    # (configured external /v1/rerank service) | lexical (model-free) | none
    # (fused order as-is).
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
    # Optional provider-registry override. Endpoint/key facts still come only
    # from config.yaml; a caller cannot inject an arbitrary URL or secret.
    provider: str | None = None
    model: str | None = None


class SearchIn(BaseModel):
    q: str
    top_k: int | None = Field(default=None, ge=1, le=50)
    dense_top_k: int | None = Field(default=None, ge=1, le=200)
    sparse_top_k: int | None = Field(default=None, ge=1, le=200)
    preset: str | None = None
    auto_preset: bool = True
    hyde: bool | None = None
    omnisearch: bool | None = None
    parent_context: bool | None = None
    neighbor_context: bool | None = None
    hype: bool | None = None
    # Rerank method for this call: cross_encoder | http | lexical | none.
    # Unset = the configured retrieval.rerank_mode (cross_encoder).
    rerank: str | None = None
    include_text: int = Field(default=1200, ge=0, le=6000)
    max_sources: int | None = Field(default=None, ge=1)


class CitationOut(BaseModel):
    n: int
    label: str


class SourceOut(BaseModel):
    # Stable evidence id for overlap/follow-up lookup. Parent-expanded
    # sections use ``parent:<parent_id>`` so two branches are compared by the
    # text they actually received, not by whichever child chunk found it.
    id: str
    # Original retrieval id that produced this evidence. It differs from id
    # for parent-context expansion and live-vault evidence.
    origin_id: str
    # False for live Omnisearch excerpts, which are not records in Chroma or
    # the parent sidecar and therefore cannot be served by GET /chunks/{id}.
    lookup_available: bool = True
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
    retrieval: dict[str, Any] = Field(default_factory=dict)
    generation: dict[str, Any] = Field(default_factory=dict)


class CompareBranchIn(BaseModel):
    """One branch of a query comparison tree."""
    id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    label: str | None = Field(default=None, max_length=100)
    top_k: int | None = Field(default=None, ge=1, le=50)
    dense_top_k: int | None = Field(default=None, ge=1, le=200)
    sparse_top_k: int | None = Field(default=None, ge=1, le=200)
    preset: str | None = None
    auto_preset: bool = True
    hyde: bool | None = None
    omnisearch: bool | None = None
    parent_context: bool | None = None
    neighbor_context: bool | None = None
    hype: bool | None = None
    rerank: str | None = None
    provider: str | None = None
    model: str | None = None


class CompareIn(BaseModel):
    q: str = Field(min_length=1)
    mode: Literal["search", "query"] = "search"
    branches: list[CompareBranchIn] = Field(min_length=2, max_length=6)
    include_text: int = Field(default=900, ge=0, le=6000)
    max_sources: int | None = Field(default=None, ge=1)
    max_tokens: int | None = Field(default=None, ge=64, le=8192)


class ConfigIn(BaseModel):
    rerank_top_k: int | None = Field(default=None, ge=1, le=50)
    dense_top_k: int | None = Field(default=None, ge=1, le=200)
    sparse_top_k: int | None = Field(default=None, ge=1, le=200)
    use_hyde: bool | None = None
    # Toggle the live-vault lane default. LIVE-ONLY (not persisted to yaml —
    # flip retrieval.omnisearch.enabled in config.yaml by hand to make it stick).
    use_omnisearch: bool | None = None
    # true = also rewrite config.yaml (comment-preserving) so the new values
    # survive a restart. The safe default is false: persistence is a mutation
    # and callers must opt in only with operator authorization.
    persist: bool = False


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
        origin_id = str(doc.id)
        parent_id = str(
            (getattr(doc, "debug", {}) or {}).get("parent_swap") or ""
        )
        live = bool(doc.metadata.get("live"))
        if parent_id:
            evidence_id = f"parent:{parent_id}"
        elif live:
            # Omnisearch ids contain vault paths (including '/'), and excerpts
            # are query-shaped rather than indexed records. Use a safe content
            # identity for comparison while making lookup support explicit.
            digest = hashlib.sha256(
                f"{origin_id}\0{doc.text}".encode("utf-8")
            ).hexdigest()[:20]
            evidence_id = f"live:{digest}"
        else:
            evidence_id = origin_id
        text = None
        if include_text:
            t = (doc.text or "").strip()
            text = t if len(t) <= include_text else t[: include_text - 1].rstrip() + "…"
        out.append(
            SourceOut(
                id=evidence_id,
                origin_id=origin_id,
                lookup_available=not live,
                n=i,
                label=label_by_n.get(i, _label_for(doc)),
                cited=(i in cited_nums),
                live=live,
                score=_doc_score(doc),
                text=text,
            )
        )
    return out


_RETRIEVAL_FIELDS = (
    "preset", "auto_preset", "top_k", "dense_top_k", "sparse_top_k",
    "hyde", "omnisearch", "parent_context", "neighbor_context", "hype",
    "rerank",
)


def _retrieval_kwargs(body: BaseModel) -> dict[str, Any]:
    """Extract only per-call search knobs from any request/branch model."""
    return {name: getattr(body, name) for name in _RETRIEVAL_FIELDS}


def _retrieval_cache_key(branch: CompareBranchIn) -> str:
    """Stable key used to share evidence across provider-only branches."""
    return json.dumps(
        _retrieval_kwargs(branch), sort_keys=True, separators=(",", ":"))


def _provider_catalog() -> list[dict[str, Any]]:
    """Configured generation backends without secret values."""
    cfg = _STATE["cfg"]
    rows: list[dict[str, Any]] = []
    for name, raw_spec in (cfg.get("providers", {}) or {}).items():
        if name in LLMClient.RESERVED_PROVIDERS:
            continue
        spec = raw_spec or {}
        env_name = spec.get("api_key_env")
        key = os.environ.get(str(env_name), "") if env_name else ""
        prefix = str(spec.get("api_key_prefix") or "")
        key_present = bool(key) if env_name else True
        key_compatible = (not prefix or key.startswith(prefix)) if key_present else None
        optional = bool(spec.get("api_key_optional"))
        available = bool(
            (not env_name)
            or (not key_present and optional)
            or (key_present and key_compatible is not False)
        )
        rows.append({
            "name": name,
            "label": spec.get("label") or name,
            "description": spec.get("description"),
            "kind": spec.get("kind", "openai"),
            "base_url": spec.get("base_url"),
            "model": spec.get("model"),
            "api_key_env": env_name,
            "key_optional": optional,
            "key_present": key_present,
            "key_compatible": key_compatible,
            "available": available,
        })
    return rows


def _generator_for(provider: str | None, model: str | None) -> Generator:
    """Resolve a configured backend for one request, cached by backend/model."""
    default = _rag().generator
    if not provider and not model:
        return default

    backend = provider or getattr(default.llm, "backend", default.llm.provider)
    catalog = {row["name"]: row for row in _provider_catalog()}
    row = catalog.get(backend)
    if row and row["key_present"] and row["key_compatible"] is False:
        raise ValueError(
            f"Provider {backend!r} has the wrong API-key type in "
            f"{row['api_key_env']}; replace it with the key type documented "
            "for that provider.")
    if row and not row["available"]:
        raise ValueError(
            f"Provider {backend!r} is missing its required "
            f"{row['api_key_env']} secret.")

    key = (backend, model or "")
    cache = _STATE.setdefault("generators", {})
    if key not in cache:
        llm = LLMClient.from_provider_override(
            _STATE["cfg"], backend, model=model, role="generation")
        cache[key] = Generator.from_config(_STATE["cfg"], llm)
    return cache[key]


def _generation_out(generator: Generator, usage: dict | None = None) -> dict[str, Any]:
    llm = generator.llm
    return {
        "backend": getattr(llm, "backend", llm.provider),
        "protocol": llm.provider,
        "model": llm.model,
        "usage": usage,
    }


def _comparison_summary(branches: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare membership/ranks only; score scales differ across rerank modes."""
    good = [b for b in branches
            if not b.get("retrieval_error") and b.get("sources") is not None]
    ids_by_branch = {
        b["id"]: [str(s["id"]) for s in b.get("sources", [])]
        for b in good
    }
    origin_ids_by_branch = {
        b["id"]: [
            str(s.get("origin_id") or s["id"]) for s in b.get("sources", [])
        ]
        for b in good
    }
    membership: dict[str, list[str]] = {}
    ranks: dict[str, dict[str, int]] = {}
    ordered_ids: list[str] = []
    for branch_id, ids in ids_by_branch.items():
        for rank, source_id in enumerate(ids, start=1):
            if source_id not in membership:
                membership[source_id] = []
                ordered_ids.append(source_id)
            membership[source_id].append(branch_id)
            ranks.setdefault(source_id, {})[branch_id] = rank

    branch_ids = list(ids_by_branch)
    common = [
        source_id for source_id in ordered_ids
        if len(membership[source_id]) == len(branch_ids)
    ] if branch_ids else []
    unique = {
        branch_id: [
            source_id for source_id in ids
            if len(membership.get(source_id, [])) == 1
        ]
        for branch_id, ids in ids_by_branch.items()
    }
    pairwise = []
    for i, left in enumerate(branch_ids):
        for right in branch_ids[i + 1:]:
            left_ids, right_ids = ids_by_branch[left], ids_by_branch[right]
            depth = min(len(left_ids), len(right_ids))
            left_at_k, right_at_k = set(left_ids[:depth]), set(right_ids[:depth])
            overlap = len(left_at_k & right_at_k)
            union = left_at_k | right_at_k
            pairwise.append({
                "left": left,
                "right": right,
                "depth": depth,
                "overlap": overlap,
                "overlap_rate": round(overlap / depth, 4) if depth else None,
                "jaccard": round(overlap / len(union), 4) if union else None,
            })
    rank_spread = {
        source_id: {
            "ranks": by_branch,
            "spread": max(by_branch.values()) - min(by_branch.values()),
        }
        for source_id, by_branch in ranks.items()
        if len(by_branch) > 1
    }
    origin_membership: dict[str, list[str]] = {}
    origin_ranks: dict[str, dict[str, int]] = {}
    for branch_id, ids in origin_ids_by_branch.items():
        for rank, source_id in enumerate(ids, start=1):
            origin_membership.setdefault(source_id, []).append(branch_id)
            origin_ranks.setdefault(source_id, {})[branch_id] = rank
    return {
        "successful_branches": branch_ids,
        "common_source_ids": common,
        "unique_source_ids": unique,
        "membership": membership,
        "ranks": ranks,
        "rank_spread": rank_spread,
        "pairwise": pairwise,
        "origin_membership": origin_membership,
        "origin_ranks": origin_ranks,
        "note": "Primary overlap follows effective evidence ids; origin_* "
                "retains the indexed child identities. Raw scores are "
                "intentionally not compared because cross-encoder logits, "
                "lexical scores and fused RRF scores use different scales.",
    }


def _current_config() -> dict[str, Any]:
    rag = _rag()
    omni = rag.retriever.omnisearch
    return {
        "rerank_top_k": rag.rerank_top_k,
        "dense_top_k": rag.retriever.dense_top_k,
        "sparse_top_k": rag.retriever.sparse_top_k,
        "use_hyde": rag.hyde.enabled,
        "reranker": {
            "mode": rag.reranker.mode,
            "model": rag.reranker.model_name,
            "max_length": rag.reranker.max_length,
        },
        "generation": _generation_out(rag.generator),
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
    t0 = time.time()

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
                auto_preset=body.auto_preset,
            )
        except (KeyError, ValueError) as e:
            return QueryOut(answer=f"Bad request: {e.args[0]}",
                            confidence="ERROR", citations=[], sources=[])
        except RerankerExecutionError as e:
            return QueryOut(answer=f"Reranking failed: {e}",
                            confidence="ERROR", citations=[], sources=[])
        except Exception as e:
            return QueryOut(
                answer=f"Retrieval failed ({type(e).__name__}: {e})",
                confidence="ERROR", citations=[], sources=[])
        inc = 1200 if body.include_text is None else body.include_text
        _record_history("/query(retrieve_only)", body, info,
                        "RETRIEVE_ONLY", len(docs), t0)
        return QueryOut(
            answer="",
            confidence="RETRIEVE_ONLY",
            citations=[],
            sources=_sources_out(docs, [], inc, body.max_sources),
            retrieval=info,
        )

    # ---- full path: retrieve first, then grounded generation ----
    # Keeping the phases separate means a provider/key failure can still return
    # the exact evidence that was successfully retrieved.
    try:
        docs, info = rag.search(body.q, **_retrieval_kwargs(body))
    except (KeyError, ValueError) as e:
        # Unknown preset name — tell the caller what IS available.
        return QueryOut(
            answer=f"Bad request: {e.args[0]}",
            confidence="ERROR",
            citations=[],
            sources=[],
        )
    except RerankerExecutionError as e:
        return QueryOut(
            answer=f"Reranking failed: {e}",
            confidence="ERROR",
            citations=[],
            sources=[],
        )
    except Exception as e:
        return QueryOut(
            answer=f"Retrieval failed ({type(e).__name__}: {e})",
            confidence="ERROR",
            citations=[],
            sources=[],
        )

    try:
        generator = _generator_for(body.provider, body.model)
    except Exception as e:
        backend = (
            body.provider
            or getattr(rag.generator.llm, "backend", rag.generator.llm.provider)
        )
        generation = {
            "backend": backend,
            "model": body.model,
            "error": str(e),
        }
        return QueryOut(
            answer=f"Generation configuration failed: {e}",
            confidence="ERROR",
            citations=[],
            sources=_sources_out(
                docs, [], body.include_text or 0, body.max_sources),
            retrieval=info,
            generation=generation,
        )

    try:
        ans = generator.generate(body.q, docs, max_tokens=body.max_tokens)
        ans.retrieval = info
    except Exception as e:
        # Return a provider-aware payload the agent can relay instead of a raw
        # 500; the selected backend may be remote, local, or request-specific.
        backend = getattr(
            generator.llm, "backend", generator.llm.provider)
        model = generator.llm.model
        return QueryOut(
            answer=(
                f"Generation backend {backend!r} "
                f"(model {model!r}) failed — "
                f"{type(e).__name__}: {e}. "
                "Tip: POST /search (or retrieve_only=true) still works without it."
            ),
            confidence="ERROR",
            citations=[],
            sources=_sources_out(
                docs, [], body.include_text or 0, body.max_sources),
            retrieval=info,
            generation={
                "backend": backend,
                "model": model,
                "error": f"{type(e).__name__}: {e}",
            },
        )

    citations = [CitationOut(n=c.number, label=c.source_label) for c in ans.citations]
    _record_history("/query", body, ans.retrieval, ans.confidence,
                    len(ans.sources), t0)
    return QueryOut(
        answer=ans.text,
        confidence=ans.confidence,
        citations=citations,
        sources=_sources_out(ans.sources, ans.citations,
                             body.include_text or 0, body.max_sources),
        retrieval=ans.retrieval or {},
        generation=_generation_out(generator, ans.usage),
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
    t0 = time.time()
    try:
        docs, info = rag.search(
            body.q, preset=body.preset, top_k=body.top_k,
            dense_top_k=body.dense_top_k, sparse_top_k=body.sparse_top_k,
            hyde=body.hyde, omnisearch=body.omnisearch,
            parent_context=body.parent_context,
            neighbor_context=body.neighbor_context,
            hype=body.hype, rerank=body.rerank,
            auto_preset=body.auto_preset,
        )
    except (KeyError, ValueError) as e:
        return {"error": e.args[0], "results": [], "retrieval": {}}
    except RerankerExecutionError as e:
        return {"error": f"Reranking failed: {e}",
                "results": [], "retrieval": {}}
    except Exception as e:
        return {"error": f"Retrieval failed ({type(e).__name__}: {e})",
                "results": [], "retrieval": {}}
    _record_history("/search", body, info, "RETRIEVE_ONLY", len(docs), t0)
    results = _sources_out(docs, [], body.include_text, body.max_sources)
    return {
        "results": [r.model_dump() for r in results],
        "retrieval": info,
    }


@app.get("/providers")
def providers() -> dict:
    """Configured generation backends and readiness, never secret values."""
    return {
        "active": _generation_out(_rag().generator),
        "providers": _provider_catalog(),
        "note": "available means the configured key is present and, when a "
                "provider declares a key type, its prefix is compatible. "
                "Endpoint reachability is checked only when a query runs.",
    }


@app.get("/chunks/{chunk_id}")
def chunk(chunk_id: str, include_text: int = 6000) -> dict:
    """Fetch one stable evidence id returned by /search or /compare."""
    cap = max(0, min(int(include_text), 20000))
    if chunk_id.startswith("live:"):
        raise HTTPException(
            status_code=404,
            detail="Live Omnisearch evidence is not stored in the index; "
                   "its source reports lookup_available=false.",
        )
    if chunk_id.startswith("parent:"):
        parent_id = chunk_id.removeprefix("parent:")
        parent_ctx = getattr(_rag(), "parent_ctx", None)
        if not parent_id or parent_ctx is None:
            raise HTTPException(
                status_code=404, detail=f"Unknown evidence id {chunk_id!r}")
        record = parent_ctx._load().get(parent_id)
        if record is None:
            raise HTTPException(
                status_code=404, detail=f"Unknown evidence id {chunk_id!r}")
        text = str(record.get("text") or "").strip()
        if cap and len(text) > cap:
            text = text[: cap - 1].rstrip() + "..."
        elif not cap:
            text = None
        return {
            "id": chunk_id,
            "kind": "parent_section",
            "text": text,
            "metadata": {
                key: value for key, value in record.items()
                if key not in {"text", "parent_id"}
            },
        }
    try:
        row = _rag().retriever._get_collection().get(
            ids=[chunk_id], include=["documents", "metadatas"])
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Chunk store unavailable ({type(e).__name__}: {e})") from e
    ids = row.get("ids") or []
    if not ids:
        raise HTTPException(status_code=404, detail=f"Unknown chunk id {chunk_id!r}")
    text = ((row.get("documents") or [""])[0] or "").strip()
    if cap and len(text) > cap:
        text = text[: cap - 1].rstrip() + "..."
    elif not cap:
        text = None
    return {
        "id": str(ids[0]),
        "kind": "chunk",
        "text": text,
        "metadata": (row.get("metadatas") or [{}])[0] or {},
    }


@app.post("/compare")
def compare(body: CompareIn) -> dict:
    """Run one query as a bounded tree of retrieval/provider branches.

    Branches execute sequentially to protect the shared cross-encoder. Branches
    that differ only by provider/model reuse one exact evidence set, making the
    answer comparison about the LLM rather than retrieval noise.
    """
    branch_ids = [branch.id for branch in body.branches]
    if len(set(branch_ids)) != len(branch_ids):
        raise HTTPException(status_code=400, detail="compare branch ids must be unique")
    if body.mode == "query" and len(body.branches) > 3:
        raise HTTPException(
            status_code=400,
            detail="query comparisons are capped at 3 generated branches; "
                   "use mode='search' for wider evidence fan-out")

    rag = _rag()
    t0 = time.time()
    evidence: dict[str, tuple[list, dict]] = {}
    out: list[dict[str, Any]] = []
    for branch in body.branches:
        request = branch.model_dump(exclude_none=True)
        row: dict[str, Any] = {
            "id": branch.id,
            "label": branch.label or branch.id,
            "request": request,
            "answer": "",
            "confidence": "RETRIEVE_ONLY",
            "citations": [],
            "sources": [],
            "retrieval": {},
            "generation": {},
            "error": None,
            "retrieval_error": False,
        }
        cache_key = _retrieval_cache_key(branch)
        try:
            if cache_key not in evidence:
                evidence[cache_key] = rag.search(
                    body.q, **_retrieval_kwargs(branch))
            docs, info = evidence[cache_key]
            row["retrieval"] = info
            row["sources"] = [
                source.model_dump()
                for source in _sources_out(
                    docs, [], body.include_text, body.max_sources)
            ]
        except (KeyError, ValueError) as e:
            row["error"] = f"Bad branch configuration: {e.args[0]}"
            row["retrieval_error"] = True
            out.append(row)
            continue
        except RerankerExecutionError as e:
            row["error"] = f"Reranking failed: {e}"
            row["retrieval_error"] = True
            out.append(row)
            continue
        except Exception as e:
            row["error"] = f"Retrieval failed ({type(e).__name__}: {e})"
            row["retrieval_error"] = True
            out.append(row)
            continue

        if body.mode == "query":
            generator = None
            try:
                generator = _generator_for(branch.provider, branch.model)
            except Exception as e:
                backend = (
                    branch.provider
                    or getattr(
                        rag.generator.llm,
                        "backend",
                        rag.generator.llm.provider,
                    )
                )
                row["confidence"] = "ERROR"
                row["generation"] = {
                    "backend": backend,
                    "model": branch.model,
                    "stage": "configuration",
                    "error": f"{type(e).__name__}: {e}",
                }
                row["error"] = (
                    f"Generation configuration failed "
                    f"({type(e).__name__}: {e})")
            else:
                row["generation"] = _generation_out(generator)
            if generator is not None:
                try:
                    answer = generator.generate(
                        body.q, docs, max_tokens=body.max_tokens)
                    row["answer"] = answer.text
                    row["confidence"] = answer.confidence
                    row["citations"] = [
                        {"n": citation.number, "label": citation.source_label}
                        for citation in answer.citations
                    ]
                    row["sources"] = [
                        source.model_dump()
                        for source in _sources_out(
                            docs, answer.citations, body.include_text,
                            body.max_sources)
                    ]
                    row["generation"] = _generation_out(generator, answer.usage)
                except Exception as e:
                    row["confidence"] = "ERROR"
                    row["generation"] = {
                        **_generation_out(generator),
                        "stage": "request",
                        "error": f"{type(e).__name__}: {e}",
                    }
                    row["error"] = (
                        f"Generation failed ({type(e).__name__}: {e})")
        out.append(row)

    summary = _comparison_summary(out)
    _record_history(
        "/compare", body,
        {"mode": body.mode,
         "branches": [b["id"] for b in out],
         "successful_branches": summary["successful_branches"]},
        "RETRIEVE_ONLY" if body.mode == "search" else "COMPARE",
        len(summary["membership"]), t0,
    )
    return {
        "q": body.q,
        "mode": body.mode,
        "branches": out,
        "comparison": summary,
        "ms": int((time.time() - t0) * 1000),
    }


@app.get("/history")
def history(limit: int = 20) -> dict:
    """
    The last calls to /search, /query and /compare (newest first, in-memory,
    resets on restart): question, the knobs the caller sent, the retrieval echo
    of what actually ran, confidence and timing. An agent tuning
    hyperparameters reads this instead of re-deriving what it already tried.
    """
    return {"calls": list(_HISTORY)[: max(1, min(limit, 50))],
            "note": "in-memory since the last :8051 restart; knobs shows only "
                    "what the caller explicitly set — everything else ran on "
                    "preset/config defaults (see retrieval for the effective "
                    "values)."}


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
    rerank_choices = list(RERANK_MODES)
    retrieval_body = {
        "q": "str",
        "top_k": "1-50? (reranked chunks returned)",
        "dense_top_k": "1-200? (vector lane pool into RRF)",
        "sparse_top_k": "1-200? (BM25 lane pool into RRF)",
        "preset": "str? (name from the live presets map)",
        "auto_preset": "bool (false suppresses implicit code preset; default true)",
        "hyde": "bool?",
        "omnisearch": "bool?",
        "parent_context": "bool? (swap note chunks for full sections)",
        "neighbor_context": "bool? (add adjacent-page PDF context)",
        "hype": "bool? (question-matching lane; needs build_hype.py)",
        "rerank": f"{'|'.join(rerank_choices)}? (per-call method)",
    }
    return {
        "service": "personal-rag",
        "version": app.version,
        "endpoints": {
            "GET /health": "readiness probe -> {ready: bool}",
            "GET /schema": "this machine-readable capability map",
            "GET /config": "live retrieval defaults, presets, active reranker "
                           "and generation provenance",
            "POST /config": "live-update retrieval defaults; persist defaults "
                            "to false. persist=true rewrites config.yaml and "
                            "requires operator authorization",
            "GET /providers": "configured generation backends, default models "
                              "and key readiness; never returns secret values",
            "GET /chunks/{chunk_id}?include_text=": "fetch one stable evidence "
                                                       "id returned by /search, "
                                                       "/query or /compare when "
                                                       "lookup_available=true",
            "GET /stats": "corpus size + per-domain/file-type breakdown",
            "GET /omnisearch?q=&k=": "raw live-vault results (Obsidian must be open)",
            "POST /search": {
                "purpose": "retrieval only — chunks + labels + text; no LLM needed",
                "body": {
                    **retrieval_body,
                    "include_text": "0-6000 chars (default 1200)",
                    "max_sources": "int?",
                },
                "response": "stable evidence ids + origin ids, lookup support, "
                            "labels/text/scores, and an "
                            "effective retrieval/provenance echo",
            },
            "POST /query": {
                "purpose": "full RAG: retrieve + grounded, cited answer",
                "body": {
                    **retrieval_body,
                    "provider": "str? (configured backend alias from /providers)",
                    "model": "str? (optional model override on that backend)",
                    "retrieve_only": "bool (skip generation)",
                    "include_text": "0-6000 chars?",
                    "max_sources": "int?",
                    "max_tokens": "64-8192? (cap answer output tokens)",
                },
                "response": "answer/citations, stable evidence + origin ids, effective "
                            "retrieval echo, and generation "
                            "backend/protocol/model/usage provenance",
                "confidence_values": ["HIGH", "MEDIUM", "LOW", "UNKNOWN",
                                      "RETRIEVE_ONLY", "ERROR"],
            },
            "POST /compare": {
                "purpose": "bounded query tree across presets, retrieval "
                           "methods and configured generation backends",
                "body": {
                    "q": "str",
                    "mode": "search|query (default search)",
                    "branches": "2-6 branches; generated query mode is capped "
                                "at 3",
                    "branch": {
                        "id": "stable unique branch id",
                        "label": "str?",
                        **{k: v for k, v in retrieval_body.items() if k != "q"},
                        "provider": "str?",
                        "model": "str?",
                    },
                    "include_text": "0-6000 chars",
                    "max_sources": "int?",
                    "max_tokens": "64-8192? (query mode)",
                },
                "response": "per-branch results plus effective-evidence and "
                            "origin-chunk membership/ranks, common/unique ids "
                            "and pairwise overlap; raw scores are intentionally "
                            "not compared",
                "evidence_reuse": "branches differing only by provider/model "
                                  "reuse the exact same retrieved chunks",
            },
            "GET /history?limit=": "last /search, /query and /compare calls "
                                   "(newest first): caller knobs, effective "
                                   "retrieval echo, confidence and timing",
        },
        "ingestion": "this API is read-only over the corpus. Web fetch "
                     "(URL -> staged .md or printed .pdf), file conversion, "
                     "ingest and index jobs live on the management console "
                     "(:8052) — GET :8052/api/schema for that capability map.",
        "presets": {name: dict(vals) for name, vals in (rag.presets or {}).items()},
        "rerank_modes": rerank_choices,
        "provenance": {
            "retrieval": "effective preset/mode/reranker model/max length and "
                         "lane/context decisions",
            "generation": "configured backend alias, wire protocol, model and usage",
            "sources": "stable evidence ids support overlap comparison; "
                       "origin_id preserves the indexed child and "
                       "lookup_available gates GET /chunks",
        },
        "auto_preset": "the 'code' preset self-applies on code-intent queries "
                       "when present; send auto_preset=false for a config-only "
                       "baseline",
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
