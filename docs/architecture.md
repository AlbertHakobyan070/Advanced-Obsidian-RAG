# Architecture

Every stage below is configured from `config.yaml` and can be swapped, tuned, or
turned off. Solid paths are always on; optional lanes open only when a query calls for
them.

## The query path

```mermaid
flowchart TD
    Q["Question"] --> R{Intent routing}
    R -->|prose| H["HyDE query expansion<br/>LLM drafts a hypothetical answer to embed"]
    R -->|code intent| C0["Skip HyDE · widen pool · open code lane"]
    H --> HY["Hybrid retrieval"]
    C0 --> HY
    HY --> D["Dense · ChromaDB<br/>bge-small-en-v1.5"]
    HY --> S["Sparse · bm25s"]
    HY --> SL["Scope lanes<br/>domain / path / file-type filters"]
    HY --> CL["Code lane<br/>.ipynb / .py / .R / .sql / …"]
    D --> RRF["Reciprocal Rank Fusion (k=60)<br/>+ metadata boosts"]
    S --> RRF
    SL --> RRF
    CL --> RRF
    RRF --> RR["Configured rerank policy<br/>cross-encoder / HTTP / lexical / none"]
    RR --> EX["Optional small-to-big<br/>context expansion"]
    EX --> G["Grounded generation<br/>answer + [n] citations + confidence"]
    G --> V["Optional second-pass<br/>citation verification"]
```

## Ingestion

The index is built from JSONL chunk files, one per source family. Each loader
normalises its inputs into the same chunk schema (text + metadata: `source_file`,
`domain`, `course`, `file_type`, `has_code`, `wikilinks`, …):

| Loader | Handles |
|---|---|
| `obsidian_parser` | Markdown notes; heading-aware sectioning; course/domain tagging; wikilink capture. |
| `pdf_loader` | Lecture PDFs and textbooks; OCR-capable for scanned pages. |
| `ipynb_loader` | Jupyter and R notebooks (`.ipynb`, `.R`, `.Rmd`, `.py`). |
| `code_loader` | Other source languages (`.js/.ts/.sql/.go/.java/.c/.cpp/.rs/.sh/…`) into a dedicated code lane. |
| `ocr_vlm` | Vision-model OCR path for hard scanned material. |

!!! note "Selectable chunking"
    Oversized sections can be split by **`heading`** (the default — respects document
    structure and wins on well-structured PDFs) or **`fixed`** (a sliding window, better
    for OCR walls of text with no paragraph breaks). Choose per run with
    `--chunking heading|fixed`.

## Hybrid retrieval

- **Dense** — sentence embeddings (bge-small-en-v1.5) in ChromaDB capture meaning and
  paraphrase.
- **Sparse** — BM25 (`bm25s`) captures exact terminology, symbols, rare names, and code
  tokens that embeddings blur.
- **Fusion** — the two ranked lists are combined by **Reciprocal Rank Fusion**:

    $$\text{RRF}(d) = \sum_{\ell \in \text{lanes}} \frac{1}{k + r_\ell(d)}, \quad k = 60$$

    where $r_\ell(d)$ is document $d$'s rank in lane $\ell$. RRF needs no score
    normalisation across incompatible scales, and $k=60$ damps the influence of any one
    lane's top ranks so a single list can't dominate.

!!! info "Why fusion is unweighted"
    Per-lane weights (`w_ℓ / (k + r)`) were evaluated and **deliberately skipped**. The
    downstream cross-encoder already decides final order once the right chunks are in
    the pool — weighting only re-introduces a tuning burden for gains inside the noise.

## Scope routing

Queries that name a domain or content type are routed toward where they point:

- `retrieval.domain_signals` maps aliases (e.g. "BI", "DataViz", "pytorch") to `domain`
  metadata values.
- `retrieval.content_signals` maps phrases ("homework", "lecture files", "tech books",
  "cheat sheet") to path substrings or file types.

A detected scope adds **filtered dense + sparse lanes** to the fusion. Routing is
**soft**: in-scope chunks are guaranteed seats in the candidate pool, but the reranker
still makes the final call — so a wrong hint degrades gracefully instead of returning
nothing. Both dictionaries are config-only; extend them without touching code.

## The code lane

Code and notebook chunks are a small slice of the corpus. For a query like *"show me a
complex ggplot from my code"*, a prose-oriented pipeline writes a prose HyDE draft that
lands near lecture notes, and BM25's code hits get outvoted in fusion by textbook pages
that merely repeat the keyword — so the reranker never even sees the user's own code.

The fix, triggered by a configurable code-intent signal list:

1. **Skip HyDE** (a prose hypothetical hurts code retrieval).
2. **Widen** the dense/sparse candidate pool.
3. **Reserve a filtered lane** for script/notebook chunks so they always reach the
   reranker.

Concept queries are untouched; code queries get their own material back at the top.

## Reranking and context expansion

The configured rerank policy is one of four deliberately different lanes:

- **`cross_encoder`** reads each query/candidate pair in process.
- **`http`** delegates the same candidates to a configured OpenAI-style
  `/v1/rerank` service.
- **`lexical`** measures model-free query-term coverage.
- **`none`** preserves the fused RRF order for a no-rerank baseline.

Known in-process models carry their context limits in the reranker registry.
`BAAI/bge-reranker-base`, for example, accepts at most 512 input tokens; an
over-length configuration is rejected explicitly instead of failing later with an
opaque tensor-index error. Runtime model failures are returned as reranking errors,
distinct from generation-provider failures.

After reranking, an optional **small-to-big** step expands each survivor with its
surrounding parent context before generation, trading prompt length for completeness.

## Grounded generation

The generator answers from the retrieved excerpts **only**, emits inline `[n]`
citations and a confidence line, and can run a **second pass** that checks each citation
actually supports the sentence it's attached to. A provider registry resolves the
wire protocol, endpoint, default model, and secret environment variable; a query may
override only the configured provider/model pair, never inject an endpoint or key. If
generation is unavailable, the API returns a provider-aware error object and
retrieval-only continues to work.

## Serving

Two FastAPI services, designed to run side by side:

- **`serve_api` (`:8051`)** — the warm query endpoint. `/query` and `/search` return
  stable evidence ids plus their retrieval origins; `/chunks/{id}` dereferences
  evidence when `lookup_available` is true; `/providers` reports backend readiness
  without secrets; and `/compare` runs bounded preset, reranker, or
  generation-provider branches. It loads the indexes and models once at startup and
  stays hot.
- **`manage_api` (`:8052`)** — the **Corpus Ledger** console backend: browse, retag, and
  delete documents; run ingest and maintenance jobs; and an in-app **Info** tab that
  draws this pipeline with a query/ingestion toggle.
