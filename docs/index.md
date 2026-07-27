# Advanced Obsidian RAG

**Grounded, cited question-answering over your own documents.**

Point it at a folder — an Obsidian vault, a research library, a documentation
tree, a pile of scanned PDFs — and ask questions in plain language. Answers are
assembled *only* from what you indexed, with inline `[n]` citations and a
per-answer confidence line. When your corpus is silent on something, it says so
rather than inventing an answer.

!!! quote "At a glance"
    - **Reads what you already have**: Markdown (Obsidian wikilinks and
      frontmatter included), PDFs — scanned ones too, via OCR — Jupyter and R
      notebooks, source code in most languages, and Office documents.
    - **Free / local by default**: CPU embeddings, on-disk vector and sparse
      indexes, no managed service. Generation is the only part that can be
      remote, and **retrieval works with no LLM at all**.
    - **Metadata-aware**: domain, path, file-type and user-tag metadata support
      scoped retrieval without splitting the corpus into separate indexes.
    - **Read the live shape from `GET /stats`** — no copied total stays accurate
      as a corpus grows.

## Why a purpose-built pipeline

A general chatbot answers from the open web. It cannot tell you what *your*
policy says, how *your* codebase solved a problem, or which page of *your*
reference carries a proof. This system indexes a document collection and answers
strictly from it, so every claim traces back to a source you already trust.

It is a full retrieval pipeline rather than a thin wrapper:

- **Hybrid retrieval** — dense embeddings + BM25, fused by Reciprocal Rank Fusion.
- **Query expansion** — HyDE (and optional HyPE) to bridge the vocabulary gap
  between a short question and long-form documents.
- **Intent-aware scope routing** — soft-routes a query toward the right subject
  area, path or file type, without ever being able to empty the result set.
- **A dedicated code lane** so scripts and notebooks surface for code questions
  instead of being buried under prose.
- **Swappable reranking** — cross-encoder, external HTTP reranker, model-free
  lexical, or none.
- **Citation-audited generation** — answers cite their sources and can self-verify.
- **A management console and an agent-facing API**, plus a reproducible
  evaluation suite.

## Where to go next

<div class="grid cards" markdown>

-   :material-rocket-launch: **[Getting started](getting-started.md)**

    Install, configure, build the indexes, ask your first question.

-   :material-sitemap: **[Architecture](architecture.md)**

    The full query path, scope routing, the code lane, and the RRF math.

-   :material-console: **[Usage](usage.md)**

    CLI, the warm HTTP API, the console, presets, and per-query knobs.

-   :material-api: **[API reference](api.md)**

    Every endpoint on both services, with permission tiers.

-   :material-robot: **[Agent integration](agents.md)**

    Driving this from an LLM agent, efficiently and without fabricated citations.

-   :material-chart-line: **[Evaluation](evaluation.md)**

    The golden suite, the metrics, and an honest read of what they mean.

-   :material-wrench: **[Operations](operations.md)**

    Content-addressed IDs, the swap playbook, index integrity, retagging.

-   :material-docker: **[Docker deployment](deployment-docker.md)**

    Run the whole stack in two commands, and stop the parts you aren't using.

</div>

## Recent additions

- **A PaddleOCR sidecar** gives scanned PDFs a real detection + recognition
  pipeline — far better than Tesseract on skewed, multi-column and non-Latin
  pages — in its own container, so it can be stopped when you are not ingesting.
- **Container profiles** make the heavy optional services (OCR, an external
  reranker) opt-in and independently stoppable, so a laptop can reclaim their
  memory.
- **Readiness that explains itself**: `GET /health` distinguishes *loading* from
  *failed* and carries the reason, and the console can tail the query API's
  startup log — so a restart that fails says why instead of spinning.
- **Reranker preflight** checks whether a cross-encoder is reachable and cached
  *before* you commit to the restart that applies it.
- **Query comparison trees** run one question under multiple presets, rerank
  methods, or generation backends and report source membership, rank shifts and
  overlap.
- **Stable evidence ids** connect `/search`, `/query` and `/compare` results to
  direct `/chunks/{id}` lookup when the source reports `lookup_available: true`.
- **Provider discovery and per-query overrides** expose resolved backend/model
  provenance without exposing secrets; provider-only branches reuse identical
  evidence for fair answer comparisons.
