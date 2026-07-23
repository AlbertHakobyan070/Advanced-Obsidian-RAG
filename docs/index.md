# Advanced Obsidian RAG

**Grounded, cited question-answering over a large personal Obsidian vault.**

Ask a question in plain language and get an answer assembled *only* from your own
notes, textbooks, homework, and notebooks — with inline `[n]` citations and a
per-answer confidence line. When the vault is silent on something, the system says so
rather than inventing an answer.

!!! quote "At a glance"
    - A mixed corpus of markdown notes, textbooks, lecture PDFs, passed homework,
      Jupyter/R notebooks, scripts, and OCR'd scanned books. Read the live shape from
      `GET /stats`; no copied total stays accurate as the vault evolves.
    - Course, domain, path, file-type, and user-tag metadata support scoped retrieval
      without splitting the corpus into separate indexes.
    - **Free / local by default** — CPU embeddings, on-disk vector + sparse indexes, and
      configurable OpenAI- or Anthropic-compatible generation providers.

## Why a purpose-built pipeline

A general chatbot answers from the open web. It can't tell you what *your* course
emphasised, how *your* homework solved a problem, or which page of *your* textbook
carries a proof. This system indexes a personal knowledge vault and answers strictly
from it, so every claim traces back to a source you already trust.

It is a full retrieval pipeline rather than a thin wrapper:

- **Hybrid retrieval** — dense embeddings + BM25, fused by Reciprocal Rank Fusion.
- **Query expansion** — HyDE (and optional HyPE) to bridge the vocabulary gap between a
  short question and long-form notes.
- **Intent-aware scope routing** — soft-routes queries toward the right domain, path, or
  file type without ever being able to empty the result set.
- **A dedicated code lane** so your own scripts and notebooks surface for code questions.
- **Cross-encoder reranking** for precision at the top.
- **Citation-audited generation** — answers cite their sources and can self-verify.
- **A management console + agent API** and a **reproducible evaluation suite**.

## Recent additions

- **Query comparison trees** run one question under multiple presets, rerank methods,
  or generation providers and report source membership, rank shifts, and overlap.
- **Stable evidence ids** connect `/search`, `/query`, and `/compare` results to direct
  `/chunks/{id}` lookup when the source reports `lookup_available: true`.
- **Provider discovery and per-query overrides** expose resolved backend/model
  provenance without exposing secrets; provider-only branches reuse identical
  evidence for fair answer comparisons.
- **Explicit reranker failures** keep retrieval-model errors distinct from generation
  outages. Known model context limits are enforced instead of surfacing opaque tensor
  errors.
- **MiniMax M3 Token Plan support** uses the subscription credential type and rejects a
  pay-as-you-go key before it can be mistaken for plan quota.

## Where to go next

<div class="grid cards" markdown>

-   :material-sitemap: **[Architecture](architecture.md)**

    The full query path, scope routing, the code lane, and the RRF math.

-   :material-rocket-launch: **[Getting started](getting-started.md)**

    Install, configure, build the indexes, ask your first question.

-   :material-console: **[Usage](usage.md)**

    CLI, the warm HTTP API, the console, presets, and per-query knobs.

-   :material-chart-line: **[Evaluation](evaluation.md)**

    The golden suite, the metrics, and an honest read of what they mean.

-   :material-wrench: **[Operations](operations.md)**

    Content-addressed IDs, the swap playbook, index integrity, retagging.

-   :material-docker: **[Docker deployment](deployment-docker.md)**

    Run both services plus a generation backend on another machine in two commands.

</div>
