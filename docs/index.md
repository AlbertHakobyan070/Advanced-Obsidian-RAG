# Advanced Obsidian RAG

**Grounded, cited question-answering over a large personal Obsidian vault.**

Ask a question in plain language and get an answer assembled *only* from your own
notes, textbooks, homework, and notebooks — with inline `[n]` citations and a
per-answer confidence line. When the vault is silent on something, the system says so
rather than inventing an answer.

!!! quote "At a glance"
    - **~170,000 retrieval chunks** across **4,400+ documents** — markdown notes, 280+
      textbooks, lecture PDFs, passed homework, Jupyter/R notebooks, scripts, and OCR'd
      scanned books.
    - **24+ courses**, **9 knowledge domains**, served from **~4 GB** of prebuilt
      indexes on a laptop.
    - **Free / local by default** — CPU embeddings, on-disk vector + sparse indexes, and
      any OpenAI-compatible endpoint for generation.

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
