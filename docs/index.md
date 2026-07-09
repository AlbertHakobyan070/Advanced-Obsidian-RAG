# Advanced Obsidian RAG

**Grounded, cited question-answering over a personal knowledge base of
markdown notes, textbooks, lecture PDFs, and notebooks.**

Ask a question in plain language and get an answer assembled *only* from your
own materials — with inline `[n]` citations and a per-answer confidence line.
When your materials are silent on something, the system says so rather than
inventing an answer.

!!! quote "At a glance"
    - **A full retrieval-augmented pipeline** over markdown notes, PDFs,
      textbooks, scanned books (OCR), Jupyter/R notebooks, and scripts.
    - **Free / local by default** — CPU embeddings, on-disk vector + sparse
      indexes, and any OpenAI-compatible endpoint for generation.
    - **Citation-audited answers** — every claim is traceable to a source you
      already trust, with a per-answer confidence line.

## Why a purpose-built pipeline

A general chatbot answers from the open web. This system indexes *your*
documents and answers strictly from them, so every claim traces back to a
source you already trust, and a wrong-but-confident answer is impossible.

It is a full retrieval pipeline rather than a thin wrapper:

- **Hybrid retrieval** — dense embeddings + BM25, fused by Reciprocal Rank
  Fusion.
- **Query expansion** — HyDE (and optional HyPE) to bridge the vocabulary
  gap between a short question and long-form notes.
- **Intent-aware scope routing** — soft-routes queries toward the right
  domain, content type, or path without ever being able to empty the
  result set.
- **A dedicated code lane** so scripts and notebooks surface for code
  questions.
- **Cross-encoder reranking** for precision at the top.
- **Citation-audited generation** — answers cite their sources and can
  self-verify.
- **A management console + agent API** and a **reproducible evaluation
  suite**.

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

    Run both services plus a generation backend on another machine in two
    commands.

</div>
