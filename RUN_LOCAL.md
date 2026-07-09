# Running Fully Local (KoboldCPP + a small open model + optional agent)

This stack runs with **zero API cost** and **zero data leaving your machine** —
everything from embeddings to generation happens locally. Here's the wiring.

## 1. Launch KoboldCPP with a small open model (Q8 quant)

KoboldCPP exposes an OpenAI-compatible API at `/v1`, so the pipeline's
existing `llm_client.py` talks to it with no code changes — only config.

```bash
# Windows
koboldcpp.exe --model <your-q8-model>.gguf ^
              --contextsize 8192 ^
              --gpulayers 99 ^
              --port 5001

# Linux / Mac
./koboldcpp --model <your-q8-model>.gguf \
            --contextsize 8192 \
            --gpulayers 99 \
            --port 5001
```

**Critical:** context length is set HERE (`--contextsize`), not per-request.
The OpenAI-compatible `/v1/chat/completions` route can't change it. 8192 is
plenty for RAG (5 reranked chunks ~ 4K tokens of context + the answer).
Bump it if you raise `rerank_top_k` or `max_chunk_size`.

`--gpulayers 99` offloads all layers to GPU. At Q8, a ~4-5B param model
wants ~12-16 GB VRAM for comfortable operation. If you run out, lower
the number — KoboldCPP runs the overflow on CPU.

When it boots you'll see:
`Starting OpenAI Compatible API on port 5001 at http://localhost:5001/v1/`

## 2. Point the RAG pipeline at it

In `config.yaml`, flip the generation provider:

```yaml
generation:
  provider: local        # <- the one line that matters
  local:
    base_url: "http://localhost:5001/v1"
    api_key: "koboldcpp"
    model: "<whatever /v1/models reports>"
```

Optionally go fully offline by also running embeddings locally:

```yaml
embedding:
  provider: local        # sentence-transformers, on-device, free
  local_model: "BAAI/bge-small-en-v1.5"
```

Then re-index (local embeddings produce different vectors than OpenAI's, so
the ChromaDB collection must be rebuilt):

```bash
python main.py index
python main.py query "Explain the central limit theorem from my notes"
```

## 3. (Optional) Route through a coding agent

A coding agent (e.g. Claude Code, Hermes Agent, OpenCode) that supports any
OpenAI-compatible endpoint can sit in front of KoboldCPP, giving the RAG
an agent layer — memory, tools, multi-step reasoning — on top of retrieval.

Point the agent at `http://localhost:5001/v1` as a custom endpoint. The
agent then exposes its own OpenAI-compatible API server, which the RAG
pipeline can target instead of KoboldCPP directly — chaining:

```
KoboldCPP (:5001) -> coding agent -> RAG pipeline (config.yaml -> generation.local.base_url)
```

Point `base_url` at the agent's API server's address rather than
KoboldCPP's when you want the agent in the loop, or keep it on KoboldCPP
for raw single-pass RAG.

## Quality note for a small local model

A 4-5B param Q8 model is capable but smaller than flagship closed-source
models. Two things keep citation quality high:
- `src/prompts/generation.yaml` is written with explicit, strict citation
  rules (small models need tighter constraints than large ones).
- `verify_citations: true` runs a second local pass to audit each
  citation. It's free locally — just slower. Keep
  `src/prompts/verify.yaml` short.
