# Running Fully Local (KoboldCPP + Gemma-4-E4B + optional the agent)

This stack runs with **zero API cost** and **zero data leaving your machine** —
everything from embeddings to generation happens locally. Here's the wiring.

## 1. Launch KoboldCPP with Gemma-4-E4B-it-Q8

KoboldCPP exposes an OpenAI-compatible API at `/v1`, so the pipeline's existing
`llm_client.py` talks to it with no code changes — only config.

```bash
# Windows
koboldcpp.exe --model gemma-4-E4B-it-Q8_k.gguf ^
              --contextsize 8192 ^
              --gpulayers 99 ^
              --port 5001

# Linux / Mac
./koboldcpp --model gemma-4-E4B-it-Q8_k.gguf \
            --contextsize 8192 \
            --gpulayers 99 \
            --port 5001
```

**Critical:** context length is set HERE (`--contextsize`), not per-request. The
OpenAI-compatible `/v1/chat/completions` route can't change it. 8192 is plenty
for RAG (5 reranked chunks ≈ 4K tokens of context + the answer). Bump it if you
raise `rerank_top_k` or `max_chunk_size`.

`--gpulayers 99` offloads all layers to GPU. At Q8, E4B wants ~12–16GB VRAM for
comfortable operation. If you run out, lower the number — KoboldCPP runs the
overflow on CPU.

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
    model: "gemma-4-E4B-it-Q8_k"
```

Optionally go fully offline by also running embeddings locally:

```yaml
embedding:
  provider: local        # sentence-transformers, on-device, free
  local_model: "BAAI/bge-small-en-v1.5"
```

Then re-index (local embeddings produce different vectors than OpenAI's, so the
ChromaDB collection must be rebuilt):

```bash
python main.py index
python main.py query "Explain knowledge distillation in my capstone"
```

## 3. (Optional) Route through the agent Agent

the agent (Nous Research) supports any OpenAI-compatible endpoint. You can put
the agent in front of KoboldCPP so the RAG system gains an agent layer — memory,
tools, multi-step reasoning — on top of retrieval.

Configure the agent interactively:

```bash
rag model      # choose "Custom OpenAI-compatible endpoint"
                  # URL: http://localhost:5001/v1
                  # the agent verifies against /v1/models and confirms the model
```

the agent itself then exposes an OpenAI-compatible API server, which the RAG
pipeline can target instead of KoboldCPP directly — chaining:

```
KoboldCPP (:5001) -> the agent agent -> RAG pipeline (config.yaml -> generation.local.base_url)
```

Point `base_url` at the the agent API server's address rather than KoboldCPP's when
you want the agent in the loop, or keep it on KoboldCPP for raw single-pass RAG.

## Quality note for the small model

Gemma-4-E4B is capable but smaller than Claude Sonnet. Two things keep citation
quality high:
- `src/prompts/generation.yaml` is written with explicit, strict citation rules
  (small models need tighter constraints than large ones).
- `verify_citations: true` runs a second local pass to audit each citation. It's
  free locally — just slower. Keep `src/prompts/verify.yaml` short.
