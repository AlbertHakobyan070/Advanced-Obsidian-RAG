"""
Embedding sidecar -- EmbeddingGemma 2 on a GPU behind an OpenAI-compatible
/v1/embeddings, in its own venv.

Why a sidecar and not a model inside the project venv: EmbeddingGemma 2 needs
sentence-transformers >= 6.1.0, transformers >= 5.19 and torchvision, and the
project venv pins an older stack (and a CPU-only torch). Serving it from here
lets the rest of the project reach it through the existing `embedding.providers`
registry (kind: openai) with nothing upgraded.

Contract:

    GET  /health          -> {"ok": true, "model", "device", "dtype",
                              "max_seq_length", "dimension", "truncate_dim",
                              "batch_size", "versions": {...}}
    POST /v1/embeddings   {"model": "<the served model id>",
                           "input": "text" | ["text", ...],
                           "dimensions": 512,                # optional, Matryoshka
                           "encoding_format": "float"|"base64"}
                          -> the OpenAI embeddings response, plus one extra
                             top-level field, "max_seq_length", which the
                             client checks against the length its fingerprint
                             declares.

Deliberate choices:

  * NO PREFIXES. The client sends already-prefixed text (the registry entry's
    query_prefix / doc_prefix). A prefix applied here too would be applied
    twice, and the fingerprint could not see it.
  * max_seq_length is ALWAYS bounded. sentence-transformers' own default for
    this model is ~1e30, i.e. unbounded: 500 real chunks reached 7.6 GB RSS.
    EMBED_MAX_SEQ_LENGTH defaults to 512 (like-for-like with the bge-small
    index) and must be a positive integer.
  * fp32 only. Pascal has no native bf16 and fp16 is NaN or silently degraded
    on this model.
  * Text-only load (vision and audio towers dropped): 271M params, ~1.7 GB.
  * A request for another model id, a bad `dimensions`, or non-string input is
    an HTTP 400. Nothing is substituted and nothing falls back to the CPU: if
    EMBED_DEVICE asks for CUDA and torch cannot see it, startup fails.
  * `dimensions` (or EMBED_TRUNCATE_DIM as the server default) truncates the
    768-d vector to its first N values and RE-NORMALISES it, as the model card
    says Matryoshka truncation requires.

Configuration, all environment variables:

    EMBED_MODEL            google/embeddinggemma-2
    EMBED_MAX_SEQ_LENGTH   512
    EMBED_TRUNCATE_DIM     (unset = keep all 768)
    EMBED_BATCH_SIZE       16      sentence-transformers' internal GPU batch
    EMBED_DEVICE           cuda    cuda | cuda:N | cpu. Pick the physical card
                                   with CUDA_VISIBLE_DEVICES (index 0 inside).
    EMBED_HOST / EMBED_PORT  127.0.0.1 / 8104

Standard library HTTP server on purpose: the verified venv already holds the
whole dependency set, and one GPU serialises requests anyway.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("embed-sidecar")

DEFAULT_MODEL = "google/embeddinggemma-2"


class RequestError(Exception):
    """A bad request: becomes an HTTP 400 with the message in the body."""


class EmbedService:
    """The loaded model plus the settings that decide what its vectors are.
    `model` needs `.encode(texts, batch_size=, normalize_embeddings=,
    show_progress_bar=)` and `.get_embedding_dimension()` (a SentenceTransformer,
    or a fake in tests)."""

    def __init__(self, model, model_id: str, device: str, max_seq_length: int,
                 truncate_dim: int | None, batch_size: int, versions: dict | None = None):
        if max_seq_length < 1:
            raise ValueError(f"max_seq_length must be a positive integer, got {max_seq_length}")
        self.model = model
        self.model_id = model_id
        self.device = device
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.native_dim = int(model.get_embedding_dimension())
        if truncate_dim is not None and not 1 <= truncate_dim <= self.native_dim:
            raise ValueError(f"truncate_dim {truncate_dim} is outside 1..{self.native_dim}")
        self.truncate_dim = truncate_dim
        self.versions = versions or {}
        self._lock = threading.Lock()

    def embed(self, texts: list[str], dimensions: int | None = None) -> np.ndarray:
        dim = dimensions if dimensions is not None else self.truncate_dim
        if dim is not None and not 1 <= dim <= self.native_dim:
            raise RequestError(f"dimensions must be between 1 and {self.native_dim}, got {dim}")
        with self._lock:                                  # one GPU: one encode at a time
            vecs = np.asarray(self.model.encode(
                texts, batch_size=self.batch_size, normalize_embeddings=True,
                show_progress_bar=False), dtype=np.float32)
        if dim is not None and dim < self.native_dim:
            vecs = vecs[:, :dim]
            vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs

    def health(self) -> dict:
        return {
            "ok": True, "model": self.model_id, "device": self.device, "dtype": "float32",
            "max_seq_length": self.max_seq_length, "dimension": self.truncate_dim or self.native_dim,
            "truncate_dim": self.truncate_dim, "batch_size": self.batch_size,
            "versions": self.versions,
        }


def embeddings_response(service: EmbedService, body: dict) -> dict:
    """The OpenAI embeddings response for one request body; RequestError on a
    request this server will not answer."""
    if not isinstance(body, dict):
        raise RequestError("request body must be a JSON object")
    if body.get("model") != service.model_id:
        raise RequestError(f"this server serves {service.model_id!r}, not {body.get('model')!r}")
    texts = body.get("input")
    if isinstance(texts, str):
        texts = [texts]
    if not isinstance(texts, list) or not texts or not all(isinstance(t, str) for t in texts):
        raise RequestError("input must be a string or a non-empty list of strings")
    fmt = body.get("encoding_format", "float")
    if fmt not in ("float", "base64"):
        raise RequestError(f"encoding_format must be 'float' or 'base64', got {fmt!r}")
    dimensions = body.get("dimensions")
    if dimensions is not None and (isinstance(dimensions, bool) or not isinstance(dimensions, int)):
        raise RequestError(f"dimensions must be an integer, got {dimensions!r}")

    vecs = service.embed(texts, dimensions)
    data = [{"object": "embedding", "index": i,
             "embedding": (base64.b64encode(v.astype("<f4").tobytes()).decode("ascii")
                           if fmt == "base64" else v.tolist())}
            for i, v in enumerate(vecs)]
    return {"object": "list", "data": data, "model": service.model_id,
            # ponytail: tokens are not counted (the SDK only needs the fields to exist)
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
            "max_seq_length": service.max_seq_length}


def make_handler(service: EmbedService):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, service.health())
            else:
                self._send(404, {"error": {"message": f"no route GET {self.path}"}})

        def do_POST(self):
            if self.path != "/v1/embeddings":
                self._send(404, {"error": {"message": f"no route POST {self.path}"}})
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                self._send(200, embeddings_response(service, body))
            except (RequestError, ValueError) as e:        # ValueError covers bad JSON
                self._send(400, {"error": {"message": str(e), "type": "invalid_request_error"}})
            except Exception as e:                         # a real fault: say so, and log it
                log.error("embedding failed:\n%s", traceback.format_exc())
                self._send(500, {"error": {"message": f"{type(e).__name__}: {e}", "type": "server_error"}})

        def log_message(self, fmt, *args):
            log.info("%s %s", self.address_string(), fmt % args)

    return Handler


def _env_int(name: str, default: int | None) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        raise SystemExit(f"{name}={raw!r} is not an integer")


def load_service() -> EmbedService:
    # sentence-transformers honours SENTENCE_TRANSFORMERS_HOME over HF_HOME, and
    # this machine sets it user-wide to a folder that does not hold the weights:
    # left alone it starts a second 1.5 GB download. Drop it so HF_HUB_CACHE rules.
    os.environ.pop("SENTENCE_TRANSFORMERS_HOME", None)

    import sentence_transformers
    import torch
    import transformers
    from sentence_transformers import SentenceTransformer

    model_id = os.environ.get("EMBED_MODEL", DEFAULT_MODEL)
    device = os.environ.get("EMBED_DEVICE", "cuda")
    max_seq_length = _env_int("EMBED_MAX_SEQ_LENGTH", 512)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"EMBED_DEVICE={device} but torch cannot see a CUDA device "
                         f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}); "
                         f"set EMBED_DEVICE=cpu to run on the CPU on purpose")
    log.info("loading %s on %s, fp32, text-only", model_id, device)
    model = SentenceTransformer(
        model_id, device=device, model_kwargs={"torch_dtype": torch.float32},
        config_kwargs={"vision_config": None, "audio_config": None})
    model.max_seq_length = max_seq_length
    return EmbedService(model, model_id, device, max_seq_length,
                        _env_int("EMBED_TRUNCATE_DIM", None), _env_int("EMBED_BATCH_SIZE", 16),
                        versions={"torch": torch.__version__,
                                  "sentence_transformers": sentence_transformers.__version__,
                                  "transformers": transformers.__version__})


def main() -> None:
    service = load_service()
    host, port = os.environ.get("EMBED_HOST", "127.0.0.1"), _env_int("EMBED_PORT", 8104)
    log.info("ready: %s", json.dumps(service.health()))
    ThreadingHTTPServer((host, port), make_handler(service)).serve_forever()


if __name__ == "__main__":
    main()
