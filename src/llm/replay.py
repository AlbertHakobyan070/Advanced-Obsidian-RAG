"""
replay.py — an LLM whose completions were written ahead of time.

The eval measures the pipeline with a FIXED, trustworthy LLM: the completions
are written outside the pipeline (by Claude agents, by hand, by any model) and
replayed from a JSONL store, so a run is repeatable and never depends on a
provider chain that may fall back to another model mid-run.

    record pass   every prompt the pipeline sends is appended to
                  <store>.pending.jsonl and the call raises ReplayMiss
    fill          a completion is written for each pending prompt into
                  <store> (one JSON object per line: key, model, completion)
    replay pass   every call is answered from <store>; a miss raises

The prompts are rendered by the pipeline's own code (HyDE, the generator, the
decomposer), so the replayed run is the real system with that LLM. The model
label comes from the store and every completion in it must carry the same
one: HyDE's cache key includes the model, so a store's drafts can never be
served to a run of another model, nor mixed with a provider's drafts.

    llm = ReplayLLM("eval/llm/sonnet.jsonl", mode="replay")
    llm.complete(system="...", user="...")      # -> LLMResponse, or ReplayMiss
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.llm.llm_client import LLMResponse

MODES = ("record", "replay")
UNFILLED = "replay:unfilled"        # the label of a store with no completion yet


class ReplayMiss(RuntimeError):
    """The store has no completion for this prompt."""


def prompt_key(system: str, user: str) -> str:
    return hashlib.sha256(json.dumps([system, user], ensure_ascii=False).encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{n}: not JSON ({e})") from None
    return rows


class ReplayLLM:
    provider = "replay"

    def __init__(self, store: str | Path, mode: str = "replay"):
        if mode not in MODES:
            raise ValueError(f"replay mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self.store = Path(store)
        self.pending = self.store.with_name(self.store.stem + ".pending.jsonl")
        self._done: dict[str, str] = {}
        models = set()
        for n, r in enumerate(_read_jsonl(self.store), 1):
            missing = [k for k in ("key", "model", "completion") if not isinstance(r.get(k), str)]
            if missing:
                raise ValueError(f"{self.store}: completion {n} lacks {', '.join(missing)}")
            self._done[r["key"]] = r["completion"]
            models.add(r["model"])
        if len(models) > 1:
            raise ValueError(f"{self.store} mixes models {sorted(models)}: one store per model")
        self.model = models.pop() if models else UNFILLED
        self._queued = {r.get("key") for r in _read_jsonl(self.pending)}

    def complete(self, system: str, user: str, temperature: float | None = None,
                 max_tokens: int | None = None) -> LLMResponse:
        key = prompt_key(system, user)
        if key in self._done:
            return LLMResponse(text=self._done[key], model=self.model, provider=self.provider)
        if self.mode == "record" and key not in self._queued:
            rec = {"key": key, "system_fp": hashlib.sha256(system.encode("utf-8")).hexdigest()[:12],
                   "system": system, "user": user, "temperature": temperature,
                   "max_tokens": max_tokens}
            self.pending.parent.mkdir(parents=True, exist_ok=True)
            with open(self.pending, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._queued.add(key)
        raise ReplayMiss(f"no completion for prompt {key[:12]} in {self.store}"
                         + (" (recorded to pending)" if self.mode == "record" else ""))
