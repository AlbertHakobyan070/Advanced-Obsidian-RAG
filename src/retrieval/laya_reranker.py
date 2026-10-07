"""
laya_reranker.py — P(relevant) per passage from a fine-tuned Laya checkpoint.

EXPERIMENTAL. Laya is a ~420M-parameter ModernBERT encoder with a small
decision head: given a STATE (here, one passage) and a typed QUESTION it
answers in one forward pass. The question asked here is a yes/no (`noul`) one
— "Does the passage answer: <query>?" — and the number kept is P(true). It
backs two optional features, both off by default: the `laya` rerank mode
(reranker.py) and the relevance gate's `laya` scorer. They share ONE
LayaScorer, so the checkpoint is loaded once.

    scorer = LayaScorer("models/laya-noetrix")
    scorer.score("what is ARIMA?", [passage, passage])      # -> [0.93, 0.04]

Four things are handled here because each fails silently otherwise:

  lazy         Nothing is imported or loaded until the first score(). `laya`
               is an optional package (pip install laya==0.3.28, deliberately
               NOT in requirements.txt) and the checkpoint is ~2 GB of RAM, so
               building a scorer is free and a disabled feature costs nothing
               at startup.
  checkpoint   Checked HERE, before laya.load, by absolute path. laya.load
               takes a missing RELATIVE path (the default model_dir is one)
               for a Hugging Face Hub repo id and goes looking on the network,
               and a directory without tokenizer/ or encoder/ gets the same
               treatment: neither is ever reported as "no checkpoint here".
  fp16         laya forces fp16 autocast on CUDA cards below compute
               capability 8 and offers no flag to turn it off; see
               _disable_forced_fp16.
  one question The question's wording is part of the model's input, so it
               lives in ONE place (NOUL_QUESTION) and the fine-tune's training
               rows (tools/laya/noetrix_laya_kaggle.ipynb, its RLCD-rows cell)
               must use the same text: copy it, never retype it.

Failures raise (LayaScorerError, or whatever laya itself raised): a scorer
that returned something for every passage regardless would turn a broken
install into a ranking that merely looks like one.
"""
from __future__ import annotations

import copy
import math
import os
import threading
import time

from src.utils.logger import get_logger

log = get_logger(__name__)


# The laya release this adapter was written and checked against. The package
# ships a release every few days and its API moves with them, so the install
# hint pins it.
LAYA_VERSION = "0.3.28"

# The ONE question every passage is judged against; the query fills {query}.
# `criteria` is not decoration: upstream documents that a `noul` question with
# none answers "no" whatever the state is, so it carries the decision.
QUESTION_ID = "rel"
NOUL_QUESTION = {
    "type": "noul",
    "instructions": "Does the passage answer: {query}?",
    "criteria": {"false": "Irrelevant.", "true": "Answers the query."},
}

# What a checkpoint directory holds (laya/agent.py's local layout):
# (name, is_dir).
_CHECKPOINT_PARTS = (
    ("rl_agent_config.json", False),
    ("model.safetensors", False),
    ("tokenizer", True),
    ("encoder", True),
)

_CHECKPOINT_HELP = ("(retrieval.laya.model_dir; docs/laya-finetune.md says "
                    "how to get a checkpoint)")

# laya forces fp16 autocast on every CUDA card whose compute capability major
# is below this (laya/agent.py draws the same line). Turing and Volta (7.x)
# have fp16 tensor cores, where the forced autocast helps; cutting at 8 turns
# it off there too, which is slower but always correct (fp32). Cut at 7 to
# keep it on them.
_FORCED_FP16_BELOW = 8


class LayaScorerError(RuntimeError):
    """The Laya scorer cannot run: the package or the checkpoint is missing,
    or laya returned something a ranking cannot be built on."""


def laya_questions(query: str) -> dict:
    """The `questions` argument of Agent.predict_batch: NOUL_QUESTION with the
    query filled in. str.format does not re-read the query, so braces in it
    are safe."""
    question = copy.deepcopy(NOUL_QUESTION)
    question["instructions"] = question["instructions"].format(query=query)
    return {QUESTION_ID: question}


class LayaScorer:
    """P(true) per passage from a Laya checkpoint. Safe to call from many
    threads."""

    def __init__(self, model_dir, device: str = "cpu", batch_size: int = 16):
        # Absolute from the first moment: see the module docstring on
        # relative paths.
        self.model_dir = os.path.abspath(
            os.path.expanduser(os.fspath(model_dir)))
        # Explicit on purpose. Left blank, laya picks a device itself (CUDA if
        # it sees one, then MPS, then XPU), and a scorer that quietly lands on
        # the wrong card is the kind of thing nobody finds until the latency
        # numbers.
        if not (isinstance(device, str) and device.strip()):
            raise ValueError(
                "retrieval.laya.device must name a device such as 'cpu' or "
                f"'cuda:0', got {device!r}: left blank, laya would pick one "
                "itself")
        if (isinstance(batch_size, bool) or not isinstance(batch_size, int)
                or batch_size < 1):
            raise ValueError(
                "retrieval.laya.batch_size must be a positive integer, got "
                f"{batch_size!r}")
        self.device = device.strip()
        self.batch_size = batch_size
        self._agent = None
        # :8051 runs requests on a thread pool; two first calls must not each
        # load ~2 GB of model.
        self._lock = threading.Lock()

    def score(self, query: str, texts: list[str]) -> list[float]:
        """P(true) that each passage answers `query`: one float per text, in
        the order given. [] for no texts, without loading anything. The
        passages are scored `batch_size` at a time.

        P(true) comes back rounded to 4 places, so ties near 0 and 1 are
        normal; a stable sort keeps the retrieval order among them. A passage
        longer than the checkpoint's max_len is cut on the right without a
        word from laya, so the count is logged."""
        texts = list(texts)
        if not texts:
            return []
        agent = self._get_agent()
        questions = laya_questions(query)
        scores: list[float] = []
        truncated = 0
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start:start + self.batch_size]
            results = agent.predict_batch(chunk, questions)
            if len(results) != len(chunk):
                # zip() would pin scores on the wrong passages.
                raise LayaScorerError(
                    f"laya returned {len(results)} result(s) for "
                    f"{len(chunk)} passage(s): refusing to rank on a "
                    "misaligned list")
            for result in results:
                p = float(result["answers"][QUESTION_ID]["noul"])
                if not math.isfinite(p):
                    # NaN compares False against everything: it would sort
                    # anywhere and still read as a ranking.
                    raise LayaScorerError(
                        f"laya returned a non-finite P(true) ({p}): "
                        "refusing to rank on it")
                scores.append(p)
                if (result.get("usage") or {}).get("truncated"):
                    truncated += 1
        if truncated:
            log.info("laya: %d of %d passage(s) truncated to the "
                     "checkpoint's max_len (the tail was not scored)",
                     truncated, len(texts))
        return scores

    # ---- loading ---------------------------------------------------------

    def _get_agent(self):
        with self._lock:
            if self._agent is None:
                self._agent = self._load()
            return self._agent

    def _checkpoint_problem(self) -> str | None:
        """What is wrong with model_dir, or None when it holds a whole
        checkpoint."""
        if not os.path.isdir(self.model_dir):
            return f"no Laya checkpoint: {self.model_dir} does not exist"
        missing = [name + ("/" if is_dir else "")
                   for name, is_dir in _CHECKPOINT_PARTS
                   if not (os.path.isdir if is_dir else os.path.isfile)(
                       os.path.join(self.model_dir, name))]
        if missing:
            return (f"{self.model_dir} is not a complete Laya checkpoint: "
                    f"missing {', '.join(missing)}")
        return None

    def _load(self):
        problem = self._checkpoint_problem()
        if problem:
            raise LayaScorerError(f"{problem} {_CHECKPOINT_HELP}")
        try:
            import laya
        except ImportError as e:
            raise LayaScorerError(
                f"the Laya reranker needs the laya package "
                f"({type(e).__name__}: {e}): pip install laya=="
                f"{LAYA_VERSION} (experimental and optional, so it is not in "
                "requirements.txt)") from e
        log.info("Loading Laya checkpoint: %s (device=%s)",
                 self.model_dir, self.device)
        t0 = time.time()
        # A problem inside laya itself (a missing torch, a corrupt file) is
        # raised as it is: only `import laya` above means "the package is not
        # installed".
        agent = laya.load(self.model_dir, device=self.device)
        self._disable_forced_fp16(agent)
        log.info("Laya ready in %.1fs", time.time() - t0)
        return agent

    @staticmethod
    def _disable_forced_fp16(agent) -> None:
        """Switch off laya's forced fp16 autocast on CUDA cards below compute
        capability 8.

        laya.load turns autocast on for every such card and has no argument or
        environment variable to stop it (LAYA_CUDA_AMP is only read at 8 and
        above). On GP104-class Pascal cards (6.1) fp16 runs at ~1/64 of the
        fp32 rate, so scoring crawls. The forward pass re-reads `amp_enabled`
        on every call, so setting it after load is enough; `dtype` is set with
        it to keep the agent consistent. A CPU agent is left alone — nothing
        here asks CUDA about it.
        """
        if agent.device.type != "cuda":
            return
        import torch
        major = torch.cuda.get_device_capability(agent.device)[0]
        if major >= _FORCED_FP16_BELOW:
            return
        if not (hasattr(agent, "amp_enabled") and hasattr(agent, "dtype")):
            # Not a documented API. A release that drops these would make the
            # assignments below do nothing while fp16 stayed forced.
            raise LayaScorerError(
                "laya's Agent has no amp_enabled / dtype attribute, so its "
                "forced fp16 autocast cannot be switched off on this compute "
                f"capability {major} card; this adapter is written for "
                f"laya=={LAYA_VERSION}")
        agent.amp_enabled = False
        agent.dtype = torch.float32
        log.info("laya: CUDA compute capability %d is below %d, so its forced "
                 "fp16 autocast is switched off (fp32 instead)",
                 major, _FORCED_FP16_BELOW)
