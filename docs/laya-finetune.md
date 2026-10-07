# Fine-tuning Laya on your vault

[Laya](https://github.com/NandhaKishorM/laya) is a ~421M-parameter encoder that answers a typed
yes/no question about a text in one forward pass and returns a probability, `P(true)`. This
package fine-tunes it on passages from your own vault so it can be tried in two roles, both
**experimental and off by default**:

- a **reranker** (`rerank: laya`): reorders the candidate pool after retrieval;
- a **relevance gate**: after reranking, drops chunks whose `P(relevant)` is below a threshold
  before generation, and abstains ("not in your notes") without calling the LLM when nothing passes.

It is a time-boxed experiment with a kill rule, not a feature. Section 2 says why, and what to
check before spending any GPU time. Training happens off the machine, on Kaggle; this repository
owns the export tool, the notebook, the inference adapter and this manual.

## 1. Privacy first

!!! warning "Uploading the export sends your vault text to a third party"
    `data/laya/pairs.jsonl` is **verbatim passages from your vault**: lecture notes, textbook
    pages, homework. Putting it on Kaggle (or Colab) hands them to a company that is not you.
    **No code in this repository uploads anything.** The export tool is local and read-only on
    the store; uploading is a manual step, and only you take it.

What follows is the whole privacy contract:

1. **The Kaggle dataset must be PRIVATE.** The `kaggle datasets create` command creates private
   datasets by default (its `-u` / `--public` flag is what makes one public: never pass it). If
   you use the web UI instead, check the visibility switch before you save.
2. **Private is not invisible.** Kaggle's dataset documentation describes private as visible to
   you and your collaborators, and to Kaggle for purposes consistent with its Privacy Policy
   (read that on the live page before relying on it; it was relayed second-hand when this was
   written).
3. **Keep the notebook private too.** Its cells print passage excerpts next to generated
   questions, and its output holds a checkpoint trained on your text.
4. **The notebook needs Internet On** (for `pip` and the model downloads), and a notebook with
   Internet On can send its inputs anywhere. The shipped notebook contains no upload code: a test
   (`tests/test_laya_notebook.py`) fails if any code cell calls an upload API, reads a secret, or
   starts a program other than `nvidia-smi`, `pip` and `torchrun`. If you edit it, you own that
   guarantee. Never put a Hugging Face **write** token (or any secret) in Kaggle Secrets for this
   notebook, and do not copy upstream's last cell: it publishes the model to a **public**
   Hugging Face repository.
5. **Delete everything after training** (section 6): the dataset, the notebook and its saved
   versions, and the local export.
6. **The checkpoint is derived from vault text.** Keep `models/` and `data/laya/` out of any
   public repository (both are gitignored).

The export also guards the *other* way a model can be tainted: it leaves out every chunk of every
gold file named in `eval/sets/` and `eval/smoke/`, so nothing you will be scored on is trained on.

## 2. Before you spend GPU hours

!!! info "The verdict: spike-only"
    Zero-shot Laya scored **below "no reranker"** in three independent third-party benchmarks,
    and a student distilled from `bge-reranker-v2-m3` is **capped by that teacher**, which the
    RAG can already run locally (`retrieval.cross_encoder_model`, or `rerank_mode: http`). No
    published result exists for a Laya fine-tuned as a reranker. Upstream itself says to treat
    Laya "as a fast base to specialise, not as a zero-shot decision engine". So build and run this
    only as a time-boxed experiment, and stop at the first kill below.

As a **reranker** it has the weakest case: there is no evidence it can beat its own teacher. As a
**relevance gate** it is the only plausible use, and there it must beat a free baseline: a cutoff on
the active reranker's own score (`scorer: rerank_score`, section 5).

### The evidence

All three benchmarks are third-party, zero-shot, and on different Laya versions; none of them
tested a fine-tune. Read from their repositories on 2026-10-06.

| Source | Setup | Zero-shot Laya |
|---|---|---|
| [`dchristopoulos/system1-rerank-bench`](https://github.com/dchristopoulos/system1-rerank-bench) (README, "measured 2026-10-05") | BEIR SciFact and NFCorpus, 100 test queries each, hybrid top-50, nDCG@10, paired bootstrap | 0.472 / 0.293 against 0.702 / 0.373 with no reranker: paired difference -0.230 [-0.307, -0.155] and -0.081 [-0.123, -0.043]. "... ranked far below no reranker under all nine wordings tried." `bge-reranker-v2-m3` there: 0.720 / 0.380, within noise of no reranker |
| [`nadeem4/ai-experiments`](https://github.com/nadeem4/ai-experiments/blob/main/rerank/RESULTS.md) (`rerank/RESULTS.md`, run 2026-09-26) | BEIR NFCorpus, 323 queries, 20 hybrid candidates, T4 | 0.3065 (-0.0241 [-0.0390, -0.0093]) and 0.3028 (-0.0278 [-0.0436, -0.0120]) against a hybrid floor of 0.3306: "worse than doing nothing at all". Its own caveat: "This is zero-shot by design" |
| [`anessbelbati/jev-rerank-bench`](https://github.com/anessbelbati/jev-rerank-bench) (laya 0.3.3, runs 2026-09-19 to 25) | 8 English datasets, BM25 top-30, one passage per prompt | 0.471 (yes/no) and 0.483 (4-level rubric) against a BM25 floor of 0.486 ("within noise"); `bge-reranker-v2-m3` 0.588 |

Upstream's own relevance number (`BENCHMARKS.md`: 0.625 plain accuracy on a balanced MS MARCO
yes/no task, chance 0.5, on data in the checkpoint's training mix) is not a ranking result.

### The cheapest spike

Ordered by cost. Stop at the first kill.

0. **Free, no GPU: does the teacher beat MiniLM at all?** On the dev split, compare
   `BAAI/bge-reranker-v2-m3` with the MiniLM cross-encoder the RAG ships, with a paired CI (the
   "bge-v2-m3 over http" branch of phase 5 in section 13 of the eval design spec,
   `docs/superpowers/specs/2026-10-04-eval-system-design.md`; later "spec section N" means that
   document). If it does not win, a student
   trained to imitate it has no reason to either: **stop before running anything else.** (The
   RAG's own config notes that on this corpus bge-reranker-v2-m3 "reordered ~half the top-5
   without a measured quality win", at about 22 times MiniLM's CPU cost.)
1. **A mini-run on one Kaggle T4** (about 1 GPU-hour, budget 2): export with
   `--n-passages 500 --negatives 7`, run the notebook with `NPROC = 1` and `epochs=2` in its
   `TRAIN_CFG`. On the held-out passages, kill if Laya's nDCG@10 is not above MiniLM's, or if the
   run collapsed (cell 8 reports both, and the calibration reliability table). Promote to a full
   run only on a pass.
   *Caution:* this run is small enough (about 100 optimizer updates, against roughly 70 in the
   upstream report, by arithmetic from their figures) to sit near the regime where upstream issue
   963 saw collapse. A collapsed spike says "too little data", not "no signal": rerun once with
   more epochs or passages before reading it as a verdict on the idea.
2. **Only if both pass**: the full export, the full run, and the evaluation in section 5.

## 3. Environment matrix

| Where | Role | Notes |
|---|---|---|
| **Kaggle, "GPU T4 x2"** | **Training (recommended)** | Two T4s (about 15 GB each) under DDP. Sessions last at most 12 hours (Kaggle's own CLI source). The weekly GPU quota is per user and reported by the server: check it with `kaggle quota` (columns `resource`, `used`, `remaining`, `total`, `refreshAt`). The commonly cited figure is **about 30 GPU-hours a week**, from secondary sources and approximate; Kaggle has reportedly varied it. Whether a dual-T4 hour counts as one quota hour or two is **unverified**: run a 10-minute dual-T4 session and diff `kaggle quota` before and after. The P100 is retired (asking for it gives a T4). |
| **Colab, free T4** | Fallback | One GPU: set `NPROC = 1` in the notebook (cell 7 then accumulates more steps to keep the effective batch at 64; about twice the wall-clock) and change its three path constants. Upstream's single-process `laya-train` CLI (new in 0.3.28) reads the same `{state, questions, gold}` rows that cell 6 writes to `rows_train.jsonl` in the scratch directory. Colab is also a third party: section 1 applies unchanged. |
| **Local GTX 1070 Ti** | **Inference only**, in a **separate venv** | Pascal. See below. |
| **Local CPU** | Default inference | The RAG venv after `pip install laya==0.3.28`. See below. |

### Local CPU (the default)

`pip install laya==0.3.28` into the RAG venv adds only a 360 KB pure-Python wheel: the
repository's `requirements.txt` already resolves to a `torch`, `transformers` and `safetensors`
that satisfy it. Pin the version: the project is weeks old and ships weekly. The fine-tuned
directory is about 0.84 GB on disk; loaded in fp32 it needs roughly 2 to 3 GB of RAM.

Do not expect it to be fast. Upstream measured a p50 of 580 ms for **one short question** on a
4-core server CPU, and "batching saves little on CPU". Passage-length inputs are several times
slower: extrapolating from upstream's figures gives about 3 s per passage and about 30 s for a
top-10 pool on 4 cores (an estimate, not a measurement), and no reason to expect a **CPU speed
advantage over the teacher**, whose per-token cost is similar. Measure it with the bench before
believing any of that (section 5).

### Local GPU inference on a Pascal card

The GTX 1070 Ti is compute capability 6.1. Use it for **inference only**, from a **second venv**:
the RAG venv's `torch` is a CPU build (per `config.yaml`), and swapping a GPU build into it would
change what the rest of the system runs on.

```powershell
# a second environment: a CUDA torch that still carries Pascal kernels, then the project's requirements
py -3.11 -m venv <runtime-dir>\.venv-laya-gpu
<runtime-dir>\.venv-laya-gpu\Scripts\pip install "torch==2.14.*" --index-url https://download.pytorch.org/whl/cu126
<runtime-dir>\.venv-laya-gpu\Scripts\pip install -r requirements.txt
<runtime-dir>\.venv-laya-gpu\Scripts\pip install laya==0.3.28
```

Why 2.14, and what is **not** verified:

- PyTorch's own `RELEASE.md` (read 2026-10-06) says 2.14 ships CUDA 12.6 wheels for Maxwell (5.0),
  Pascal (6.0), Volta, Turing, Ampere and Hopper, while 2.15 (release scheduled 2026-10-28) is
  CUDA 13.2 only, and an open RFC (`pytorch/pytorch#190385`) proposes dropping CUDA 12.6 and with
  it Maxwell, Pascal and Volta. So 2.14 is the last release with Pascal kernels.
- **Not verified:** that a `torch-2.14.x+cu126` wheel exists for Windows / CPython 3.11 (the index
  was unreachable when this was written), and that the sm_60 binary runs on a 6.1 card. CUDA's
  binary-compatibility rules say it should. If `pip` cannot find the wheel, stop; do not
  substitute another build. Check the install with a smoke test, and trust that over any comment:

  ```powershell
  <runtime-dir>\.venv-laya-gpu\Scripts\python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list()); x = torch.randn(512, 512, device='cuda'); print(float((x @ x).sum()))"
  ```

  The arch list should include `sm_60` and the matmul should run; the smoke test, not this
  reading of `RELEASE.md`, is the arbiter.
- `TORCH_DISABLE_NATIVE_JIT=1` (from upstream's `docs/docker.md`): PyTorch 2.14 otherwise replaces
  some CUDA ops with Triton kernels compiled on first use, which needs a C compiler that a Windows
  box does not have, and `predict` can fail with `Failed to find C compiler`. Set it in the shell that
  starts the service (`$env:TORCH_DISABLE_NATIVE_JIT = "1"`), not machine-wide.
- The adapter turns off Laya's forced fp16 autocast on Pascal: fp16 runs at about 1/64 of fp32 speed
  on GP104 (a figure from the design plan, not verified), and fp32 is also the more accurate dtype
  (upstream issue 443). In fp32 the model needs about 1.7 GB for weights and roughly 3 GB at a batch
  of 16 passages (an estimate).
- Set `retrieval.laya.device` to the **headless** card (find its index with `nvidia-smi`). The
  adapter loads Laya inside the RAG process, so GPU inference means running the pipeline itself from
  this venv. `rag.bat` always uses the main venv, so start the service by hand with that venv's
  `python.exe -m uvicorn serve_api:app --host 127.0.0.1 --port 8051`, after copying the cache
  variables `rag.bat` sets (`HF_HOME`, `HF_HUB_CACHE`, `SENTENCE_TRANSFORMERS_HOME`) into the shell;
  otherwise every model downloads again into the default caches.

### Why not train locally

- **Memory.** A full fine-tune in fp32 holds weights, gradients and two Adam moments: 16 bytes per
  parameter, so 421M parameters is about 6.3 GiB before a single activation. With activations that
  comes to roughly **7 to 7.5 GiB per GPU** (the design plan's estimate): razor-thin on the headless
  8 GiB card, and impossible on the display card, which the desktop already uses.
- **fp16 does not rescue it.** It would halve the memory, but runs at about 1/64 of the fp32 rate
  on GP104 (a figure from the design plan; not verified).
- **Two cards do not pool memory.** DDP gives each card a full replica of the model and splits
  only the data.
- **NCCL, the backend upstream's DDP script requires, is Linux-only.**

## 4. Step by step

Run `python ...` commands with the project venv's interpreter (`<runtime-dir>\.venv\Scripts\python.exe`),
from the repository root. `rag.bat` only forwards to `main.py`, so tool scripts need the interpreter
called directly.

### 4.1 Export the pairs (local, no LLM, read-only on the store)

```powershell
python tools/laya/export_pairs.py
```

Flags: `--n-passages` (how many passages to sample), `--negatives` (hard negatives per passage),
`--seed` (the fixed sampling seed) and `--out` (the output path; by default `data/laya/pairs.jsonl`).
By default it samples 4,000 passages with 15 negatives each; those are **defaults** (`--help` is
authoritative), and the tool prints the real counts.

It samples prose and code passages (at least 300 characters, no table-of-contents chunks),
**excludes every chunk of every gold file in `eval/sets/` and `eval/smoke/`** and prints how many it
excluded, takes each passage's nearest neighbours **from other files** (by its stored embedding) as
hard negatives, and writes one row per passage:
`{pid, passage, negatives: [{id, text}], meta: {source_file, file_type}}`. For the mini-run of
section 2, add `--n-passages 500 --negatives 7`.

### 4.2 Create a PRIVATE Kaggle dataset

Install the `kaggle` CLI (it is not part of the RAG's requirements) and sign in with your own account
(`kaggle auth login`, or a token in the `KAGGLE_API_TOKEN` environment variable; never write a token
into a file in this repository).

```powershell
mkdir data\laya\upload
copy data\laya\pairs.jsonl data\laya\upload\
kaggle datasets init -p data\laya\upload      # writes dataset-metadata.json
# edit it: "title" (6 to 50 characters), "id": "<your-username>/<slug>" (slug 3 to 50 characters),
# and exactly one entry in "licenses" ("unknown" is allowed)
kaggle datasets create -p data\laya\upload    # PRIVATE by default. Do not add -u or --public
kaggle datasets status <your-username>/<slug> # wait until it reports ready
```

The folder holds exactly one data file because the CLI uploads everything in it.

### 4.3 Run the notebook

Create a Kaggle notebook, import `tools/laya/noetrix_laya_kaggle.ipynb`, and in its options set
**Accelerator = GPU T4 x2**, **Internet = On**, visibility **Private**, then add your private dataset
as an input. Run the cells in order; each is explained by the markdown above it. The slow ones are
cell 4 (doc2query), cell 5 (teacher labels) and cell 7 (training), and the log of cell 7 prints an
ETA: stop and shrink the export if it is not what you budgeted. Two guards raise on purpose, so a bad
run stops early: cell 5 if the generated questions are too generic, cell 8 if the student collapsed.

When it finishes, read cell 8's output and `metrics.json`. The file holds `ndcg_at_10` (Laya, MiniLM
and the teacher, with the teacher's probability also used as the gain, and paired differences with
95% intervals), `calibration` (ECE and reliability bins of `P(true)` against the source passage and
against the teacher's label, the teacher's own ECE, and upstream's answer-confidence ECE for
reference), `collapse`, `latency` (GPU timings), `laya_truncated_share` (passages cut at 512 tokens),
`batch_vs_single_max_abs_diff`, and `provenance` (seed, library versions, model commit hashes, the
recipe and the counts). The notebook adapts cells from `NandhaKishorM/laya` (Apache-2.0); each
adapted cell says what it changed.

If training dies part-way, the last epoch's checkpoint is in `/tmp/noetrix-laya/checkpoint_latest`
for as long as the session lives. It loads, but its temperature is the base checkpoint's, not a fitted
one.

### 4.4 Download, unzip, install, switch on

```powershell
# the notebook's Output tab, or (the zip is the only file you want):
kaggle kernels output <your-username>/<notebook-slug> --file-pattern "noetrix-laya\.zip" -p downloads
Expand-Archive downloads\noetrix-laya.zip -DestinationPath models\laya-noetrix
pip install laya==0.3.28        # into the RAG venv
```

The archive's root **is** the checkpoint (`model.safetensors`, `encoder/`, `tokenizer/`,
`rl_agent_config.json`, plus `metrics.json`), so `models/laya-noetrix` is what
`retrieval.laya.model_dir` expects by default.

Then turn the experiment on: in the management console (`rag console`, :8052) open **Settings >
Experimental features** and switch on `laya_rerank`, which sets `retrieval.laya.enabled` (its other
defaults are `model_dir: models/laya-noetrix`, `device: cpu` and `batch_size: 16`). The console writes
`config.yaml` immediately; **restart :8051** (`rag serve`) so the service picks it up. A first check:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8051/search -Method Post -ContentType "application/json" `
  -Body '{"q": "what is a stationary process", "rerank": "laya", "top_k": 5}'
```

While the flag is off, or when `model_dir` holds no checkpoint, `rerank: laya` fails with a readable
`Reranking failed: ...` that names the flag or the path. It never falls back to another reranker.

## 5. Testing and graduation

### 5.1 Reranker: compare on the dev split

```bash
python main.py bench run --configs laya-rerank,default-k10 --split dev
```

`laya-rerank` (`{top_k: 10, rerank: laya}` in `eval/configs.yaml`) and `default-k10` (the shipped
pipeline, with the MiniLM cross-encoder) are scored on the same questions, so the comparison is
paired. Do the same for `bge-reranker-v2-m3`, the second reference, with that reranker configured;
the runner runs reranker models one at a time. (`bench` is the v2 harness specified in the eval
design spec; [Evaluation](evaluation.md) documents the older `main.py eval` and its golden set.)

The **`P-rerank-pool` probe** (spec section 5) isolates ranking skill: the pool is fixed at the gold
chunks plus 30 hard negatives, so the retriever's recall cannot hide or flatter the reranker. That
makes it the cleanest place to put Laya next to MiniLM and bge. The spec's CLI (section 10) runs it
with `--probes`; that belongs to the analysis phase, so check `python main.py bench run --help` for
the flag before relying on it.

### 5.2 The graduation rule

Spec section 11: *an experimental component (graph mode, Jev, Laya, a new embedding model) graduates
iff it beats the current default on dev with a paired CI excluding zero, and then holds on test.* For
Laya this package makes that concrete: it graduates as a reranker only if **all** of these hold:

1. **It beats MiniLM and `bge-reranker-v2-m3` on the dev split**, each with a paired 95% CI of the
   difference that excludes zero. MiniLM is the shipped default; `bge-reranker-v2-m3` is the teacher,
   and a student that cannot beat its own teacher adds nothing the RAG cannot already run.
2. **It holds on the test split.** Section 11 does not define "holds" numerically. A defensible
   reading, borrowed from the bar it sets for components that stay on by default: a test-split
   delta of at least zero against each reference with the paired CI not entirely below zero, and no
   suite with at least 30 questions regressing by more than 5 points. Open the test split **once**,
   after the dev decisions are final: `--split test --unseal-test`, and every opening is appended to
   `eval/test_ledger.jsonl`.
3. **Its p95 CPU rerank latency is no worse than `bge-reranker-v2-m3`'s.** This is the condition
   most likely to bind: no CPU speed advantage over the teacher is expected (section 3). The
   notebook's latency numbers are GPU timings and only a hint; measure CPU latency with the bench.

### 5.3 The second role: the relevance gate

The gate is a decision step **after** reranking. It drops chunks whose `P(relevant to the original
query)` is below a threshold before parent/neighbour expansion and generation, and when nothing is
left, `/query` returns the generator's existing no-docs answer (confidence LOW, no citations)
**without calling the LLM**; the echo says `abstained: true`. Both roles are independent toggles,
default off. The keys live under `retrieval.relevance_gate`:

```yaml
retrieval:
  relevance_gate:
    enabled: false
    scorer: rerank_score   # rerank_score | laya
    threshold: null        # required when enabled: true (null + enabled is a configuration error)
```

- `scorer: rerank_score` cuts on the **active reranker's own score** (the threshold is in that
  model's units). It is free, and it is the **baseline every learned gate must beat**.
- `retrieval.relevance_gate.scorer: laya` cuts on this package's `P(true)`, a number in [0, 1]. It
  needs `retrieval.laya.enabled` too: with Laya disabled it fails readably instead of quietly
  switching to `rerank_score`.
- Per call: `gate` (true/false) and `gate_threshold` (a number) on `/search`, `/query` and
  `/compare`; the retrieval echo has a `gate` key with `{scorer, threshold, kept, dropped, abstained}`.
- Bench configs `gate-rerank-score` and `gate-laya`. **The threshold is chosen on dev by a sweep**
  that reports, per threshold, the gold-chunk loss on answerable questions (the fraction of retrieved
  gold chunks the gate removed), nugget recall, and abstention accuracy on `unanswerable`. It is then
  **reported once on test** (spec section 11).

!!! warning "Why calibration (cell 8) matters for the gate"
    A threshold only means something if `P(true)` does: "keep chunks above 0.7" must correspond to
    a stable hit rate, or the cutoff drifts with the topic. Do not borrow anyone's calibration
    figure; measure your own. Upstream says its shipped checkpoints are "over-confident as shipped",
    and a third-party benchmark (jev-rerank-bench, laya 0.3.3) measured an ECE for the yes/no
    `P(true)` against graded relevance of 0.280 and 0.313 (8- and 14-dataset averages) against
    0.097 and 0.079 for Jev. Calibration is task-dependent, though: on AG News another third party
    measured Laya's raw ECE at 0.0275, better than Jev's. So rely on the **reliability table and ECE
    in `metrics.json`**. The student's calibration is bounded by the teacher's, and its labels are the
    teacher's, not yours: only the human-reviewed dev questions can show that Laya adds anything over
    the free baseline.

Practical points when you pick a threshold: cut on `P(true)`, never on the `confidence` Laya also
returns (that is `max(p, 1-p)`, so a confident "no" scores high); `P(true)` is rounded to 4 decimals,
so a threshold finer than that is meaningless and ties near 0 and 1 are possible.

## 6. Cleanup

When you have the zip and have read `metrics.json`:

```powershell
kaggle datasets delete <your-username>/<slug> -y
kaggle kernels delete <your-username>/<notebook-slug> -y   # the notebook, its saved versions and their output
```

- The notebook's saved versions keep their output, which includes the checkpoint trained on your
  text, so delete the notebook as well as the dataset. Look at your Kaggle profile afterwards to
  confirm nothing is left.
- Delete the local export: `data/laya/` (the pairs file and the `upload` folder), and the downloaded
  zip. Keep `models/laya-noetrix`: it is the product, it is gitignored, and it is derived from vault
  text, so it never goes in a public repository.
- If you made a Kaggle token only for this, revoke it.
- Deleting removes your copy from your account. It cannot take back what a third party has already
  read, which is why section 1 comes first.
