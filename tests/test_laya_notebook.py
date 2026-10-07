"""The Kaggle notebook for the Laya fine-tune, and the manual that drives it.

The notebook cannot run in CI: it needs two GPUs, Kaggle's Internet, three model
downloads and a private dataset of vault passages. So these tests pin what CAN be
checked offline, and what is expensive or embarrassing to get wrong:

  1. IT IS A VALID NOTEBOOK. Kaggle imports it as nbformat 4: cells numbered 0-9,
     a markdown cell explaining each code cell, no stored outputs (outputs of a
     real run quote vault passages).
  2. IT NEVER UPLOADS. The dataset is vault text and the checkpoint is trained on
     it. The upstream notebook ends with a cell that pushes the model to a PUBLIC
     Hugging Face repo; this one must not, and must not run any command that talks
     to a third party other than pip / model downloads / nvidia-smi / torchrun.
  3. TRAINING AND INFERENCE SEE THE SAME QUESTION. The noul wording is defined
     once and reused by the evaluation cell; the inference adapter in the repo uses
     the identical text, so a drift here silently trains the wrong prompt.
  4. THE SEQUENCE BUDGET IS PINNED. Upstream's DDP script saves max_len 1024 /
     head_max_len 256 into the config while training at 512 / 192.
  5. THE PURE LOGIC RUNS HERE. Query cleaning, the split by passage, the row
     format, nDCG, the reliability bins / ECE (checked against upstream's own
     ece_score), the bootstrap, the collapse tripwire and the checkpoint check are
     plain Python in the notebook; they are pulled out of the notebook's AST and
     executed, so a bug in them is found here and not after a 4 hour GPU run.
  6. THE MANUAL has its sections in the promised order, privacy first, and does
     not quote a calibration figure its sources could not confirm.

Nothing here executes a notebook cell as a whole.
"""
import ast
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tools" / "laya" / "noetrix_laya_kaggle.ipynb"
MANUAL = ROOT / "docs" / "laya-finetune.md"
MKDOCS = ROOT / "mkdocs.yml"

# cell number -> a word its heading must contain (a renumbered or dropped cell shows up here)
EXPECTED_HEADINGS = {
    0: "privacy",
    1: "environment",
    2: "install",
    3: "load",
    4: "doc2query",
    5: "teacher",
    6: "rows",
    7: "fine-tune",
    8: "evaluation",
    9: "zip",
}

# Anything that could move vault text, or a model trained on it, off the machine, plus the secret stores that would
# make an upload possible. Matched against the RAW source of every code cell, comments included, so the cells must
# not even mention them (the markdown cells may: they explain what was deliberately left out).
FORBIDDEN_IN_CODE = [
    r"push_to_hub", r"upload_folder", r"upload_file", r"upload_large_folder", r"HfApi", r"create_repo",
    r"create_commit", r"kaggle\s+datasets\s+(create|version)", r"kaggle\s+kernels\s+push",
    r"import\s+requests", r"from\s+requests", r"requests\.(post|put|patch)", r"urllib", r"httpx", r"\bcurl\b",
    r"\bwget\b", r"\bscp\b", r"\brsync\b", r"\bftp", r"smtplib", r"\bsocket\b", r"os\.system", r"os\.popen",
    r"kaggle_secrets", r"UserSecretsClient", r"notebook_login", r"HF_TOKEN", r"KAGGLE_KEY", r"hf_[A-Za-z0-9]{20,}",
]

# The only external programs a code cell may start (the first element of the argv list handed to run()).
ALLOWED_COMMANDS = {"nvidia-smi", "sys.executable", "torchrun"}


# ----------------------------------------------------------------------------------------------- helpers

def _source(cell):
    s = cell["source"]
    return s if isinstance(s, str) else "".join(s)


@pytest.fixture(scope="module")
def nb():
    assert NOTEBOOK.exists(), f"{NOTEBOOK} does not exist"
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code_cells(nb):
    return [c for c in nb["cells"] if c["cell_type"] == "code"]


def _code_cells_text(nb):
    return [_source(c) for c in _code_cells(nb)]


def _train_script(nb):
    """The text of the constant TRAIN_SCRIPT in the fine-tune cell: the program torchrun actually runs."""
    found = []
    for cell in _code_cells(nb):
        for node in ast.parse(_source(cell)).body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "TRAIN_SCRIPT" for t in node.targets):
                found.append(ast.literal_eval(node.value))
    assert len(found) == 1, "expected exactly one TRAIN_SCRIPT constant (a single string literal)"
    return found[0]


def _sections(nb):
    """{n: {"title": str, "markdown": str, "code": [str, ...]}} - a numbered markdown heading opens section n and owns
    every following code cell up to the next numbered heading."""
    out, current = {}, None
    for cell in nb["cells"]:
        text = _source(cell)
        if cell["cell_type"] == "markdown":
            m = re.match(r"\s*#{1,6}\s*(\d+)\.\s+(.+)", text)
            if m:
                current = int(m.group(1))
                assert current not in out, f"section {current} is numbered twice"
                out[current] = {"title": m.group(2).strip(), "markdown": text, "code": []}
                continue
            if current is not None:
                out[current]["markdown"] += "\n" + text
        elif current is not None:
            out[current]["code"].append(text)
    return out


def _is_pure_constant(node):
    """A top-level `NAME = <literal or re.compile(<literals>)>` - safe to execute without the notebook's globals."""
    if not (isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) and t.id.isupper() for t in node.targets)):
        return False
    try:
        ast.literal_eval(node.value)
        return True
    except (ValueError, TypeError, SyntaxError):
        pass
    v = node.value
    return (isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and v.func.attr == "compile"
            and isinstance(v.func.value, ast.Name) and v.func.value.id == "re"
            and all(isinstance(a, (ast.Constant, ast.Attribute)) for a in v.args))


def _pure(nb, names):
    """Execute the named top-level functions of the notebook (plus its pure constants) in a bare namespace."""
    wanted, found, nodes = set(names), set(), []
    for cell in _code_cells(nb):
        for node in ast.parse(_source(cell)).body:
            if isinstance(node, ast.FunctionDef) and node.name in wanted:
                nodes.append(node)
                found.add(node.name)
            elif _is_pure_constant(node):
                nodes.append(node)
    assert found == wanted, f"not defined at the top level of a code cell: {sorted(wanted - found)}"
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {"re": re, "math": math, "np": np, "json": json, "random": __import__("random"), "os": __import__("os")}
    exec(compile(module, str(NOTEBOOK), "exec"), ns)
    return ns


def _upstream_ece_score(conf, correct, bins=15):
    """laya.common.ece_score as shipped in laya 0.3.28 (Apache-2.0, NandhaKishorM/laya), copied so the notebook's own
    binning can be held against the definition it claims to follow without installing laya."""
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        sel = (conf >= lo if i == 0 else conf > lo) & (conf <= hi)
        if sel.any():
            e += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(e)


# ------------------------------------------------------------------------- 1. it is a valid, empty notebook

def test_notebook_is_valid_nbformat4_json(nb):
    assert nb["nbformat"] == 4
    assert isinstance(nb["nbformat_minor"], int)
    assert nb["metadata"]["kernelspec"]["name"] == "python3"
    assert nb["metadata"]["language_info"]["name"] == "python"
    assert nb["cells"], "no cells"
    ids = []
    for i, cell in enumerate(nb["cells"]):
        assert cell["cell_type"] in ("markdown", "code"), f"cell {i}: {cell['cell_type']!r}"
        assert isinstance(cell["metadata"], dict)
        assert isinstance(cell["source"], (str, list))
        if isinstance(cell["source"], list):
            assert all(isinstance(line, str) for line in cell["source"])
        if nb["nbformat_minor"] >= 5:
            ids.append(cell["id"])
        if cell["cell_type"] == "code":
            assert set(cell) >= {"cell_type", "metadata", "source", "outputs", "execution_count"}
        else:
            assert "outputs" not in cell and "execution_count" not in cell, f"markdown cell {i} carries code fields"
    assert len(ids) == len(set(ids))


def test_notebook_validates_against_the_official_schema(nb):
    nbformat = pytest.importorskip("nbformat")
    nbformat.validate(nbformat.reads(NOTEBOOK.read_text(encoding="utf-8"), as_version=4))


def test_notebook_stores_no_outputs_or_execution_counts(nb):
    """A saved output of a real run quotes the vault (cell 4 prints sample queries next to their passages)."""
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [], f"cell {i} has stored outputs"
            assert cell["execution_count"] is None, f"cell {i} has an execution count"


def test_every_code_cell_parses_as_plain_python(nb):
    """No IPython magics: a failed `!cmd` is silently ignored, so shell-outs go through run(), which raises."""
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            ast.parse(_source(cell), filename=f"cell {i}")


def test_embedded_training_script_parses_as_plain_python(nb):
    ast.parse(_train_script(nb), filename="train_ddp.py")


def test_nothing_swallows_an_exception(nb):
    """Upstream's script falls back to a temperature of 1.2 when the fit raises; a fine-tune that quietly ships a
    guessed temperature is worse than one that stops."""
    programs = [(f"code cell {i}", _source(c)) for i, c in enumerate(_code_cells(nb))] + [("TRAIN_SCRIPT", _train_script(nb))]
    for where, text in programs:
        for node in ast.walk(ast.parse(text)):
            assert not isinstance(node, ast.ExceptHandler), f"{where} has an except clause (line {node.lineno})"
            assert not (isinstance(node, ast.keyword) and node.arg == "ignore_errors"), f"{where} passes ignore_errors"


# ----------------------------------------------------------------- 2. cells 0-9, in order, each one explained

def test_cells_zero_to_nine_are_present_in_order(nb):
    sections = _sections(nb)
    assert list(sections) == list(range(10)), f"numbered sections are {list(sections)}"
    for n, word in EXPECTED_HEADINGS.items():
        assert word in sections[n]["title"].lower(), f"cell {n}'s heading is {sections[n]['title']!r}, expected {word!r}"
    assert sections[0]["code"] == [], "cell 0 is the privacy contract: it must not run anything"
    for n in range(1, 10):
        assert sections[n]["code"], f"cell {n} has no code"


def test_every_code_cell_is_preceded_by_a_markdown_cell_that_explains_it(nb):
    cells = nb["cells"]
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            assert i > 0 and cells[i - 1]["cell_type"] == "markdown", f"code cell {i} has no markdown cell before it"
            words = len(_source(cells[i - 1]).split())
            assert words >= 40, f"the markdown before code cell {i} is {words} words: it should say what the cell does and why"


def test_every_numbered_section_has_exactly_one_code_cell_so_the_numbering_is_not_ambiguous(nb):
    for n, sec in _sections(nb).items():
        assert len(sec["code"]) <= 1, f"cell {n} has {len(sec['code'])} code cells"


# ---------------------------------------------------------------------------- 3. nothing leaves the machine

def test_no_code_cell_uploads_anything_or_reads_a_secret(nb):
    for i, cell in enumerate(_code_cells(nb)):
        text = _source(cell)
        for pattern in FORBIDDEN_IN_CODE:
            assert not re.search(pattern, text), f"code cell {i} matches {pattern!r}"


def test_the_only_programs_started_are_nvidia_smi_pip_and_torchrun(nb):
    seen = set()
    for cell in _code_cells(nb):
        for node in ast.walk(ast.parse(_source(cell))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "run":
                argv = node.args[0]
                assert isinstance(argv, ast.List), "run() must be given a literal argv list"
                first = argv.elts[0]
                seen.add(first.value if isinstance(first, ast.Constant) else ast.unparse(first))
    assert seen, "no run() call found"
    assert seen <= ALLOWED_COMMANDS, f"unexpected commands: {sorted(seen - ALLOWED_COMMANDS)}"


def test_run_helper_raises_when_a_command_fails(nb):
    """A bare `!torchrun ...` ignores the exit status, so a crashed fine-tune would sail on into the evaluation cell."""
    for cell in _code_cells(nb):
        for node in ast.parse(_source(cell)).body:
            if isinstance(node, ast.FunctionDef) and node.name == "run":
                assert any(isinstance(n, ast.Raise) for n in ast.walk(node)), "run() never raises"
                return
    pytest.fail("no run() helper")


def test_privacy_cell_says_private_delete_and_no_upload(nb):
    text = _sections(nb)[0]["markdown"].lower()
    for needle in ("private", "delete", "upload", "internet"):
        assert needle in text, f"cell 0 never says {needle!r}"


def test_attribution_to_upstream_is_present(nb):
    everything = "\n".join(_source(c) for c in nb["cells"])
    assert "Apache-2.0" in everything and "NandhaKishorM/laya" in everything


# ------------------------------------------------------------------------------------------------- 4. pins

def test_installs_are_exactly_pinned_and_torch_is_left_alone(nb):
    cell = _sections(nb)[2]["code"][0]
    pins = None
    for node in ast.parse(cell).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PINS" for t in node.targets):
            pins = ast.literal_eval(node.value)
    assert pins is not None, "cell 2 defines no PINS list"
    assert sorted(pins) == sorted(["laya==0.3.28", "transformers==5.19.0", "sentence-transformers==5.5.1",
                                   "FlagEmbedding==1.4.2"])
    assert all("==" in p for p in pins)
    assert not any(p.lower().startswith("torch") for p in pins), "Kaggle's CUDA build of torch must be left alone"
    assert not re.search(r"--upgrade|\s-U\b", cell), "an unpinned upgrade defeats the pins"


def test_models_are_the_ones_the_plan_names(nb):
    text = "\n".join(_code_cells_text(nb))
    assert 'GEN_ID = "Qwen/Qwen2.5-1.5B-Instruct"' in text
    assert 'TEACHER_ID = "BAAI/bge-reranker-v2-m3"' in text
    assert 'MINILM_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"' in text
    assert "do_sample=False" in text, "doc2query must decode greedily"
    assert "use_fp16=True" in text and "normalize=True" in text, "the teacher call the notes document"


# ------------------------------------------------------------------- 5. the wording and the sequence budget

def test_noul_wording_is_verbatim_and_defined_exactly_once(nb):
    text = "\n".join(_code_cells_text(nb))
    assert text.count("Does the passage answer: {query}?") == 1, "the instruction must be defined once and reused"
    assert text.count('"false": "Irrelevant."') == 1
    assert text.count('"true": "Answers the query."') == 1
    assert '"type": "noul"' in text
    # ... and the evaluation cell asks the model with that same function, not with a copy of the text
    evaluation = _sections(nb)[8]["code"][0]
    assert "make_question(" in evaluation and "Does the passage" not in evaluation


def test_sequence_budget_is_512_192_and_forced_into_the_saved_config(nb):
    text = "\n".join(_code_cells_text(nb))
    assert re.search(r"^MAX_LEN, HEAD_MAX_LEN = 512, 192\s*(#.*)?$", text, re.M)
    assert "max_len=MAX_LEN" in text and "head_max_len=HEAD_MAX_LEN" in text, "the recipe must carry the budget"
    script = _train_script(nb)
    assert 'max_len=H["max_len"]' in script and 'head_max_len=H["head_max_len"]' in script
    # every config the script writes goes through that budget (rolling checkpoint and final)
    assert script.count("**budget") >= 2
    assert "save_checkpoint(" in script, "the final directory must be written by laya's own save_checkpoint (config last)"


def test_training_launches_the_documented_ddp_recipe(nb):
    code = _sections(nb)[7]["code"][0]
    assert '"torchrun", "--standalone", f"--nproc_per_node={NPROC}"' in code
    for knob in ("epochs=4", "micro_batch=8", "effective_batch=64", "encoder_lr=2.5e-5", "head_lr=1e-4",
                 "weight_decay=0.01", "min_lr=1e-6", "grad_clip=1.0", "rl_samples=4", "sigma_start=0.4", "sigma_end=0.1",
                 "w_sph=0.75", "w_rps=1.0", "calib_max=400", "calib_frac=0.1", "calib_seed=20260922"):
        assert knob in code, f"hyper-parameter {knob} is not in the recipe"
    assert "NPROC = 2" in "\n".join(_code_cells_text(nb)), "two processes (T4 x2) is the default"


def test_collapse_risk_is_explained_where_training_happens(nb):
    text = _sections(nb)[7]["markdown"].lower()
    assert "963" in text, "upstream's collapse issue should be cited"
    assert "collaps" in text and "logit" in text, "say how to spot it"


# ----------------------------------------------------------------------------- 6. the pure logic, executed

def test_clean_query_keeps_questions_and_rejects_what_would_poison_the_labels(nb):
    clean = _pure(nb, ["clean_query"])["clean_query"]
    assert clean("What is the variance of a sample?") == "What is the variance of a sample?"
    assert clean('  "What is a Markov chain?"  ') == "What is a Markov chain?"
    assert clean("“What is a Markov chain?”") == "What is a Markov chain?"          # curly quotes
    assert clean("• Why does gradient descent diverge?") == "Why does gradient descent diverge?"   # a bullet glyph
    assert clean("1. How do I compute a rolling mean in pandas?\nA second line the model added") == \
        "How do I compute a rolling mean in pandas?"
    assert clean("- Why does gradient descent diverge?") == "Why does gradient descent diverge?"
    assert clean("Question: What does ARIMA stand for?") == "What does ARIMA stand for?"
    assert clean("how to compute the sample variance of a numeric vector") is not None   # no '?', but >= 6 words
    for bad in ("", "   \n", "Why?", "variance", "Define entropy.",
                "What does the passage say about entropy?", "Summarise the text above?", "Explain this document?"):
        assert clean(bad) is None, bad


def test_split_by_passage_is_a_disjoint_deterministic_90_10_split(nb):
    split = _pure(nb, ["split_by_passage"])["split_by_passage"]
    pids = [f"p{i}" for i in range(200)]
    train, held = split(pids, 0.10, 0)
    assert not train & held
    assert train | held == set(pids)
    assert len(held) == 20
    assert split(pids, 0.10, 0) == (train, held)
    assert split(pids, 0.10, 1)[1] != held, "a different seed must draw a different held-out set"
    assert len(split(["a", "b", "c"], 0.01, 0)[1]) == 1, "never an empty held-out set"
    with pytest.raises(ValueError):
        split(["only-one"], 0.1, 0)


def _scored(pid, n_neg=3):
    return {"pid": pid, "passage": f"passage {pid}", "meta": {"source_file": f"{pid}.md", "file_type": "note"},
            "negatives": [{"id": f"{pid}-n{j}", "text": f"negative {pid} {j}"} for j in range(n_neg)],
            "query": f"What is {pid}?", "p": [0.9] + [0.05 * (j + 1) for j in range(n_neg)]}


def test_rows_are_in_the_documented_laya_format(nb):
    make_row = _pure(nb, ["make_question", "make_row"])["make_row"]
    row = make_row("some passage", "What is X?", 0.07)
    assert set(row) == {"state", "questions", "gold"}
    assert row["state"] == "some passage"
    assert row["questions"] == {"rel": {"type": "noul", "instructions": "Does the passage answer: What is X??",
                                         "criteria": {"false": "Irrelevant.", "true": "Answers the query."}}}
    probs = row["gold"]["rel"]["probabilities"]
    assert probs["true"] == pytest.approx(0.07) and probs["false"] == pytest.approx(0.93)
    assert row["gold"]["rel"]["label"] == "false"
    assert make_row("t", "q", 0.5)["gold"]["rel"]["label"] == "true"
    json.dumps(row)   # native JSON objects, not JSON-in-strings
    for bad in (1.2, -0.1, float("nan")):
        with pytest.raises(ValueError):
            make_row("t", "q", bad)


def test_a_heldout_positive_never_appears_in_training_rows(nb):
    ns = _pure(nb, ["split_by_passage", "make_question", "make_row", "build_train_rows"])
    scored = [_scored(f"p{i}") for i in range(10)]
    train, held = ns["split_by_passage"]([r["pid"] for r in scored], 0.2, 0)
    held_texts = {r["passage"] for r in scored if r["pid"] in held}
    victim = next(r for r in scored if r["pid"] in train)
    victim["negatives"][0]["text"] = next(iter(held_texts))      # a distractor that IS a held-out positive
    rows, shared = ns["build_train_rows"](scored, train, held_texts)
    assert shared == 1
    assert not {r["state"] for r in rows} & held_texts
    assert len(rows) == sum(1 + len(r["negatives"]) for r in scored if r["pid"] in train) - 1
    # a passage and its negatives share ONE query, and each row's label is that candidate's own teacher probability
    first = next(r for r in scored if r["pid"] in train and r is not victim)
    mine = [row for row in rows if row["state"] in [first["passage"]] + [n["text"] for n in first["negatives"]]]
    assert len(mine) == 1 + len(first["negatives"])
    assert {row["questions"]["rel"]["instructions"] for row in mine} == {f"Does the passage answer: {first['query']}?"}
    assert [row["gold"]["rel"]["probabilities"]["true"] for row in mine] == pytest.approx(first["p"])


def test_heldout_groups_keep_one_positive_and_shuffle_it_deterministically(nb):
    ns = _pure(nb, ["split_by_passage", "build_heldout"])
    scored = [_scored(f"p{i}", n_neg=5) for i in range(40)]
    _, held = ns["split_by_passage"]([r["pid"] for r in scored], 0.25, 0)
    groups = ns["build_heldout"](scored, held, 0)
    assert {g["pid"] for g in groups} == held
    for g in groups:
        assert sum(g["label"]) == 1 and len(g["label"]) == len(g["texts"]) == len(g["teacher_p"]) == 6
        at = g["label"].index(1)
        assert g["texts"][at] == f"passage {g['pid']}" and g["teacher_p"][at] == 0.9
    assert groups == ns["build_heldout"](scored, held, 0)
    assert any(g["label"][0] != 1 for g in groups), "positives must not always come first: ties would favour them"


def test_ndcg_at_k_on_hand_computed_rankings(nb):
    ndcg = _pure(nb, ["ndcg_at_k"])["ndcg_at_k"]
    assert ndcg([0.9, 0.1, 0.2], [1, 0, 0]) == pytest.approx(1.0)
    assert ndcg([0.1, 0.9, 0.5], [1, 0, 0]) == pytest.approx(1 / math.log2(4))          # positive ranked 3rd
    assert ndcg([0.1, 0.9, 0.5], [1, 0, 0], k=2) == 0.0                                  # ... and outside the cut
    assert ndcg([0.5, 0.5], [0, 1]) == pytest.approx(1 / math.log2(3)), "equal scores keep candidate order"
    graded = [0.9, 0.1, 0.5]
    assert ndcg(graded, graded) == pytest.approx(1.0)
    assert ndcg([0.1, 0.5, 0.9], graded) < 1.0
    assert ndcg([0.3, 0.2], [0, 0]) == 0.0, "no relevant candidate: 0, not a division error"


def test_reliability_bins_and_ece_agree_with_upstreams_definition(nb):
    ns = _pure(nb, ["reliability_bins", "ece_from_bins"])
    p, y = np.array([0.05, 0.05, 0.95, 0.95]), np.array([0.0, 0.0, 1.0, 1.0])
    rows = ns["reliability_bins"](p, y, 10)
    assert [r["n"] for r in rows] == [2, 2]
    assert rows[0]["mean_p"] == pytest.approx(0.05) and rows[0]["frac_true"] == 0.0
    assert rows[1]["mean_p"] == pytest.approx(0.95) and rows[1]["frac_true"] == 1.0
    assert ns["ece_from_bins"](rows) == pytest.approx(0.05)
    assert ns["ece_from_bins"](ns["reliability_bins"](np.full(8, 0.5), np.array([0, 1] * 4), 10)) == pytest.approx(0.0)
    rng = np.random.default_rng(0)
    p = np.concatenate([rng.random(500), [0.0, 1.0, 0.1, 0.2, 0.5, 0.9]])        # includes values exactly on bin edges
    y = (rng.random(len(p)) < p).astype(float)
    for bins in (10, 15):
        mine = ns["ece_from_bins"](ns["reliability_bins"](p, y, bins))
        assert mine == pytest.approx(_upstream_ece_score(p, y, bins), abs=1e-12)
    assert sum(r["n"] for r in ns["reliability_bins"](p, y, 10)) == len(p), "every prediction lands in some bin"


def test_paired_bootstrap_ci_is_deterministic_and_means_what_it_says(nb):
    ci = _pure(nb, ["paired_bootstrap_ci"])["paired_bootstrap_ci"]
    assert ci([0.1] * 50, 500, 1) == pytest.approx((0.1, 0.1, 0.1))
    d = np.random.default_rng(0).normal(0.05, 0.1, 400)
    mean, lo, hi = ci(d, 2000, 3)
    assert mean == pytest.approx(d.mean()) and lo < mean < hi
    assert ci(d, 2000, 3) == (mean, lo, hi), "a fixed seed must give the same interval"
    assert 0 < lo, "a clear positive effect: the interval excludes zero"
    noise = [0.1, -0.1] * 200
    m2, lo2, hi2 = ci(noise, 2000, 3)
    assert m2 == pytest.approx(0.0) and lo2 < 0 < hi2


def test_collapse_check_flags_a_student_that_answers_the_same_for_everything(nb):
    check = _pure(nb, ["collapse_check"])["collapse_check"]
    teacher = [[0.95, 0.05, 0.02, 0.01], [0.80, 0.10, 0.01, 0.30]]
    assert check(teacher, teacher)["collapsed"] is False
    assert check([[0.50] * 4, [0.50] * 4], teacher)["collapsed"] is True          # the uniform plateau
    assert check([[0.06] * 4, [0.06] * 4], teacher)["collapsed"] is True          # the label prior
    assert check([[0.7, 0.1, 0.05, 0.2], [0.6, 0.2, 0.1, 0.3]], teacher)["collapsed"] is False
    out = check(teacher, teacher)
    assert out["std_ratio"] == pytest.approx(1.0) and out["range_ratio"] == pytest.approx(1.0)


def test_checkpoint_problem_names_what_laya_load_would_trip_over(nb, tmp_path):
    problem = _pure(nb, ["checkpoint_problem"])["checkpoint_problem"]
    d = tmp_path / "ckpt"
    assert "does not exist" in problem(str(d))
    d.mkdir()
    for name, is_dir in (("model.safetensors", False), ("tokenizer", True), ("encoder", True)):
        (d / name).mkdir() if is_dir else (d / name).write_bytes(b"x")
    # a DDP-style rolling checkpoint: weights, encoder, tokenizer, but no rl_agent_config.json
    assert "rl_agent_config.json" in problem(str(d))
    (d / "rl_agent_config.json").write_text("{}")
    assert problem(str(d)) is None
    for name in ("model.safetensors", "rl_agent_config.json"):
        (d / name).unlink()
        assert name in problem(str(d))
        (d / name).write_text("x")
    for name in ("tokenizer", "encoder"):       # laya.load falls through to a Hugging Face lookup without these two
        (d / name).rmdir()
        assert name in problem(str(d))
        (d / name).mkdir()
    assert problem(str(d)) is None


# ----------------------------------------------------------------------------------- 7. the manual and nav

@pytest.fixture(scope="module")
def manual():
    assert MANUAL.exists(), f"{MANUAL} does not exist"
    return MANUAL.read_text(encoding="utf-8")


def _manual_sections(text):
    parts = re.split(r"^## (\d+)\. (.+)$", text, flags=re.M)
    # parts = [preamble, n, title, body, n, title, body, ...]
    return {int(parts[i]): (parts[i + 1].strip(), parts[i + 2]) for i in range(1, len(parts), 3)}


def test_manual_sections_come_in_the_promised_order_privacy_first(manual):
    secs = _manual_sections(manual)
    assert list(secs) == [1, 2, 3, 4, 5, 6], f"sections are {list(secs)}"
    wants = {1: "privacy", 2: "before you spend", 3: "environment", 4: "step by step", 5: "testing", 6: "cleanup"}
    for n, word in wants.items():
        assert word in secs[n][0].lower(), f"section {n} is titled {secs[n][0]!r}"
    privacy = secs[1][1].lower()
    for needle in ("third party", "private", "delete", "no code", "upload"):
        assert needle in privacy, f"the privacy section never says {needle!r}"
    first_heading = re.search(r"^## ", manual, re.M)
    assert "privacy" in manual[first_heading.start():first_heading.start() + 40].lower(), "privacy must be the first section"


def test_manual_carries_the_commands_and_rules_the_package_promises(manual):
    needles = [
        "laya==0.3.28", "kaggle datasets create", "kaggle datasets delete", "kaggle quota", "cu126", "TORCH_DISABLE_NATIVE_JIT",
        "tools/laya/export_pairs.py", "--n-passages", "--negatives", "--seed", "--out", "models/laya-noetrix",
        "retrieval.laya.enabled", "laya-rerank", "default-k10", "P-rerank-pool", "--unseal-test",
        "retrieval.relevance_gate", "gate-laya", "gate_threshold", "bge-reranker-v2-m3", "MiniLM", "spike",
    ]
    for needle in needles:
        assert needle in manual, f"the manual never mentions {needle!r}"


def test_manual_does_not_quote_the_calibration_figure_nobody_could_source(manual):
    assert not re.search(r"0\.32\s*(vs|against|versus|and)\b", manual, re.I)
    assert not re.search(r"ECE\s*(of\s*)?0\.32", manual, re.I)


def test_manual_never_tells_the_reader_to_make_the_dataset_public(manual):
    for line in manual.splitlines():
        if re.search(r"--public|\s-u\b", line):
            assert re.search(r"\bnot\b|never|don'?t|without|omit", line, re.I), f"suspicious line: {line!r}"


def test_mkdocs_nav_lists_the_manual():
    assert "laya-finetune.md" in MKDOCS.read_text(encoding="utf-8")
