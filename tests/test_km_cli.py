"""`rag map`: parser, argument refusals, --estimate, exit codes. No LLM, no network."""
import json
import sys
from pathlib import Path

import fitz
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]


def _parse(*argv):
    import main                  # imported here, as test_bench_cli does: it configures logging
    return main.build_parser().parse_args(list(argv))


def _run(*argv):
    args = _parse(*argv)
    args.func(args)


@pytest.fixture
def env(tmp_path):
    """A config file whose vault holds one text PDF and whose staging root is outside it."""
    vault, staging = tmp_path / "vault", tmp_path / "staging"
    (vault / "books").mkdir(parents=True)
    doc = fitz.open()
    for i in range(3):
        doc.new_page().insert_text((72, 72), f"Page {i + 1}. " + "A line of ordinary text. " * 8)
    doc.save(vault / "books" / "b.pdf")
    doc.close()
    cfg = yaml.safe_load((ROOT / "config.example.yaml").read_text(encoding="utf-8"))
    cfg["parser"]["vault_path"] = str(vault)
    cfg["knowledge_map"]["staging_dir"] = str(staging)
    cfgfile = tmp_path / "config.yaml"
    cfgfile.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return type("Env", (), {"cfg": str(cfgfile), "vault": vault, "staging": staging})


def test_parser_map_args():
    a = _parse("map", "book.pdf", "--chapter", "Storage", "--template", "concept", "--dry-run",
               "--export-prompts", "out", "--import-extract", "in", "--title", "DDIA")
    assert (a.pdf, a.chapter, a.pages, a.template, a.dry_run) == ("book.pdf", "Storage", None, "concept", True)
    assert (a.export_prompts, a.import_extract, a.title, a.estimate) == ("out", "in", "DDIA", None)
    assert _parse("map", "book.pdf", "--pages", "140-240").pages == (140, 240)
    assert _parse("map", "--estimate", "books").pdf is None
    with pytest.raises(SystemExit):
        _parse("map", "book.pdf", "--pages", "140")


def test_map_requires_chapter_or_pages(env, capsys):
    for extra in ([], ["--chapter", "A", "--pages", "1-2"]):
        with pytest.raises(SystemExit) as e:
            _run("--config", env.cfg, "map", str(env.vault / "books" / "b.pdf"), *extra)
        assert e.value.code == 2
        assert "ERROR:" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:       # and no PDF at all, without --estimate
        _run("--config", env.cfg, "map", "--chapter", "A")
    assert e.value.code == 2


def test_scout_error_exits_2_without_writing(env, capsys):
    with pytest.raises(SystemExit) as e:
        _run("--config", env.cfg, "map", str(env.vault / "books" / "b.pdf"), "--pages", "1-99", "--dry-run")
    assert e.value.code == 2 and "ERROR:" in capsys.readouterr().err
    assert not env.staging.exists()


def test_vault_relative_pdf_and_missing_pdf(env, capsys):
    """I3: a vault-relative PDF works from the repo cwd; M2: a missing one is ERROR, exit 2."""
    _run("--config", env.cfg, "map", "books/b.pdf", "--pages", "1-2", "--template", "concept", "--dry-run")
    out = capsys.readouterr().out
    assert "plan:" in out and "NEXT:" in out and not env.staging.exists()
    with pytest.raises(SystemExit) as e:
        _run("--config", env.cfg, "map", "books/missing.pdf", "--pages", "1-2", "--dry-run")
    assert e.value.code == 2 and "ERROR: PDF not found" in capsys.readouterr().err


def test_unknown_template_exits_2(env, capsys):
    """M7: the names come from templates.yaml (run_map's check), not a second list in the parser."""
    with pytest.raises(SystemExit) as e:
        _run("--config", env.cfg, "map", "books/b.pdf", "--pages", "1-2", "--template", "nope", "--dry-run")
    err = capsys.readouterr().err
    assert e.value.code == 2 and "ERROR: --template 'nope' is not one of: concept" in err


def test_estimate_prints_inventory(env, capsys):
    _run("--config", env.cfg, "map", "--estimate", str(env.vault))
    out = capsys.readouterr().out
    assert "books/b.pdf" in out and "pages" in out
    assert "after the first runs" in out         # no ok runs yet: inventory only, no projection


def test_estimate_projects_from_ok_runs(env, capsys):
    for name, status, in_tok, secs in (("a", "ok", 600, 30.0), ("b", "failed", 9999, 999.0)):
        d = env.staging / name / "ch"
        d.mkdir(parents=True)
        (d / "run.json").write_text(json.dumps({
            "status": status, "source": {"pages": [1, 10]},
            "totals": {"input_tokens": in_tok, "output_tokens": 400, "wall_seconds": secs}}), encoding="utf-8")
    _run("--config", env.cfg, "map", "--estimate", str(env.vault))
    out = capsys.readouterr().out
    assert "1 finished run" in out
    assert "100.0 tokens/page" in out and "3.0 s/page" in out        # (600+400)/10 and 30/10 pages
    assert "300 tokens" in out                                        # 3 text pages x 100


def test_failed_run_exits_1_with_reason(env, capsys, monkeypatch):
    import src.knowledge_map.run as run
    monkeypatch.setattr(run, "run_map", lambda *a, **k: {"status": "failed", "reason": "grounded ratio 0.50",
                                                         "folder": "x"})
    with pytest.raises(SystemExit) as e:
        _run("--config", env.cfg, "map", str(env.vault / "books" / "b.pdf"), "--pages", "1-2")
    assert e.value.code == 1 and "grounded ratio 0.50" in capsys.readouterr().err


def test_ok_run_prints_next_lines(env, capsys, monkeypatch):
    import src.knowledge_map.run as run
    seen = {}
    monkeypatch.setattr(run, "run_map", lambda cfg, pdf, **k: seen.update(k, pdf=pdf) or {
        "status": "ok", "reason": None, "folder": "STAGED"})
    _run("--config", env.cfg, "map", str(env.vault / "books" / "b.pdf"), "--pages", "1-2")
    out = capsys.readouterr().out
    assert "NEXT:" in out and "STAGED" in out and "generated_paths" in out and "--append" in out
    assert seen["pages"] == (1, 2) and seen["chapter"] is None and seen["dry_run"] is False
