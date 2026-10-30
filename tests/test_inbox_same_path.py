"""An inbox PDF whose exact path is already indexed cannot be ingested again, force or not.

THE BUG THIS PINS (slice B review DR-3). `force: true` is the inbox's "ingest anyway" for a
file that merely looks indexed (same stem, elsewhere in the corpus). But a PDF at the SAME
path as indexed rows is a re-ingest of that source: an edited file mints new doc_ids, and
`index --append` now refuses to keep the old rows beside them. By then the ingest job has
already archived the PDFs to Inbox/_ingested/ (it does that inside the job), so one
re-ingested file stranded every other file in the batch: archived, never indexed, and gone
from the inbox. The console now refuses up front, before anything is queued or moved.

Run:  python -m pytest tests/test_inbox_same_path.py -q
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.utils.config_loader import Config


@pytest.fixture
def inbox(management_module, monkeypatch, tmp_path):
    m = management_module
    vault = tmp_path / "vault"
    folder = vault / "Inbox"
    folder.mkdir(parents=True)
    for name in ("A.pdf", "B.pdf"):
        (folder / name).write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(m, "CFG", Config({"pdf": {"vault_path": str(vault)},
                                          "webui": {"inbox_dir": "Inbox"}}, tmp_path))
    queued = []
    monkeypatch.setattr(m, "enqueue", lambda kind, prm: queued.append((kind, prm))
                        or SimpleNamespace(public=lambda: {"kind": kind}))
    indexed = {}
    monkeypatch.setattr(m, "build_manifest", lambda force=False: indexed)
    return SimpleNamespace(m=m, folder=folder, queued=queued, indexed=indexed)


def _index(w, source):
    w.indexed[source] = {"filename": Path(source).stem, "jsonls": ["inbox_old_chunks.jsonl"]}


def test_force_cannot_re_ingest_a_file_whose_path_is_indexed(inbox):
    _index(inbox, str(Path("Inbox") / "A.pdf"))             # A was ingested from here before

    r = inbox.m.ingest_inbox(inbox.m.InboxIngestIn(force=True))

    body = json.loads(r.body)
    assert r.status_code == 409 and body["ok"] is False
    assert body["already_indexed"] == ["A.pdf"]
    assert "documents/delete" in body["error"] and "REPLACE" in body["error"]
    # review SB2-C4: a DIFFERENT document that only shares the name must be renamed, not
    # have the indexed one deleted (its original sits in _ingested/, which no lane rediscovers)
    assert "DIFFERENT document" in body["error"] and "rename" in body["error"]
    assert inbox.queued == []                                # no job, so no PDF is archived
    assert sorted(p.name for p in inbox.folder.iterdir()) == ["A.pdf", "B.pdf"]


def test_force_still_ingests_a_file_whose_stem_is_indexed_elsewhere(inbox):
    _index(inbox, str(Path("Tech Books") / "A.pdf"))        # same stem, another path: a duplicate, not a REPLACE

    r = inbox.m.ingest_inbox(inbox.m.InboxIngestIn(force=True))

    assert r["ok"] is True and [k for k, _ in inbox.queued] == ["ingest_pdfs", "index_append"]


def test_the_custom_designer_refuses_the_same_path_for_a_pdf_group(inbox):
    _index(inbox, str(Path("Inbox") / "A.pdf"))
    groups = [inbox.m.CustomGroupIn(kind="pdf", files=["A.pdf"]),
              inbox.m.CustomGroupIn(kind="pdf", files=["B.pdf"])]

    r = inbox.m.ingest_custom(inbox.m.CustomIngestIn(groups=groups, force=True))

    assert r.status_code == 409 and json.loads(r.body)["already_indexed"] == ["A.pdf"]
    assert inbox.queued == []
    assert sorted(p.name for p in inbox.folder.iterdir()) == ["A.pdf", "B.pdf"]


def test_a_page_range_top_up_of_an_indexed_inbox_pdf_is_queued(inbox):
    """REGRESSION (review B2-2): the documented top-up (a wider --pages range only ADDS
    pages) went through the Custom job's Pages field until the same-path refusal blocked
    it, and the refusal's advice (delete, then repeat) would have dropped the indexed
    pages. A group with `pages` is left to the append's page-by-page check."""
    _index(inbox, str(Path("Inbox") / "A.pdf"))             # pages 1-50 went in earlier
    groups = [inbox.m.CustomGroupIn(kind="pdf", files=["A.pdf"], pages="51-100")]

    r = inbox.m.ingest_custom(inbox.m.CustomIngestIn(groups=groups, force=True))

    assert r["ok"] is True and [k for k, _ in inbox.queued] == ["ingest_pdfs", "index_append"]
    assert inbox.queued[0][1]["pages"] == "51-100"
