"""`GET /api/ocr/status` must report the PaddleOCR sidecar's THREE states.

The sidecar is its own container, and it can fail in a way neither of the other
OCR lanes can: up, answering HTTP, and completely unable to OCR because its
engine will not import (a missing system library, or a base image that moved
under a pinned wheel). That happened, and nothing noticed — /health answered 200
without ever importing paddle, the compose healthcheck called the container
healthy, and the first real page came back a 500.

So the sidecar now answers 503 with `engine_importable: false`, and the console
has to keep the three states apart:

    not configured   -> no sidecar in the config at all
    container stopped-> configured, nothing answers (the normal way to give the
                        RAM back between ingests — not a fault)
    up but not ready -> answers, cannot OCR

The trap these tests exist for: `reachable` used to be `status_code < 500`, so
adopting a 503 would have quietly filed "up but broken" under "stopped" — the
same conflation, one layer up.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.utils.config_loader import Config

ROOT = Path(__file__).resolve().parents[1]

SIDECAR = "http://sidecar:8103"


class FakeResponse:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self.ok = status_code < 400
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


@pytest.fixture
def ocr_status(monkeypatch, tmp_path, management_module):
    """Call ocr_status() against a scripted sidecar, with the rest stubbed out.

    Only the paddle branch is under test, so Tesseract detection and the VLM
    probe are pinned to "absent": otherwise the result would depend on whether
    the machine running the tests happens to have Tesseract installed.
    """
    manage_api = management_module
    monkeypatch.setattr("src.ingestion.pdf_loader.detect_ocr_engine",
                        lambda: None)
    monkeypatch.setattr(manage_api.shutil, "which", lambda name: None)

    def run(health, *, configured=True):
        cfg_data = {"pdf": {"ocr_engine": "paddle"}}
        if configured:
            cfg_data["pdf"]["paddle_ocr"] = {"base_url": SIDECAR, "lang": "en"}
        monkeypatch.setattr(
            "src.utils.config_loader.load_config",
            lambda *a, **k: Config(cfg_data, tmp_path))

        import requests

        def fake_get(url, **kwargs):
            if url.endswith("/health"):
                if isinstance(health, Exception):
                    raise health
                return health
            raise requests.ConnectionError("no vlm endpoint in this test")

        monkeypatch.setattr(requests, "get", fake_get)
        return manage_api.ocr_status()["paddle"]

    return run


def test_engine_that_cannot_import_is_up_but_not_ready(ocr_status):
    """The failure the field exists for: answering, and useless."""
    paddle = ocr_status(FakeResponse(503, {
        "ok": False, "engine": "paddleocr", "engine_importable": False,
        "engine_import_error": "free(): invalid pointer (exit -6)",
        "loaded": [], "langs": ["en", "ch"]}))

    assert paddle["reachable"] is True          # NOT "container stopped"
    assert paddle["engine_importable"] is False
    assert "invalid pointer" in paddle["engine_error"]
    assert paddle["langs"] == ["en", "ch"]


def test_healthy_sidecar_is_ready(ocr_status):
    paddle = ocr_status(FakeResponse(200, {
        "ok": True, "engine": "paddleocr", "engine_importable": True,
        "engine_import_error": None, "loaded": [], "langs": ["en"]}))

    assert paddle["reachable"] is True
    assert paddle["engine_importable"] is True
    assert paddle["engine_error"] is None
    # An engine is built on the first page, so "nothing loaded yet" is the
    # normal state of a healthy sidecar and must not read as unready.
    assert paddle["configured"] is True


def test_stopped_container_stays_distinguishable(ocr_status):
    """Stopping the sidecar between ingests is routine, not a fault."""
    import requests
    paddle = ocr_status(requests.ConnectionError("connection refused"))

    assert paddle["configured"] is True
    assert paddle["reachable"] is False
    assert paddle["engine_importable"] is None   # unknown, nothing answered
    assert "ConnectionError" in paddle["error"]


def test_sidecar_without_the_field_is_unknown_not_broken(ocr_status):
    """An older sidecar image predates engine_importable.

    Reporting that as False would paint a working container red, so the absent
    field has to stay unknown — the console only calls it broken on a literal
    false.
    """
    paddle = ocr_status(FakeResponse(200, {
        "ok": True, "engine": "paddleocr", "loaded": [], "langs": ["en"]}))

    assert paddle["reachable"] is True
    assert paddle["engine_importable"] is None
    assert paddle["engine_error"] is None


def test_unconfigured_sidecar_is_not_probed_at_all(ocr_status):
    paddle = ocr_status(FakeResponse(200, {}), configured=False)

    assert paddle["configured"] is False
    assert paddle["reachable"] is False
    assert paddle["engine_importable"] is None
