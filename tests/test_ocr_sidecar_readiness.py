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


# --- which ENGINE, and on what hardware -------------------------------------
#
# The sidecar reported `device` and `paddleocr_version` from the day the GPU
# lane landed, and nothing read them, so "is the fast sidecar the one actually
# running?" still meant reading `docker ps`. These pin the wiring, because the
# symptom of losing it is not an error — it is a CPU sidecar quietly doing a
# GPU sidecar's job at a fraction of the speed, which looks like a slow ingest.


def test_console_reports_device_version_and_engine(ocr_status):
    paddle = ocr_status(FakeResponse(200, {
        "ok": True, "engine": "paddleocr", "engine_importable": True,
        "engine_import_error": None, "loaded": ["ocr:en"], "langs": ["en"],
        "device": "gpu:0", "paddleocr_version": "3.7.0",
        "pipeline": "ocr",
        "pipelines": {
            "ocr": {"available": True, "reason": None,
                    "models": ["PP-OCRv6_medium_det", "PP-OCRv6_medium_rec"]},
            "vl": {"available": True, "reason": None,
                   "models": ["PP-DocLayoutV3", "PaddleOCR-VL-1.6-0.9B"]},
        }}))

    assert paddle["device"] == "gpu:0"
    assert paddle["paddleocr_version"] == "3.7.0"
    assert paddle["pipeline"] == "ocr"
    # The MODEL name is the part that cannot be inferred from anything else:
    # `PaddleOCR()` resolves to whatever the installed paddlex pins as current,
    # so the same code is PP-OCRv4 on one image and PP-OCRv6 on another with no
    # other visible difference.
    assert "PP-OCRv6_medium_rec" in paddle["pipelines"]["ocr"]["models"]
    assert paddle["pipelines"]["vl"]["available"] is True


def test_cpu_sidecar_reports_vl_as_unavailable_with_a_reason(ocr_status):
    """The CPU image pins PaddleOCR 2.x, which has no PaddleOCRVL at all.

    "Unavailable" is useless without the why: one cause is fixed by a rebuild
    and the other by using the GPU image, and the console cannot tell an
    operator which unless the sidecar says.
    """
    paddle = ocr_status(FakeResponse(200, {
        "ok": True, "engine": "paddleocr", "engine_importable": True,
        "engine_import_error": None, "loaded": [], "langs": ["en"],
        "device": "auto", "paddleocr_version": "2.9.1", "pipeline": "ocr",
        "pipelines": {
            "ocr": {"available": True, "reason": None, "models": []},
            "vl": {"available": False,
                   "reason": "PaddleOCR 2.9.1 does not ship PaddleOCRVL "
                             "(needs 3.x)",
                   "models": []},
        }}))

    assert paddle["device"] == "auto"
    assert paddle["pipelines"]["vl"]["available"] is False
    assert "3.x" in paddle["pipelines"]["vl"]["reason"]


def test_sidecar_predating_the_engine_fields_is_unknown_not_broken(ocr_status):
    """Same rule as engine_importable: absent means unknown.

    A sidecar built before the engine switch reports none of these. Defaulting
    them to a string would invent a fact; defaulting `pipelines` to a populated
    dict would claim VL is available on an image that has never heard of it.
    """
    paddle = ocr_status(FakeResponse(200, {
        "ok": True, "engine": "paddleocr", "engine_importable": True,
        "loaded": [], "langs": ["en"]}))

    assert paddle["device"] is None
    assert paddle["paddleocr_version"] is None
    assert paddle["pipeline"] is None
    assert paddle["pipelines"] == {}


# --- the ingest client's half of the same contract ---------------------------


def _client(cfg_data, tmp_path):
    from src.ingestion.ocr_paddle import PaddleOCRClient
    return PaddleOCRClient.from_config(Config(cfg_data, tmp_path))


def test_client_omits_pipeline_when_config_does_not_set_one(monkeypatch, tmp_path):
    """Sending nothing is what keeps this working against an OLD sidecar.

    A client that always sent `pipeline` would 400 against every sidecar built
    before the field existed, turning an optional feature into a hard
    requirement on the container version.
    """
    client = _client({"pdf": {"paddle_ocr": {"base_url": SIDECAR}}}, tmp_path)
    assert client.pipeline is None

    sent = {}

    class R:
        status_code = 200

        @staticmethod
        def raise_for_status():
            pass

        @staticmethod
        def json():
            return {"text": "hello", "lines": 1}

    import requests
    monkeypatch.setattr(requests, "post",
                        lambda url, json=None, **kw: (sent.update(json or {}), R)[1])

    assert client.ocr_image(b"\x89PNG") == "hello"
    assert "pipeline" not in sent
    assert "image_b64" in sent and sent["lang"] == "en"


def test_client_forwards_the_configured_pipeline(monkeypatch, tmp_path):
    client = _client({"pdf": {"paddle_ocr": {"base_url": SIDECAR,
                                             "pipeline": "vl"}}}, tmp_path)
    assert client.pipeline == "vl"

    sent = {}

    class R:
        status_code = 200

        @staticmethod
        def raise_for_status():
            pass

        @staticmethod
        def json():
            return {"text": "# Heading\n\n$x^2$", "lines": 2, "pipeline": "vl"}

    import requests
    monkeypatch.setattr(requests, "post",
                        lambda url, json=None, **kw: (sent.update(json or {}), R)[1])

    assert client.ocr_image(b"\x89PNG") == "# Heading\n\n$x^2$"
    assert sent["pipeline"] == "vl"


def test_client_raises_when_the_sidecar_cannot_serve_the_engine(monkeypatch,
                                                                tmp_path):
    """A 501 must NOT come back as a page of PP-OCRv6 text.

    The whole point of asking for `vl` is the markdown, so silently accepting
    the other engine's output would relabel the result as something it is not —
    and the caller would only find out by noticing there is no LaTeX in a
    corpus they believe has it.
    """
    client = _client({"pdf": {"paddle_ocr": {"base_url": SIDECAR,
                                             "pipeline": "vl"}}}, tmp_path)

    import requests

    class R:
        status_code = 501

        @staticmethod
        def raise_for_status():
            raise requests.HTTPError("501 Server Error: Not Implemented")

        @staticmethod
        def json():
            return {"error": 'pipeline "vl" unavailable: paddlex[ocr] extras '
                             'are not installed'}

    monkeypatch.setattr(requests, "post", lambda *a, **kw: R)

    with pytest.raises(requests.HTTPError):
        client.ocr_image(b"\x89PNG")
