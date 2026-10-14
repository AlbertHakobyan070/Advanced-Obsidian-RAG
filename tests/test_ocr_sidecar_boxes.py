"""The PaddleOCR sidecar keeps its boxes and scores when asked (return_boxes).

It used to compute both and drop them. OCR Studio shows them, so /ocr now
returns them on request — read through BOTH result shapes, because the CPU image
pins PaddleOCR 2.x and the GPU image 3.x, and they disagree. Paddle itself is not
installed here: the engines are fakes returning each major's real structure.
"""
import base64
import importlib.util
import io
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("ocr_sidecar_server",
                                               ROOT / "ocr-sidecar" / "server.py")
server = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(server)


class V3:
    def predict(self, image):
        return [{"rec_texts": ["first line", "", "second"],
                 "rec_scores": np.array([0.9, 0.1, 0.5]),
                 "rec_boxes": np.array([[1, 2, 30, 12], [0, 0, 1, 1], [1, 20, 25, 30]])}]


class V2:
    def ocr(self, image, cls=True):
        return [[[[[1, 2], [30, 2], [30, 12], [1, 12]], ("first line", 0.9)],
                 [[[1, 20], [25, 20], [25, 30], [1, 30]], ("second", 0.5)]]]


def test_both_result_shapes_keep_text_score_and_box(monkeypatch):
    for major, engine in ((3, V3()), (2, V2())):
        monkeypatch.setattr(server, "_paddleocr_major", lambda m=major: m)
        got = server._run_ocr_detailed(engine, None)
        assert [d["text"] for d in got] == ["first line", "second"]
        assert [d["score"] for d in got] == [0.9, 0.5]
        assert got[0]["box"] == [1.0, 2.0, 30.0, 12.0]
        # the plain-text path the ingest lane uses is unchanged
        assert server._run_ocr(engine, None) == ["first line", "second"]


def test_ocr_returns_boxes_only_when_asked(monkeypatch):
    monkeypatch.setattr(server, "_paddleocr_major", lambda: 3)
    monkeypatch.setattr(server, "_engine", lambda pipeline, lang: V3())
    buf = io.BytesIO()
    Image.new("RGB", (40, 40), "white").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    client = TestClient(server.app)

    plain = client.post("/ocr", json={"image_b64": b64}).json()
    assert plain["text"] == "first line\nsecond" and "boxes" not in plain

    rich = client.post("/ocr", json={"image_b64": b64, "return_boxes": True}).json()
    assert rich["text"] == plain["text"]
    assert rich["confidence"] == 0.7 and len(rich["boxes"]) == 2
