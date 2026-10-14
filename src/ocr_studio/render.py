"""Page -> pixels, plus the clean-up applied before OCR.

Region boxes are stored in PAGE POINTS (PyMuPDF's page space, 72 per inch), so
they mean the same thing at every DPI: the browser draws them over a small
preview and the OCR call crops them out of a large render.

Clean-up is non-destructive: a page carries an op list, and every render
re-applies it to a fresh pixmap. The source file is only ever opened read-only.

Healing deviates from the design spec: it asked for OpenCV's `cv2.inpaint`, and
OpenCV is not in the venv (nothing new is installed). `heal` instead fills the
mask from the nearest unmasked pixel (scipy's distance transform) and blurs the
fill so the seam does not read as an edge. On paper (a stain, a pen mark, a
shadow over a white margin) that is what Telea would converge to; it is worse on
textured photographs, which are not what this tab is for.
"""
from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
DOC_EXTS = {".pdf"} | IMAGE_EXTS

# op name -> (param, low, high). One table, so the browser, the store and the
# renderer cannot disagree about what a valid op is.
OPS = {
    "deskew": ("angle", -15.0, 15.0),       # degrees, counter-clockwise
    "contrast": ("factor", 0.2, 4.0),
    "threshold": ("level", 1, 254),
    "denoise": ("size", 3, 7),              # median filter, odd sizes only
    "heal": (None, None, None),             # uses the page's mask
}


def validate_ops(ops) -> list[dict]:
    """Return a clean copy of an op list, or raise ValueError naming the bad op.

    A deskew angle may be None, meaning "detect it" — resolve_deskew() turns it
    into a number before the page is saved, so stored pages are reproducible."""
    if ops is None:
        return []
    if not isinstance(ops, list):
        raise ValueError("ops must be a list")
    out = []
    for op in ops:
        if not isinstance(op, dict) or op.get("op") not in OPS:
            raise ValueError(f"unknown op {op!r}; expected one of {sorted(OPS)}")
        name = op["op"]
        param, lo, hi = OPS[name]
        clean = {"op": name}
        if param:
            v = op.get(param)
            if v is None and name == "deskew":
                clean[param] = None
            else:
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    raise ValueError(f"{name}.{param} must be a number, got {v!r}")
                if not lo <= v <= hi:
                    raise ValueError(f"{name}.{param} must be within {lo}..{hi}, got {v}")
                if name in ("threshold", "denoise"):
                    v = int(round(v))
                if name == "denoise" and v % 2 == 0:
                    raise ValueError("denoise.size must be odd (3, 5 or 7)")
                clean[param] = v
        out.append(clean)
    return out


def _open(path: Path):
    import pymupdf
    return pymupdf.open(str(path))


def page_count(path: Path) -> int:
    with _open(path) as doc:
        return doc.page_count


def page_size(path: Path, page: int) -> tuple[float, float]:
    """(width, height) of a 1-based page, in points."""
    with _open(path) as doc:
        if not 1 <= page <= doc.page_count:
            raise ValueError(f"page {page} is out of range 1..{doc.page_count}")
        r = doc[page - 1].rect
        return float(r.width), float(r.height)


def render(path: Path, page: int, dpi: int, max_edge: int | None = None
           ) -> tuple[Image.Image, float]:
    """Render a 1-based page to RGB. Returns (image, pixels-per-point).

    `max_edge` caps the longest side: DPI is relative to the PHYSICAL page, so a
    poster-sized page at a modest DPI can still be enormous."""
    import pymupdf
    with _open(path) as doc:
        if not 1 <= page <= doc.page_count:
            raise ValueError(f"page {page} is out of range 1..{doc.page_count}")
        pg = doc[page - 1]
        zoom = dpi / 72.0
        if max_edge:
            zoom = min(zoom, max_edge / (max(pg.rect.width, pg.rect.height) or 1.0))
        pix = pg.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        return img, pix.width / (pg.rect.width or 1.0)


def detect_skew(img: Image.Image, limit: float = 5.0, step: float = 0.25) -> float:
    """The rotation (degrees, PIL's counter-clockwise sign) that straightens text.

    Projection profile: rotate the ink mask through candidate angles and keep the
    one whose row sums are spikiest — text lines line up with pixel rows exactly
    when the page is straight. Done on a fixed-width copy, so the answer does not
    depend on the render DPI."""
    g = img.convert("L")
    scale = 800.0 / max(g.size)
    if scale < 1:
        g = g.resize((max(1, int(g.width * scale)), max(1, int(g.height * scale))))
    ink = Image.fromarray(((np.asarray(g) < 128) * 255).astype(np.uint8))
    best, best_score = 0.0, -1.0
    for a in np.arange(-limit, limit + 1e-9, step):
        rows = (np.asarray(ink.rotate(float(a), resample=Image.NEAREST, fillcolor=0)) > 0
                ).sum(axis=1).astype(np.float64)
        score = float(np.sum(np.diff(rows) ** 2))
        if score > best_score:
            best, best_score = float(a), score
    return round(best, 2)


def heal(img: Image.Image, mask: Image.Image | None) -> Image.Image:
    """Inpaint the white part of `mask` (any size; resized to the image)."""
    if mask is None:
        return img
    from scipy import ndimage
    m = np.asarray(mask.convert("L").resize(img.size, Image.NEAREST)) > 127
    if not m.any():
        return img
    arr = np.asarray(img).astype(np.float32)
    _, (iy, ix) = ndimage.distance_transform_edt(m, return_indices=True)
    filled = arr[iy, ix]
    sigma = (2.0, 2.0, 0) if arr.ndim == 3 else 2.0
    blurred = ndimage.gaussian_filter(filled, sigma=sigma)
    out = arr.copy()
    out[m] = blurred[m]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), img.mode)


def apply_ops(img: Image.Image, ops: list[dict], mask: Image.Image | None = None
              ) -> Image.Image:
    """Apply a validated op list in order. A deskew with angle None is detected."""
    for op in ops:
        name = op["op"]
        if name == "deskew":
            angle = op.get("angle")
            if angle is None:
                angle = detect_skew(img)
            if angle:
                img = img.rotate(angle, resample=Image.BICUBIC, expand=False,
                                 fillcolor="white" if img.mode == "RGB" else 255)
        elif name == "contrast":
            img = ImageEnhance.Contrast(img).enhance(op["factor"])
        elif name == "threshold":
            lvl = op["level"]
            img = img.convert("L").point(lambda v: 255 if v > lvl else 0)
        elif name == "denoise":
            img = img.filter(ImageFilter.MedianFilter(op["size"]))
        elif name == "heal":
            img = heal(img, mask)
    return img


def resolve_deskew(img: Image.Image, ops: list[dict]) -> list[dict]:
    """Replace every `deskew: {angle: None}` with the detected angle, measured on
    the image as the ops before it leave it."""
    out = []
    for op in ops:
        if op["op"] == "deskew" and op.get("angle") is None:
            op = {"op": "deskew", "angle": detect_skew(apply_ops(img, out))}
        out.append(op)
    return out


def deskew_bbox(bbox, ops: list[dict], size: tuple[float, float]) -> list[float]:
    """Where a box found on the RAW page lands once the page's deskew ops have
    rotated it (layout proposals come from the raw page; regions live on the
    cleaned one). Rotation is about the page centre, counter-clockwise on
    screen, with y pointing down; the result is the rotated corners' bounds,
    clamped to the page."""
    angle = sum(op.get("angle") or 0.0 for op in ops if op["op"] == "deskew")
    if not angle:
        return [float(v) for v in bbox]
    t = math.radians(angle)
    c, s = math.cos(t), math.sin(t)
    cx, cy = size[0] / 2, size[1] / 2
    x0, y0, x1, y1 = (float(v) for v in bbox)
    pts = [(x - cx, y - cy) for x in (x0, x1) for y in (y0, y1)]
    xs = [cx + dx * c + dy * s for dx, dy in pts]
    ys = [cy - dx * s + dy * c for dx, dy in pts]
    return [round(max(0.0, min(xs)), 2), round(max(0.0, min(ys)), 2),
            round(min(size[0], max(xs)), 2), round(min(size[1], max(ys)), 2)]


def crop(img: Image.Image, scale: float, bbox, pad: int
         ) -> tuple[Image.Image, tuple[int, int]]:
    """Cut a points-space box out of a render, padded (engines miss glyphs that
    touch the edge) and clamped to the image. Returns (crop, its top-left in
    the render's pixels), so boxes found inside the crop map back to the page."""
    x0, y0, x1, y1 = (float(v) for v in bbox)
    box = (max(0, math.floor(x0 * scale) - pad), max(0, math.floor(y0 * scale) - pad),
           min(img.width, math.ceil(x1 * scale) + pad),
           min(img.height, math.ceil(y1 * scale) + pad))
    if box[2] - box[0] < 2 or box[3] - box[1] < 2:
        raise ValueError(f"region {list(bbox)} is empty at this page size")
    return img.crop(box), (box[0], box[1])


def to_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
