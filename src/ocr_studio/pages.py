"""Per-page studio state, its assembly into markdown, and the OCR quality report.

State lives OUTSIDE the vault, one folder per document:

    <data>/ocr_studio/<sha1(vault-relative posix path)[:16]>/
        doc.json            {"source": "<vault-relative posix path>",
                             "pages": [1-based ints]}   (the working page set; absent = all)
        page_<n>.json       {regions, ops, status, dpi, updated_at}   (n is 1-based)
        page_<n>_mask.png   the healing-brush mask, white = heal

A region is {id, bbox (page points), type, order, lang, engine, status, text,
format, confidence, raw, lines, error, engine_used, figure, ms}. `raw` is what
the engine returned and `text` is what the operator kept, which is what the CER
report compares.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

from eval.bench.questions import _retry_locked
from src.ocr_studio.engines import ENGINES
from src.ocr_studio.render import validate_ops

REGION_TYPES = ("text", "table", "formula", "figure", "ignore")
REGION_STATUS = ("todo", "done", "error")
PAGE_STATUS = ("todo", "done")
FORMATS = ("plain", "markdown", "latex", "image")
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
_LANG_RE = re.compile(r"^[A-Za-z0-9_+-]{0,40}$")
MAX_REGIONS = 500
MAX_TEXT = 200_000


def doc_key(rel: str) -> str:
    return hashlib.sha1(rel.replace("\\", "/").encode("utf-8")).hexdigest()[:16]


def store_root(cfg) -> Path:
    """Beside the active vault's chunk files, so a vault switch moves it too."""
    return cfg.path("paths.chunks_file").parent / "ocr_studio"


def figures_root(cfg) -> Path:
    """The PDF lane's figure folder (PDFLoader.from_config resolves it the same way)."""
    return (cfg.path("pdf.figures_dir") if cfg.get("pdf.figures_dir")
            else cfg.project_root / "data" / "figures")


def atomic_write(path: Path, data: bytes) -> None:
    """Temp file + replace, retried while Windows holds the target open."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_bytes(data)
        _retry_locked(os.replace, tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def empty_page() -> dict:
    return {"regions": [], "ops": [], "status": "todo", "dpi": None, "updated_at": None}


def _num(v, what: str) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a number, got {v!r}")


def validate_region(r) -> dict:
    if not isinstance(r, dict):
        raise ValueError("a region must be an object")
    rid = str(r.get("id") or "")
    if not _ID_RE.match(rid):
        raise ValueError(f"bad region id {rid!r}")
    bbox = r.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f"region {rid}: bbox must be [x0, y0, x1, y1]")
    x0, y0, x1, y1 = (_num(v, f"region {rid} bbox") for v in bbox)
    if not (x1 > x0 and y1 > y0):
        raise ValueError(f"region {rid}: bbox must have x1 > x0 and y1 > y0")
    rtype = r.get("type") or "text"
    if rtype not in REGION_TYPES:
        raise ValueError(f"region {rid}: type must be one of {REGION_TYPES}")
    engine = r.get("engine") or ""
    if engine and engine not in ENGINES:
        raise ValueError(f"region {rid}: engine must be one of {sorted(ENGINES)}")
    lang = str(r.get("lang") or "")
    if not _LANG_RE.match(lang):
        raise ValueError(f"region {rid}: bad language code {lang!r}")
    status = r.get("status") or "todo"
    if status not in REGION_STATUS:
        raise ValueError(f"region {rid}: status must be one of {REGION_STATUS}")
    fmt = r.get("format") or "plain"
    if fmt not in FORMATS:
        raise ValueError(f"region {rid}: format must be one of {FORMATS}")
    text = r.get("text") or ""
    raw = r.get("raw")
    for name, v in (("text", text), ("raw", raw or "")):
        if not isinstance(v, str) or len(v) > MAX_TEXT:
            raise ValueError(f"region {rid}: {name} must be a string under {MAX_TEXT} chars")
    conf = r.get("confidence")
    lines = r.get("lines") or []
    if not isinstance(lines, list):
        raise ValueError(f"region {rid}: lines must be a list")
    return {
        "id": rid, "bbox": [round(v, 2) for v in (x0, y0, x1, y1)], "type": rtype,
        "order": int(_num(r.get("order", 0), f"region {rid} order")),
        "lang": lang, "engine": engine, "status": status, "text": text,
        "format": fmt, "raw": raw,
        "confidence": None if conf is None else round(_num(conf, "confidence"), 4),
        "lines": [{"text": str(ln.get("text", "")), "bbox": list(ln.get("bbox") or []),
                   "conf": ln.get("conf")} for ln in lines[:2000] if isinstance(ln, dict)],
        "error": (str(r["error"])[:2000] if r.get("error") else None),
        "engine_used": (str(r["engine_used"]) if r.get("engine_used") else None),
        "figure": (str(r["figure"]) if r.get("figure") else None),
        "ms": (int(r["ms"]) if isinstance(r.get("ms"), (int, float)) else None),
    }


def validate_page(data) -> dict:
    """A clean copy of a page record, or ValueError naming the first problem."""
    if not isinstance(data, dict):
        raise ValueError("page must be an object")
    regions = data.get("regions") or []
    if not isinstance(regions, list) or len(regions) > MAX_REGIONS:
        raise ValueError(f"regions must be a list of at most {MAX_REGIONS}")
    clean = [validate_region(r) for r in regions]
    ids = [r["id"] for r in clean]
    if len(ids) != len(set(ids)):
        raise ValueError("region ids must be unique on a page")
    status = data.get("status") or "todo"
    if status not in PAGE_STATUS:
        raise ValueError(f"page status must be one of {PAGE_STATUS}")
    dpi = data.get("dpi")
    if dpi is not None:
        dpi = int(_num(dpi, "dpi"))
        if not 72 <= dpi <= 600:
            raise ValueError("dpi must be within 72..600")
    return {"regions": clean, "ops": validate_ops(data.get("ops")), "status": status,
            "dpi": dpi, "updated_at": data.get("updated_at")}


class Store:
    def __init__(self, root: Path):
        self.root = Path(root)

    def doc_dir(self, rel: str) -> Path:
        return self.root / doc_key(rel)

    def page_file(self, rel: str, page: int) -> Path:
        return self.doc_dir(rel) / f"page_{int(page)}.json"

    def mask_file(self, rel: str, page: int) -> Path:
        return self.doc_dir(rel) / f"page_{int(page)}_mask.png"

    def load(self, rel: str, page: int) -> dict:
        p = self.page_file(rel, page)
        if not p.is_file():
            return empty_page()
        return json.loads(p.read_text(encoding="utf-8"))

    def save(self, rel: str, page: int, data: dict) -> dict:
        clean = validate_page(data)
        clean["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        d = self.doc_dir(rel)
        if not (d / "doc.json").is_file():
            atomic_write(d / "doc.json", json.dumps(
                {"source": rel.replace("\\", "/")}, ensure_ascii=False).encode("utf-8"))
        atomic_write(self.page_file(rel, page),
                     json.dumps(clean, ensure_ascii=False, indent=1).encode("utf-8"))
        return clean

    def save_mask(self, rel: str, page: int, png: bytes | None) -> None:
        p = self.mask_file(rel, page)
        if png is None:
            p.unlink(missing_ok=True)
        else:
            atomic_write(p, png)

    def pages(self, rel: str) -> dict[int, dict]:
        """Every saved page of a document, keyed by 1-based page number."""
        out = {}
        for p in self.doc_dir(rel).glob("page_*.json"):
            m = re.fullmatch(r"page_(\d+)\.json", p.name)
            if m:
                out[int(m.group(1))] = json.loads(p.read_text(encoding="utf-8"))
        return dict(sorted(out.items()))

    def meta(self, rel: str) -> dict:
        f = self.doc_dir(rel) / "doc.json"
        return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {}

    def page_set(self, rel: str) -> list[int] | None:
        """The document's working pages (1-based, sorted), or None for all of them."""
        return self.meta(rel).get("pages") or None

    def set_page_set(self, rel: str, pages: list[int] | None) -> None:
        """Store the working pages; None clears the set (= all pages). Clearing
        on a document with no record writes nothing."""
        meta = self.meta(rel)
        if not pages and not meta:
            return
        meta["source"] = rel.replace("\\", "/")
        if pages:
            meta["pages"] = sorted({int(p) for p in pages})
        else:
            meta.pop("pages", None)
        atomic_write(self.doc_dir(rel) / "doc.json",
                     json.dumps(meta, ensure_ascii=False).encode("utf-8"))

    def mask_pages(self, rel: str) -> set[int]:
        return {int(m.group(1)) for p in self.doc_dir(rel).glob("page_*_mask.png")
                if (m := re.fullmatch(r"page_(\d+)_mask\.png", p.name))}

    def docs(self) -> list[str]:
        """Vault-relative paths of every document that has studio state."""
        out = []
        for d in sorted(self.root.glob("*/doc.json")):
            try:
                out.append(json.loads(d.read_text(encoding="utf-8"))["source"])
            except (OSError, ValueError, KeyError):
                continue
        return out


def page_worked(page: dict) -> bool:
    """Saved work on a page: a region, a clean-up op or the done mark. (A mask
    is a file beside it — see Store.mask_pages.) A record saved with none of
    these is a Save pressed on an untouched page, not work."""
    return bool(page.get("regions") or page.get("ops") or page.get("status") == "done")


def doc_state(store: Store, rel: str) -> dict:
    """Modified = any saved region, clean-up op, heal mask or done page.
    `edited` counts regions whose kept text differs from the engine's raw
    output (the CER report's notion of a hand correction)."""
    pages = store.pages(rel)
    worked = {n for n, p in pages.items() if page_worked(p)} | store.mask_pages(rel)
    regions = [r for p in pages.values() for r in p.get("regions") or []]
    return {"modified": bool(worked), "pages_worked": len(worked), "regions": len(regions),
            "ocr_done": sum(r.get("status") == "done" for r in regions),
            "edited": sum(r.get("raw") is not None and r.get("type") not in ("figure", "ignore")
                          and (r.get("text") or "").strip() != (r.get("raw") or "").strip()
                          for r in regions),
            "pages_done": sum(p.get("status") == "done" for p in pages.values())}


# ---- assembly ---------------------------------------------------------------

def assemble_page(page: dict, figure_link=None) -> str:
    """Regions in reading order -> page markdown. `ignore` regions and empty
    regions drop out; formulas get $$ fences unless they already carry LaTeX
    delimiters; figures become image links (the PDF lane's chunker turns those
    into `[image: caption]` and keeps the path in figure_images metadata).
    `figure_link(path) -> str` rewrites a figure's link (the export points them
    into its zip); by default the link is the figure's own path."""
    parts = []
    for r in sorted(page.get("regions") or [], key=lambda r: (r.get("order", 0), r["bbox"][1])):
        rtype = r.get("type")
        text = (r.get("text") or "").strip()
        if rtype == "ignore":
            continue
        if rtype == "figure":
            if r.get("figure"):
                alt = re.sub(r"[\]\n]+", " ", text).strip() or "Figure"
                # <...>: vault and project paths have spaces, which a bare
                # markdown link destination cannot hold
                href = figure_link(r["figure"]) if figure_link else Path(r["figure"]).as_posix()
                parts.append(f"![{alt}](<{href}>)")
            continue
        if not text:
            continue
        if rtype == "formula" and not text.startswith(("$$", "\\[", "\\begin")):
            text = f"$$\n{text}\n$$"
        parts.append(text)
    return "\n\n".join(parts)


def page_figures(page: dict) -> list[str]:
    return [r["figure"] for r in sorted(page.get("regions") or [], key=lambda r: r.get("order", 0))
            if r.get("type") == "figure" and r.get("figure") and Path(r["figure"]).is_file()]


# ---- OCR quality: character error rate per engine ---------------------------

def levenshtein(a: str, b: str) -> int:
    # ponytail: O(len(a)*len(b)) pure Python, ~1 s for two 2,000-char regions;
    # fine for a hand-checked page set, swap in rapidfuzz if the set grows
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def quality(store: Store) -> dict:
    """CER of each engine's raw output against the operator's kept text, over
    the regions of pages marked DONE (done = hand-checked; an untouched region
    on a done page counts as zero errors, because it was read and accepted)."""
    per: dict[str, dict] = {}
    pages_counted = 0
    for rel in store.docs():
        for _, page in store.pages(rel).items():
            if page.get("status") != "done":
                continue
            pages_counted += 1
            for r in page.get("regions") or []:
                eng = r.get("engine_used")
                if not eng or r.get("raw") is None or r.get("type") in ("figure", "ignore"):
                    continue
                ref = (r.get("text") or "").strip()
                hyp = (r.get("raw") or "").strip()
                e = per.setdefault(eng, {"engine": eng, "regions": 0, "edited": 0,
                                         "chars": 0, "errors": 0})
                e["regions"] += 1
                e["chars"] += len(ref)
                d = levenshtein(hyp, ref) if hyp != ref else 0
                e["errors"] += d
                e["edited"] += bool(d)
    for e in per.values():
        e["cer"] = round(e["errors"] / e["chars"], 4) if e["chars"] else None
    return {"pages": pages_counted, "engines": sorted(per.values(), key=lambda e: e["engine"])}
