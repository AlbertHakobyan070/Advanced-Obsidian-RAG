"""One map run, end to end (spec 5, 5.3, 3.6, 3.7): scout, classify, extract (or import),
merge, locate, verify, link, edges, caps, gate, assemble, write.

Writes go only to `<staging>/<book>/<chapter>/` and only through atomic_write. A failed
run (a unit out of retries, a budget, the grounding gate, a bad import) writes nothing
but `runs/<ts>.json`; its checkpoints under `.work/<key>/` stay, so a rerun resumes.
"""
import dataclasses
import functools
import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import psutil
from pydantic import BaseModel, ValidationError

from src.ocr_studio.pages import atomic_write

from .assemble import apply_caps, build_map, merge_nodes, render_canvas
from .config import KMConfig, load_km_config
from .coverage import gaps, heading_spans, need
from .graph import Edge, Node, edge_id, node_id, slug
from .llm_step import Budget, BudgetExceeded, StepRunner, UnitFailed
from .locate import PageText, locate
from .schema import Classification, Extraction, Links, Revision, Revisions, Verdicts
from .scout import Unit, scout
from .steps import (classify, extract_prompt, extract_unit, link, node_target, prompt_versions,
                    trim_nodes, verify_edges, verify_nodes)
from .templates import check_extraction, load_templates


class RunFailed(RuntimeError):
    """The run ends `failed` with this reason (gate, import); not a bug."""


def _dump(obj) -> bytes:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str).encode("utf-8")


class _ExtractCheckpoint(BaseModel):
    served: str | None          # the model that answered, kept so resumed rejects still name it
    extraction: Extraction


class _ExportManifest(BaseModel):
    """DIR/export.json: what the exported prompts were rendered for (I1)."""
    template: str
    classifier_reason: str
    pdf_sha256: str
    pages: list[int]
    units: list[str]
    prompt_versions: dict[str, int]


def _read_manifest(folder: Path, sc, chap, template: str | None) -> _ExportManifest | None:
    """The export's manifest, checked against this run. Absent is allowed only with --template."""
    f = folder / "export.json"
    if not f.is_file():
        if template is None:
            raise RunFailed(f"{f} is missing, so the template the prompts were rendered for is unknown; "
                            "pass --template or re-export with --export-prompts")
        return None
    try:
        m = _ExportManifest.model_validate_json(f.read_bytes())
    except ValidationError as e:
        raise RunFailed(f"export manifest invalid: {f}: {e}") from e
    here = {"pdf_sha256": sc.sha256, "pages": list(chap), "units": [u.id for u in sc.units],
            "template": template if template is not None else m.template}
    for field, want in here.items():
        if getattr(m, field) != want:
            raise RunFailed(f"{f}: {field} {getattr(m, field)!r} does not match this run's {want!r}; "
                            "import replies made for this PDF, page range and template, or re-export")
    return m


def _key(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _outside_vault(path: Path, kcfg: KMConfig) -> Path:
    if Path(path).resolve().is_relative_to(kcfg.vault_root.resolve()):
        raise ValueError(f"{path} is inside the vault ({kcfg.vault_root}); the map harness never writes "
                         "under parser.vault_path (set knowledge_map.staging_dir / the export dir outside it)")
    return Path(path)


def _cached(path: Path, model_cls, fn, write: bool = True):
    """The step's checkpoint if present (no model call), else run the step and write one
    (dry-run and export read checkpoints but never write them)."""
    if path.exists():
        return model_cls.model_validate_json(path.read_text(encoding="utf-8"))
    obj = fn()
    if write:
        atomic_write(path, obj.model_dump_json(indent=2).encode("utf-8"))
    return obj


def _slug_or_none(title: str) -> str | None:
    try:
        return slug(title)
    except ValueError:      # a title with no letters names no node; the caller counts it discarded
        return None


def _read_imports(folder: Path, units, tpl) -> dict[str, tuple[Extraction, str]]:
    """Ruling R1: DIR/extract-<unit>.json per unit, valid as Extraction and the template."""
    out = {}
    for u in units:
        f = folder / f"extract-{u.id}.json"
        if not f.is_file():
            raise RunFailed(f"import file missing: {f}")
        raw = f.read_bytes()
        try:
            ext = Extraction.model_validate_json(raw)
        except ValidationError as e:
            raise RunFailed(f"import file invalid: {f}: {e}") from e
        errs = check_extraction(ext, tpl)
        if errs:
            raise RunFailed(f"import file invalid: {f}: {'; '.join(errs)}")
        out[u.id] = (ext, hashlib.sha256(raw).hexdigest())
    return out


def run_map(cfg, pdf: Path, *, chapter: str | None = None, pages: tuple[int, int] | None = None,
            template: str | None = None, dry_run: bool = False, export_prompts: Path | None = None,
            import_extract: Path | None = None, title: str | None = None, argv: list[str] | None = None,
            client_factory=None, progress: Callable[[str], None] = print) -> dict:
    """`cfg` is the app config, or a KMConfig when `client_factory` supplies the clients."""
    kcfg = cfg if isinstance(cfg, KMConfig) else load_km_config(cfg)
    if isinstance(cfg, KMConfig) and client_factory is None:
        raise ValueError("run_map needs the app config (load_config()) to build LLM clients; "
                         "a bare KMConfig works only with client_factory")
    t0 = time.monotonic()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    sc = scout(Path(pdf), kcfg, chapter=chapter, pages=pages)          # ScoutError raises before any write
    if title:
        sc = dataclasses.replace(sc, book_title=title)
    folder = _outside_vault(kcfg.staging_dir / slug(sc.book_title) / slug(sc.chapter_title), kcfg)
    if export_prompts:
        export_prompts = _outside_vault(export_prompts, kcfg)
    templates = load_templates()
    if template is not None and template not in templates:
        raise ValueError(f"--template {template!r} is not one of: {', '.join(templates)}")

    attempts: list[dict] = []
    rejects: list[dict] = []
    budget = Budget(kcfg.max_llm_calls, kcfg.max_wall_minutes)
    proc, peak = psutil.Process(), [0]
    writes = not (dry_run or export_prompts)
    chap = (sc.page_start, sc.page_end)
    rec = {"version": 1, "argv": argv, "status": None, "reason": None, "started_utc": ts,
           "config": dataclasses.asdict(kcfg),
           "source": {"file": sc.vault_rel, "sha256": sc.sha256, "pages": list(chap),
                      "title": sc.book_title, "chapter": sc.chapter_title},
           "folder": str(folder), "checkpoint": None, "template": None, "classifier_reason": None,
           "gate": None, "counts": {}, "totals": {}, "attempts": attempts, "rejects": rejects}

    def sample():
        peak[0] = max(peak[0], proc.memory_info().rss)

    @functools.cache
    def runner(role):
        return StepRunner(cfg, role, kcfg.roles[role], budget, attempts, kcfg.retries,
                          kcfg.rate_limit_sleep_s, client_factory=client_factory,
                          request_timeout_s=kcfg.request_timeout_s)

    def finish(status, reason=None, files=()):
        sample()
        llm = [a for a in attempts if a["provider"] != "import"]
        rec.update(status=status, reason=reason, totals={
            "calls": len(llm),
            "input_tokens": sum(a["input_tokens"] for a in llm),
            "output_tokens": sum(a["output_tokens"] for a in llm),
            "retries": len(llm) - sum(a["outcome"] == "ok" for a in llm),
            "rate_limit_waits": sum(a["outcome"] == "rate_limited" for a in llm),
            "wall_seconds": round(time.monotonic() - t0, 3),
            "peak_rss_mb": round(peak[0] / 2**20, 1)})
        if writes:
            for name, data in [*files, *([("run.json", rec)] if status == "ok" else []), (f"runs/{ts}.json", rec)]:
                atomic_write(folder / name, _dump(data))
        progress(f"map {status}" + (f": {reason}" if reason else ""))
        return rec

    def reject(n: Node, reason: str, failed=(), detail=None):
        rejects.append({"id": n.id, "title": n.title, "kind": n.kind, "summary": n.summary, "unit": n.unit,
                        "reason": reason, "detail": detail, "served": served.get(n.unit),
                        "quotes": [{"text": q, "pages": list(p)} for q, p in n.located] + list(failed)})

    def verdicts(step, items, fn, cls=Verdicts):
        out = {}
        key = next(iter(cls.model_fields))                           # "verdicts" | "revisions"
        batches = [items[i:i + kcfg.verify_batch] for i in range(0, len(items), kcfg.verify_batch)]
        for k, b in enumerate(batches, 1):
            progress(f"[{k}/{len(batches)}] {step} {b[0].id}")
            name = f"{step}-{hashlib.sha256('|'.join(x.id for x in b).encode()).hexdigest()[:12]}.json"
            v = _cached(work / name, cls, lambda: cls(**{key: list(fn(b).values())}))
            sample()
            out.update((x.id, x) for x in getattr(v, key))
        return out

    try:
        manifest = _read_manifest(Path(import_extract), sc, chap, template) if import_extract else None
        if manifest is not None and manifest.template not in templates:
            raise RunFailed(f"export.json template {manifest.template!r} is not one of: {', '.join(templates)}")
        if template is not None:
            tpl, reason = templates[template], "set with --template"
        elif manifest is not None:         # the export's classification, never a second classify
            tpl, reason = templates[manifest.template], manifest.classifier_reason
        else:
            progress("[1/1] classify")
            cr = kcfg.roles["classify"]
            ck = _key([sc.sha256, list(chap), prompt_versions()["map_classify"], cr.provider, cr.model, cr.fallbacks])
            c = _cached(folder / ".work" / f"classify-{ck[:16]}.json", Classification,
                        lambda: classify(runner("classify"), sc, templates), write=writes)
            sample()
            tpl, reason = templates[c.template], c.reason
        rec.update(template=tpl.name, classifier_reason=reason)
        chapter_chars = sum(u.chars for u in sc.units)
        targets = {u.id: node_target(u.chars, chapter_chars, kcfg.node_cap) for u in sc.units}

        if dry_run:
            rec["plan"] = [{"unit": u.id, "section": u.section_id, "pages": [u.page_start, u.page_end],
                            "chars": u.chars, "node_target": targets[u.id]} for u in sc.units]
            progress(f"plan: {sc.book_title} / {sc.chapter_title}, pp. {chap[0]}-{chap[1]}, "
                     f"template {tpl.name} ({reason}), staging {folder}")
            progress(f"headings ({len(sc.headings)}): " + "; ".join(f"{h.title} (p. {h.page})" for h in sc.headings))
            for p in rec["plan"]:
                progress(f"  {p['unit']}  pp. {p['pages'][0]}-{p['pages'][1]}  {p['chars']} chars  "
                         f"<= {p['node_target']} nodes")
            return finish("dry_run")
        if export_prompts:
            for i, u in enumerate(sc.units, 1):
                system, user = extract_prompt(u, tpl, targets[u.id], quote_words=kcfg.quote_words)
                atomic_write(Path(export_prompts) / f"extract-{u.id}.prompt.json",
                             _dump({"system": system, "user": user, "unit": u.id, "section": u.section_id}))
                progress(f"[{i}/{len(sc.units)}] export {u.id}")
            atomic_write(Path(export_prompts) / "export.json", _ExportManifest(
                template=tpl.name, classifier_reason=reason, pdf_sha256=sc.sha256, pages=list(chap),
                units=[u.id for u in sc.units], prompt_versions=prompt_versions()).model_dump_json(indent=2).encode())
            return finish("exported")

        imported = _read_imports(Path(import_extract), sc.units, tpl) if import_extract else {}
        key_src = {"pdf": sc.sha256, "pages": list(chap), "template": tpl.name, "prompts": prompt_versions(),
                   "roles": {r: [s.provider, s.model, s.fallbacks] for r, s in sorted(kcfg.roles.items())},
                   "node_cap": kcfg.node_cap, "max_unit_chars": kcfg.max_unit_chars,
                   "window_pages": kcfg.window_pages, "max_span_pages": kcfg.max_span_pages,
                   "verify_batch": kcfg.verify_batch, "quote_words": kcfg.quote_words,
                   "coverage": [kcfg.heading_min_ratio, kcfg.min_section_chars, kcfg.chars_per_node]}
        if imported:
            key_src["imports"] = {uid: sha for uid, (_, sha) in imported.items()}
        key = _key(key_src)
        rec["checkpoint"] = key
        work = folder / ".work" / key[:16]

        # Extract: node ids from kind + title; edges name endpoints by title within their reply.
        nodes, raw_edges, discarded, served = [], [], 0, {}

        def take(ext: Extraction, u, unit_id: str) -> list[Node]:
            nonlocal discarded
            titles = {slug(n.title) for n in ext.nodes}
            for e in ext.edges:
                s, t = _slug_or_none(e.source), _slug_or_none(e.target)
                if s in titles and t in titles:
                    raw_edges.append((s, t, e.type, e.quote))
                else:
                    discarded += 1
            return [Node(node_id(n.kind, n.title), n.kind, n.title, n.summary, u.section_id, unit_id,
                         n.importance, list(n.quotes)) for n in ext.nodes]

        def extract_fresh(u, short=None):
            ext = extract_unit(runner("extract"), u, tpl, sum(g.target for g in short) if short
                               else targets[u.id], quote_words=kcfg.quote_words,
                               headings=[g.line() for g in short] if short else None)
            return _ExtractCheckpoint(served=attempts[-1]["served"], extraction=ext)   # the ok attempt is last

        for i, u in enumerate(sc.units, 1):
            progress(f"[{i}/{len(sc.units)}] extract {u.id}")
            if u.id in imported:
                ext, sha = imported[u.id]
                attempts.append({"role": "extract", "unit": u.id, "provider": "import", "requested": None,
                                 "served": None, "input_tokens": 0, "output_tokens": 0, "seconds": 0.0,
                                 "outcome": "ok", "sha256": sha})
                served[u.id] = None
            else:
                ck = _cached(work / f"extract-{u.id}.json", _ExtractCheckpoint, lambda: extract_fresh(u))
                ext, served[u.id] = ck.extraction, ck.served
            sample()
            nodes += take(ext, u, u.id)

        text = PageText(sc.pages)
        secs = {s.id: (s.page_start, s.page_end) for s in sc.sections}

        def ground(cands: list[Node]) -> list[Node]:
            """Locate every quote (section pages first, then the chapter), verify, trim, re-verify.
            Returns the verified nodes; every other candidate is rejected with its reason."""
            located = []
            for n in cands:
                failed = []
                for q in n.quotes:
                    r = locate(q, text, secs[n.section], chap, words=kcfg.quote_words)
                    if r.status == "located":
                        n.located.append((q, r.pages))
                    else:
                        failed.append({"text": q, "status": r.status})
                if not n.located:
                    reject(n, failed[0]["status"], failed)
                elif len({p for _, ps in n.located for p in ps}) > kcfg.max_span_pages:
                    reject(n, "too_wide", failed)
                else:
                    located.append(n)

            nv = verdicts("verify-nodes", located, lambda b: verify_nodes(runner("verify"), b, text))
            # A summary that claims more than its located quotes is cut down by the verifier and
            # checked again by a different model (the reverify role), so a node with good quotes
            # is not lost to an overclaiming summary and no model grades its own rewrite.
            bad = [n for n in located if not nv[n.id].supported]
            tv = verdicts("trim-nodes", bad, lambda b: {i: Revision(id=i, summary=s) for i, s in trim_nodes(
                runner("verify"), [(n, nv[n.id].reason) for n in b], text).items()}, cls=Revisions)
            for n in bad:
                if tv[n.id].summary is not None:
                    n.summary_revised_from, n.summary = n.summary, tv[n.id].summary
            rv = verdicts("reverify-nodes", [n for n in bad if n.summary_revised_from is not None],
                          lambda b: verify_nodes(runner("reverify"), b, text))
            verified = []
            for n in located:
                if nv[n.id].supported or (n.id in rv and rv[n.id].supported):
                    n.status = "verified"
                    verified.append(n)
                elif n.id in rv:
                    n.summary, n.summary_revised_from = n.summary_revised_from, None    # reject the extracted one
                    reject(n, "unsupported", detail=f"{nv[n.id].reason}; after trim: {rv[n.id].reason}")
                else:
                    reject(n, "unsupported", detail=f"{nv[n.id].reason}; trim: no supported summary")
            return verified

        merged = merge_nodes(nodes)
        verified = ground(merged)

        # Coverage: a heading with fewer kept nodes than its length needs gets one more extract
        # call over the pages under it, naming the nodes already there and asking for the rest;
        # the new nodes pass the same gate. An imported run only reports its gaps, so a
        # Claude-extracted map never mixes in extract-model nodes.
        spans, unplaced = heading_spans(sc.headings, text, kcfg.min_section_chars)
        short = gaps(spans, verified, text, kcfg.chars_per_node) if kcfg.chars_per_node > 0 else []
        if short and not imported:
            fresh = []
            for i, g in enumerate(short, 1):
                u = next(u for u in sc.units if u.page_start <= g.span.page <= u.page_end)
                hi = text._page_at(g.span.end - 1)
                body = "\n".join(sc.pages[p] for p in range(g.span.page, hi + 1))
                part = Unit(f"{slug(g.span.title)[:40]}-p{g.span.page}", u.section_id, g.span.page, hi,
                            body, len(body))
                progress(f"[{i}/{len(short)}] gap-fill {part.id}: {len(g.have)} of {g.need} nodes")
                ck = _cached(work / f"gapfill-{part.id}-{_key(g.line())[:8]}.json",
                             _ExtractCheckpoint, lambda: extract_fresh(part, [g]))
                served[f"gapfill-{part.id}"] = ck.served
                sample()
                fresh += take(ck.extraction, part, f"gapfill-{part.id}")
            known = {slug(n.title) for n in merged}
            new = [n for n in merge_nodes(fresh) if slug(n.title) not in known]   # already tried once
            merged += new
            verified += ground(new)

        def still_short(nodes: list[Node]) -> list[str]:
            left = gaps(spans, nodes, text, kcfg.chars_per_node) if kcfg.chars_per_node > 0 else []
            return [f"{g.span.title} (p. {g.span.page}): {len(g.have)} of {g.need} nodes"
                    + ("" if g.named else ", none named after the heading") for g in left]

        # The cap never cuts below what the headings need (their length, plus one node for each
        # heading whose own idea was missing), so it cannot undo a gap-fill; node_cap stays the floor.
        cap = max(kcfg.node_cap, sum(need(s, kcfg.chars_per_node) for s in spans) + sum(not g.named for g in short)
                  if kcfg.chars_per_node > 0 else 0)
        rec["coverage"] = {"headings": len(spans), "short_before_gapfill": len(short), "node_cap": cap,
                           "short": still_short(verified), "unplaced": unplaced}   # redone on the kept nodes
        by_slug = {slug(n.title): n.id for n in merged}

        # Gate (ruling R6): grounded / candidates is final here, so a failing run stops before link.
        ec = {"stated": 0, "inferred": 0, "downgraded": 0, "duplicates": 0, "discarded": discarded}
        rec["counts"] = {"candidates": len(merged), "kept": 0,
                         "revised": sum(n.summary_revised_from is not None for n in verified),
                         "dropped": dict(Counter(r["reason"] for r in rejects)), "edges": ec}
        ratio = len(verified) / len(merged) if merged else 0.0
        rec["gate"] = {"grounded": len(verified), "candidates": len(merged), "ratio": round(ratio, 4),
                       "min_grounded_ratio": kcfg.min_grounded_ratio,
                       "passed": bool(merged) and ratio >= kcfg.min_grounded_ratio}
        if not merged:
            raise RunFailed("no candidates")
        if not rec["gate"]["passed"]:
            raise RunFailed(f"grounded ratio {ratio:.2f} ({len(verified)}/{len(merged)}) is below "
                            f"min_grounded_ratio {kcfg.min_grounded_ratio}")

        links = Links(edges=[])
        if len(verified) >= 2:
            progress("[1/1] link")
            links = _cached(work / "link.json", Links,
                            lambda: link(runner("link"), verified, tpl, quote_words=kcfg.quote_words))
            ec["discarded"] += links.dropped
            sample()

        # Edges between surviving nodes; a duplicate id keeps the quoted copy. Quoted edges are
        # located and verified; either failure downgrades the edge to inferred.
        alive = {n.id: n for n in verified}
        by_id: dict[str, Edge] = {}
        for s, t, typ, q in ([(by_slug[s], by_slug[t], typ, q) for s, t, typ, q in raw_edges]
                             + [(e.source, e.target, e.type, e.quote) for e in links.edges]):
            if s not in alive or t not in alive or s == t:
                ec["discarded"] += 1
                continue
            eid = edge_id(s, t, typ)
            if eid in by_id:
                ec["duplicates"] += 1
                if q and not by_id[eid].quote:
                    by_id[eid].quote, by_id[eid].status = q, "stated"
                continue
            by_id[eid] = Edge(eid, s, t, typ, q, status="stated" if q else "inferred")
        edges = list(by_id.values())
        quoted = []
        for e in edges:
            if e.quote:
                r = locate(e.quote, text, chap, chap, words=kcfg.quote_words)
                if r.status == "located":
                    e.pages = r.pages
                    quoted.append(e)
                else:
                    e.quote, e.status = None, "inferred"
                    ec["downgraded"] += 1
        ev = verdicts("verify-edges", quoted, lambda b: verify_edges(runner("verify"), b, alive, text))
        for e in quoted:
            if not ev[e.id].supported:
                e.quote, e.pages, e.status = None, (), "inferred"
                ec["downgraded"] += 1

        kept, kept_edges, cap_rejects = apply_caps(verified, edges, dataclasses.replace(kcfg, node_cap=cap))
        rejects += cap_rejects
        rec["counts"].update(kept=len(kept), dropped=dict(Counter(r["reason"] for r in rejects)))
        rec["coverage"]["short"] = still_short(kept)            # the cap can undo a gap-fill
        ec.update(stated=sum(e.status == "stated" for e in kept_edges),
                  inferred=sum(e.status == "inferred" for e in kept_edges))
        graph = build_map(sc, tpl, reason, kept, kept_edges, kcfg.generated_markers)
        canvas = render_canvas(graph, tpl.layer_edges)
        return finish("ok", files=[("map.json", graph), (graph["canvas"], canvas)])
    except UnitFailed as e:
        return finish("failed", f"unit {e.unit!r} failed: {e}")
    except BudgetExceeded as e:
        return finish("failed", e.reason)
    except RunFailed as e:
        return finish("failed", str(e))
    except Exception as e:      # a bug still raises, but leaves its run record behind first
        finish("failed", f"crash: {e!r}")
        raise
