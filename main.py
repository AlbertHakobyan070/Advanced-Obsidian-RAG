#!/usr/bin/env python3
"""
main.py — CLI entry point for Noetrix.

Build the corpus
    python -m src.ingestion.obsidian_parser <vault> -o data/chunks.jsonl
                                               # whole-vault markdown parse
    python main.py index                       # FULL rebuild from chunks.jsonl only
    python main.py index --append FILE.jsonl   # add one more chunk file
    python main.py stamp                       # verify, then record, which embedder built
                                               # the collection (:8051 refuses a mismatch)
    python main.py index --rebuild-into NAME [--embedding PROVIDER]
                                               # re-embed into a NEW collection, resumably;
                                               # the active one is never written
    python main.py graph-embed                 # rebuild <collection>__graph from scratch

Ingest lanes (each writes its own *_chunks.jsonl; then `index --append` it)
    ingest-pdfs | ingest-notebooks | ingest-code | ingest-canvas | ingest-md
    fetch-web | convert-files                  # stage web pages / office files
                                               # into <inbox>/_converted first

Ask and measure
    python main.py query "What is ARIMA?"      # one-shot question
    python main.py chat                        # interactive REPL
    python main.py bench run --configs ladder  # the v2 eval: labelled questions scored per
                                               # named pipeline configuration; also
                                               # bench validate | split | report | sample
    python main.py serve                       # launch the Streamlit app

`index` WITHOUT --append deletes the dense collection and rebuilds both
indexes from chunks.jsonl alone, so every appended lane (PDFs, notebooks,
code, canvases) drops out of both — see Embedder.build_indexes. The console
runs these same commands as subprocesses (manage_api._build_argv), including
that full rebuild as the `index_rebuild` job.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make 'src' importable when run as `python main.py`
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval.bench.cli import add_bench_parser
from src.utils.config_loader import load_config
from src.utils.console import force_utf8_console
from src.utils.logger import configure_logging, get_logger

# This import-time call configures logging with the DEFAULTS, which turns
# every _bootstrap_logging(cfg) below into a no-op: config.yaml's
# logging.level / logging.file are not applied. See src/utils/logger.py.
log = get_logger(__name__)


def _bootstrap_logging(cfg):
    configure_logging(
        level=cfg.get("logging.level", "INFO"),
        log_file=cfg.path("logging.file") if cfg.get("logging.file") else None,
        console=cfg.get("logging.console", True),
    )


def _cmd_rebuild_into(cfg, args):
    from src.embeddings.embedder import Embedder
    from src.embeddings.reembed import ReembedError, rebuild_into
    from src.embeddings.registry import resolve_embedding_spec
    from src.embeddings.sidecar import StampError

    try:
        spec = resolve_embedding_spec(cfg, args.embedding)
        emb = Embedder.from_config(cfg, spec=spec)
        print(f"Re-embedding {cfg.get('paths.collection_name')!r} into {args.rebuild_into!r} "
              f"with {spec.label()} (doc_prefix {spec.doc_prefix!r}, max_seq_length "
              f"{spec.max_seq_length})")
        sidecar = rebuild_into(cfg, args.rebuild_into, emb, limit=args.limit)
    except (ReembedError, StampError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

    build = sidecar["build"]
    print(f"\n{'OK' if sidecar['status'] == 'complete' else 'PARTIAL'} {args.rebuild_into!r}: "
          f"{sidecar['count']} of {sidecar['expected']} chunks, {sidecar['dimension']}-d, "
          f"status {sidecar['status']} ({build['embedded']} embedded this run, "
          f"{build['chunks_per_s']} chunks/s; sample cosine {sidecar['verified']['min_cosine']:.5f})")
    if sidecar["status"] == "building":
        print("   --limit run: left 'building' on purpose, :8051 will refuse to serve it. "
              "Delete it when done.")
        return
    print("   HyPE: NOT re-embedded (deferred); keep retrieval.hype.enabled off if you activate it.")
    print(f"   Score it:  python main.py bench run --collection {args.rebuild_into} ...")
    print(f"   Activate:  paths.collection_name: {args.rebuild_into}  and  "
          f"embedding.provider: {sidecar['provider']}, then restart :8051")


def cmd_graph_embed(args):
    cfg = load_config(args.config)
    _bootstrap_logging(cfg)
    from src.embeddings.embedder import Embedder
    from src.embeddings.graph_twin import build_twin, graph_collection_name, graph_spec
    from src.embeddings.reembed import ReembedError
    from src.embeddings.sidecar import StampError

    name = graph_collection_name(cfg)
    try:
        spec = graph_spec(cfg)
        if spec is None:
            raise ValueError("graph.embedding_provider is null: graph mode seeds from the main "
                             "collection with its own embedder, so there is no graph index to "
                             "build. Set it to an embedding.providers name first")
        if spec.api_key_env and not spec.api_key_optional:
            # Embedder.from_config reads this key with Config.require_secret, whose
            # plain RuntimeError is too broad to catch around the build (torch's
            # out-of-memory error is one too), so the key is asked for here, alone.
            try:
                cfg.require_secret(spec.api_key_env)
            except RuntimeError as e:
                raise ValueError(f"{spec.label()} needs its API key: {e}") from e
        emb = Embedder.from_config(cfg, spec=spec)
        print(f"Rebuilding {name!r} from scratch: the {cfg.get('graph.file_type', 'canvas')!r} "
              f"chunks of {cfg.get('paths.collection_name')!r}, embedded by {spec.label()} "
              f"(doc_prefix {spec.doc_prefix!r}, max_seq_length {spec.max_seq_length})")
        sidecar = build_twin(cfg, emb, limit=args.limit)
    except (ReembedError, StampError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

    build = sidecar["build"]
    print(f"\n{'OK' if sidecar['status'] == 'complete' else 'PARTIAL'} {name!r}: "
          f"{sidecar['count']} of {sidecar['expected']} chunks, {sidecar['dimension']}-d, "
          f"status {sidecar['status']} ({build['embedded']} embedded, "
          f"{build['chunks_per_s']} chunks/s; sample cosine {sidecar['verified']['min_cosine']:.5f})")
    if sidecar["status"] == "building":
        print("   --limit run: left 'building' on purpose; graph-mode query seeding refuses it. "
              "Run graph-embed without --limit to replace it.")
        return
    print("   Restart :8051 so graph mode seeds from it.")


def cmd_index(args):
    cfg = load_config(args.config)
    _bootstrap_logging(cfg)
    rebuild_only = {"--embedding": args.embedding, "--limit": args.limit}
    if args.rebuild_into and args.append:
        print("ERROR: --rebuild-into and --append cannot be combined", file=sys.stderr)
        sys.exit(2)
    if not args.rebuild_into and any(v is not None for v in rebuild_only.values()):
        print(f"ERROR: {' / '.join(rebuild_only)} only apply with --rebuild-into", file=sys.stderr)
        sys.exit(2)
    if args.rebuild_into:
        return _cmd_rebuild_into(cfg, args)
    from src.embeddings.embedder import Embedder

    emb = Embedder.from_config(cfg)
    if getattr(args, "append", None):
        stats = emb.append_indexes(Path(args.append))
        if stats["appended"] == 0:
            # An append that appends nothing is always an upstream mistake
            # (empty/misfiltered ingest). Fail loudly so job queues show it red.
            print(f"\n❌ {args.append} contains 0 chunks — nothing appended. "
                  f"The ingest that produced it found no documents; "
                  f"check its log/filters.")
            sys.exit(3)
        print(f"\n✅ Appended {stats['appended']} chunks to existing index.")
    else:
        stats = emb.build_indexes()
        print(f"\n✅ Indexed {stats['chunks']} chunks.")
        print(f"   Dense:  {cfg.path('paths.chroma_dir')}")
        print(f"   Sparse: {cfg.path('paths.bm25_index')}")


def cmd_ingest_pdfs(args):
    try:
        from rapidocr_onnxruntime import RapidOCR
        if not hasattr(RapidOCR, 'text_detector'):
            @property
            def text_detector_patch(self):
                # Map what PyMuPDF looks for to what RapidOCR actually uses
                return getattr(self, 'text_detect', None)
            RapidOCR.text_detector = text_detector_patch
    except ImportError:
        pass
        
    cfg = load_config(args.config)
    _bootstrap_logging(cfg)
    from src.ingestion.pdf_loader import PDFLoader, load_skip_set

    loader = PDFLoader.from_config(cfg)
    if args.vault:
        loader.vault_path = Path(args.vault)
    if args.only_books:
        loader.only_book_folders = True
    if args.skip_books:
        loader.skip_books = True
    if args.include_path:
        loader.include_path = args.include_path.lower()
    if args.exclude_path:
        loader.exclude_path = args.exclude_path.lower()
    if getattr(args, "include_files", None):
        loader.include_files = {f.strip().lower()
                                for f in args.include_files.split(",") if f.strip()}
    if args.output:
        loader.output_file = Path(args.output)
    if args.max_pages:
        loader.max_pages_per_pdf = args.max_pages
    if getattr(args, "pages", None):
        from src.ingestion.pdf_loader import parse_page_spec
        try:
            parse_page_spec(args.pages)          # validate format up front
        except ValueError as e:
            print(f"ERROR: --pages: {e}")
            sys.exit(2)
        loader.page_spec = args.pages
    if args.ocr_engine:
        loader.set_ocr_engine(args.ocr_engine)
    if args.no_ocr:
        loader.ocr_enabled = False
        loader.ocr_engine = None
    if args.skip_list:
        loader.skip_set = load_skip_set(Path(args.skip_list))
    if args.no_images:
        loader.extract_images = False
    if args.archive_processed:
        loader.archive_processed = True
    if getattr(args, "chunking", None):
        loader.chunking = args.chunking
    if args.force_domain:
        loader.force_domain = args.force_domain.strip().lower()
    if args.force_tags:
        loader.force_tags = [t.strip().lstrip("#").lower()
                             for t in args.force_tags.split(",") if t.strip()]
    out = loader.ingest_vault()
    s = loader.stats
    if s["pdfs_found"] == 0:
        # 3558 PDFs seen / 0 kept has silently produced "done" no-op job chains
        # before (only_books vs inbox). A zero-document ingest is never success.
        print("\n❌ No PDFs matched the discovery filters — NOTHING was ingested.")
        print("   See the 'Where they went' warning above for which filter bit.")
        sys.exit(3)
    if s["chunks_total"] == 0:
        print(f"\n❌ {s['pdfs_found']} PDF(s) matched but produced 0 chunks "
              f"({s['pdfs_failed']} failed) — nothing to index.")
        sys.exit(3)
    print(f"\n✅ PDF chunks written to {out}")
    print(f"   Now run: python main.py index --append {out}")


def cmd_ingest_notebooks(args):
    cfg = load_config(args.config)
    _bootstrap_logging(cfg)
    from src.ingestion.ipynb_loader import NotebookLoader

    loader = NotebookLoader.from_config(cfg)
    if args.vault:
        loader.vault_path = Path(args.vault)
    if args.output:
        loader.output_file = Path(args.output)
    if args.no_outputs:
        loader.include_outputs = False
    if args.save_figures:
        loader.save_figures = True
    if args.exts:
        loader.exts = {e.strip().lower() for e in args.exts.split(",") if e.strip()}
    if getattr(args, "include_path", None):
        loader.include_path = args.include_path
    if getattr(args, "include_files", None):
        loader.include_files = {f.strip() for f in args.include_files.split(",") if f.strip()}
    if getattr(args, "force_domain", None):
        loader.force_domain = args.force_domain.strip().lower()
    if getattr(args, "force_tags", None):
        loader.force_tags = [t.strip().lstrip("#").lower()
                             for t in args.force_tags.split(",") if t.strip()]
    out = loader.ingest_vault()
    print(f"\n✅ Notebook/code chunks written to {out}")
    print(f"   Now run: python main.py index --append {out}")


def cmd_ingest_code(args):
    cfg = load_config(args.config)
    _bootstrap_logging(cfg)
    from src.ingestion.code_loader import CodeLoader

    loader = CodeLoader.from_config(cfg)
    if args.vault:
        loader.vault_path = Path(args.vault)
    if args.output:
        loader.output_file = Path(args.output)
    if args.include_path:
        loader.include_path = args.include_path.lower()
    if args.exclude_path:
        loader.exclude_path = args.exclude_path.lower()
    if getattr(args, "include_files", None):
        loader.include_files = {f.strip().lower()
                                for f in args.include_files.split(",") if f.strip()}
    if args.exts:
        loader.exts = {e.strip().lower() for e in args.exts.split(",") if e.strip()}
    if getattr(args, "force_domain", None):
        loader.force_domain = args.force_domain.strip().lower()
    if getattr(args, "force_tags", None):
        loader.force_tags = [t.strip().lstrip("#").lower()
                             for t in args.force_tags.split(",") if t.strip()]
    out = loader.ingest_vault()
    if loader.stats["chunks_total"] == 0:
        print("\n❌ 0 code chunks produced — no matching files in scope "
              "(agent-project roots are skipped unless --include-path names them).")
        sys.exit(3)
    print(f"\n✅ Code chunks written to {out}")
    print(f"   Now run: python main.py index --append {out}")


def cmd_ingest_canvas(args):
    cfg = load_config(args.config)
    _bootstrap_logging(cfg)
    from src.ingestion.canvas_loader import CanvasLoader

    loader = CanvasLoader.from_config(cfg)
    if args.vault:
        loader.vault_path = Path(args.vault)
    if args.output:
        loader.output_file = Path(args.output)
    if args.include_path:
        loader.include_path = args.include_path.lower()
    if getattr(args, "max_chunk_size", None):
        loader.max_chunk = int(args.max_chunk_size)
    if getattr(args, "chunking", None):
        loader.chunking = args.chunking
    if getattr(args, "context_depth", None) is not None:
        loader.context_depth = int(args.context_depth)
    if getattr(args, "force_domain", None):
        loader.force_domain = args.force_domain.strip().lower()
    if getattr(args, "force_tags", None):
        loader.force_tags = [t.strip().lstrip("#").lower()
                             for t in args.force_tags.split(",") if t.strip()]
    out = loader.ingest_vault()
    if loader.stats["chunks_total"] == 0:
        print("\n❌ 0 canvas chunks produced — no matching files in scope.")
        sys.exit(3)
    print(f"\n✅ Canvas chunks written to {out}")
    print(f"   Now run: python main.py index --append {out}")


def cmd_ocr_studio_ingest(args):
    """OCR Studio pages of ONE document -> its own chunk file (never a lane's
    canonical one). The console queues `index --append` of it next."""
    cfg = load_config(args.config)
    _bootstrap_logging(cfg)
    from src.ocr_studio.ingest import build_chunks, write_chunks

    vault = Path(cfg.get("pdf.vault_path") or cfg.get("parser.vault_path"))
    chunks, info = build_chunks(cfg, vault, args.doc, args.pages)
    print(f"{args.doc}: {info['pages_selected']} of {info['pages_saved']} saved page(s) "
          f"selected ({args.pages}), {info['pages_with_text']} with text, "
          f"{info['chunks']} chunk(s)")
    if not chunks:
        print("\n❌ 0 chunks produced — mark pages done (or ingest --pages all) and "
              "make sure their regions hold text.")
        sys.exit(3)
    out = write_chunks(chunks, Path(args.output))
    print(f"\n✅ OCR Studio chunks written to {out}")


def cmd_ingest_md(args):
    """Scoped markdown parse (inbox md lane). Unlike the vault-wide parser
    run, this REQUIRES an include filter + its own output so the canonical
    chunks.jsonl can never be clobbered. E2 parent sidecars are skipped for
    scoped runs (parents_md.jsonl is a whole-vault artifact)."""
    cfg = load_config(args.config)
    _bootstrap_logging(cfg)
    from src.ingestion.obsidian_parser import ObsidianParser

    vault = args.vault or cfg.get("parser.vault_path") or cfg.get("pdf.vault_path")
    parser = ObsidianParser(vault, config={
        "chunking": args.chunking or cfg.get("parser.chunking", "heading"),
        "include_path": args.include_path,
        "skip_roots": [],                      # include_path is the scope here
    })
    docs = parser.parse_all()
    if not docs:
        print("\n❌ 0 markdown chunks produced — no matching .md files in scope.")
        sys.exit(3)
    fd = (args.force_domain or "").strip().lower() or None
    ft = [t.strip().lstrip("#").lower()
          for t in (args.force_tags or "").split(",") if t.strip()]
    if fd or ft:
        from src.ingestion.obsidian_parser import apply_forced_meta
        for d in docs:
            apply_forced_meta(d.metadata, fd, ft)
    parser.export_jsonl(docs, args.output)
    print(f"\n✅ {len(docs)} markdown chunks written to {args.output}")
    print(f"   Now run: python main.py index --append {args.output}")


def cmd_fetch_web(args):
    cfg = load_config(args.config)
    _bootstrap_logging(cfg)
    from src.ingestion.web_import import fetch_urls

    urls = [u.strip() for u in args.urls.split(",") if u.strip()]
    out_dir = Path(args.out_dir) if args.out_dir else _inbox_dir(cfg) / "_converted"
    res = fetch_urls(urls, out_dir, backend=args.backend, fmt=args.format)
    ok = sum(1 for r in res if r["ok"])
    for r in res:
        print(("✅" if r["ok"] else "❌") + f" {r.get('url')}: "
              f"{r.get('file') or r.get('error')}")
    print(f"\n{ok}/{len(res)} page(s) fetched into {out_dir}")
    if ok < len(res):
        sys.exit(3)


def cmd_convert_files(args):
    cfg = load_config(args.config)
    _bootstrap_logging(cfg)
    from src.ingestion.web_import import convert_files

    inbox = _inbox_dir(cfg)
    out_dir = Path(args.out_dir) if args.out_dir else inbox / "_converted"
    files = [f.strip() for f in args.files.split(",") if f.strip()]
    res = convert_files(files, inbox, out_dir, ocr_pages=args.ocr_pages or "",
                        ocr_language=cfg.get("pdf.ocr_language", "eng"))
    ok = sum(1 for r in res if r["ok"])
    for r in res:
        print(("✅" if r["ok"] else "❌") + f" {r.get('file')}: "
              f"{r.get('output') or r.get('error')}")
    print(f"\n{ok}/{len(res)} file(s) converted into {out_dir}")
    if ok < len(res):
        sys.exit(3)


def _inbox_dir(cfg) -> Path:
    vault = Path(cfg.get("pdf.vault_path") or cfg.get("parser.vault_path"))
    return vault / (cfg.get("webui.inbox_dir") or "Inbox")


def _render_answer(answer) -> None:
    print("\n" + "=" * 70)
    print(answer.text)
    print("=" * 70)
    print(f"Confidence: {answer.confidence}")
    if getattr(answer, "retrieval", None):
        r = answer.retrieval
        line = (f"Retrieval:  preset={r.get('preset') or 'default'}"
                f" · k={r.get('rerank_top_k')}"
                f" · hyde={'on' if r.get('hyde_used') else 'off'}"
                f" · candidates={r.get('candidates')}")
        if r.get("scope"):
            line += f" · scope={','.join(r['scope'])}"
        print(line)
    if answer.citations:
        print("\nSources:")
        for c in answer.citations:
            mark = ""
            if c.supported is True:
                mark = " ✓"
            elif c.supported is False:
                mark = f" ⚠ unsupported ({c.note})"
            print(f"  [{c.number}] {c.source_label}{mark}")
    if answer.verification:
        print(f"\nCitation audit: {answer.verification.get('overall', 'N/A')}")
    if answer.usage:
        print(f"\nTokens: {answer.usage}")


def cmd_query(args):
    cfg = load_config(args.config)
    _bootstrap_logging(cfg)
    from src.pipeline import RAGPipeline

    rag = RAGPipeline.from_config(cfg)
    try:
        answer = rag.query(args.question, preset=args.preset, top_k=args.top_k,
                           max_tokens=args.max_tokens)
    except KeyError as e:
        print(f"ERROR: {e.args[0]}")
        sys.exit(2)
    _render_answer(answer)


def cmd_chat(args):
    cfg = load_config(args.config)
    _bootstrap_logging(cfg)
    from src.pipeline import RAGPipeline

    rag = RAGPipeline.from_config(cfg)
    print("Noetrix — interactive mode. Type 'exit' or Ctrl-C to quit.\n")
    try:
        while True:
            q = input("❓ ").strip()
            if q.lower() in {"exit", "quit", ""}:
                break
            answer = rag.query(q)
            _render_answer(answer)
            print()
    except (KeyboardInterrupt, EOFError):
        print("\nbye.")


def cmd_stamp(args):
    cfg = load_config(args.config)
    _bootstrap_logging(cfg)
    from src.embeddings.embedder import Embedder
    from src.embeddings.sidecar import (
        EmbeddingMismatchError, StampError, read_sidecar, sidecar_label, sidecar_path,
        stamp_collection)

    emb = Embedder.from_config(cfg)
    try:
        sidecar, outcome = stamp_collection(cfg, emb, args.collection, force=args.force)
    except (StampError, EmbeddingMismatchError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

    name, chroma_dir = sidecar["collection"], cfg.path("paths.chroma_dir")
    label = sidecar_label(sidecar)
    if outcome == "already":
        print(f"✅ {name!r} already has a matching fingerprint: {label}, "
              f"{sidecar['dimension']}-d. Nothing to do.")
    else:
        verified = sidecar["verified"]
        print(f"Verifying {name!r} ({sidecar['count']} chunks) against {label} ...")
        print(f"  {verified['sample']} sampled chunks re-embedded: min cosine "
              f"{verified['min_cosine']:.5f} (threshold {verified['threshold']})")
        print(f"✅ Stamped {name!r}: {label}, {sidecar['dimension']}-d, {sidecar['count']} chunks")
        print(f"   {sidecar_path(chroma_dir, name)}")
    # The HyPE twin is a collection of its own: say so when it exists unstamped.
    twin = sidecar.get("hype_collection")
    if sidecar["role"] == "chunks" and twin:
        from src.utils.chroma_client import persistent_client
        existing = {c.name for c in persistent_client(chroma_dir).list_collections()}
        if twin in existing and read_sidecar(chroma_dir, twin) is None:
            print(f"   HyPE collection {twin!r} is not stamped: "
                  f"`rag stamp --collection {twin}` verifies it too.")


def cmd_serve(args):
    import subprocess

    app_path = Path(__file__).resolve().parent / "app.py"
    print(f"Launching Streamlit app: {app_path}")
    subprocess.run(["streamlit", "run", str(app_path)])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Grounded, cited question-answering over your own documents.")
    p.add_argument("--config", default=None, help="Path to config.yaml")
    sub = p.add_subparsers(dest="command", required=True)

    idx = sub.add_parser("index", help="Build dense+sparse indexes from chunks.jsonl")
    idx.add_argument("--append", default=None, metavar="JSONL",
                     help="Append chunks from this file to the existing index "
                          "(e.g. data/pdf_chunks.jsonl) instead of rebuilding")
    idx.add_argument("--rebuild-into", default=None, metavar="NAME",
                     help="Re-embed the active collection's corpus into a NEW collection NAME "
                          "(ids from the active collection, text from the chunk JSONLs). "
                          "Resumable: run the same command again after an interruption. "
                          "The active collection is never written")
    idx.add_argument("--embedding", default=None, metavar="PROVIDER",
                     help="With --rebuild-into: the embedder to build with, a name from "
                          "embedding.providers (or local); default embedding.provider")
    idx.add_argument("--limit", type=int, default=None, metavar="N",
                     help="With --rebuild-into: embed only N ids, for a smoke test of a "
                          "throwaway collection. The result stays 'building' (unservable)")
    idx.set_defaults(func=cmd_index)

    pdf = sub.add_parser("ingest-pdfs", help="Extract vault PDFs into data/pdf_chunks.jsonl")
    pdf.add_argument("--only-books", action="store_true",
                     help="Only process PDFs inside Books/ folders")
    pdf.add_argument("--skip-books", action="store_true",
                     help="Process everything EXCEPT Books/ folders (for the lecture-notes pass; "
                          "guarantees already-ingested books aren't re-processed)")
    pdf.add_argument("--include-path", default=None, metavar="SUBSTR",
                     help="Only process PDFs whose path contains SUBSTR, e.g. \"Current Courses\"")
    pdf.add_argument("--exclude-path", default=None, metavar="SUBSTR",
                     help="Skip PDFs whose path contains SUBSTR, e.g. \"Current Courses\" "
                          "to avoid re-processing the lecture pass")
    pdf.add_argument("--output", default=None, metavar="JSONL",
                     help="Write chunks here instead of data/pdf_chunks.jsonl "
                          "(e.g. data/lecture_chunks.jsonl)")
    pdf.add_argument("--max-pages", type=int, default=None,
                     help="Cap pages per PDF from page 1 (for huge books / quick tests)")
    pdf.add_argument("--pages", default=None, metavar="SPEC",
                     help="Ingest only an arbitrary 1-based page subset, e.g. "
                          "\"1-50,60,70-80\" (inclusive ranges + singletons). "
                          "Trims front-matter/index/back-matter. Takes precedence "
                          "over --max-pages. Kept pages keep stable doc_ids, so a "
                          "wider re-ingest only ADDS pages (upsert).")
    pdf.add_argument("--no-ocr", action="store_true", help="Disable OCR")
    pdf.add_argument("--ocr-engine", default=None,
                     choices=["auto", "tesseract", "paddle", "vlm", "none"],
                     help="OCR engine for scanned pages: auto (Tesseract-first "
                          "probe, the default), tesseract, paddle (PaddleOCR "
                          "sidecar from pdf.paddle_ocr), vlm (vision endpoint "
                          "from pdf.vlm_ocr), or none")
    pdf.add_argument("--skip-list", default=None, metavar="JSONL",
                     help="Path to dedup_skiplist.json (overrides config pdf.skip_list_file)")
    pdf.add_argument("--vault", default=None, metavar="PATH",
                     help="Override the vault root for this run (e.g. to test the correct path)")
    pdf.add_argument("--no-images", action="store_true",
                     help="Skip figure extraction (faster test runs)")
    pdf.add_argument("--archive-processed", action="store_true",
                     help="After a PDF yields chunks, move it into an _ingested/ "
                          "folder beside it (inbox lane: keeps re-runs from "
                          "re-processing or clobbering previous batches)")
    pdf.add_argument("--force-domain", default=None, metavar="DOMAIN",
                     help="Stamp every chunk's domain (inbox uploads have no "
                          "course path, so they land 'general' otherwise)")
    pdf.add_argument("--include-files", default=None, metavar="NAME,NAME",
                     help="Only process PDFs with these exact filenames "
                          "(comma-separated; file-scoped custom jobs)")
    pdf.add_argument("--chunking", default=None,
                     choices=("heading", "fixed", "document", "none"),
                     help="How oversized sections are split: 'heading' = "
                          "paragraph packing (default), 'fixed' = strict "
                          "sliding window (for OCR/wall-of-text PDFs)")
    pdf.add_argument("--force-tags", default=None, metavar="TAG,TAG",
                     help="Comma-separated tags stamped on every chunk's "
                          "metadata (feeds tag search + the retrieval tag boost)")
    pdf.set_defaults(func=cmd_ingest_pdfs)

    nb = sub.add_parser("ingest-notebooks",
                        help="Ingest native .ipynb/.py/.R/.Rmd into data/ipynb_chunks.jsonl")
    nb.add_argument("--vault", default=None, metavar="PATH", help="Override vault root")
    nb.add_argument("--output", default=None, metavar="JSONL",
                    help="Write chunks here instead of data/ipynb_chunks.jsonl")
    nb.add_argument("--no-outputs", action="store_true", help="Do not include cell outputs")
    nb.add_argument("--save-figures", action="store_true",
                    help="Decode notebook image outputs to data/notebook_figures/ "
                         "and link them via metadata (default OFF)")
    nb.add_argument("--exts", default=None,
                    help="Comma-separated subset, e.g. '.ipynb,.py' (default: all four)")
    nb.add_argument("--include-path", default=None, metavar="SUBSTR",
                    help="Only files whose vault-relative path contains SUBSTR")
    nb.add_argument("--include-files", default=None, metavar="NAME,NAME",
                    help="Only process files with these exact filenames "
                         "(comma-separated; file-scoped custom jobs)")
    nb.add_argument("--force-domain", default=None, metavar="DOMAIN",
                    help="Stamp this domain on every chunk (metadata only)")
    nb.add_argument("--force-tags", default=None, metavar="TAG,TAG",
                    help="Stamp these #tags on every chunk (metadata only)")
    nb.set_defaults(func=cmd_ingest_notebooks)

    code = sub.add_parser("ingest-code",
                          help="Ingest raw source code the notebook loader doesn't "
                               "cover (.js/.ts/.sql/.go/.java/.c/.cpp/.rs/…) into "
                               "data/code_chunks.jsonl")
    code.add_argument("--vault", default=None, metavar="PATH", help="Override vault root")
    code.add_argument("--output", default=None, metavar="JSONL",
                      help="Write chunks here instead of data/code_chunks.jsonl")
    code.add_argument("--include-path", default=None, metavar="SUBSTR",
                      help="Only files whose path contains SUBSTR (also scopes the "
                           "skipped agent-project roots back in)")
    code.add_argument("--exclude-path", default=None, metavar="SUBSTR",
                      help="Skip files whose path contains SUBSTR")
    code.add_argument("--include-files", default=None, metavar="NAME,NAME",
                      help="Only process files with these exact filenames "
                           "(comma-separated; file-scoped custom jobs)")
    code.add_argument("--exts", default=None,
                      help="Comma-separated subset, e.g. '.js,.ts,.sql' (default: all mapped)")
    code.add_argument("--force-domain", default=None, metavar="DOMAIN",
                      help="Stamp this domain on every chunk (metadata only)")
    code.add_argument("--force-tags", default=None, metavar="TAG,TAG",
                      help="Stamp these #tags on every chunk (metadata only)")
    code.set_defaults(func=cmd_ingest_code)

    canvas = sub.add_parser("ingest-canvas",
                            help="Ingest Obsidian .canvas files (nodes + labelled "
                                 "edges) into data/canvas_chunks.jsonl")
    canvas.add_argument("--vault", default=None, metavar="PATH", help="Override vault root")
    canvas.add_argument("--output", default=None, metavar="JSONL",
                        help="Write chunks here instead of data/canvas_chunks.jsonl")
    canvas.add_argument("--include-path", default=None, metavar="SUBSTR",
                        help="Only files whose vault-relative path contains SUBSTR")
    canvas.add_argument("--force-domain", default=None, metavar="DOMAIN",
                        help="Stamp this domain on every chunk (metadata only)")
    canvas.add_argument("--max-chunk-size", type=int, default=None, metavar="N",
                        help="Split node bodies longer than N chars (default: "
                             "canvas.max_chunk_size, null = no splitting)")
    canvas.add_argument("--chunking", default=None,
                        choices=["heading", "fixed", "document", "none"],
                        help="Splitter used when --max-chunk-size applies")
    canvas.add_argument("--context-depth", type=int, default=None,
                        choices=[0, 1, 2], metavar="0|1|2",
                        help="How much of a NEIGHBOUR is inlined into each "
                             "chunk: 0 titles only, 1 neighbour text, 2 also "
                             "second-hop titles. Duplicates text across "
                             "chunks; the run reports the measured cost")
    canvas.add_argument("--force-tags", default=None, metavar="TAG,TAG",
                        help="Stamp these #tags on every chunk (metadata only)")
    canvas.set_defaults(func=cmd_ingest_canvas)

    mdp = sub.add_parser("ingest-md",
                         help="SCOPED markdown parse (inbox md lane) — needs "
                              "--include-path and its own --output, never "
                              "touches chunks.jsonl")
    mdp.add_argument("--include-path", required=True, metavar="SUBSTR",
                     help="Only .md files whose vault-relative path contains this")
    mdp.add_argument("--output", required=True, metavar="JSONL",
                     help="Chunk file to write (e.g. data/inbox_md_chunks.jsonl)")
    mdp.add_argument("--chunking", default=None,
                     choices=("heading", "fixed", "document", "none"),
                     help="How oversized sections split (default: parser.chunking)")
    mdp.add_argument("--vault", default=None, metavar="PATH",
                     help="Override vault root")
    mdp.add_argument("--force-domain", default=None, metavar="DOMAIN",
                     help="Stamp this domain on every chunk (metadata only)")
    mdp.add_argument("--force-tags", default=None, metavar="TAG,TAG",
                     help="Stamp these #tags on every chunk (metadata only)")
    mdp.set_defaults(func=cmd_ingest_md)

    osi = sub.add_parser("ocr-studio-ingest",
                         help="Chunk one document's OCR Studio pages into its own "
                              "chunk file (the console then appends it)")
    osi.add_argument("--doc", required=True, metavar="REL",
                     help="Vault-relative path of the PDF or image")
    osi.add_argument("--output", required=True, metavar="JSONL",
                     help="Chunk file to write (data/ocrstudio_*_chunks.jsonl)")
    osi.add_argument("--pages", default="done", choices=("done", "all"),
                     help="done = pages marked done (default); all = every saved page")
    osi.set_defaults(func=cmd_ocr_studio_ingest)

    fw = sub.add_parser("fetch-web",
                        help="Fetch web pages to markdown (inbox _converted "
                             "staging) via requests/crawl4ai/scrapling/crawlee "
                             "+ markitdown")
    fw.add_argument("--urls", required=True, metavar="URL,URL",
                    help="Comma-separated http(s) URLs")
    fw.add_argument("--backend", default="auto",
                    choices=("auto", "requests", "crawl4ai", "scrapling", "crawlee"),
                    help="Fetch backend (auto = best installed of crawl4ai/"
                         "scrapling/requests; requests always works; crawlee "
                         "runs only when named)")
    fw.add_argument("--format", default="md", choices=("md", "pdf"),
                    help="md = convert to markdown (markitdown); pdf = print "
                         "the rendered page via headless Chromium (keeps "
                         "LaTeX/tables/code exactly as the site shows them)")
    fw.add_argument("--out-dir", default=None, metavar="DIR",
                    help="Output folder (default: <inbox>/_converted)")
    fw.set_defaults(func=cmd_fetch_web)

    cv = sub.add_parser("convert-files",
                        help="Convert inbox files (pdf/docx/pptx/xlsx/html/…) "
                             "to markdown via markitdown, optional PDF-page OCR")
    cv.add_argument("--files", required=True, metavar="NAME,NAME",
                    help="Comma-separated inbox filenames")
    cv.add_argument("--ocr-pages", default=None, metavar="SPEC",
                    help='Also OCR these 1-based PDF pages (e.g. "1-4,9") and '
                         "append the text (Tesseract)")
    cv.add_argument("--out-dir", default=None, metavar="DIR",
                    help="Output folder (default: <inbox>/_converted)")
    cv.set_defaults(func=cmd_convert_files)

    q = sub.add_parser("query", help="Ask one question")
    q.add_argument("question")
    q.add_argument("--top-k", type=int, default=None, dest="top_k",
                   help="Override rerank_top_k for this query (default: config value)")
    q.add_argument("--preset", default=None,
                   help="Retrieval preset from config (code | concept | synthesis); "
                        "omit for auto (code preset fires on code-intent queries)")
    q.add_argument("--max-tokens", type=int, default=None, dest="max_tokens",
                   help="Cap the answer's output tokens for this query "
                        "(default: generation.max_tokens from config)")
    q.set_defaults(func=cmd_query)

    sub.add_parser("chat", help="Interactive REPL").set_defaults(func=cmd_chat)

    st = sub.add_parser("stamp",
                        help="Verify which embedder built a collection, then record it")
    st.add_argument("--collection", default=None, metavar="NAME",
                    help="Collection to stamp (default: paths.collection_name; a HyPE "
                         "question collection is verified query-side)")
    st.add_argument("--force", action="store_true",
                    help="Replace an existing fingerprint that disagrees with the "
                         "configured embedder. The sample check still has to pass")
    st.set_defaults(func=cmd_stamp)

    ge = sub.add_parser("graph-embed",
                        help="Rebuild graph mode's seed index <collection>__graph from scratch: "
                             "the graph.file_type chunks embedded by graph.embedding_provider. "
                             "Never writes the main collection or a chunk file")
    ge.add_argument("--limit", type=int, default=None, metavar="N",
                    help="Embed only N graph chunks, for a smoke test. The result stays "
                         "'building' (never served)")
    ge.set_defaults(func=cmd_graph_embed)

    add_bench_parser(sub)

    sub.add_parser("serve", help="Launch the Streamlit app").set_defaults(func=cmd_serve)
    return p


def main():
    force_utf8_console()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
