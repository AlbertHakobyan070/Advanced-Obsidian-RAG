"""OCR Studio: hand-guided OCR of one document, page by page.

The console's OCR tab (webui/index.html, /api/ocrstudio/* in manage_api.py)
drives these modules; the design is docs/superpowers/specs/2026-10-05-ocr-studio-design.md.

  render   page -> pixels, the non-destructive clean-up ops, the healing brush
  engines  one call shape over Tesseract, the PaddleOCR sidecar and the VLM client
  pages    per-page state (regions, ops, mask), assembly to markdown, CER report
  ingest   assembled pages -> PDF-lane chunks -> a NON-canonical chunk file

Nothing here writes into the vault: state lives under the active data dir
(`<data>/ocr_studio/`), figures under `pdf.figures_dir`.
"""
