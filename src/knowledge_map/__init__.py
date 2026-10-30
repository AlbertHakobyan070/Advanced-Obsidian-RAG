"""
knowledge_map — turn a document into a grounded, routable knowledge map.

Every pointer on the map is derived, never trusted: the extractor returns
verbatim quotes and `locate.py` finds them in the PDF page text to write the
pages. See docs/superpowers/specs/2026-10-29-knowledge-map-harness-design.md.
"""
