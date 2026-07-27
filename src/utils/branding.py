"""
branding.py — the product's name, in exactly one place.

Everything user-visible that names the product imports from here: the two
FastAPI titles, the machine-readable `service` slugs in both capability maps,
the console header, the CLI banner.

WHY THIS EXISTS. The project was renamed once already, and the rename touched
the README, the docs site, two API contracts, the console, the Docker bundle
and the release sanitizer. A name that lives in one constant costs a one-line
edit instead of a repo-wide sweep, and a sweep is where you miss the string
that ends up published in a JSON contract.

WHAT IS DELIBERATELY *NOT* HERE — these look like the product name but are
load-bearing identifiers, and changing them breaks data rather than text:

  * `paths.collection_name` (default "obsidian_vault") is the ChromaDB
    collection that existing indexes were WRITTEN INTO. Renaming it does not
    rename the collection; it points the retriever at a new, empty one and
    every query silently returns nothing.
  * `.obsidian`, `.obsidian-git`, `.trash` in the ingest skip-lists are real
    directory names on disk.
  * `src/ingestion/obsidian_parser.py` is named for the markdown dialect it
    parses (wikilinks, frontmatter), it is a documented CLI entry point, and
    every other loader imports from it.

Naming a format or an integration you interoperate with is a statement of
fact, not branding. Naming the PRODUCT is branding, and that is what this file
owns.
"""
from __future__ import annotations

# The product name, as a person reads it.
PRODUCT_NAME = "Noetrix"

# One line, used under the name wherever a subtitle fits.
TAGLINE = "Grounded, cited answers from your own documents"

# Lowercase, hyphenated, stable. This is what agents match on in the `service`
# field of GET /schema and GET /api/schema, so treat a change here as a
# contract change and not a cosmetic one.
SERVICE_SLUG = "noetrix"

QUERY_SERVICE = f"{SERVICE_SLUG}-query"
CONSOLE_SERVICE = f"{SERVICE_SLUG}-console"

QUERY_API_TITLE = f"{PRODUCT_NAME} — Query API"
CONSOLE_API_TITLE = f"{PRODUCT_NAME} — Management Console"
