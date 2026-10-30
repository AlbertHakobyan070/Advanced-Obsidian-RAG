# Knowledge maps

`rag map` reads one chapter of a PDF and builds a **knowledge map**: an Obsidian
canvas of the chapter's concepts, each node carrying verbatim quotes from the
book and the PDF pages they were found on, plus a typed graph (`map.json`)
behind it. A run writes only to a **staging folder**. It never writes the
vault, never touches the index, and never ingests anything.

## What is guaranteed

- **Pages are derived, not written by a model.** The extractor returns quotes;
  Python finds each quote in the PDF text and records the pages. A quote that
  is not found, or is found on more than one page, fails its node.
- **A pinned verifier checks each node** against the cited pages. When a
  node's quotes are good but its summary claims more than they say, the
  verifier cuts the summary down to what the quotes state, and a different
  pinned model (`knowledge_map.reverify`, which may share no model with
  `verify`) checks it again;
  a node that passes is kept, with `summary_revised_from` holding the
  extracted summary in `map.json`, and the run record counts it under
  `revised`. Nodes that still fail are dropped and listed in the run record,
  never drawn.
- **A coverage pass.** The scout finds the chapter's headings from font size
  (no PDF outline needed). After the first check, a heading is short when it
  holds fewer kept nodes than its length needs (one per
  `knowledge_map.coverage.chars_per_node` characters), or when none of its nodes
  is named after it (its title holds every significant word of the heading).
  A short heading gets one more extract call over the pages under it. The call names the
  nodes already there and asks for the rest, and the new nodes pass the same
  checks. An `--import-extract` run makes no such call; it only reports the
  gaps. The node cap never cuts below what the headings need together, so it
  cannot undo a gap-fill; `knowledge_map.node_cap` is the floor. The run
  record's `coverage` block gives the cap used and lists the headings still
  short of nodes in the final map. `--dry-run` prints the headings it found.
- **A run gate.** If too small a share of candidate nodes end up located and
  supported (`knowledge_map.min_grounded_ratio`), the run fails and writes
  nothing but its run record.
- **The canvas name carries `(generated)`**, so the canvas lane stamps
  `provenance: generated` on it when it is eventually ingested.

## The command

```text
rag map PDF (--chapter TITLE | --pages A-B) [--template T] [--dry-run]
            [--export-prompts DIR] [--import-extract DIR] [--title SHORT]
rag map --estimate FOLDER
```

- `--chapter` takes a title from the PDF's outline; `--pages` takes a 1-based
  PDF page range and works on PDFs without an outline. Pass exactly one.
- `--template` skips the classifier; it takes a template name from
  `src/knowledge_map/templates.yaml`.
- `--dry-run` scouts, classifies and prints the plan. Nothing is written.
- `--export-prompts DIR` writes each extraction unit's exact prompt to `DIR`
  and stops; fill `DIR/extract-<unit>.json` with a reply that follows the same
  schema as a model reply, then rerun with `--import-extract DIR`. Imported
  units skip the extract model; link, verify and everything after run as usual.
  The export also writes `DIR/export.json` (template, the classifier's reason,
  the PDF's sha256, the page range, the unit ids, the prompt versions). The
  import reads it: without `--template` it uses the exported template and does
  not classify again, and it refuses to run if the PDF, the page range, the
  units or a given `--template` differ from the export, naming the field. With
  no `export.json` in `DIR`, the import needs `--template`.
- `--title` overrides the short book title used in folder and canvas names.
- The PDF must live inside `parser.vault_path`; the staging folder must not.
  Pass the PDF as an absolute path or relative to `parser.vault_path` (not to
  the current directory). It must exist and end in `.pdf`.

The command exits 2 with `ERROR: ...` for a bad argument or an unusable PDF
(for example a scanned chapter that needs OCR first), and 1 when the run itself
fails, printing the reason. It ends with `NEXT:` lines saying where the map is.

The console runs the same command as the `map` job kind (permission:
`mutating`), and agents can queue it through the MCP change tool; it builds into
staging only. Its params are `pdf`, `chapter` or `pages`, and optionally
`template`, `title` and `dry_run`.

## Estimating before you run

`rag map --estimate FOLDER` prints one row per PDF under `FOLDER`: pages,
text-layer and scanned pages, outline chapters, caption and image counts. It
calls no model. Once finished runs exist under the staging folder, it also
projects tokens and minutes from their measured per-page rates.

## What a run leaves in staging

```text
<staging_dir>/<book>/<chapter>/
    <Book> - <Chapter> (generated).canvas
    map.json          the typed graph behind the canvas
    run.json          the latest successful run's record
    runs/<utc>.json   every run, failed ones included
    .work/<key>/      step checkpoints
```

A rerun with the same input and settings reuses the checkpoints and makes no
model call for finished steps; a changed setting gets a new key.

## The run record

`run.json` and `runs/<utc>.json` hold: the command line, the `knowledge_map`
settings in force, the source file, hash and page range, the chosen template and
why, every model attempt (role, requested and served model, tokens in and out,
seconds, outcome), totals (calls, tokens, retries, rate-limit waits, wall
seconds, peak memory), node and edge counts, the grounding gate result, the
`status` (`ok` or `failed`) with its `reason`, and the **rejects**: every dropped
node with its quotes, cited pages and the reason it was dropped.

## Placing a map (by hand)

Placement is deliberately manual. After you have reviewed a staged map:

1. Copy the `.canvas` into a vault folder listed in `provenance.generated_paths`,
   or keep the `(generated)` name anywhere in the vault.
2. Ingest it with the canvas lane, then add the chunks with
   `index --append`.

The canvas lane stamps the chunks `provenance: generated`, so how they enter
answers follows the `provenance` settings. Moving the canvas later changes its
path and therefore its chunk ids, so choose its final place before ingesting.

## Settings

Everything is under `knowledge_map:` in `config.yaml` (staging folder, grounding
gate, node cap, quote length, budgets, and one block per model role); the
commented keys in `config.example.yaml` describe each one.
