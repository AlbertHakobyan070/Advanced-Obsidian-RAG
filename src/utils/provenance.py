"""
provenance.py — where a chunk's evidence comes from: authored, source or generated.

THE ONE CONSUMER RULE (decision 4, revised by route-then-cite §5.2)
  In order:
    1. a stored `provenance: generated` is generated;
    2. a source_file the path rule names is generated: under a
       provenance.generated_paths prefix, an agent-instruction file
       (provenance.agent_instruction_files / _dirs, spec §7 call 11), or a
       .canvas path holding a canvas.generated_markers marker. This beats a
       stored `authored`, which every md row carries by default;
    3. any other stored key (a value outside PROVENANCE raises);
    4. provenance.by_file_type decides by file_type;
    5. authored.
  So code, notebooks and canvases with no key and no path match are authored,
  and the chunks indexed before the key existed resolve with no migration.

  The generator's label and confidence cap, graph_search, the evidence pack and
  the bench validator all resolve through provenance_of, so no two of them
  disagree about a chunk. It imports nothing from the project, so a caller takes
  the rule without the retriever or the LLM client.
"""
from __future__ import annotations

import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

PROVENANCE = ("authored", "source", "generated")

# retrieval.generated_mode: what generated chunks do in retrieval (spec §5.1).
GENERATED_MODES = ("off", "drop", "route")


class UnstampedProvenanceError(RuntimeError):
    """Under drop or route, stored rows the rule calls generated whose stored key is
    not "generated": the dense lanes filter on that key, so the pipeline refuses to
    build (spec §5.2) and names the files and the fix."""


class RouteUnavailableError(RuntimeError):
    """generated_mode route while the store holds generated rows and no pointer sidecar
    is loaded (spec §5.4): a configured route refuses to build, a per-call route refuses
    the call (the API's 409). Names the fix, python main.py route-pointers."""


def check_mode(value, where: str) -> str:
    """A generated_mode value, or ValueError naming `where`. A bool is refused, not
    read as off: YAML reads a bare off (or no/false) as False, and false could as
    well mean "no generated chunks", which is drop."""
    if isinstance(value, bool) or value not in GENERATED_MODES:
        hint = ': YAML reads a bare off as false, so quote it ("off")' if isinstance(value, bool) else ""
        raise ValueError(f"{where} must be one of {', '.join(GENERATED_MODES)}, got {value!r}{hint}")
    return value


# What a key the config leaves out means, as by_file_type defaults to {pdf: source}:
# an older config still keeps agent files out of the evidence (spec §7 call 11).
_DEFAULTS = {
    "generated_paths": [],
    "agent_instruction_files": ["AGENTS.md", "CLAUDE.md", "GEMINI.md", "SKILL.md"],
    "agent_instruction_dirs": [".claude"],
}
# canvas.generated_markers when the config leaves it out. The canvas loader and the
# rule both read the key through canvas_markers(), so they agree on which canvases
# are generated.
CANVAS_MARKERS_DEFAULT = ["(generated)"]


def _norm(p) -> str:
    """canvas_loader.norm_vault_path's body (NFC, '/', no edge slashes, casefolded),
    written here because importing it would cycle once obsidian_parser stamps
    through this module. A test pins that the two agree."""
    return unicodedata.normalize("NFC", str(p)).replace("\\", "/").strip("/").casefold()


def canvas_markers(value) -> tuple[str, ...]:
    """canvas.generated_markers, casefolded, blanks dropped; None means none."""
    if isinstance(value, str):     # its characters would each match nearly every path
        raise ValueError('canvas.generated_markers must be a list, e.g. ["(generated)"], '
                         f"not the string {value!r}: fix it in config.yaml")
    return tuple(str(m).casefold() for m in (value or []) if str(m).strip())


def marks_generated_canvas(path, markers) -> bool:
    """A .canvas path containing one of `markers` (canvas_markers' output) is LLM-written."""
    p = _norm(path)
    return p.endswith(".canvas") and any(m in p for m in markers)


@dataclass(frozen=True)
class ProvenanceRule:
    by_file_type: Mapping[str, str] = field(default_factory=dict)
    generated_paths: tuple[str, ...] = ()            # normalised prefixes, no trailing "/"
    instruction_files: frozenset[str] = frozenset()  # casefolded file names
    instruction_dirs: frozenset[str] = frozenset()   # casefolded directory names
    generated_markers: tuple[str, ...] = ()          # casefolded canvas.generated_markers; .canvas paths only

    def instruction_file(self, source_file) -> bool:
        """The file name is an agent-instruction name, or a directory on its path is."""
        *dirs, name = _norm(source_file).split("/")
        return name in self.instruction_files or any(d in self.instruction_dirs for d in dirs)

    def generated_by_path(self, source_file) -> bool:
        """Whole-segment prefix match, an instruction file, or the canvas loader's
        marker test (marks_generated_canvas)."""
        p = _norm(source_file)
        return (any(p == pre or p.startswith(pre + "/") for pre in self.generated_paths)
                or self.instruction_file(source_file)
                or marks_generated_canvas(source_file, self.generated_markers))


def provenance_map(cfg) -> dict[str, str]:
    """provenance.by_file_type with its keys lowercased, because file_type is
    matched lowercased. Read when a component is built, so a bad value fails at
    startup, naming the key, instead of on the first query that meets it."""
    raw = cfg.get("provenance.by_file_type", {"pdf": "source"})
    if not isinstance(raw, Mapping):
        raise ValueError(f"provenance.by_file_type must map a file_type to one of "
                         f"{', '.join(PROVENANCE)}, got {raw!r}: fix it in config.yaml")
    bad = {k: v for k, v in raw.items() if v not in PROVENANCE}
    if bad:
        raise ValueError(f"provenance.by_file_type values must be one of "
                         f"{', '.join(PROVENANCE)}, got {bad!r}: fix them in config.yaml")
    return {str(k).lower(): v for k, v in raw.items()}


def _names(cfg, key: str) -> list[str]:
    """provenance.<key> as a list of non-empty strings; absent = the shipped value,
    present but anything else (null included) raises."""
    raw = cfg.get(f"provenance.{key}", _DEFAULTS[key])
    if not isinstance(raw, list) or not all(isinstance(v, str) and _norm(v).strip() for v in raw):
        raise ValueError(f"provenance.{key} must be a list of non-empty strings, got {raw!r}: "
                         f"fix it in config.yaml")
    return raw


def provenance_rule(cfg) -> ProvenanceRule:
    """The whole rule from config, built when a component is, so a bad value fails
    at startup naming its key."""
    return ProvenanceRule(
        by_file_type=provenance_map(cfg),
        generated_paths=tuple(_norm(p) for p in _names(cfg, "generated_paths")),
        instruction_files=frozenset(_norm(n) for n in _names(cfg, "agent_instruction_files")),
        instruction_dirs=frozenset(_norm(n) for n in _names(cfg, "agent_instruction_dirs")),
        generated_markers=canvas_markers(cfg.get("canvas.generated_markers", CANVAS_MARKERS_DEFAULT)),
    )


def provenance_of(meta: dict, rule: ProvenanceRule | Mapping[str, str]) -> str:
    """This chunk's provenance by the one rule above. A plain mapping is read as
    ProvenanceRule(by_file_type=mapping). An explicit value outside PROVENANCE
    raises rather than reading as authored: a typo would otherwise turn a
    generated summary into evidence. None counts as absent, as it does in
    Chroma, which never stores a None value."""
    if not isinstance(rule, ProvenanceRule):
        rule = ProvenanceRule(by_file_type=rule)
    explicit = meta.get("provenance")
    if explicit == "generated" or rule.generated_by_path(meta.get("source_file") or ""):
        return "generated"
    if explicit is not None:
        if explicit not in PROVENANCE:
            raise ValueError(
                f"chunk provenance {explicit!r} of {meta.get('source_file')!r} is not one of "
                f"{', '.join(PROVENANCE)}: fix that file's provenance and re-ingest it")
        return explicit
    return rule.by_file_type.get(str(meta.get("file_type", "")).lower(), "authored")


def stamp_generated(meta: dict, rule: ProvenanceRule | None) -> dict:
    """Set provenance: generated when the path rule names meta's source_file.
    None (direct test constructions only) leaves meta unchanged. Returns meta."""
    if rule is not None and rule.generated_by_path(meta.get("source_file") or ""):
        meta["provenance"] = "generated"
    return meta


def unstamped_generated(metas: Iterable[dict], rule: ProvenanceRule) -> dict[str, int]:
    """source_file -> rows the rule calls generated whose stored key is not
    "generated": the rows the dense lanes' stored-key drop cannot exclude."""
    counts: Counter = Counter()
    for m in metas:
        if m.get("provenance") != "generated" and provenance_of(m, rule) == "generated":
            counts[m.get("source_file") or ""] += 1
    return dict(counts)
