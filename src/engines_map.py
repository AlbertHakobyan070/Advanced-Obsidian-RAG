"""
engines_map.py — every swappable engine, per component, built from probe facts.

Pure: manage_api gathers the facts with its own fast helpers and passes them in
(GET /api/engines), and this module decides what they mean. The reliability
rule therefore lives in one place and is tested on hand-built input.

An engine is offered as SWAPPABLE only if a probe can say whether it is
available, a switch is checked before it is written, a failure raises instead
of falling back to another engine, and the cost of the switch is stated. One
that fails a criterion is still listed, with swappable False and the reason.
The one declared fallback is the `auto` web-fetch backend: its chain IS what
`auto` means. (docs/superpowers/specs/2026-10-15-mcp-token-engines-design.md §4)

`switch` is the cheapest way an option can be put to use:
  per_call  a per-request knob (/search, /query, /compare, the fetch request)
  per_job   read when a job starts
  restart   a config write plus a :8051 restart
  build     a one-off build job first (the tantivy index)
  reembed   a re-embed into a new collection first
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

COMPONENTS = ("generation", "reranker", "embeddings", "sparse", "ocr", "fetch", "chunking")
SWITCH = ("per_call", "per_job", "restart", "build", "reembed")

# The config key POST /api/engines/set writes for each component. None: there is
# no persistent setting, the engine is chosen per request.
SETTINGS = {
    "generation": "generation.provider",
    "reranker": "retrieval.rerank_mode",
    "embeddings": "paths.collection_name",
    "sparse": "retrieval.sparse_backend",
    "ocr": "pdf.ocr_engine",
    "fetch": None,
    "chunking": "parser.chunking",
}

# The second key a `model` writes. The reranker's applies to cross_encoder only.
MODEL_SETTINGS = {
    "generation": "generation.model",
    "reranker": "retrieval.cross_encoder_model",
}

NO_PROBE = "src/engines_map.py has no probe for this engine, so nothing checks it"


@dataclass(frozen=True)
class Option:
    engine: str
    available: bool
    reason: str | None
    swappable: bool
    switch: str

    def __post_init__(self):
        if self.switch not in SWITCH:
            raise ValueError(f"{self.engine}: switch {self.switch!r} is not one of {SWITCH}")
        if self.reason is None and not (self.available and self.swappable):
            raise ValueError(f"{self.engine}: an unavailable or unswappable option needs a reason")


def _opt(engine: str, switch: str, *, unavailable: str | None = None,
         unswappable: str | None = None) -> Option:
    """Each keyword is the reason it is NOT so, or None."""
    why = "; ".join(r for r in (unswappable, unavailable) if r) or None
    return Option(engine, unavailable is None, why, unswappable is None, switch)


def _unknown(engine: str, switch: str) -> Option:
    return Option(engine, False, NO_PROBE, False, switch)


def _status_error(status: dict) -> str | None:
    return None if status.get("available") else (status.get("error") or "not available")


def _generation(p: dict) -> list[Option]:
    opts = []
    for row in p["providers"]:
        why = None
        if not row["available"]:
            why = (f"{row['api_key_env']} holds the wrong credential type"
                   if row.get("key_compatible") is False
                   else f"{row['api_key_env']} is not set")
        opts.append(_opt(row["name"], "per_call", unavailable=why))
    for name in p["reserved"]:
        opts.append(_opt(name, "restart",
                         unavailable="nothing probes a legacy provider name",
                         unswappable="add a providers: entry for this backend instead"))
    return opts


def _reranker(p: dict) -> list[Option]:
    opts = []
    for mode in p["modes"]:
        if mode == "cross_encoder":
            opts.append(_opt(mode, "per_call", unavailable=None if p["cached"] else (
                f"{p['model']} is not in the local model cache, so the next :8051 start "
                f"would download it (set checks the hub first)")))
        elif mode == "http":
            opts.append(_opt(mode, "per_call",
                             unavailable="nothing probes the external rerank endpoint "
                                         "(retrieval.rerank_http.base_url)",
                             unswappable="it cannot be checked before a switch; use it "
                                         "per call (rerank: http)"))
        elif mode in ("lexical", "none"):
            opts.append(_opt(mode, "per_call"))
        elif mode == "laya":
            laya = p["laya"]
            why = ("experimental and off: turn on retrieval.laya.enabled first"
                   if not laya["enabled"] else
                   "the optional laya package is not installed" if not laya["package"] else
                   f"no checkpoint at {laya['model_dir']}" if not laya["checkpoint"] else None)
            opts.append(_opt(mode, "per_call", unavailable=why))
        else:
            opts.append(_unknown(mode, "per_call"))
    return opts


def _embeddings(p: dict) -> list[Option]:
    opts, names, built_with = [], set(), set()
    for c in p["collections"]:
        names.add(c["name"])
        if "error" in c:
            opts.append(_opt(c["name"], "restart", unavailable=c["error"],
                             unswappable="its embedding fingerprint cannot be read"))
            continue
        built = f"{c['provider']} · {c['model']}"
        unavailable = None
        if c["status"] == "complete":
            built_with.add(c["provider"])
        else:
            unavailable = f"an unfinished re-embed by {built}: finish it first"
        mismatch = None
        if c["diff"]:
            mismatch = (f"built by {built}, but the configured embedder is {p['model']} "
                        f"(they differ in {', '.join(c['diff'])}); point embedding.* in "
                        f"config.yaml at its embedder first")
        opts.append(_opt(c["name"], "restart", unavailable=unavailable, unswappable=mismatch))
    if p["active"] not in names:
        opts.append(_opt(p["active"], "restart",
                         unswappable="it has no embedding fingerprint: run `main.py stamp` "
                                     "to record one"))
    for prov in p["providers"]:
        if prov != p["provider"] and prov not in built_with:
            opts.append(_opt(prov, "reembed",
                             unavailable=f"no collection is built with {prov} yet: "
                                         f"main.py index --rebuild-into NAME --embedding {prov}",
                             unswappable="a re-embed is a long job of its own, not a switch"))
    return opts


def _sparse(p: dict) -> list[Option]:
    opts = []
    for name in p["backends"]:
        if name == "bm25s":
            opts.append(_opt(name, "restart"))
        elif name == "tantivy":
            if not p["tantivy_package"]:
                opts.append(_opt(name, "build", unavailable="the optional tantivy package "
                                                            "is not installed"))
            elif not p["tantivy_index"]:
                opts.append(_opt(name, "build", unavailable=f"no index at {p['tantivy_dir']}: "
                                                            f"run the build_tantivy job first"))
            else:
                opts.append(_opt(name, "restart"))
        else:
            opts.append(_unknown(name, "restart"))
    return opts


def _ocr(p: dict) -> list[Option]:
    st = p["status"]
    paddle = st["paddle-vl"] if p.get("pipeline") == "vl" else st["paddle"]
    opts = []
    for name in p["engines"]:
        if name == "auto":
            opts.append(_opt(name, "per_job", unavailable=_status_error(st["tesseract"]),
                             unswappable="auto falls back from Tesseract to RapidOCR, so it "
                                         "is a chain, not one engine; pin tesseract instead, "
                                         "or set auto back from the Settings tab (pdf.ocr_engine)"))
        elif name == "tesseract":
            opts.append(_opt(name, "per_job", unavailable=_status_error(st["tesseract"])))
        elif name == "paddle":
            opts.append(_opt(name, "per_job", unavailable=_status_error(paddle)))
        elif name == "vlm":
            opts.append(_opt(name, "per_job", unavailable=_status_error(st["vlm"])))
        elif name == "none":
            opts.append(_opt(name, "per_job"))
        else:
            opts.append(_unknown(name, "per_job"))
    return opts


def _fetch(p: dict) -> list[Option]:
    opts = []
    for name in p["backends"]:
        if name in ("auto", "requests"):          # requests is a hard dependency
            opts.append(_opt(name, "per_call"))
        elif name in p["installed"]:
            opts.append(_opt(name, "per_call", unavailable=None if p["installed"][name]
                             else f"the {name} package is not installed"))
        else:
            opts.append(_unknown(name, "per_call"))
    return opts


def _chunking(p: dict) -> list[Option]:
    return [_opt(name, "per_job") for name in p["values"]]


_BUILDERS = {"generation": _generation, "reranker": _reranker, "embeddings": _embeddings,
             "sparse": _sparse, "ocr": _ocr, "fetch": _fetch, "chunking": _chunking}


def build_engines(probes: dict) -> dict:
    """{"components": {name: {active, setting, model, options}}} for each
    component present in `probes`, in COMPONENTS order."""
    unknown = sorted(set(probes) - set(COMPONENTS))
    if unknown:
        raise ValueError(f"unknown component(s) {unknown}; expected {list(COMPONENTS)}")
    return {"components": {
        name: {"active": probes[name]["active"], "setting": SETTINGS[name],
               "model": probes[name].get("model"),
               "options": [asdict(o) for o in _BUILDERS[name](probes[name])]}
        for name in COMPONENTS if name in probes}}


def switch_option(engines: dict, component: str, engine: str) -> dict:
    """The option POST /api/engines/set may write. ValueError says why not:
    an unknown component or engine, no persistent setting, or unswappable.
    Availability is the caller's preflight, not checked here."""
    comp = engines["components"].get(component)
    if comp is None:
        raise ValueError(f"unknown component {component!r}; expected one of {list(COMPONENTS)}")
    if comp["setting"] is None:
        raise ValueError(f"{component} has no persistent setting: it is chosen per request")
    opt = next((o for o in comp["options"] if o["engine"] == engine), None)
    if opt is None:
        raise ValueError(f"unknown {component} engine {engine!r}; expected one of "
                         f"{[o['engine'] for o in comp['options']]}")
    if not opt["swappable"]:
        raise ValueError(f"{component} engine {engine!r} is not switchable: {opt['reason']}")
    return opt


def switch_changes(component: str, engine: str, model: str | None) -> dict:
    """The {dotted.key: value} a switch writes through the Settings writer."""
    changes = {SETTINGS[component]: engine}
    if model is None:
        return changes
    key = MODEL_SETTINGS.get(component)
    if key is None or (component == "reranker" and engine != "cross_encoder"):
        raise ValueError(f"model applies only to generation and to the reranker's "
                         f"cross_encoder, not to {component} {engine!r}")
    changes[key] = model
    return changes
