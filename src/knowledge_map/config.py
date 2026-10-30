"""The `knowledge_map:` config section, validated once at load.

No silent defaults: a missing key raises ValueError naming it (the one exception
is the scanned-page threshold, which belongs to the `pdf:` section).
"""
from dataclasses import dataclass
from pathlib import Path

from src.utils.provenance import CANVAS_MARKERS_DEFAULT, canvas_markers

ROLES = ("classify", "extract", "link", "verify", "reverify")


@dataclass(frozen=True)
class RoleSpec:
    provider: str
    model: str
    max_tokens: int
    temperature: float
    fallbacks: tuple[tuple[str, str], ...]   # (provider, model), tried in order


@dataclass(frozen=True)
class KMConfig:
    staging_dir: Path
    vault_root: Path
    min_grounded_ratio: float
    max_span_pages: int
    node_cap: int
    fanout_max: int
    quote_words: tuple[int, int]
    max_wall_minutes: float
    max_llm_calls: int
    retries: int
    verify_batch: int
    max_unit_chars: int
    window_pages: int
    max_scanned_ratio: float
    scanned_threshold: int
    rate_limit_sleep_s: float
    request_timeout_s: float
    heading_min_ratio: float                 # coverage: a line this many times the body font size is a heading
    min_section_chars: int                   # coverage: a heading with less text under it is not checked
    chars_per_node: int                      # coverage: a heading needs one node per this many chars; 0 = off
    generated_markers: tuple[str, ...]       # canvas.generated_markers, read as the canvas lane reads it
    roles: dict[str, RoleSpec]


def _req(cfg, key: str):
    val = cfg.get(key)
    if val is None:
        raise ValueError(f"config key '{key}' is required and missing (knowledge_map section of config.yaml)")
    return val


def _role(cfg, role: str) -> RoleSpec:
    base = f"knowledge_map.{role}"
    fallbacks = _req(cfg, f"{base}.fallbacks")
    if not isinstance(fallbacks, list):
        raise ValueError(f"config key '{base}.fallbacks' must be a list of {{provider, model}}")
    fb = []
    for i, f in enumerate(fallbacks):
        for k in ("provider", "model"):
            if not isinstance(f, dict) or not f.get(k):
                raise ValueError(f"config key '{base}.fallbacks[{i}].{k}' is required and missing")
        fb.append((f["provider"], f["model"]))
    spec = RoleSpec(
        provider=str(_req(cfg, f"{base}.provider")),
        model=str(_req(cfg, f"{base}.model")),
        max_tokens=int(_req(cfg, f"{base}.max_tokens")),
        temperature=float(_req(cfg, f"{base}.temperature")),
        fallbacks=tuple(fb),
    )
    if role in ("verify", "reverify"):   # verifiers are pinned: a router may never pick them (spec 3.6)
        if spec.model == "auto":
            raise ValueError(f"config key '{base}.model' may not be 'auto': the verifier must be a pinned model")
        for i, (_, m) in enumerate(spec.fallbacks):
            if m == "auto":
                raise ValueError(f"config key '{base}.fallbacks[{i}].model' may not be 'auto': the verifier must be a pinned model")
    return spec


def _check_reverify(roles: dict[str, RoleSpec]) -> None:
    """A trimmed summary is written by a verify model, so it is re-checked by another one."""
    writers = {roles["verify"].model, *(m for _, m in roles["verify"].fallbacks)}
    re_ = roles["reverify"]
    for key, m in [("model", re_.model), *((f"fallbacks[{i}].model", m) for i, (_, m) in enumerate(re_.fallbacks))]:
        if m in writers:
            raise ValueError(f"config key 'knowledge_map.reverify.{key}' is {m!r}, which is also a "
                             f"knowledge_map.verify model: re-verification must use a different model")


def load_km_config(cfg) -> KMConfig:
    k = "knowledge_map"
    vault_root = Path(_req(cfg, "parser.vault_path"))
    staging = Path(_req(cfg, f"{k}.staging_dir"))
    if staging.resolve().is_relative_to(vault_root.resolve()):
        raise ValueError(f"config key 'knowledge_map.staging_dir' ({staging}) must be outside parser.vault_path ({vault_root})")
    roles = {r: _role(cfg, r) for r in ROLES}
    _check_reverify(roles)
    qw = _req(cfg, f"{k}.quote_words")
    if not (isinstance(qw, list) and len(qw) == 2 and 0 < qw[0] <= qw[1]):
        raise ValueError("config key 'knowledge_map.quote_words' must be [min, max] with 0 < min <= max")
    return KMConfig(
        staging_dir=staging,
        vault_root=vault_root,
        min_grounded_ratio=float(_req(cfg, f"{k}.min_grounded_ratio")),
        max_span_pages=int(_req(cfg, f"{k}.max_span_pages")),
        node_cap=int(_req(cfg, f"{k}.node_cap")),
        fanout_max=int(_req(cfg, f"{k}.fanout_max")),
        quote_words=(int(qw[0]), int(qw[1])),
        max_wall_minutes=float(_req(cfg, f"{k}.max_wall_minutes")),
        max_llm_calls=int(_req(cfg, f"{k}.max_llm_calls")),
        retries=int(_req(cfg, f"{k}.retries")),
        verify_batch=int(_req(cfg, f"{k}.verify_batch")),
        max_unit_chars=int(_req(cfg, f"{k}.extract.max_unit_chars")),
        window_pages=int(_req(cfg, f"{k}.scout.window_pages")),
        max_scanned_ratio=float(_req(cfg, f"{k}.scout.max_scanned_ratio")),
        scanned_threshold=int(cfg.get("pdf.skip_scanned_threshold", 50)),
        rate_limit_sleep_s=float(_req(cfg, f"{k}.rate_limit_sleep_s")),
        request_timeout_s=float(_req(cfg, f"{k}.request_timeout_s")),
        heading_min_ratio=float(_req(cfg, f"{k}.coverage.heading_min_ratio")),
        min_section_chars=int(_req(cfg, f"{k}.coverage.min_section_chars")),
        chars_per_node=int(_req(cfg, f"{k}.coverage.chars_per_node")),
        generated_markers=canvas_markers(cfg.get("canvas.generated_markers", CANVAS_MARKERS_DEFAULT)),
        roles=roles,
    )
