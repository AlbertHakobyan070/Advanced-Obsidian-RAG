"""The deployment assets under deploy/ -- checked statically, without Docker.

What these guard is not "does it build" (CI builds the images) but the properties
that rot silently:

  * the container config is GENERATED, and the generator must keep working as
    config.example.yaml changes -- including the failure it used to miss, a loopback
    URL with no rewrite (the EmbeddingGemma sidecar URLs shipped pointing at the
    container itself);
  * every image is pinned, every image we build runs as a numeric non-root user, and
    no compose service can publish a port beyond 127.0.0.1;
  * the build contexts keep the corpus and secrets out;
  * the Kubernetes manifests render, every field name is one the API accepts, every
    container has probes and resource limits, and no Secret carries a value.

Rendering needs `kubectl` (it embeds kustomize); without it those tests skip rather
than pretend. Field names are checked offline against the typed models in the
`kubernetes` client, so no cluster and no kubeconform is needed.
"""
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy"
DOCKER = DEPLOY / "docker"
K8S = DEPLOY / "k8s"

pytestmark = pytest.mark.skipif(not DOCKER.is_dir(), reason="no deploy/ in this checkout")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def generator():
    return _load(DOCKER / "make_docker_config.py", "make_docker_config")


# ---------------------------------------------------------------------------
# The config generator
# ---------------------------------------------------------------------------

def _generate(generator, tmp_path: Path, text: str | None = None) -> dict:
    src = tmp_path / "in.yaml"
    src.write_text(text if text is not None else (ROOT / "config.example.yaml").read_text(encoding="utf-8"),
                   encoding="utf-8")
    out = tmp_path / "out.yaml"
    assert generator.main(["x", str(src), str(out)]) == 0
    return yaml.safe_load(out.read_text(encoding="utf-8"))


def test_the_example_config_generates_a_container_config(generator, tmp_path):
    tree = _generate(generator, tmp_path)
    # Service names, not loopback: localhost inside the container is the container.
    assert tree["embedding"]["providers"]["eg2"]["base_url"] == "http://embed:8104/v1"
    assert tree["embedding"]["providers"]["eg2_l512"]["base_url"] == "http://embed-l512:8105/v1"
    assert tree["pdf"]["paddle_ocr"]["base_url"] == "http://paddleocr:8103"
    assert tree["paths"]["chroma_dir"] == "/rag_data/chroma_db"


def test_a_new_loopback_url_with_no_rewrite_fails_the_build(generator, tmp_path):
    """The anchors only catch a CHANGED line. A NEW local URL has no anchor and used
    to ship unreachable; the loopback sweep is what stops it."""
    text = (ROOT / "config.example.yaml").read_text(encoding="utf-8")
    text += '\nnew_sidecar:\n  base_url: "http://127.0.0.1:9999/v1"\n'
    with pytest.raises(SystemExit, match="loopback"):
        _generate(generator, tmp_path, text)


def test_a_drifted_anchor_fails_loudly(generator, tmp_path):
    text = (ROOT / "config.example.yaml").read_text(encoding="utf-8")
    with pytest.raises(SystemExit, match="expected 1 occurrence"):
        _generate(generator, tmp_path, text.replace('chroma_dir: "data/chroma_db"', 'chroma_dir: "elsewhere"'))


def test_the_personal_generator_runs_on_the_live_config(tmp_path):
    personal = DEPLOY / "package" / "make_personal_config.py"
    if not personal.is_file():
        pytest.skip("deploy/package is the author's own packaging; not in the public release")
    if not (ROOT / "config.yaml").is_file():
        pytest.skip("no live config.yaml in this checkout")
    mod = _load(personal, "make_personal_config")
    out = tmp_path / "personal.yaml"
    assert mod.main(["x", str(ROOT / "config.yaml"), str(out)]) == 0


# ---------------------------------------------------------------------------
# Images and compose
# ---------------------------------------------------------------------------

def _images(path: Path):
    """Every `image:` in a compose file, with the service it belongs to."""
    tree = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [(name, svc["image"]) for name, svc in tree["services"].items() if "image" in svc]


def _pinned(ref: str) -> bool:
    ref = re.sub(r"\$\{[^}]*:-([^}]*)\}", r"\1", ref)       # ${VAR:-default} -> default
    if "@sha256:" in ref:
        return True
    tag = ref.rsplit(":", 1)[1] if ":" in ref.rsplit("/", 1)[-1] else "latest"
    return tag != "latest"


COMPOSE_FILES = [DOCKER / "docker-compose.yml", DOCKER / "docker-compose.gpu.yml"]


@pytest.mark.parametrize("path", COMPOSE_FILES, ids=lambda p: p.name)
def test_compose_images_are_pinned_and_ports_stay_on_loopback(path):
    for service, ref in _images(path):
        assert _pinned(ref), f"{path.name}: {service} uses an unpinned image {ref!r}"
    tree = yaml.safe_load(path.read_text(encoding="utf-8"))
    for name, svc in tree["services"].items():
        for port in svc.get("ports", []):
            assert str(port).startswith("127.0.0.1:"), f"{path.name}: {name} publishes {port!r} beyond loopback"


def test_compose_has_no_secret_values_and_demands_the_ones_it_needs():
    text = (DOCKER / "docker-compose.yml").read_text(encoding="utf-8")
    assert re.search(r"sk-[A-Za-z0-9_\-]{16,}|hf_[A-Za-z0-9]{20,}", text) is None
    assert "ENCRYPTION_KEY:?" in text and "RAG_HOST_HOME:?" in text   # fail at `up`, not later
    services = yaml.safe_load(text)["services"]
    # The proxy gets ONLY the variables it needs, not every provider key in .env.
    assert all("env_file" not in svc for svc in services.values())


def test_the_api_token_is_optional_everywhere_it_is_passed():
    """RAG_API_TOKEN is optional: an unset one must not fail `docker compose up`, and
    every place a deployment takes it from names it."""
    rag = yaml.safe_load((DOCKER / "docker-compose.yml").read_text(encoding="utf-8"))["services"]["rag"]
    assert rag["environment"]["RAG_API_TOKEN"] == "${RAG_API_TOKEN:-}"
    assert '# RAG_API_TOKEN: ""' in (K8S / "base" / "secret.template.yaml").read_text(encoding="utf-8")
    for example in (ROOT / ".env.example", DOCKER / ".env.example"):
        assert re.search(r"(?m)^RAG_API_TOKEN=$", example.read_text(encoding="utf-8")), example


def _sh():
    found = shutil.which("sh")
    if found:
        return found
    git = shutil.which("git")                                        # Git for Windows ships its own
    cand = Path(git).resolve().parents[1] / "bin" / "sh.exe" if git else None
    return str(cand) if cand and cand.is_file() else None


@pytest.mark.skipif(_sh() is None, reason="no POSIX shell")
@pytest.mark.parametrize("token", ["", "s3cret"])
def test_the_live_probe_sends_the_token_to_the_console_only_when_one_is_set(tmp_path, token):
    """/health is exempt from the token and /docs is not, so `healthcheck.sh live`
    must carry it on the console probe when RAG_API_TOKEN is set, and never otherwise."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "curl").write_text('#!/bin/sh\necho "$*" >> "$FAKE_LOG"\n', encoding="utf-8", newline="\n")
    (bin_dir / "curl").chmod(0o755)
    log = tmp_path / "curl.log"
    env = {**os.environ, "FAKE_LOG": log.as_posix(), "RAG_API_TOKEN": token}
    # The stand-in goes first INSIDE the shell: Git for Windows' sh puts its own
    # /usr/bin (with a real curl) ahead of the PATH it inherits.
    done = subprocess.run([_sh(), "-c", 'PATH="$PWD/bin:$PATH"; export PATH; exec sh "$0" live',
                           (DOCKER / "healthcheck.sh").as_posix()],
                          cwd=tmp_path, capture_output=True, text=True, env=env, timeout=60)
    assert done.returncode == 0, done.stderr
    health, docs = log.read_text(encoding="utf-8").splitlines()
    assert health.endswith("/health") and "Authorization" not in health
    assert docs.endswith("/docs")
    assert ("Authorization: Bearer s3cret" in docs) is bool(token)


DOCKERFILES = [DOCKER / "Dockerfile", ROOT / "ocr-sidecar" / "Dockerfile",
               ROOT / "ocr-sidecar" / "Dockerfile.gpu", ROOT / "embed-sidecar" / "Dockerfile"]


@pytest.mark.parametrize("path", DOCKERFILES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_dockerfiles_pin_their_base_and_drop_root(path):
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    froms = [ln for ln in lines if ln.upper().startswith("FROM ")]
    assert froms and all("@sha256:" in ln for ln in froms), f"{path.name}: base image not pinned by digest"
    users = [ln.split(None, 1)[1] for ln in lines if ln.upper().startswith("USER ")]
    assert users, f"{path.name}: no USER, so the container runs as root"
    # Numeric (or an ARG that is): Kubernetes' runAsNonRoot cannot verify a name.
    assert not re.match(r"^[A-Za-z_][\w-]*$", users[-1]), f"{path.name}: USER {users[-1]!r} is a name, not a uid"


def test_the_app_image_copies_a_list_not_the_whole_context():
    lines = (DOCKER / "Dockerfile").read_text(encoding="utf-8").splitlines()
    copies = [ln for ln in lines if ln.strip().upper().startswith("COPY ")]
    assert copies
    for ln in copies:
        assert not re.search(r"\bCOPY\b(\s+--\S+)*\s+\.\s", ln), f"whole-context COPY: {ln.strip()}"


def test_the_build_context_rules_keep_the_corpus_and_secrets_out():
    rules = {ln.strip() for ln in (DOCKER / "Dockerfile.dockerignore").read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.lstrip().startswith("#")}
    for must in ("**/.env", "**/data", "**/rag_data", "**/*.jsonl", "**/*.pkl", "**/config.yaml",
                 "eval/sets", "eval/runs", "eval/seeds"):
        assert must in rules, f"Dockerfile.dockerignore does not exclude {must}"
    assert "!**/.env.example" in rules
    for sidecar in ("ocr-sidecar", "embed-sidecar"):
        allow = [ln.strip() for ln in (ROOT / sidecar / ".dockerignore").read_text(encoding="utf-8").splitlines()
                 if ln.strip() and not ln.lstrip().startswith("#")]
        assert allow[0] == "*", f"{sidecar}/.dockerignore is not an allowlist"


def test_ci_builds_on_pull_requests_without_pushing_and_publishes_only_by_hand():
    workflow = ROOT / ".github" / "workflows" / "docker-build.yml"
    if not workflow.is_file():
        pytest.skip("the workflow is not part of this checkout")
    wf = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    triggers = wf.get("on") or wf.get(True)         # YAML 1.1 reads a bare `on` as True
    assert "pull_request" in triggers and "workflow_dispatch" in triggers
    assert "push" not in triggers and "schedule" not in triggers
    for job in ("manifests", "images"):
        text = yaml.safe_dump(wf["jobs"][job])
        assert "login-action" not in text and "packages" not in text, f"{job} can reach a registry"
        for step in wf["jobs"][job]["steps"]:
            if "build-push-action" in step.get("uses", ""):
                assert step["with"]["push"] is False
    publish = wf["jobs"]["publish"]
    assert "workflow_dispatch" in publish["if"] and "inputs.publish" in publish["if"]
    assert wf["on" if "on" in wf else True]["workflow_dispatch"]["inputs"]["publish"]["default"] is False


def _ignore_rules(path: Path) -> list[tuple[bool, re.Pattern]]:
    """A .dockerignore as (is_exception, regex), compiled the way moby's
    patternmatcher does: `**` spans directories, `*` and `?` stop at `/`."""
    rules = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        negated = line.startswith("!")
        pattern = line.lstrip("!").strip("/")
        out, i = "^", 0
        while i < len(pattern):
            if pattern[i:i + 2] == "**":
                i += 2
                if pattern[i:i + 1] == "/":
                    i += 1
                    out += "(?:.*/)?"
                else:
                    out += ".*"
                continue
            out += "[^/]*" if pattern[i] == "*" else "[^/]" if pattern[i] == "?" else re.escape(pattern[i])
            i += 1
        rules.append((negated, re.compile(out + "$")))
    return rules


def _ignored(rel: str, rules) -> bool:
    """moby's MatchesOrParentMatches: the last matching rule wins, and a rule that
    matches a parent directory excludes everything beneath it."""
    parts = rel.split("/")
    matched = False
    for negated, rx in rules:
        if negated != matched:
            continue
        if rx.match(rel) or any(rx.match("/".join(parts[:n])) for n in range(1, len(parts))):
            matched = not negated
    return matched


RULES = DOCKER / "Dockerfile.dockerignore"


def test_every_file_the_app_dockerfile_copies_is_in_the_build_context():
    """A COPY source that Dockerfile.dockerignore excludes is not in the context, and
    the build fails. This bit once: `deploy` was excluded as a folder while the
    Dockerfile copies three files out of deploy/docker -- invisible in the flat
    release-archive layout, fatal in a checkout (and in CI)."""
    rules = _ignore_rules(RULES)
    args = {"APP_DIR": ".", "DEPLOY_DIR": "deploy/docker"}      # the checkout layout's defaults
    checked = 0
    for ln in (DOCKER / "Dockerfile").read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln.upper().startswith("COPY ") or "--from" in ln:
            continue
        tokens = [t for t in ln.split()[1:] if not t.startswith("--")]
        for src in tokens[:-1]:                                   # the last token is the destination
            src = re.sub(r"\$\{(\w+)\}", lambda m: args[m.group(1)], src)
            src = re.sub(r"^\./", "", src)
            found = sorted(ROOT.glob(src))
            assert found, f"COPY source {src!r} matches nothing in the repo"
            for hit in found:
                files = [p for p in hit.rglob("*") if p.is_file() and "__pycache__" not in p.parts] \
                    if hit.is_dir() else [hit]
                kept = [p for p in files if not _ignored(p.relative_to(ROOT).as_posix(), rules)]
                assert kept, f"COPY {src!r}: every file under {hit.relative_to(ROOT)} is excluded by Dockerfile.dockerignore"
                checked += 1
    assert checked >= 6


def test_the_ignore_rules_behave_as_documented_in_both_layouts():
    rules = _ignore_rules(RULES)
    kept_in_checkout = ["main.py", "src/utils/config_loader.py", "webui/index.html", "eval/bench/cli.py",
                        "eval/configs.yaml", "requirements.txt", ".env.example",
                        "deploy/docker/constraints.txt", "deploy/docker/docker-entrypoint.sh",
                        "deploy/docker/healthcheck.sh"]
    sent_never = ["data/chunks.jsonl", "data/vaults/x/chroma_db/chroma.sqlite3", "rag_data/bm25_index.pkl",
                  ".env", "deploy/docker/.env", "config.yaml", "deploy/docker/config.docker.yaml",
                  "eval/sets/books.yaml", "eval/runs/r1/run.json", "eval/seeds/s.jsonl", ".git/config",
                  "logs/jobs/j1.log", "src/ingestion/__pycache__/x.pyc"]
    assert not [p for p in kept_in_checkout if _ignored(p, rules)]
    assert [p for p in sent_never if not _ignored(p, rules)] == []
    # The release-archive layout: code under app/, the CORPUS at the root.
    kept_in_archive = ["app/main.py", "app/src/utils/config_loader.py", "app/webui/index.html",
                       "app/eval/sets/books.yaml",              # the author's own sets ship on purpose
                       "constraints.txt", "docker-entrypoint.sh", "healthcheck.sh", "Dockerfile"]
    never_in_archive = ["data/chunks.jsonl", "rag_data/chroma_db/chroma.sqlite3", "rag_data/bm25_index.pkl",
                        ".env", "config.docker.yaml", "app/data/x.jsonl", "app/logs/j.log"]
    assert not [p for p in kept_in_archive if _ignored(p, rules)]
    assert [p for p in never_in_archive if not _ignored(p, rules)] == []


def test_shell_scripts_have_unix_line_endings():
    for sh in DOCKER.glob("*.sh"):
        assert b"\r" not in sh.read_bytes(), f"{sh.name} has CRLF; the container cannot run it"


def test_the_mcp_server_starts_in_the_background_and_is_published_only_on_loopback():
    entry = (DOCKER / "docker-entrypoint.sh").read_text(encoding="utf-8")
    launch = "python mcp_server.py --http --host 0.0.0.0 &"
    assert launch in entry.splitlines()
    assert entry.index(launch) < entry.index("exec python -m uvicorn manage_api:app")   # before the console execs
    rag = yaml.safe_load((DOCKER / "docker-compose.yml").read_text(encoding="utf-8"))["services"]["rag"]
    assert "127.0.0.1:${NOETRIX_MCP_PORT:-8053}:8053" in rag["ports"]
    # MCP is optional: a down MCP server must never mark the RAG unhealthy.
    assert "mcp" not in (DOCKER / "healthcheck.sh").read_text(encoding="utf-8").lower()
    deployment = next(d for d in yaml.safe_load_all((K8S / "base" / "rag.yaml").read_text(encoding="utf-8"))
                      if d and d["kind"] == "Deployment")
    assert {"name": "mcp", "containerPort": 8053} in deployment["spec"]["template"]["spec"]["containers"][0]["ports"]


# ---------------------------------------------------------------------------
# Kubernetes
# ---------------------------------------------------------------------------

KUBECTL = shutil.which("kubectl")
needs_kubectl = pytest.mark.skipif(KUBECTL is None, reason="kubectl (which embeds kustomize) is not installed")


@pytest.fixture(scope="module")
def k8s_tree(tmp_path_factory):
    """A scratch copy of deploy/k8s with the generated config in place, so the repo
    tree is never written to and the test does not depend on a leftover file."""
    gen = _load(DOCKER / "make_docker_config.py", "make_docker_config_k8s")
    root = tmp_path_factory.mktemp("k8s") / "k8s"
    shutil.copytree(K8S, root, ignore=shutil.ignore_patterns("config.docker.yaml"))
    assert gen.main(["x", str(ROOT / "config.example.yaml"), str(root / "base" / "config.docker.yaml")]) == 0
    # An overlay that turns on every component, to prove they all render together.
    sink = root / "overlays" / "everything"
    sink.mkdir()
    sink.joinpath("kustomization.yaml").write_text(
        "apiVersion: kustomize.config.k8s.io/v1beta1\nkind: Kustomization\n"
        "resources: [../../base]\n"
        "components: [../../components/paddleocr, ../../components/rerank, ../../components/embed-gpu]\n"
        "images:\n  - {name: noetrix-rag, newTag: t}\n  - {name: noetrix-paddleocr, newTag: t}\n"
        "  - {name: noetrix-embed, newTag: t}\n", encoding="utf-8")
    gpu = root / "overlays" / "gpu-ocr"
    gpu.mkdir()
    gpu.joinpath("kustomization.yaml").write_text(
        "apiVersion: kustomize.config.k8s.io/v1beta1\nkind: Kustomization\n"
        "resources: [../../base]\ncomponents: [../../components/paddleocr-gpu]\n"
        "images:\n  - {name: noetrix-rag, newTag: t}\n  - {name: noetrix-paddleocr-gpu, newTag: t}\n", encoding="utf-8")
    return root


def _render(root: Path, overlay: str) -> list[dict]:
    done = subprocess.run([KUBECTL, "kustomize", str(root / "overlays" / overlay)],
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    return [d for d in yaml.safe_load_all(done.stdout) if d]


OVERLAYS = ["local", "cloud", "everything", "gpu-ocr"]


def _unknown_fields(value, model, path="") -> list[str]:
    """Field names the API would reject, found by walking a manifest against the
    typed models in the `kubernetes` client (generated from the OpenAPI spec)."""
    from kubernetes import client
    problems: list[str] = []
    if isinstance(model, str):
        m = re.match(r"list\[(.+)\]", model)
        if m:
            for i, item in enumerate(value or []):
                problems += _unknown_fields(item, m.group(1), f"{path}[{i}]")
            return problems
        m = re.match(r"dict\(str, (.+)\)", model)
        if m:
            for k, item in (value or {}).items():
                problems += _unknown_fields(item, m.group(1), f"{path}.{k}")
            return problems
        cls = getattr(client, model, None)
        if cls is None or not hasattr(cls, "openapi_types"):
            return problems              # a scalar or an opaque type
        model = cls
    if not isinstance(value, dict):
        return problems
    by_json = {json_name: attr for attr, json_name in model.attribute_map.items()}
    for key, sub in value.items():
        if key not in by_json:
            problems.append(f"{path}.{key} is not a field of {model.__name__}")
            continue
        problems += _unknown_fields(sub, model.openapi_types[by_json[key]], f"{path}.{key}")
    return problems


KINDS = {"Deployment": "V1Deployment", "Service": "V1Service", "PersistentVolumeClaim": "V1PersistentVolumeClaim",
         "ConfigMap": "V1ConfigMap", "Namespace": "V1Namespace", "Secret": "V1Secret"}


@needs_kubectl
@pytest.mark.parametrize("overlay", OVERLAYS)
def test_every_overlay_renders_with_only_fields_the_api_accepts(k8s_tree, overlay):
    pytest.importorskip("kubernetes")
    docs = _render(k8s_tree, overlay)
    assert docs
    for doc in docs:
        assert doc["kind"] in KINDS, f"unexpected kind {doc['kind']}"
        bad = _unknown_fields(doc, KINDS[doc["kind"]], doc["kind"] + "/" + doc["metadata"]["name"])
        assert not bad, "\n".join(bad)


@needs_kubectl
@pytest.mark.parametrize("overlay", OVERLAYS)
def test_workloads_have_probes_limits_pinned_images_and_no_added_privilege(k8s_tree, overlay):
    for doc in _render(k8s_tree, overlay):
        if doc["kind"] != "Deployment":
            continue
        name = doc["metadata"]["name"]
        pod = doc["spec"]["template"]["spec"]
        assert doc["spec"]["replicas"] == 1 and doc["spec"]["strategy"]["type"] == "Recreate", \
            f"{name}: single writer on a ReadWriteOnce volume"
        # Kubernetes injects RAG_PORT=tcp://ip:8051 for a Service named `rag` -- the
        # variable the entrypoint reads as a port NUMBER -- unless service links are off.
        assert pod.get("enableServiceLinks") is False, f"{name}: service-link variables can shadow its settings"
        for c in pod["containers"]:
            where = f"{name}/{c['name']}"
            assert c["resources"]["requests"]["memory"] and c["resources"]["limits"]["memory"], where
            assert {"startupProbe", "readinessProbe", "livenessProbe"} <= set(c), f"{where}: missing a probe"
            assert not c.get("securityContext", {}).get("privileged"), where
            assert ":latest" not in c["image"], f"{where}: {c['image']}"
            # Our own images are overlay-named (no tag or digest in the base); a
            # full reference must be pinned.
            if "/" in c["image"] or "@" in c["image"]:
                assert "@sha256:" in c["image"] or not c["image"].endswith(":latest"), where
        if name != "freellmapi" and name != "rerank":     # the upstream images that declare root
            sc = pod["securityContext"]
            assert sc["runAsNonRoot"] is True and sc["runAsUser"] == 10001 and sc["fsGroup"] == 10001, name


@needs_kubectl
def test_the_rag_pod_is_ready_only_when_the_index_is_and_the_console_stays_reachable(k8s_tree):
    docs = _render(k8s_tree, "local")
    rag = next(d for d in docs if d["kind"] == "Deployment" and d["metadata"]["name"] == "rag")
    c = rag["spec"]["template"]["spec"]["containers"][0]
    assert c["readinessProbe"]["exec"]["command"] == ["healthcheck.sh"]            # "ready":true
    assert c["livenessProbe"]["exec"]["command"] == ["healthcheck.sh", "live"]     # not the index
    assert c["startupProbe"]["exec"]["command"] == ["healthcheck.sh", "live"]
    console = next(d for d in docs if d["kind"] == "Service" and d["metadata"]["name"] == "rag-console")
    assert console["spec"]["publishNotReadyAddresses"] is True
    cm = [d for d in docs if d["kind"] == "ConfigMap"]
    assert len(cm) == 1 and "config.yaml" in cm[0]["data"]
    assert re.fullmatch(r"noetrix-config-[a-z0-9]+", cm[0]["metadata"]["name"])    # hashed: a config change rolls the pod


@needs_kubectl
def test_the_proxy_is_probed_on_the_endpoint_that_needs_no_key(k8s_tree):
    """The pinned FreeLLMAPI answers /v1/models with 401 without a key (found by running
    it), so a probe there would never go ready. /api/ping is what the image's own
    HEALTHCHECK uses."""
    proxy = next(d for d in _render(k8s_tree, "local")
                 if d["kind"] == "Deployment" and d["metadata"]["name"] == "freellmapi")
    c = proxy["spec"]["template"]["spec"]["containers"][0]
    for probe in ("startupProbe", "readinessProbe", "livenessProbe"):
        assert c[probe]["httpGet"]["path"] == "/api/ping", probe


@needs_kubectl
def test_no_overlay_renders_a_secret(k8s_tree):
    """Secrets are created out of band; a rendered Secret would mean a value, or an
    empty one that overwrites the real thing on every apply."""
    for overlay in OVERLAYS:
        assert all(d["kind"] != "Secret" for d in _render(k8s_tree, overlay)), overlay


def test_the_secret_template_has_keys_but_no_values():
    docs = [d for d in yaml.safe_load_all((K8S / "base" / "secret.template.yaml").read_text(encoding="utf-8")) if d]
    assert {d["metadata"]["name"] for d in docs} == {"noetrix-proxy", "noetrix-provider-keys"}
    for d in docs:
        assert d["kind"] == "Secret"
        assert not any(d.get("stringData", {}).values()), f"{d['metadata']['name']} carries a value"
        assert "data" not in d
    base = yaml.safe_load((K8S / "base" / "kustomization.yaml").read_text(encoding="utf-8"))
    assert "secret.template.yaml" not in base["resources"]
