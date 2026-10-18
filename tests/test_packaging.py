"""Packaging: the archives, the launchers and the skills -- checked without Docker.

tests/test_deploy_assets.py guards the images and the manifests. This file guards what
turns them into something a person can install on ANY machine:

  * the app allowlist (deploy/package/common.ps1) names files that exist, and together
    with the deploy files it satisfies every COPY in the Dockerfile in the ARCHIVE
    layout -- the layout a stranger builds from, which the checkout-layout test does not
    see;
  * compose keeps working across the three host shapes (macOS/Linux, Windows with a
    drive-letter home, remapped ports) -- evaluated here with a small interpolator, and
    cross-checked against `docker compose config` when the CLI is present;
  * the launchers (start.sh, start.ps1) install the skills byte for byte, idempotently,
    and refuse a mislabelled one -- run for real wherever a shell is available;
  * no .env.example carries a secret;
  * the cold archive builds end to end (Windows PowerShell 5.1, where the packager runs)
    and holds exactly what it should: no corpus, no key, current app, both skills.

Anything that needs a tool the machine lacks skips, rather than pretending.
"""
import fnmatch
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
DOCKER = ROOT / "deploy" / "docker"
PKG = ROOT / "deploy" / "package"

pytestmark = pytest.mark.skipif(not PKG.is_dir(),
                                reason="deploy/package is the author's own packaging; not in the public release")


def _ps_list(name: str) -> list[str]:
    """The quoted items of `$script:<name> = @( ... )` in common.ps1."""
    text = (PKG / "common.ps1").read_text(encoding="utf-8")
    body = re.search(rf"\$script:{name}\s*=\s*@\((.*?)\)", text, re.S)
    assert body, f"common.ps1 has no $script:{name} list"
    return re.findall(r'"([^"]+)"', body.group(1))


def _helpers():
    """The ignore-rule matcher from test_deploy_assets (tests/ is not a package)."""
    spec = importlib.util.spec_from_file_location("deploy_assets_helpers", ROOT / "tests" / "test_deploy_assets.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _copy_sources(args: dict[str, str]) -> list[str]:
    """Every COPY source in the app Dockerfile, with its build args filled in."""
    out = []
    for ln in (DOCKER / "Dockerfile").read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln.upper().startswith("COPY ") or "--from" in ln:
            continue
        tokens = [t for t in ln.split()[1:] if not t.startswith("--")]
        for src in tokens[:-1]:                                  # the last token is the destination
            out.append(re.sub(r"^\./", "", re.sub(r"\$\{(\w+)\}", lambda m: args[m.group(1)], src)))
    return out


# ---------------------------------------------------------------------------
# The allowlists and the Dockerfile
# ---------------------------------------------------------------------------

def test_the_allowlists_name_things_that_exist():
    for f in _ps_list("AppFiles"):
        assert (ROOT / f).is_file(), f"common.ps1 AppFiles lists {f}, which is not in the repo"
    for d in _ps_list("AppDirs"):
        assert (ROOT / d).is_dir(), f"common.ps1 AppDirs lists {d}\\, which is not in the repo"
    for f in _ps_list("DeployFiles"):
        assert (DOCKER / f).is_file(), f"common.ps1 DeployFiles lists {f}, which is not in deploy/docker"
    for f in _ps_list("ArchiveRootFiles"):
        assert (PKG / f).is_file(), f"common.ps1 ArchiveRootFiles lists {f}, which is not in deploy/package"
    assert set(_ps_list("PublicSkipFiles")) <= set(_ps_list("AppFiles"))


def test_every_root_script_ships():
    """Console jobs run root scripts (rebuild_bm25.py, build_hype.py, ...). One missing
    from AppFiles leaves its job failing inside the container, and nothing else notices."""
    missing = sorted(p.name for p in ROOT.glob("*.py") if p.name not in _ps_list("AppFiles"))
    assert not missing, f"root scripts not in common.ps1 AppFiles: {missing}"


def test_the_archive_layout_satisfies_every_dockerfile_copy():
    """The archive builds with APP_DIR=app and DEPLOY_DIR=. (docker-compose.override.yml).
    Each COPY source must then be something the packager puts there."""
    override = yaml.safe_load((PKG / "docker-compose.override.yml").read_text(encoding="utf-8"))
    args = override["services"]["rag"]["build"]["args"]
    assert args == {"APP_DIR": "app", "DEPLOY_DIR": "."}, "the override and this test disagree about the layout"
    files, dirs, deploy = _ps_list("AppFiles"), _ps_list("AppDirs"), _ps_list("DeployFiles")
    sources = _copy_sources(args)
    assert len(sources) >= 6
    for src in sources:
        if src.startswith("app/"):
            rest = src[len("app/"):]
            if "*" in rest:
                assert any(fnmatch.fnmatch(f, rest) for f in files), f"nothing in AppFiles matches COPY {src}"
            else:
                assert rest in files or rest in dirs, f"COPY {src}: common.ps1 does not stage it into app/"
        else:
            assert src in deploy, f"COPY {src}: common.ps1 does not stage it at the archive root"


def test_every_script_the_windows_shell_reads_is_ascii():
    """Windows PowerShell 5.1 reads a BOM-less file as the ANSI code page: one typographic
    dash is a parse error on somebody else's machine."""
    for ps1 in PKG.glob("*.ps1"):
        bad = [(n, ln) for n, ln in enumerate(ps1.read_text(encoding="utf-8").splitlines(), 1) if not ln.isascii()]
        assert not bad, f"{ps1.name} has non-ASCII text, e.g. line {bad[0][0]}"


def test_the_launchers_have_the_right_line_endings():
    assert b"\r" not in (PKG / "start.sh").read_bytes(), "start.sh has CRLF; it cannot run on macOS or Linux"


def test_the_stack_runs_on_cpu_alone_and_gpu_services_stay_opt_in():
    """A Mac has no NVIDIA runtime: the base file may not request a GPU anywhere, and the
    GPU overlay may not be pulled into a bare `docker compose up`."""
    base = yaml.safe_load((DOCKER / "docker-compose.yml").read_text(encoding="utf-8"))
    for name, svc in base["services"].items():
        text = yaml.safe_dump(svc)
        assert "nvidia" not in text and "gpus" not in svc and "runtime" not in svc, f"{name} wants a GPU"
        assert "platform" not in svc, f"{name} pins a platform; it would run emulated on the other architecture"
    gpu = yaml.safe_load((DOCKER / "docker-compose.gpu.yml").read_text(encoding="utf-8"))
    assert all(svc.get("profiles") for svc in gpu["services"].values()), "a GPU service starts without its profile"
    optional = {n for n, s in base["services"].items() if s.get("profiles")}
    assert optional == {"paddleocr", "paddle-cache-perms", "rerank"}
    assert not (PKG / "docker-compose.gpu.yml").exists()


# ---------------------------------------------------------------------------
# .env.example: no secret, ever
# ---------------------------------------------------------------------------

SECRET_SHAPE = re.compile(r"sk-[A-Za-z0-9_\-]{16,}|hf_[A-Za-z0-9]{20,}|gsk_[A-Za-z0-9]{20,}")


@pytest.mark.parametrize("path", [ROOT / ".env.example", DOCKER / ".env.example"], ids=["root", "deploy"])
def test_env_examples_carry_no_secret(path):
    text = path.read_text(encoding="utf-8")
    assert not SECRET_SHAPE.search(text), f"{path.name} contains a key-shaped value"
    for ln in text.splitlines():
        m = re.match(r"^([A-Z][A-Z0-9_]*)=(.*)$", ln)
        if m and re.search(r"(KEY|TOKEN|SECRET)$", m.group(1)):
            assert re.fullmatch(r"(sk-(ant-)?\.\.\.)?", m.group(2).strip()), \
                f"{path.name}: {m.group(1)} has a value; ship it empty or as a placeholder"


# ---------------------------------------------------------------------------
# Compose across host shapes
# ---------------------------------------------------------------------------

def _interp(s: str, env: dict[str, str]) -> str:
    """Compose interpolation: ${V}, ${V:-default}, ${V:?message}; defaults may nest."""
    out, i = "", 0
    while i < len(s):
        if not s.startswith("${", i):
            out += s[i]
            i += 1
            continue
        depth, j = 1, i + 2
        while depth:
            if s.startswith("${", j):
                depth, j = depth + 1, j + 2
                continue
            depth -= s[j] == "}"
            j += 1
        name, op, arg = re.match(r"(\w+)(?::([-?])(.*))?$", s[i + 2:j - 1], re.S).groups()
        val = env.get(name, "")
        if op == "-" and not val:
            val = _interp(arg, env)
        elif op == "?" and not val:
            raise KeyError(_interp(arg, env))
        out, i = out + val, j
    return out


def _rag(env: dict[str, str]) -> dict:
    svc = yaml.safe_load((DOCKER / "docker-compose.yml").read_text(encoding="utf-8"))["services"]["rag"]
    home = next(v for v in svc["volumes"] if isinstance(v, str) and "RAG_HOST_HOME" in v)
    return {"ports": [_interp(p, env) for p in svc["ports"]],
            "env": {k: _interp(v, env) for k, v in svc["environment"].items()},
            "home": _interp(home, env)}


MAC = {"RAG_HOST_HOME": "/Users/a", "ENCRYPTION_KEY": "k"}
WINDOWS = {"RAG_HOST_HOME": "C:/Users/a", "RAG_CONTAINER_HOME": "/c/Users/a", "ENCRYPTION_KEY": "k"}
REMAPPED = {**MAC, "NOETRIX_API_PORT": "9051", "NOETRIX_CONSOLE_PORT": "9052", "NOETRIX_MCP_PORT": "9053"}


def test_macos_and_linux_mount_home_at_the_same_path():
    rag = _rag(MAC)
    assert rag["home"] == "/Users/a:/Users/a"
    assert rag["env"]["RAG_HOST_HOME"] == "/Users/a"               # what config.docker.yaml expands
    assert rag["ports"] == ["127.0.0.1:8051:8051", "127.0.0.1:8052:8052", "127.0.0.1:8053:8053"]


def test_windows_mounts_a_drive_path_onto_a_linux_path_and_the_config_sees_the_linux_one():
    rag = _rag(WINDOWS)
    assert rag["home"] == "C:/Users/a:/c/Users/a"                  # a drive path cannot be a container path
    assert rag["env"]["RAG_HOST_HOME"] == "/c/Users/a"             # so the vault_path the app opens is the container's


def test_a_remapped_host_port_is_the_port_inside_too():
    """The console's PAGE calls webui.rag_api from the browser: if only the host side moved,
    its Query tab would talk to whatever owns :8051."""
    rag = _rag(REMAPPED)
    # The MCP port moves on the host side only: no page in a browser names it.
    assert rag["ports"] == ["127.0.0.1:9051:9051", "127.0.0.1:9052:9052", "127.0.0.1:9053:8053"]
    assert rag["env"]["RAG_PORT"] == "9051" and rag["env"]["RAG_CONSOLE_PORT"] == "9052"


def test_compose_still_refuses_to_start_without_a_home():
    with pytest.raises(KeyError, match="RAG_HOST_HOME"):
        _rag({"ENCRYPTION_KEY": "k"})


@pytest.mark.parametrize("scenario,env", [("mac", MAC), ("remapped", REMAPPED),
                                          pytest.param("windows", WINDOWS, marks=pytest.mark.skipif(
                                              sys.platform != "win32", reason="drive paths parse on Windows hosts"))])
def test_the_interpolator_agrees_with_docker_compose(scenario, env):
    """`docker compose config` needs no daemon, so this runs wherever the CLI is installed."""
    if shutil.which("docker") is None:
        pytest.skip("no docker CLI")
    done = subprocess.run(["docker", "compose", "-f", str(DOCKER / "docker-compose.yml"), "config", "--format", "json"],
                          capture_output=True, text=True, env={**os.environ, **env}, timeout=60)
    if done.returncode != 0:
        pytest.skip(f"docker compose config unavailable: {done.stderr.strip()[:120]}")
    real = json.loads(done.stdout)["services"]["rag"]
    mine = _rag(env)
    assert real["environment"]["RAG_HOST_HOME"] == mine["env"]["RAG_HOST_HOME"]
    assert real["environment"]["RAG_PORT"] == mine["env"]["RAG_PORT"]
    assert [f"{p['published']}:{p['target']}" for p in real["ports"]] == [p.split(":", 1)[1] for p in mine["ports"]]
    home = next(v for v in real["volumes"] if v["target"] == mine["home"].rsplit(":", 1)[1])
    assert home["source"] == mine["home"].rsplit(":", 1)[0]


# ---------------------------------------------------------------------------
# The generated config follows the port
# ---------------------------------------------------------------------------

def test_the_generated_config_points_the_console_at_the_published_api_port(tmp_path, monkeypatch):
    gen = _helpers()._load(DOCKER / "make_docker_config.py", "make_docker_config_pkg")
    out = tmp_path / "out.yaml"
    assert gen.main(["x", str(ROOT / "config.example.yaml"), str(out)]) == 0
    assert b"\r" not in out.read_bytes(), "the generated config must be LF whichever OS built it"
    rag_api = yaml.safe_load(out.read_text(encoding="utf-8"))["webui"]["rag_api"]
    sys.path.insert(0, str(ROOT))
    from src.utils.config_loader import expand_env
    monkeypatch.setenv("RAG_PORT", "18051")
    assert expand_env(rag_api) == "http://127.0.0.1:18051"
    monkeypatch.delenv("RAG_PORT")
    assert expand_env(rag_api) == "http://127.0.0.1:8051"          # Kubernetes sets nothing


# ---------------------------------------------------------------------------
# The skills and the launchers
# ---------------------------------------------------------------------------

PERSONAL = "---\nname: personal-rag\ndescription: stand-in for the curated definition\n---\n\n# body\n"


def test_the_ops_skill_is_shippable_as_it_stands():
    text = (ROOT / "SKILL_RAGOPS.md").read_text(encoding="utf-8")
    assert re.search(r"(?m)^name: rag-ops\r?$", text)
    private = re.search(re.search(r"PrivateTokenRe\s*=\s*'([^']+)'", (PKG / "common.ps1").read_text(encoding="utf-8")).group(1),
                        text, re.I)
    assert not private, f"SKILL_RAGOPS.md carries a private identifier: {private and private.group(0)}"


def _bundle_with_skills(tmp_path: Path, launcher: str) -> Path:
    bundle = tmp_path / "bundle"
    (bundle / "skills" / "rag-ops").mkdir(parents=True)
    (bundle / "skills" / "personal-rag").mkdir(parents=True)
    shutil.copy(ROOT / "SKILL_RAGOPS.md", bundle / "skills" / "rag-ops" / "SKILL.md")
    (bundle / "skills" / "personal-rag" / "SKILL.md").write_text(PERSONAL, encoding="utf-8", newline="\n")
    shutil.copy(PKG / launcher, bundle / launcher)
    return bundle


def _sh():
    found = shutil.which("sh")
    if found:
        return found
    git = shutil.which("git")                                        # Git for Windows ships its own
    cand = Path(git).resolve().parents[1] / "bin" / "sh.exe" if git else None
    return str(cand) if cand and cand.is_file() else None


def _check_skills_install(run, bundle: Path, dest: Path):
    first = run(dest)
    assert first.returncode == 0, first.stderr
    for name in ("personal-rag", "rag-ops"):
        assert (dest / name / "SKILL.md").read_bytes() == (bundle / "skills" / name / "SKILL.md").read_bytes()
    assert len(re.findall(r"(?m)^installed ", first.stdout)) == 2
    again = run(dest)
    assert again.returncode == 0 and len(re.findall(r"(?m)^unchanged ", again.stdout)) == 2     # idempotent
    (bundle / "skills" / "personal-rag" / "SKILL.md").write_text(PERSONAL + "more\n", encoding="utf-8", newline="\n")
    assert re.search(r"(?m)^updated ", run(dest).stdout)                      # a newer archive updates it
    bogus = bundle / "skills" / "bogus"
    bogus.mkdir()
    (bogus / "SKILL.md").write_text("---\nname: something-else\n---\n", encoding="utf-8")
    refused = run(dest)
    assert refused.returncode != 0 and "does not declare" in (refused.stdout + refused.stderr)


@pytest.mark.skipif(_sh() is None, reason="no POSIX shell")
def test_start_sh_installs_the_skills(tmp_path):
    bundle = _bundle_with_skills(tmp_path, "start.sh")
    dest = tmp_path / "dest"
    syntax = subprocess.run([_sh(), "-n", "start.sh"], cwd=bundle, capture_output=True, text=True)
    assert syntax.returncode == 0, syntax.stderr
    _check_skills_install(lambda d: subprocess.run([_sh(), "start.sh", "skills", "--dest", d.as_posix()], cwd=bundle,
                                                   capture_output=True, text=True, timeout=60), bundle, dest)


# Stand-ins for `docker` and `curl`, so start.sh's real branches run without an engine.
FAKE_DOCKER = """#!/bin/sh
echo "$*" >> "$FAKE_LOG"
case "$1" in
  info)    [ -z "${FAKE_ENGINE_DOWN:-}" ] || exit 1
           case "$*" in *MemTotal*) echo 17179869184 ;; *OperatingSystem*) echo Ubuntu ;; esac ;;
  version) echo linux/arm64 ;;
  compose) case "$2" in
             version) ;;
             up)   exit "${FAKE_UP_EXIT:-0}" ;;
             ps)   echo "${FAKE_CONTAINER_STATE:-running}" ;;
             logs) echo "fake rag log line" ;;
           esac ;;
esac
exit 0
"""
FAKE_CURL = """#!/bin/sh
for last in "$@"; do :; done
case "$last" in
  */health) [ -n "${FAKE_HEALTH:-}" ] || exit 7; printf '%s' "$FAKE_HEALTH" ;;
  */)       exit 0 ;;                 # the console's page shell, exempt from the token
  *)        exit 22 ;;
esac
"""


def _start_sh_run(tmp_path: Path, **env):
    bundle = tmp_path / "bundle"
    bin_dir, home = tmp_path / "bin", tmp_path / "home"
    if not bundle.exists():
        bundle.mkdir()
        bin_dir.mkdir()
        home.mkdir()
        for name, body in (("docker", FAKE_DOCKER), ("curl", FAKE_CURL)):
            (bin_dir / name).write_text(body, encoding="utf-8", newline="\n")
            # Linux skips a non-executable file on PATH and runs the real one behind it;
            # Git for Windows' sh goes by the shebang, which hid this locally.
            (bin_dir / name).chmod(0o755)
        shutil.copy(PKG / "start.sh", bundle / "start.sh")
        shutil.copy(DOCKER / ".env.example", bundle / ".env.example")
        (bundle / "docker-compose.yml").write_text("services: {}\n", encoding="utf-8")
        (bundle / "config.docker.yaml").write_text("x: 1\n", encoding="utf-8")
        # Ports nothing listens on: if a real curl ever slipped in, it could not reach a real stack.
        (bundle / ".env").write_text("NOETRIX_API_PORT=9\nNOETRIX_CONSOLE_PORT=9\n", encoding="utf-8", newline="\n")
    full = {**os.environ, "HOME": home.as_posix(), "FAKE_LOG": (tmp_path / "docker.log").as_posix(),
            "NOETRIX_WAIT_S": "20", **env}                    # bounded, so a miss ends in a message, not a hang
    # Put the stand-ins first INSIDE the shell: Git for Windows' sh puts its own /usr/bin and
    # /mingw64/bin (with a real curl) ahead of whatever PATH it inherited.
    done = subprocess.run([_sh(), "-c", 'PATH="$PWD/bin:$PATH"; export PATH; cd bundle && exec sh start.sh'],
                          cwd=tmp_path, capture_output=True, text=True, env=full, timeout=90)
    return done, bundle, home


READY = '{"ready":true,"state":"ready","uptime_s":3}'


@pytest.mark.skipif(_sh() is None, reason="no POSIX shell")
def test_start_sh_sets_up_env_starts_and_waits_for_ready(tmp_path):
    done, bundle, home = _start_sh_run(tmp_path, FAKE_HEALTH=READY)
    assert done.returncode == 0, done.stderr
    env = dict(ln.split("=", 1) for ln in (bundle / ".env").read_text(encoding="utf-8").splitlines()
               if re.match(r"^[A-Z_]+=", ln))
    # the placeholder became the real home (as the shell spells it: Git's sh rewrites a drive path)
    assert env["RAG_HOST_HOME"].endswith("/" + home.name) and env["RAG_HOST_HOME"] != "/Users/yourname"
    assert re.fullmatch(r"[0-9a-f]{64}", env["ENCRYPTION_KEY"])          # generated, because there was none
    assert "compose up --build -d" in (tmp_path / "docker.log").read_text(encoding="utf-8")
    assert "Noetrix is running" in done.stdout and "http://127.0.0.1:9" in done.stdout
    key = env["ENCRYPTION_KEY"]
    again, _, _ = _start_sh_run(tmp_path, FAKE_HEALTH=READY)
    assert again.returncode == 0 and f"ENCRYPTION_KEY={key}" in (bundle / ".env").read_text(encoding="utf-8")
    assert "generated ENCRYPTION_KEY" not in again.stdout                # an existing key is never replaced


@pytest.mark.skipif(_sh() is None, reason="no POSIX shell")
@pytest.mark.parametrize("env,must_say", [
    ({"FAKE_ENGINE_DOWN": "1"}, "engine is not running"),
    ({"FAKE_UP_EXIT": "1", "FAKE_HEALTH": READY}, "NOETRIX_API_PORT"),
    ({"FAKE_HEALTH": '{"ready":false,"state":"failed","error":"boom"}'}, "boom"),
    ({"FAKE_CONTAINER_STATE": "exited"}, "stopped"),
    ({"FAKE_CONTAINER_STATE": "running", "NOETRIX_WAIT_S": "0"}, "not ready after 0s"),
], ids=["engine-down", "compose-up-fails", "pipeline-failed", "container-stopped", "timeout"])
def test_start_sh_fails_readably(tmp_path, env, must_say):
    done, _, _ = _start_sh_run(tmp_path, **env)
    assert done.returncode != 0
    assert done.stderr.startswith("ERROR:") or "\nERROR:" in done.stderr, done.stderr
    assert must_say in done.stderr, done.stderr


def test_the_launchers_probe_the_consoles_page_shell_which_needs_no_token():
    """GET / is exempt from RAG_API_TOKEN and /docs is not: a launcher that probed /docs
    would call a healthy console dead whenever a token is set."""
    for name in ("start.sh", "start.ps1"):
        text = (PKG / name).read_text(encoding="utf-8")
        assert '"http://127.0.0.1:$console/"' in text, name
        assert "$console/docs" not in text, name


@pytest.mark.parametrize("token", ["", "s3cret"])
def test_the_smoke_probe_sends_the_token_only_when_one_is_set(monkeypatch, token):
    spec = importlib.util.spec_from_file_location("noetrix_smoke", PKG / "smoke.py")
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    sent = []

    def record(req, timeout):
        sent.append(req)
        raise OSError("nothing is sent in this test")
    monkeypatch.setattr(smoke.urllib.request, "urlopen", record)
    monkeypatch.setenv("RAG_API_TOKEN", token)
    assert smoke.http("GET", "http://127.0.0.1:9/api/jobs")[0] == 0
    assert sent[0].get_header("Authorization") == ("Bearer s3cret" if token else None)


@pytest.mark.skipif(sys.platform != "win32" or shutil.which("powershell") is None, reason="Windows PowerShell only")
def test_start_ps1_installs_the_skills(tmp_path):
    bundle = _bundle_with_skills(tmp_path, "start.ps1")
    dest = tmp_path / "dest"
    _check_skills_install(
        lambda d: subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "start.ps1",
                                  "skills", "-Dest", str(d)], cwd=bundle, capture_output=True, text=True, timeout=120),
        bundle, dest)


# ---------------------------------------------------------------------------
# The cold archive, built for real
# ---------------------------------------------------------------------------

@pytest.mark.skipif(sys.platform != "win32" or shutil.which("powershell") is None,
                    reason="the packager is Windows PowerShell (it uses Windows' own tar.exe)")
def test_the_cold_archive_builds_and_holds_what_it_should(tmp_path):
    skill = tmp_path / "personal-rag-SKILL.md"
    skill.write_text(PERSONAL, encoding="utf-8", newline="\n")
    out = tmp_path / "out"
    out.mkdir()
    done = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(PKG / "package-cold.ps1"),
         "-ProjectDir", str(ROOT), "-OutDir", str(out), "-Python", sys.executable, "-PersonalRagSkill", str(skill)],
        capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    archive = out / "Noetrix-cold.tar.gz"
    digest = (out / "Noetrix-cold.tar.gz.sha256").read_text(encoding="ascii").split()
    assert digest[0] == hashlib.sha256(archive.read_bytes()).hexdigest() and digest[1] == archive.name

    unpacked = tmp_path / "unpacked"
    with tarfile.open(archive) as tf:
        tf.extractall(unpacked, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
    bundle = unpacked / "rag-docker-bundle"

    assert not (bundle / "docker-compose.gpu.yml").exists(), "a GPU overlay must not ship to machines without NVIDIA"
    assert not (bundle / "paddleocr" / "Dockerfile.gpu").exists()

    # what a stranger needs to build and run
    for f in ["docker-compose.yml", "docker-compose.override.yml", "Dockerfile", "Dockerfile.dockerignore", "constraints.txt",
              "docker-entrypoint.sh", "healthcheck.sh", "config.docker.yaml", "start.sh", "start.ps1", "INSTALL.md",
              "BUILD_INFO.txt", ".env", ".env.example", "paddleocr/server.py", "paddleocr/Dockerfile"]:
        assert (bundle / f).is_file(), f"the archive has no {f}"
    assert b"\r" not in (bundle / "start.sh").read_bytes() and b"\r" not in (bundle / "docker-entrypoint.sh").read_bytes()

    # the app is the allowlist, nothing more, and current
    for f in _ps_list("AppFiles"):
        assert (bundle / "app" / f).is_file(), f"app/{f} is missing from the archive"
    assert (bundle / "app" / "manage_api.py").read_bytes() == (ROOT / "manage_api.py").read_bytes()
    for never in ["config.yaml", ".env", "CLAUDE.md", "WORKLOG.md", "INSTRUCTIONS.md", "rag.bat"]:
        assert not (bundle / "app" / never).exists(), f"app/{never} must not ship"

    # no corpus, no key
    assert not [p for p in bundle.rglob("*") if p.suffix in {".jsonl", ".pkl", ".sqlite3"}], "a corpus file shipped"
    assert [p.name for p in (bundle / "data").iterdir()] == [".gitkeep"]
    keys = dict(ln.split("=", 1) for ln in (bundle / ".env").read_text(encoding="utf-8").splitlines() if re.match(r"^[A-Z_]+=", ln))
    assert set(keys) == {"RAG_HOST_HOME", "ENCRYPTION_KEY"} and re.fullmatch(r"[0-9a-f]{64}", keys["ENCRYPTION_KEY"])
    assert not SECRET_SHAPE.search("\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in bundle.rglob("*")
                                             if p.is_file() and p.suffix not in {".png", ".jpg", ".gz"}))
    # no question set either: the sets quote the vault, and only the with-data archive carries one
    assert (bundle / "app" / "eval" / "bench" / "cli.py").is_file()
    for sets in ["sets", "smoke"]:
        assert not (bundle / "app" / "eval" / sets).exists(), f"app/eval/{sets} must not ship in the cold archive"

    # the skills: rag-ops is the project's file byte for byte; both are recorded in BUILD_INFO
    assert (bundle / "skills" / "rag-ops" / "SKILL.md").read_bytes() == (ROOT / "SKILL_RAGOPS.md").read_bytes()
    assert (bundle / "skills" / "personal-rag" / "SKILL.md").read_bytes() == skill.read_bytes()
    info = (bundle / "BUILD_INFO.txt").read_text(encoding="utf-8")
    assert re.search(r"(?m)^commit  : ", info) and re.search(r"(?m)^skill   : rag-ops sha256 [0-9a-f]{64}$", info)

    # and the Dockerfile builds from it: every COPY source is there AND survives the ignore rules
    helpers = _helpers()
    rules = helpers._ignore_rules(bundle / "Dockerfile.dockerignore")
    for src in _copy_sources({"APP_DIR": "app", "DEPLOY_DIR": "."}):
        hits = sorted(bundle.glob(src))
        assert hits, f"COPY {src} matches nothing in the archive"
        kept = [p for h in hits for p in ([h] if h.is_file() else h.rglob("*")) if p.is_file()
                and not helpers._ignored(p.relative_to(bundle).as_posix(), rules)]
        assert kept, f"COPY {src}: Dockerfile.dockerignore excludes everything it would copy"
