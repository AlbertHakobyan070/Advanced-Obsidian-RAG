# Deploying Noetrix

Everything that builds or runs the stack in a container lives here, versioned with
the code it deploys. There is one set of Dockerfiles and one Compose file; the
release archives and the Kubernetes manifests are built from them, not beside them.

```
deploy/
  docker/            the stack as containers (Compose)
    Dockerfile                 the app image: query API + console, non-root
    Dockerfile.dockerignore    what the build context may contain
    constraints.txt            pins for the transitive ML packages
    docker-compose.yml         rag, freellmapi, + opt-in paddleocr (CPU) and rerank
    docker-compose.gpu.yml     NVIDIA-only: paddleocr-gpu, embed
    docker-entrypoint.sh       runs both servers; supervises the query API
    healthcheck.sh             the one probe: ready / live
    make_docker_config.py      generates the container config from config.example.yaml
    .env.example               settings and the keys the stack reads
  k8s/               the same stack as Kubernetes manifests (kustomize)
    base/                      rag, freellmapi, storage, the ConfigMap, a Secret template
    components/                opt-in: paddleocr, paddleocr-gpu, rerank, embed-gpu
    overlays/local/            kind / k3d
    overlays/cloud/            a placeholder with the decisions marked
  package/           builds the release archives (the author's own packaging)
    package-cold.ps1           Noetrix-cold.tar.gz   (code only, from the live tree)
    package.ps1                Noetrix-mac.tar.gz    (with the corpus)  /  -Public: Noetrix-public.tar.gz
    common.ps1                 what every archive shares: the app allowlist, skills, gates, tar
    start.sh, start.ps1        the launchers an archive carries (macOS/Linux, Windows)
    INSTALL.md, SETUP_MACOS.md, MAC_SETUP.md, FREELLMAPI.md   the guides an archive carries
    docker-compose.override.yml  re-points the build contexts for the flat archive layout
    smoke.py                   boots an archive on CPU and probes it (see "Verifying an archive")
../ocr-sidecar/      PaddleOCR sidecar: Dockerfile (CPU), Dockerfile.gpu, its own GPU compose
../embed-sidecar/    EmbeddingGemma 2 sidecar: Dockerfile (GPU only)
```

## What is in an image, and what is not

The app image holds code and nothing else. **No config, no `.env`, no corpus, no
indexes, no keys.** Those arrive at run time:

| What | How it reaches the container | Where it persists |
|---|---|---|
| config | a file mounted at `/app/config.yaml`, **generated** (below) | the host file / a ConfigMap |
| secrets | environment variables, or the mounted `.env` (Compose) | never in an image or a repo |
| JSONL chunk files | volume at `/app/data` | **primary data** — the corpus has no recorded ingest command, so these cannot be regenerated; back them up |
| indexes (Chroma + bm25) | volume at `/rag_data` | derived from the JSONLs, but a full re-embed is expensive |
| model cache | volume at `/hf_cache` | downloaded once |
| job logs and their provenance records | volume at `/app/logs` | the only record of how a document was ingested |

`Dockerfile` copies an explicit list of directories, and `Dockerfile.dockerignore`
keeps the corpus, secrets, the live `config.yaml` and vault-derived evaluation
material out of the build context in the first place. Both are checked by
`tests/test_deploy_assets.py`.

The images run as a **numeric non-root user** (uid 10001), so Kubernetes'
`runAsNonRoot` can verify them. Base images are pinned by **digest**; the Python
side by `requirements.txt` plus `docker/constraints.txt`. To move a pin, the
command is in a comment beside it (`docker buildx imagetools inspect <image>`), and
the same digest appears in the Dockerfile, the Compose file and the manifests —
change them together.

## The config is generated

A container config kept by hand drifts: it has repeatedly fallen behind a project
that keeps adding keys, and a container that starts on a stale config silently
behaves like an older build. So nothing here carries a copy.
`make_docker_config.py` derives it from `config.example.yaml` and rewrites only what
a container must change (paths onto volumes, `localhost` onto service names).
Every rewrite asserts how many times it must match, and any loopback URL left over
afterwards fails the run, so a renamed key or a new local service breaks the
**build**, not a deployment.

```
python deploy/docker/make_docker_config.py config.example.yaml <output.yaml>
```

It needs PyYAML and nothing else (the project venv has it).

## Local, with Compose

From a checkout (Docker or Docker Desktop, nothing else):

```
cd deploy/docker
cp .env.example .env              # set RAG_HOST_HOME and ENCRYPTION_KEY
python make_docker_config.py ../../config.example.yaml config.docker.yaml
docker compose -p noetrix up --build -d
```

`-p noetrix` names the project; without it Compose uses the folder name, and
volumes would be called `docker_*`. (Do **not** rename the project of an unpacked
archive — see "Upgrading" below.) The query API is on `127.0.0.1:8051`, the console
on `:8052`, the generation proxy on `:3001`. Every host port is a variable
(`NOETRIX_API_PORT`, `NOETRIX_CONSOLE_PORT`, `NOETRIX_PROXY_PORT`, ...), so a second
stack — or a test — can sit beside one that owns the defaults. Nothing is bound
beyond loopback. The two RAG ports are also the numbers the servers listen on
**inside** the container (compose passes them as `RAG_PORT` / `RAG_CONSOLE_PORT`),
because the console's page calls the query API from the *browser*: a remap on the
host side alone would leave its Query tab talking to whatever owns `:8051`.
`make_docker_config.py` writes `webui.rag_api` as `http://127.0.0.1:${RAG_PORT:-8051}`
for the same reason.

**Host shapes.** `RAG_HOST_HOME` is bind-mounted at the path named by
`RAG_CONTAINER_HOME`, which defaults to the same path (macOS, Linux). Windows sets both
(`RAG_HOST_HOME=C:/Users/you`, `RAG_CONTAINER_HOME=/c/Users/you`): a Linux container
cannot mount onto a drive letter, and the config's `${RAG_HOST_HOME}` must expand to the
path the *container* sees. `start.ps1` writes both lines.

**Health.** The `rag` container is healthy only when `/health` says `"ready":true`
(`healthcheck.sh`). `/health` itself answers 200 while the index is loading and
again when the pipeline failed to build, so a plain status check would call both
healthy. Run against an empty data volume the pipeline builds fine and reports ready
after about a minute and a half of model loading: a fresh install goes healthy, and
until the first ingest `/search` answers with an `error` naming the collection that
does not exist yet (not an empty list; `smoke.py` expects exactly that state). A
pipeline that fails to build (a bad model id, a
moved path) stays unhealthy, with the reason in `/health` and the console's service
log.

Optional services:

```
docker compose --profile ocr    up -d --build paddleocr    # scanned PDFs, CPU
docker compose --profile rerank up -d rerank               # external reranker
# NVIDIA only (linux/amd64): paddleocr-gpu and the embedding sidecar
docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile ocr-gpu up -d --build paddleocr-gpu
docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile embed   up -d --build embed
```

**Linux hosts.** A bind-mounted `data/` and `rag_data/` take their ownership from the
host, so give them to the container user: `chown -R 10001:10001 data rag_data`. Docker
Desktop on a Mac or Windows maps ownership and needs nothing. The named volumes are
handled by the one-shot `volume-perms` service, which runs on every `up`.

## Platforms

| Image | amd64 | arm64 (Apple Silicon) |
|---|---|---|
| app (`docker/Dockerfile`) | yes | yes |
| PaddleOCR CPU (`ocr-sidecar/Dockerfile`) | yes | yes |
| FreeLLMAPI, llama.cpp rerank (upstream) | yes | yes |
| PaddleOCR GPU, embedding sidecar | yes | **no** — CUDA wheels and an NVIDIA runtime only exist for amd64 |

CI builds both architectures of the CPU images (`.github/workflows/docker-build.yml`);
the arm64 build there is emulated (QEMU), and it **builds** the image — nothing in CI
*runs* it. Without building anything, `uv pip compile requirements.txt -c
deploy/docker/constraints.txt --python-platform aarch64-unknown-linux-gnu --python-version
3.11 --only-binary :all:` proves every Python pin has an arm64 wheel.

**Hosts.** An archive builds its image *on the machine that unpacks it*, so the
architecture is whatever that machine is and nothing is cross-compiled or emulated:
macOS (Apple Silicon `arm64`, Intel `amd64`), Windows (`amd64`, Docker Desktop with the
WSL 2 backend, Linux containers) and Linux (`amd64`, `arm64`). `smoke.py` (below) is how
to prove one of them; a claim that a platform works belongs next to a smoke run, not in
this page.

## The archives

The `package/` scripts build the tarballs that carry all of this to another machine.
They take **everything from the repo** — the Dockerfile and Compose files from
`docker/`, the application from the live tree through the allowlist in `common.ps1`, the
OCR sidecar from `ocr-sidecar/`, the skills, the launchers, and a freshly generated config
— so an archive cannot carry a stale copy. They are run from **native PowerShell** on the
Windows machine, and each is ONE command:

| Artifact | Command (from the repo root) | Contents | For |
|---|---|---|---|
| `Noetrix-cold.tar.gz` | `powershell -ExecutionPolicy Bypass -File deploy\package\package-cold.ps1` | code only, keyless `.env`, empty `data/` and `rag_data/` | a fresh machine, from the live tree |
| `Noetrix-mac.tar.gz` | `powershell -ExecutionPolicy Bypass -File deploy\package\package.ps1` (run it detached) | the author's corpus and prebuilt indexes, config generated from the live `config.yaml` | the author's own Mac |
| `Noetrix-public.tar.gz` | `powershell -ExecutionPolicy Bypass -File deploy\package\package.ps1 -Public` | as cold, with the application taken from the **sanitized** public checkout | anyone else |

Every build prints the version and commit it is packaging, writes `BUILD_INFO.txt` into the
archive (version, commit, whether the tree was clean, the hashes of both skills) and a
`.sha256` file beside it. `-RequireClean` refuses to package uncommitted changes (use it
for a release); without it the build warns and `BUILD_INFO.txt` says `DIRTY`. The version is
`VERSION` at the repo root when that file exists, otherwise `git describe`. From a git
worktree the personal build needs `-DataDir` and `-LiveConfig` (a worktree has neither);
`-PersonalRagSkill` names the curated skill (below).

**Adding a file to the archives** (a release-notes file, a new top-level module): put it in
`$script:AppFiles` (or `$script:AppDirs`) in `common.ps1`. `tests/test_packaging.py`
checks that every entry exists, that together with the deploy files they satisfy every
`COPY` in the Dockerfile in the archive layout, and builds a cold archive for real. A new
`*.py` at the repo root is already covered by the `*.py` rule.

**The skills.** `skills/rag-ops` is the project's `SKILL_RAGOPS.md`, byte for byte, so it
cannot be stale. `skills/personal-rag` is *not* the project's `SKILL.md` (that is the
internal playbook): it is the curated, sanitized definition the agents are given, which
lives outside this repo, so it is read from `-PersonalRagSkill` (default: the Claude Code
install, `~/.claude/skills/personal-rag/SKILL.md`). Both are refused if they do not declare
their own name or carry a private identifier. The recipient installs them with
`start.sh skills` / `start.ps1 skills` (Claude Code by default, `--dest` / `-Dest` for another
agent). Neither skill reads the stack's address from anywhere: they assume
`127.0.0.1:8051` and `:8052`, which is where the archive publishes them.

An unpacked archive is the same stack in a flat layout: the Dockerfile and Compose
file beside `app/` and `paddleocr/`. The only difference from a checkout is
`docker-compose.override.yml`, which Compose merges automatically and which does
nothing but point the two build contexts at that layout. On any machine with Docker:

```
tar -xzf Noetrix-cold.tar.gz && cd rag-docker-bundle
sh start.sh                                                  # macOS, Linux
powershell -ExecutionPolicy Bypass -File start.ps1           # Windows
```

`start` checks Docker (installed, Compose v2, the engine running, Linux containers), fills
`.env`, runs `docker compose up --build -d` and waits for `"ready":true`, failing with
a plain explanation and the log at the first thing wrong; `INSTALL.md` in the archive is its
guide. The image is built on the target, so nothing is cross-compiled and no image is
transferred. (The alternative — building multi-arch images once and pulling them — is the
manual publish job in the workflow, which has never been run.)

**Verifying an archive.** `python deploy/package/smoke.py Noetrix-cold.tar.gz` unpacks it,
runs the platform's launcher exactly as a new user would, on CPU, under its own compose
project (`noetrix-smoke`), host ports and image tag, so it can run beside a stack in use;
probes the query API, the console and the proxy; installs the skills; prints each
container's memory; and tears down only its own project. It needs the Docker engine running
and about as much memory as the stack itself.

**Upgrading an existing install.** Unpack the newer archive and run `start` (or
`docker compose up --build -d`) again. Two things carry over and must not be disturbed:

- Do not change `ENCRYPTION_KEY`, and do not rename the Compose project (the folder
  is `rag-docker-bundle`): both would orphan the provider keys stored in the proxy's
  volume. `FREELLMAPI.md` has the recovery.
- The image now runs as a non-root user. Volumes that an older image filled as root
  are handed over by `volume-perms` on the next `up`; if a volume still reports it
  is not writable, `docker compose up volume-perms` shows why.

The guides shipped inside the archives (`INSTALL.md`, `MAC_SETUP.md`, `SETUP_MACOS.md`,
`FREELLMAPI.md`) are versioned in `package/`.

The bundle folder (`rag-docker-bundle`) is only a working directory for the personal build:
it holds the corpus copy, the indexes and `.env`. Everything else in it — `app/`, `skills/`,
the Dockerfile, compose file, config, launchers and guides — is overwritten from the repo
on every personal build, and the cold and public builds never read it, so do not edit it.
The superseded `package.ps1`, `package-cold.ps1`, `make_docker_config.py` and
`config.docker.public.yaml` that still sit in it are left out of the archive.

## Kubernetes

Plain kustomize; nothing to install beyond `kubectl`. The base is the core stack, the
`components/` are opt-in features, an overlay chooses between them and supplies image
names and storage settings. `kubectl kustomize deploy/k8s/overlays/local` renders
without a cluster.

Two inputs are deliberately not in git:

1. **The config.** Generate it into the base; it becomes a ConfigMap whose name
   carries a hash of its content, so changing it rolls the pod:
   ```
   python deploy/docker/make_docker_config.py config.example.yaml deploy/k8s/base/config.docker.yaml
   ```
   A ConfigMap is read-only, so the console's Settings tab cannot save in a cluster:
   change the config by regenerating it.
2. **The Secrets.** `base/secret.template.yaml` lists the keys and holds no values and
   is not part of the build; create them with `kubectl create secret generic`
   (commands are in the template) or a secret manager. The proxy's `ENCRYPTION_KEY`
   and the RAG's provider keys are separate Secrets, so neither workload sees the
   other's.

```
# local (kind / k3d): build, load the image, apply
docker build -f deploy/docker/Dockerfile -t noetrix-rag:local .
kind load docker-image noetrix-rag:local
kubectl apply -k deploy/k8s/overlays/local
kubectl -n noetrix port-forward svc/rag-console 8052:8052
```

How it is laid out, and why:

- **One pod runs both servers, one replica, `Recreate`.** The console restarts the
  query API by signalling a pid in a shared process namespace, ChromaDB has a single
  writer, and the claims are `ReadWriteOnce`. Scaling reads out means a second,
  read-only deployment over a copy of the indexes; that is not built.
- **Three probes, three questions** (all `healthcheck.sh`). *Startup* and *liveness*
  ask whether both servers answer; Uvicorn does not open `:8051` until the pipeline
  build finishes, so answering means the boot is over, whether it succeeded or
  failed. *Readiness* asks for `"ready":true`. A pod whose pipeline failed to build is
  Running but not Ready: it gets no API traffic and is not restarted for it.
- **The console has its own Service** with `publishNotReadyAddresses`, so it stays
  reachable exactly when it is needed to fix an index that is not ready.
- **Resource requests and limits** are in the manifests, each with the measurement it
  came from and its date. The memory-hungry process is the query API, which holds
  both indexes in RAM; sidecar figures marked ESTIMATE have not been measured yet.
  Volume sizes come from the live corpus's footprint. The local overlay shrinks both
  for a laptop cluster.
- **Neither the query API nor the console has authentication.** They were built to
  listen on localhost. Every Service is ClusterIP and the cloud overlay ships no
  Ingress on purpose: put an authenticating proxy in front before exposing either,
  because the console's endpoints ingest, delete and change settings.
- The FreeLLMAPI and llama.cpp images are upstream's and run as root; they are pinned by
  digest and not pushed into a stricter security context that nobody has tried
  against their entrypoints.

`overlays/cloud` is a placeholder: every `REPLACE-*` is a decision that depends on
your registry and cluster, and nothing has been run against one.

## CI

`.github/workflows/docker-build.yml` builds every image on pull requests and pushes
nothing, and runs `tests/test_deploy_assets.py` (generator, pins, non-root, the
build-context rules, and the manifests rendered and schema-checked offline). A
separate job publishes to GHCR; it is **manual only** (a hand-started run with
"publish" ticked) and has never been run.

## Not done

- Transitive dependencies are pinned only for the ML stack (`constraints.txt`); a full
  lock file is not generated.
- The curated `personal-rag` skill lives outside the repo (see "The skills"), so a build
  takes whatever is installed on the packaging machine; there is no check that it is the
  newest. `BUILD_INFO.txt` records its hash.
- CI builds the images but never starts them; `smoke.py` is run by hand.
- The SELinux relabelling a Fedora/RHEL host needs for the bind mounts, and Docker Engine
  (as opposed to Docker Desktop) bind-mount ownership on Linux beyond the documented
  `chown`, have not been exercised.
- The public archive's application comes from a periodic snapshot of the sanitized public
  checkout; `BUILD_INFO.txt` records its commit, and it is only as current as the last
  release sync.
- Images are not signed and have no SBOM; GitHub Actions are referenced by tag, not by
  commit.
- No read-only replica, no autoscaling, no Ingress or authentication, no backup job for
  the primary-data volume.
