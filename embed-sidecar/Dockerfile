# EmbeddingGemma 2 sidecar -- GPU only, linux/amd64 only.
#
# Not built for arm64 on purpose. The CUDA 12.6 torch wheels in requirements.txt
# are x86_64 builds, and an Apple Silicon host has no NVIDIA runtime to give them
# a card. The server refuses to start without a visible CUDA device (unless
# EMBED_DEVICE=cpu is set deliberately), so an image that "ran" there would only
# ever fail at startup.
#
# The CUDA runtime comes from the pip wheels (nvidia-* packages pulled in by
# torch), so a plain Python base is enough: the only host-side requirement is the
# NVIDIA driver, which the NVIDIA container runtime injects at start.
#
#   docker build -t noetrix-embed:local embed-sidecar/
#   docker run --rm --gpus all -p 127.0.0.1:8104:8104 -e HF_TOKEN=... noetrix-embed:local
#
# See README.md in this folder for the variables and the contract.

# Same Debian release and digest as the other images; to bump it:
#   docker buildx imagetools inspect python:3.11-slim-bookworm
FROM --platform=linux/amd64 python:3.11.17-slim-bookworm@sha256:0a310eeecf4e1f5a0743f9a6520c90c88d089c903ca5fd283f501e3a805f5f89

ARG APP_UID=10001
ARG APP_GID=10001

WORKDIR /srv

# The measured working stack (see requirements.txt): torch 2.14.1+cu126,
# sentence-transformers 6.1.0, transformers 5.19.0. `--extra-index-url` inside the
# file points pip at the CUDA wheels. Roughly 5 GB of wheels, so this layer
# is cached on requirements.txt alone.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

COPY server.py /srv/server.py

# Non-root; the model cache is the one writable place. Numeric uid so Kubernetes'
# runAsNonRoot can verify it. A cache volume an earlier root container filled is
# handed over by the compose file's perms step (Kubernetes: fsGroup).
RUN groupadd --gid ${APP_GID} embed \
 && useradd --uid ${APP_UID} --gid ${APP_GID} --home-dir /hf_cache \
            --no-create-home --shell /usr/sbin/nologin embed \
 && mkdir -p /hf_cache && chown ${APP_UID}:${APP_GID} /hf_cache

# The weights (~1.7 GB) download on first start into the cache volume. The server
# binds 127.0.0.1 by default; in a container it must listen on all interfaces.
ENV HF_HOME=/hf_cache \
    HOME=/hf_cache \
    EMBED_HOST=0.0.0.0 \
    EMBED_PORT=8104 \
    PYTHONUNBUFFERED=1

USER ${APP_UID}:${APP_GID}

EXPOSE 8104

# The model loads BEFORE the port opens, so a connection that succeeds means a
# model is in memory; /health then reports the settings it embeds with. No curl in
# this image -- Python is already here.
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=5 \
  CMD ["python", "-c", "import os, urllib.request as u; u.urlopen('http://127.0.0.1:%s/health' % os.environ.get('EMBED_PORT', '8104'), timeout=4)"]

CMD ["python", "/srv/server.py"]
