# PaddleOCR sidecar, CPU build -- the PORTABLE default, and the one the Mac
# bundle ships. Built only when the `ocr` compose profile is used, so a stack
# that never ingests scanned PDFs never pays for this image.
#
# Dockerfile.gpu next to this one is the same sidecar on CUDA with PaddleOCR
# 3.x, for a host that has an NVIDIA card; it keeps the models in VRAM instead
# of system RAM. Both COPY the SAME server.py, which detects the installed
# PaddleOCR major and speaks either API. Keep them that way -- forking the
# server is how the two builds start disagreeing about the contract.
#
# The Debian release is PINNED, like every pip version below. `python:3.11-slim`
# is a moving tag: it rolled from bookworm (glibc 2.36) to trixie (glibc 2.41),
# and paddlepaddle 2.6.2's prebuilt libpaddle.so does not survive that — it
# abort()s inside `import paddle` with "free(): invalid pointer", a SIGABRT that
# no Python `except` can catch. Same image, same pins, broken by a base that
# moved underneath it.
FROM python:3.11-slim-bookworm

# OpenCV (a PaddleOCR dependency) needs libgl1 and libglib2.0-0 even in its
# headless build; without them the import fails with a bare libGL error that
# reads like a broken install.
#
# libgomp1 is the GNU OpenMP runtime that paddlepaddle's own libpaddle.so links
# against. python:3.11-slim does not carry it, and leaving it out produces the
# worst possible failure shape: /health answers 200 because it never imports
# paddle, the compose healthcheck therefore reports the container healthy, and
# the stack only breaks on the FIRST REAL PAGE with
# "ImportError: libgomp.so.1: cannot open shared object file". A sidecar that
# lies about being ready is worse than one that refuses to start.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

# CPU paddlepaddle: there is no GPU inside this container on a Mac, and on
# Linux a GPU needs explicit passthrough. Pinned because PaddleOCR's API has
# moved between majors and this sidecar reads the 2.x result shape.
#
# cython<3 is load-bearing, not tidiness. paddleocr pulls Cython in UNPINNED,
# today's is 3.2.9, and paddlepaddle 2.6.2 is from 2024 — with Cython 3
# present, `import paddle` dies in libpaddle.so with "free(): invalid pointer"
# / SIGSEGV before any of this code runs. 0.29.37 imports cleanly.
#
# What this does NOT fix, and does not need to: a COLD `import paddleocr`
# still raises "zlib.error: Error -2 while decompressing data" from pyclipper.
# That is an import-ORDER artefact, not a broken tree — pyclipper's extension
# loads fine once numpy and PIL are in memory, which is exactly what server.py
# does before it ever touches paddleocr. Verified after this pin: a rendered
# page comes back transcribed character-for-character. Pinning the ~30
# transitive deps one at a time only moves that symptom around, because the
# symptom was never about versions.
RUN pip install --no-cache-dir \
      paddlepaddle==2.6.2 \
      paddleocr==2.9.1 \
      "cython<3" \
      fastapi==0.133.1 \
      uvicorn==0.41.0 \
      pillow==11.3.0 \
      numpy==1.26.4

COPY server.py /srv/server.py

# Model weights land here on first use. Mounted as a named volume by compose so
# the ~200MB download happens once rather than on every container recreate.
ENV PADDLE_OCR_HOME=/paddle_cache \
    HOME=/paddle_cache \
    PYTHONUNBUFFERED=1

EXPOSE 8103
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8103"]
