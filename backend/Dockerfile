# Multi-stage. The single biggest practical risk here is image size: a
# torch-based sentence-transformers image lands around 7-8 GB, which is unusable
# on a free-tier container host. The ONNX runtime path is roughly 600-700 MB.
#
# The critical correctness rule that comes with it: the index must be BUILT and
# QUERIED with the identical model, backend, and version. ONNX conversions of
# MiniLM have historically produced embeddings that do not match the torch
# original, and a build-time/serve-time mismatch silently destroys retrieval
# while every local metric still looks correct. The embedding regression test in
# CI exists for exactly this.

# ---------- builder ----------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./requirements.txt

# torch is dropped from the runtime: sentence-transformers is used with the
# onnx backend. Installing both would defeat the entire point of this stage.
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ---------- runtime ----------
FROM python:3.11-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LLM_PROVIDER=none \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

# Source only. Artifacts (index, embeddings, model, audit) are NOT baked here by
# this Dockerfile — .dockerignore excludes data/ and artifacts/. Ship them
# through the deployment pipeline instead: baked in a later build stage, or
# pulled from a private store at deploy time. Never during a live request, and
# never in public git history.
COPY backend/ ./backend/
COPY pyproject.toml ./

# Non-root.
RUN useradd --create-home --uid 10001 appuser \
 && mkdir -p /app/artifacts /app/data /app/.cache \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# The app must boot and serve with no provider configured. If this fails, the
# deterministic path is broken and no key would have saved it.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=4).status==200 else 1)"

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
