# syntax=docker/dockerfile:1.7
# Multi-stage build. The default (last) stage is the runtime image; `--target tools` adds the
# ingestion, training and evaluation libraries. Secrets are passed as environment variables at run time.

ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install dependencies from the lockfile into /app/.venv.
FROM base AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app

# Dependencies first so this layer is cached, then the project (non-editable, so the venv can be copied).
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# Same, plus the optional ingest, train and eval libraries.
FROM builder AS tools-builder
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --extra ingest --extra train --extra eval

# Download the Hugging Face models at build time.
FROM base AS models
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
ARG RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
COPY --from=builder /app/.venv /app/.venv
ENV HF_HOME=/opt/hf PATH=/app/.venv/bin:$PATH
RUN python -c "\
from sentence_transformers import CrossEncoder, SentenceTransformer; \
SentenceTransformer('${EMBEDDING_MODEL}'); CrossEncoder('${RERANKER_MODEL}')"

# Non-root user (UID 1000).
FROM base AS runtime-base
RUN useradd --create-home --uid 1000 user \
    && mkdir -p /app /home/user/.cache/huggingface \
    && chown -R user:user /app /home/user
# Reported by /health and attached to Langfuse traces.
ARG GIT_SHA=unknown
ENV GIT_SHA=${GIT_SHA} \
    HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    HF_HUB_OFFLINE=1 \
    PATH=/app/.venv/bin:$PATH \
    PORT=7860
WORKDIR /app
COPY --from=models --chown=user:user /opt/hf /home/user/.cache/huggingface
COPY --chown=user:user artifacts ./artifacts
USER user
EXPOSE 7860

FROM runtime-base AS tools
COPY --from=tools-builder --chown=user:user /app/.venv /app/.venv
# Ingestion needs network access.
ENV HF_HUB_OFFLINE=0
CMD ["python", "-m", "market_research_agent.data.ingest"]

FROM runtime-base AS runtime
COPY --from=builder --chown=user:user /app/.venv /app/.venv
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '7860'), timeout=8)" || exit 1
CMD ["python", "-m", "market_research_agent.api"]
