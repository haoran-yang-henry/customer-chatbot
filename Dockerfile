# syntax=docker/dockerfile:1
# API image (Day 7): multi-stage, non-root, base images pinned by digest.
#   docker build -t customer-agent .   ·   make up-app (docker-compose.yml, profile "app")
ARG PYTHON_IMAGE=python:3.12.14-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e

FROM ghcr.io/astral-sh/uv:0.12.2@sha256:069a51314a7bb6031777a9273205fe1b0b19e914ef418207d1338b268df641dd AS uv

FROM ${PYTHON_IMAGE} AS build
COPY --from=uv /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app
# Dependencies first: this layer is rebuilt only when the lock file changes.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --no-install-project
COPY README.md alembic.ini ./
COPY src ./src
COPY prompts ./prompts
COPY migrations ./migrations
COPY data ./data
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev

FROM ${PYTHON_IMAGE}
# Debian security fixes released after the base image was built: Trivy fails the build on
# fixable HIGH/CRITICAL findings, and the upstream image is rebuilt only every few weeks.
RUN apt-get update \
    && apt-get upgrade -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*
RUN groupadd --system --gid 1001 app \
    && useradd --system --uid 1001 --gid app --no-create-home --shell /usr/sbin/nologin app
WORKDIR /app
# The project is installed in editable mode, so prompts/ is found next to src/.
COPY --from=build --chown=app:app /app /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    CA_APP_ENV=prod \
    CA_LOG_FORMAT=json
USER app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=2)"]
# --proxy-headers: behind a reverse proxy the rate limit sees the real client address.
# --timeout-keep-alive 75: longer than typical load balancer idle timeouts (60 s), so the proxy,
# not the server, closes idle connections; otherwise requests race a closing connection (EOF).
CMD ["uvicorn", "--factory", "customer_agent.main:create_app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log", "--proxy-headers", "--timeout-keep-alive", "75"]
