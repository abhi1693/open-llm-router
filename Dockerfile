# syntax=docker/dockerfile:1.7

FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY open_llm_router ./open_llm_router

RUN python -m pip install --upgrade pip build \
    && python -m build --wheel --outdir /tmp/dist


FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ROUTING_CONFIG_PATH=/app/router.profile.yaml \
    ROUTER_AUDIT_LOG_PATH=/app/logs/router_decisions.jsonl \
    ROUTER_RUNTIME_OVERRIDES_PATH=/app/logs/router.runtime.overrides.yaml

WORKDIR /app

RUN groupadd --system app && useradd --system --gid app app

COPY --from=builder /tmp/dist/*.whl /tmp/dist/
RUN python -m pip install --no-cache-dir /tmp/dist/*.whl \
    && rm -rf /tmp/dist

RUN mkdir -p /app/logs && chown -R app:app /app

COPY router.profile.yaml /app/router.profile.yaml

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import sys, urllib.request; sys.exit(0) if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else sys.exit(1)"

CMD ["open-llm-router"]
