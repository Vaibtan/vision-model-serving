ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.8.4@sha256:40775a79214294fb51d097c9117592f193bcfdfc634f4daa0e169ee965b10ef0
ARG PLAYWRIGHT_IMAGE=mcr.microsoft.com/playwright/python:v1.62.0-noble@sha256:aa81288e738725378becba5b3e06cb0f3a7f012a610e87e8d767a090ea3f740d

FROM ${UV_IMAGE} AS uv
FROM ${PLAYWRIGHT_IMAGE}

COPY --from=uv /uv /uvx /bin/
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONPATH=/app/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --only-group browser --no-install-project
COPY --chown=1000:1000 src ./src
COPY --chown=1000:1000 scripts/validate_browser_workbench.py ./scripts/validate_browser_workbench.py
USER 1000:1000
