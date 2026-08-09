ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.8.4@sha256:40775a79214294fb51d097c9117592f193bcfdfc634f4daa0e169ee965b10ef0
ARG PYTHON_IMAGE=python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7

FROM ${UV_IMAGE} AS uv
FROM ${PYTHON_IMAGE} AS build

COPY --from=uv /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra gateway --extra web --no-install-project
RUN rm -rf \
      /app/.venv/lib/python3.12/site-packages/pydicom/data/charset_files \
      /app/.venv/lib/python3.12/site-packages/pydicom/data/palettes \
      /app/.venv/lib/python3.12/site-packages/pydicom/data/test_files

FROM ${PYTHON_IMAGE} AS app

ENV DJANGO_SETTINGS_MODULE=vision_model_serving.web.settings \
    PATH=/app/.venv/bin:$PATH \
    PYTHONPATH=/app/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
RUN groupadd --gid 10001 vms \
    && useradd --no-log-init --uid 10001 --gid 10001 --home-dir /nonexistent vms
WORKDIR /app
COPY --from=build --chown=10001:10001 /app/.venv /app/.venv
COPY --chown=10001:10001 manage.py ./
COPY --chown=10001:10001 config ./config
COPY --chown=10001:10001 src ./src
USER 10001:10001

FROM app AS test
COPY --chown=10001:10001 tests/real_infra/test_django_api.py ./tests/real_infra/test_django_api.py
COPY --chown=10001:10001 scripts/benchmark_api.py ./scripts/benchmark_api.py
COPY --chown=10001:10001 scripts/smoke_api.py ./scripts/smoke_api.py
COPY --chown=10001:10001 scripts/validate_compose_api.py ./scripts/validate_compose_api.py

FROM app AS runtime
EXPOSE 8000
CMD ["gunicorn", "vision_model_serving.web.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "2", "--access-logfile", "-", "--error-logfile", "-", "--timeout", "30", "--graceful-timeout", "30"]
