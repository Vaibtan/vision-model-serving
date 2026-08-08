ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.8.4@sha256:40775a79214294fb51d097c9117592f193bcfdfc634f4daa0e169ee965b10ef0
ARG CUDA_DEVEL_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04@sha256:24c8e3581ea6330038b0d374920721983312627f8adbfcf390bdb4b399d280ed
ARG CUDA_RUNTIME_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04@sha256:ac55d124da4882b497f732d8dfd9a702d5447a5f29d08d56da6f64f0a1eb34bc

FROM ${UV_IMAGE} AS uv
FROM ${CUDA_DEVEL_IMAGE} AS build

ARG DEBIAN_FRONTEND=noninteractive
ARG FOCALNET_COMMIT=23901e021dc6ec8f66bad47983f45a25574452cc
RUN apt-get update \
    && apt-get install --yes --no-install-recommends git python3 python3-dev \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /uvx /bin/
ENV CUDA_HOME=/usr/local/cuda \
    CUDACXX=/usr/local/cuda/bin/nvcc \
    TORCH_CUDA_ARCH_LIST=8.9 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra gpu --no-install-project
RUN rm -rf \
      /app/.venv/lib/python3.12/site-packages/pydicom/data/charset_files \
      /app/.venv/lib/python3.12/site-packages/pydicom/data/palettes \
      /app/.venv/lib/python3.12/site-packages/pydicom/data/test_files
RUN git init /opt/focalnet \
    && git -C /opt/focalnet remote add origin https://github.com/FocalNet/FocalNet-DINO.git \
    && git -C /opt/focalnet fetch --depth 1 origin ${FOCALNET_COMMIT} \
    && git -C /opt/focalnet checkout --detach FETCH_HEAD
COPY config ./config
COPY patches ./patches
COPY scripts/l4_validation/prepare_focalnet.py ./scripts/l4_validation/prepare_focalnet.py
COPY src ./src
ENV PYTHONPATH=/app/src
RUN /app/.venv/bin/python scripts/l4_validation/prepare_focalnet.py build \
      --repo /opt/focalnet \
      --project-root /app \
      --spec /app/config/l4-fp32-environment.json \
      --max-jobs 4

FROM ${CUDA_RUNTIME_IMAGE} AS runtime

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install --yes --no-install-recommends git libglib2.0-0t64 libgomp1 python3 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 vms \
    && useradd --no-log-init --uid 10001 --gid 10001 --home-dir /nonexistent vms
ENV HF_HUB_OFFLINE=1 \
    PATH=/app/.venv/bin:$PATH \
    PYTHONPATH=/app/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_OFFLINE=1
WORKDIR /app
COPY --from=build --chown=10001:10001 /app/.venv /app/.venv
COPY --from=build --chown=10001:10001 /opt/focalnet /opt/focalnet
COPY --chown=10001:10001 config ./config
COPY --chown=10001:10001 config_cfg.py ./config_cfg.py
COPY --chown=10001:10001 patches ./patches
COPY --chown=10001:10001 src ./src
USER 10001:10001
CMD ["python", "-m", "vision_model_serving.execution.executor_cli", "--socket-path", "/run/vms/executor.sock", "--job-root", "/var/lib/vms/jobs", "--result-ttl-seconds", "900", "--project-root", "/app", "--artifact-root", "/models", "--tokenizer-root", "/assets/tokenizer", "--focalnet-root", "/opt/focalnet", "--mmbcd-root", "/sources/MMBCD", "--dino-root", "/sources/dino"]
