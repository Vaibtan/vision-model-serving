ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.8.4@sha256:40775a79214294fb51d097c9117592f193bcfdfc634f4daa0e169ee965b10ef0
ARG EXECUTOR_IMAGE=vision-model-serving-executor:local

FROM ${UV_IMAGE} AS uv
FROM ${EXECUTOR_IMAGE}

USER root
COPY --from=uv /uv /uvx /bin/
COPY requirements/tensorrt-l4.txt ./requirements/tensorrt-l4.txt
RUN uv pip install \
      --python /app/.venv/bin/python \
      --requirement requirements/tensorrt-l4.txt \
    && /app/.venv/bin/python -c \
      "from importlib.metadata import version; assert version('torch-tensorrt') == '2.8.0'; assert version('tensorrt') == '10.12.0.36'; assert version('cuda-python') == '12.8.0'"
COPY --chown=10001:10001 scripts/tensorrt_runtime_verify.py ./scripts/tensorrt_runtime_verify.py
COPY --chown=10001:10001 scripts/l4_validation/_common.py ./scripts/l4_validation/_common.py
COPY --chown=10001:10001 scripts/l4_validation/17_evaluate_pytorch_optimizations.py ./scripts/l4_validation/17_evaluate_pytorch_optimizations.py
COPY --chown=10001:10001 scripts/l4_validation/18_build_tensorrt_candidate.py ./scripts/l4_validation/18_build_tensorrt_candidate.py
USER 10001:10001
ENTRYPOINT ["python"]
