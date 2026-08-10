"""Composition root for web-safe execution dependencies."""

from __future__ import annotations

from functools import lru_cache

from django.conf import settings
from redis import Redis

from vision_model_serving.execution import RqExecutionConfig, RqGpuExecutionGateway
from vision_model_serving.execution.executor import GpuExecutorClient


@lru_cache(maxsize=1)
def prediction_gateway() -> RqGpuExecutionGateway:
    """Build the Redis/RQ adapter without importing CUDA or model modules."""

    redis = Redis.from_url(
        settings.VMS_REDIS_URL,
        socket_connect_timeout=settings.VMS_REDIS_CONNECT_TIMEOUT_SECONDS,
        socket_timeout=settings.VMS_REDIS_SOCKET_TIMEOUT_SECONDS,
    )
    return RqGpuExecutionGateway(
        redis_client=redis,
        job_root=settings.VMS_JOB_ROOT,
        config=RqExecutionConfig(
            capacity=settings.VMS_QUEUE_CAPACITY,
            reservation_ttl_seconds=settings.VMS_RESERVATION_TTL_SECONDS,
            job_timeout_seconds=settings.VMS_JOB_TIMEOUT_SECONDS,
            result_ttl_seconds=settings.VMS_RESULT_TTL_SECONDS,
            status_ttl_seconds=settings.VMS_STATUS_TTL_SECONDS,
            queue_name=settings.VMS_QUEUE_NAME,
            key_prefix=settings.VMS_KEY_PREFIX,
        ),
    )


@lru_cache(maxsize=1)
def executor_client() -> GpuExecutorClient:
    """Build the local status client without importing CUDA or model modules."""

    return GpuExecutorClient(
        settings.VMS_EXECUTOR_SOCKET,
        timeout_seconds=settings.VMS_EXECUTOR_STATUS_TIMEOUT_SECONDS,
    )
