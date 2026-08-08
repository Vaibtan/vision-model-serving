"""Composition root for web-safe execution dependencies."""

from __future__ import annotations

from functools import lru_cache

from django.conf import settings
from redis import Redis

from vision_model_serving.execution import RqExecutionConfig, RqGpuExecutionGateway


@lru_cache(maxsize=1)
def prediction_gateway() -> RqGpuExecutionGateway:
    """Build the Redis/RQ adapter without importing CUDA or model modules."""

    redis = Redis.from_url(settings.VMS_REDIS_URL)
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
