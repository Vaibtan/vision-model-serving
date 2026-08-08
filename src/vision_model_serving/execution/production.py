"""Production Celery/Redis composition for the GPU execution gateway."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from vision_model_serving.pipeline.pipeline import PredictionPipeline

from .celery import CeleryTaskDispatcher, configure_gpu_worker
from .gateway import StoredGpuExecutionGateway
from .redis_state import RedisPredictionStateRepository, _RedisClient
from .storage import EphemeralJobStore
from .worker import PredictionJobWorker


@dataclass(frozen=True, slots=True)
class GpuExecutionConfig:
    capacity: int
    reservation_ttl_seconds: float
    worker_loss_ttl_seconds: float
    result_ttl_seconds: float
    tombstone_ttl_seconds: float
    visibility_timeout_seconds: int
    queue_name: str = "gpu-inference"

    def __post_init__(self) -> None:
        if (
            isinstance(self.capacity, bool)
            or not isinstance(self.capacity, int)
            or self.capacity < 1
        ):
            raise ValueError("capacity must be a positive integer")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or value <= 0
            for value in (
                self.reservation_ttl_seconds,
                self.worker_loss_ttl_seconds,
                self.result_ttl_seconds,
                self.tombstone_ttl_seconds,
            )
        ):
            raise ValueError("execution TTLs must be positive")
        if (
            isinstance(self.visibility_timeout_seconds, bool)
            or not isinstance(self.visibility_timeout_seconds, int)
            or self.visibility_timeout_seconds <= self.worker_loss_ttl_seconds
        ):
            raise ValueError(
                "broker visibility timeout must exceed the worker-loss lease"
            )
        if not self.queue_name:
            raise ValueError("queue name must not be empty")


class CeleryRedisGpuExecutionGateway(StoredGpuExecutionGateway):
    """Production adapter with Redis admission and opaque Celery dispatch."""

    def __init__(
        self,
        *,
        redis_client: _RedisClient,
        celery_app: object,
        job_root: Path,
        config: GpuExecutionConfig,
        clock: Callable[[], float] = time,
    ):
        configure_gpu_worker(
            celery_app,
            visibility_timeout_seconds=config.visibility_timeout_seconds,
        )
        state = RedisPredictionStateRepository(
            redis_client,
            capacity=config.capacity,
            result_ttl_seconds=config.result_ttl_seconds,
            tombstone_ttl_seconds=config.tombstone_ttl_seconds,
            clock=clock,
        )
        store = EphemeralJobStore(job_root, clock=clock)
        store.cleanup_expired()
        super().__init__(
            state=state,
            store=store,
            dispatcher=CeleryTaskDispatcher(
                celery_app,
                queue_name=config.queue_name,
            ),
            reservation_ttl_seconds=config.reservation_ttl_seconds,
            clock=clock,
        )


def build_prediction_worker_factory(
    *,
    redis_client_factory: Callable[[], _RedisClient],
    pipeline_factory: Callable[[], PredictionPipeline],
    job_root: Path,
    config: GpuExecutionConfig,
    clock: Callable[[], float] = time,
) -> Callable[[], PredictionJobWorker]:
    """Delay Redis and CUDA construction until Celery executes in its child."""

    def build() -> PredictionJobWorker:
        state = RedisPredictionStateRepository(
            redis_client_factory(),
            capacity=config.capacity,
            result_ttl_seconds=config.result_ttl_seconds,
            tombstone_ttl_seconds=config.tombstone_ttl_seconds,
            clock=clock,
        )
        store = EphemeralJobStore(job_root, clock=clock)
        store.cleanup_expired()
        return PredictionJobWorker(
            state=state,
            store=store,
            pipeline=pipeline_factory(),
            worker_loss_ttl_seconds=config.worker_loss_ttl_seconds,
            result_ttl_seconds=config.result_ttl_seconds,
            clock=clock,
        )

    return build
