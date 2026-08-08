"""Celery adapter configured for a single long-running GPU owner."""

from __future__ import annotations

from typing import Callable, Protocol

from .contracts import PredictionId


GPU_TASK_NAME = "vision_model_serving.execute_prediction"


class _CeleryConfiguration(Protocol):
    def update(self, **kwargs: object) -> None: ...


class _CeleryApp(Protocol):
    conf: _CeleryConfiguration

    def send_task(self, name: str, **options: object) -> object: ...

    def task(self, **options: object) -> Callable[[object], object]: ...


class _Worker(Protocol):
    def execute(self, prediction_id: PredictionId, locator: str) -> None: ...


def configure_gpu_worker(
    app: _CeleryApp,
    *,
    visibility_timeout_seconds: int,
) -> None:
    """Apply queue safety settings that are also asserted by deployment tests."""

    if (
        isinstance(visibility_timeout_seconds, bool)
        or not isinstance(visibility_timeout_seconds, int)
        or visibility_timeout_seconds < 1
    ):
        raise ValueError("visibility timeout must be a positive integer")
    app.conf.update(
        task_acks_late=True,
        task_acks_on_failure_or_timeout=True,
        task_reject_on_worker_lost=False,
        task_ignore_result=True,
        task_serializer="json",
        accept_content=["json"],
        worker_concurrency=1,
        worker_prefetch_multiplier=1,
        worker_cancel_long_running_tasks_on_connection_loss=True,
        broker_connection_retry_on_startup=True,
        broker_transport_options={
            "visibility_timeout": visibility_timeout_seconds,
        },
    )


class CeleryTaskDispatcher:
    def __init__(self, app: _CeleryApp, *, queue_name: str):
        if not queue_name:
            raise ValueError("queue name must not be empty")
        self._app = app
        self._queue_name = queue_name

    def dispatch(self, prediction_id: PredictionId, locator: str) -> None:
        self._app.send_task(
            GPU_TASK_NAME,
            args=[str(prediction_id), locator],
            kwargs={},
            queue=self._queue_name,
        )


def register_prediction_task(
    app: _CeleryApp,
    worker_factory: Callable[[], _Worker],
) -> object:
    """Register a no-retry task that lazily constructs CUDA state in execution."""

    worker: _Worker | None = None

    @app.task(
        name=GPU_TASK_NAME,
        acks_late=True,
        reject_on_worker_lost=False,
        max_retries=0,
        autoretry_for=(),
        serializer="json",
        ignore_result=True,
    )
    def execute_prediction(prediction_id: str, locator: str) -> None:
        nonlocal worker
        if worker is None:
            worker = worker_factory()
        worker.execute(PredictionId(prediction_id), locator)

    return execute_prediction
