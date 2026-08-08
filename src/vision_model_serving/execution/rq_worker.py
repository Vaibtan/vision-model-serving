"""RQ task entry point for one isolated GPU prediction."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from time import time
from typing import Protocol

from vision_model_serving.pipeline.contracts import (
    CaseInput,
    PredictionMode,
    PredictionResult,
)

from .contracts import PredictionId
from .storage import EphemeralJobStore, JobResultNotFound


class _Pipeline(Protocol):
    def infer(self, case: CaseInput, mode: PredictionMode) -> PredictionResult: ...


class PredictionWorkerError(RuntimeError):
    pass


class PredictionJobWorker:
    """Execute one stored request without exposing its payload to Redis."""

    def __init__(
        self,
        *,
        job_root: Path,
        pipeline: _Pipeline,
        result_ttl_seconds: int,
        clock: Callable[[], float] = time,
    ):
        if result_ttl_seconds < 1:
            raise ValueError("result TTL must be positive")
        self._store = EphemeralJobStore(job_root, clock=clock)
        self._store.cleanup_expired()
        self._pipeline = pipeline
        self._result_ttl_seconds = result_ttl_seconds

    def execute(self, prediction_id: PredictionId, locator: str) -> None:
        try:
            try:
                self._store.load_result(locator)
                return
            except JobResultNotFound:
                pass
            request = self._store.load_request(prediction_id, locator)
            result = self._pipeline.infer(request.case, request.mode)
            result = _apply_detector_display_threshold(
                result,
                request.detector_score_threshold,
            )
            self._store.store_result(
                locator,
                result,
                ttl_seconds=self._result_ttl_seconds,
            )
        except Exception:  # noqa: BLE001 - never persist private pipeline errors
            raise PredictionWorkerError("prediction execution failed") from None
        finally:
            self._store.purge_request(locator)

    def status(self) -> object:
        status = getattr(self._pipeline, "status", None)
        if not callable(status):
            raise PredictionWorkerError("prediction runtime status is unavailable")
        return status()


_worker_factory: Callable[[], PredictionJobWorker] | None = None
_worker: PredictionJobWorker | None = None


def configure_prediction_worker(
    worker_factory: Callable[[], PredictionJobWorker],
) -> None:
    """Configure the factory in the RQ parent before it forks a job process."""

    if not callable(worker_factory):
        raise TypeError("worker factory must be callable")
    global _worker_factory, _worker
    _worker_factory = worker_factory
    _worker = None


def build_prediction_worker_factory(
    *,
    pipeline_factory: Callable[[], _Pipeline],
    job_root: Path,
    result_ttl_seconds: int,
    clock: Callable[[], float] = time,
) -> Callable[[], PredictionJobWorker]:
    """Delay pipeline and CUDA construction until a job child executes."""

    def build() -> PredictionJobWorker:
        return PredictionJobWorker(
            job_root=job_root,
            pipeline=pipeline_factory(),
            result_ttl_seconds=result_ttl_seconds,
            clock=clock,
        )

    return build


def build_executor_client_factory(
    *,
    socket_path: Path,
    timeout_seconds: float,
) -> Callable[[], object]:
    """Build only the lightweight executor client inside each RQ work-horse."""

    def build() -> object:
        from .executor import GpuExecutorClient

        return GpuExecutorClient(socket_path, timeout_seconds=timeout_seconds)

    return build


def create_prediction_rq_worker(
    *,
    redis_client: object,
    queue_name: str,
    worker_factory: Callable[[], PredictionJobWorker],
) -> object:
    """Create the production one-job-at-a-time RQ worker."""

    from rq import Queue, Worker
    from rq.serializers import JSONSerializer

    configure_prediction_worker(worker_factory)
    queue = Queue(
        queue_name,
        connection=redis_client,
        serializer=JSONSerializer,
    )

    def record_worker_loss(job: object, *_details: object) -> None:
        key_prefix = str(job.meta.get("key_prefix", ""))
        if not key_prefix:
            return
        redis_client.zrem(f"{key_prefix}:active", job.id)
        metrics_key = f"{key_prefix}:metrics"
        redis_client.hincrby(metrics_key, "failed_total", 1)
        redis_client.hincrby(metrics_key, "worker_lost_total", 1)

    return Worker(
        [queue],
        connection=redis_client,
        serializer=JSONSerializer,
        work_horse_killed_handler=record_worker_loss,
    )


def execute_prediction_job(prediction_id: str, locator: str) -> None:
    """RQ task receiving only the opaque prediction and storage identifiers."""

    from rq import get_current_job

    job = get_current_job()
    if job is None or _worker_factory is None:
        raise PredictionWorkerError("prediction worker is not configured")
    key_prefix = str(job.meta.get("key_prefix", ""))
    if not key_prefix:
        raise PredictionWorkerError("prediction worker is not configured")
    redis = job.connection
    metrics_key = f"{key_prefix}:metrics"
    active_key = f"{key_prefix}:active"
    started_at = time()
    timeout = int(job.timeout or 0)
    redis.zadd(active_key, {prediction_id: started_at + timeout})
    if job.enqueued_at is not None:
        wait_ms = max(0.0, (started_at - job.enqueued_at.timestamp()) * 1_000.0)
        redis.hincrbyfloat(metrics_key, "queue_wait_ms_total", wait_ms)
    global _worker
    if _worker is None:
        _worker = _worker_factory()
    try:
        _worker.execute(PredictionId(prediction_id), locator)
    except Exception:
        redis.hincrby(metrics_key, "failed_total", 1)
        raise
    else:
        redis.hincrby(metrics_key, "succeeded_total", 1)
    finally:
        redis.zrem(active_key, prediction_id)


def _apply_detector_display_threshold(
    result: PredictionResult,
    threshold: float | None,
) -> PredictionResult:
    """Filter display detections after inference without changing classifier ROIs."""

    if threshold is None:
        return result
    detector = replace(
        result.detector,
        top_candidates=tuple(
            detection
            for detection in result.detector.top_candidates
            if detection.score >= threshold
        ),
        post_nms=tuple(
            detection
            for detection in result.detector.post_nms
            if detection.score >= threshold
        ),
    )
    return replace(result, detector=detector)
