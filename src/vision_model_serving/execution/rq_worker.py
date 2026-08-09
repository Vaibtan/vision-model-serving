"""RQ task entry point that forwards opaque identifiers to the GPU executor."""

from __future__ import annotations

from pathlib import Path
from time import time

from vision_model_serving.observability import record_queue_wait

from .contracts import PredictionId
from .executor import GpuExecutorClient


_executor_socket_path: Path | None = None
_executor_timeout_seconds: float | None = None
_executor_client: GpuExecutorClient | None = None


def _configure_executor_client(
    *,
    socket_path: Path,
    timeout_seconds: float,
) -> None:
    if not isinstance(socket_path, Path):
        raise TypeError("executor socket path must be a pathlib.Path")
    if timeout_seconds <= 0:
        raise ValueError("executor timeout must be positive")
    global _executor_socket_path, _executor_timeout_seconds, _executor_client
    _executor_socket_path = socket_path
    _executor_timeout_seconds = float(timeout_seconds)
    _executor_client = None


def create_prediction_rq_worker(
    *,
    redis_client: object,
    queue_name: str,
    executor_socket_path: Path,
    executor_timeout_seconds: float,
) -> object:
    """Create the production RQ dispatcher for one private GPU executor."""

    from rq import Queue, Worker
    from rq.serializers import JSONSerializer

    _configure_executor_client(
        socket_path=executor_socket_path,
        timeout_seconds=executor_timeout_seconds,
    )
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
    """Forward only opaque prediction and storage identifiers over the socket."""

    from rq import get_current_job

    job = get_current_job()
    if job is None or _executor_socket_path is None or _executor_timeout_seconds is None:
        raise RuntimeError("prediction executor client is not configured")
    key_prefix = str(job.meta.get("key_prefix", ""))
    if not key_prefix:
        raise RuntimeError("prediction executor client is not configured")
    redis = job.connection
    metrics_key = f"{key_prefix}:metrics"
    active_key = f"{key_prefix}:active"
    started_at = time()
    timeout = int(job.timeout or 0)
    redis.zadd(active_key, {prediction_id: started_at + timeout})
    if job.enqueued_at is not None:
        wait_ms = max(0.0, (started_at - job.enqueued_at.timestamp()) * 1_000.0)
        redis.hincrbyfloat(metrics_key, "queue_wait_ms_total", wait_ms)
        try:
            record_queue_wait(wait_ms / 1_000.0)
        except Exception:  # noqa: BLE001, S110 - telemetry is non-authoritative
            pass
    global _executor_client
    if _executor_client is None:
        _executor_client = GpuExecutorClient(
            _executor_socket_path,
            timeout_seconds=_executor_timeout_seconds,
        )
    try:
        _executor_client.execute(PredictionId(prediction_id), locator)
    except Exception:
        redis.hincrby(metrics_key, "failed_total", 1)
        raise
    else:
        redis.hincrby(metrics_key, "succeeded_total", 1)
    finally:
        redis.zrem(active_key, prediction_id)
