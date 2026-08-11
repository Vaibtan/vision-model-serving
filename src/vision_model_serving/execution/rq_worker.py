"""RQ task entry point that forwards opaque identifiers to the GPU executor."""

from __future__ import annotations

from pathlib import Path
from time import time

from vision_model_serving.failures import is_case_input_failure

from .contracts import QUEUE_WAIT_BUCKET_SECONDS, PredictionId
from .executor import (
    GpuExecutorBusy,
    GpuExecutorCaseFailed,
    GpuExecutorClient,
    GpuExecutorUnavailable,
)
from .storage import EphemeralJobStore, JobLeaseConflict, JobPayloadNotFound
from .terminal import (
    TerminalFailureKind,
    TerminalFailureMarker,
    store_terminal_marker,
    terminal_key,
)


_executor_socket_path: Path | None = None
_executor_timeout_seconds: float | None = None
_executor_task_timeout_seconds: float | None = None
_executor_client: GpuExecutorClient | None = None


def _configure_executor_client(
    *,
    socket_path: Path,
    timeout_seconds: float,
    task_timeout_seconds: float | None = None,
) -> None:
    if not isinstance(socket_path, Path):
        raise TypeError("executor socket path must be a pathlib.Path")
    if timeout_seconds <= 0:
        raise ValueError("executor timeout must be positive")
    if task_timeout_seconds is not None and (
        task_timeout_seconds <= 0 or task_timeout_seconds >= timeout_seconds
    ):
        raise ValueError("executor task timeout must be below the socket timeout")
    global _executor_socket_path, _executor_timeout_seconds
    global _executor_task_timeout_seconds, _executor_client
    _executor_socket_path = socket_path
    _executor_timeout_seconds = float(timeout_seconds)
    _executor_task_timeout_seconds = (
        None if task_timeout_seconds is None else float(task_timeout_seconds)
    )
    _executor_client = None


def create_prediction_rq_worker(
    *,
    redis_client: object,
    queue_name: str,
    executor_socket_path: Path,
    executor_timeout_seconds: float,
    executor_task_timeout_seconds: float | None = None,
) -> object:
    """Create the production RQ dispatcher for one private GPU executor."""

    from rq import Queue, Worker
    from rq.serializers import JSONSerializer

    _configure_executor_client(
        socket_path=executor_socket_path,
        timeout_seconds=executor_timeout_seconds,
        task_timeout_seconds=executor_task_timeout_seconds,
    )
    queue = Queue(
        queue_name,
        connection=redis_client,
        serializer=JSONSerializer,
    )

    def record_worker_loss(job: object, *_details: object) -> None:
        # RQ calls this from the worker's main loop; an exception here would
        # kill the loop, so every Redis write stays best-effort.
        try:
            key_prefix = str(job.meta.get("key_prefix", ""))
            if not key_prefix:
                return
            now = time()
            submitted_at = now
            enqueued_at = getattr(job, "enqueued_at", None)
            if hasattr(enqueued_at, "timestamp"):
                submitted_at = float(enqueued_at.timestamp())
            store_terminal_marker(
                redis_client,
                key_prefix=key_prefix,
                prediction_id=str(job.id),
                marker=TerminalFailureMarker(
                    TerminalFailureKind.RUNTIME_UNAVAILABLE,
                    "prediction runtime was unavailable",
                    True,
                    submitted_at,
                    now,
                ),
                ttl_seconds=int(job.meta.get("status_ttl_seconds", 1_200)),
            )
            # Retain admission until its execution deadline. The work-horse may
            # have died while the executor still owns uncancellable CUDA work.
            metrics_key = f"{key_prefix}:metrics"
            redis_client.hincrby(metrics_key, "failed_total", 1)
            redis_client.hincrby(metrics_key, "worker_lost_total", 1)
        except Exception:  # noqa: BLE001, S110 - never break the worker loop
            pass

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
    submitted_at = float(job.enqueued_at.timestamp()) if job.enqueued_at is not None else started_at
    status_ttl = int(job.meta.get("status_ttl_seconds", 1_200))
    job_root_value = job.meta.get("job_root")
    if not isinstance(job_root_value, str) or not job_root_value:
        raise RuntimeError("prediction job store is not configured")
    job_root = Path(job_root_value)
    if timeout <= 0 or not _claim_reservation(
        redis,
        active_key=active_key,
        terminal_marker_key=terminal_key(key_prefix, prediction_id),
        prediction_id=prediction_id,
        now=started_at,
        execution_expires_at=started_at + timeout,
    ):
        store_terminal_marker(
            redis,
            key_prefix=key_prefix,
            prediction_id=prediction_id,
            marker=TerminalFailureMarker(
                TerminalFailureKind.RESERVATION_EXPIRED,
                "prediction reservation expired",
                False,
                submitted_at,
                started_at,
            ),
            ttl_seconds=status_ttl,
        )
        redis.hincrby(metrics_key, "failed_total", 1)
        raise RuntimeError("prediction reservation is no longer active")
    store = EphemeralJobStore(job_root)
    try:
        store.acquire_lease(locator, ttl_seconds=timeout + 30.0)
    except (JobPayloadNotFound, JobLeaseConflict) as error:
        kind = (
            TerminalFailureKind.RESERVATION_EXPIRED
            if isinstance(error, JobPayloadNotFound)
            else TerminalFailureKind.RUNTIME_UNAVAILABLE
        )
        store_terminal_marker(
            redis,
            key_prefix=key_prefix,
            prediction_id=prediction_id,
            marker=TerminalFailureMarker(
                kind,
                (
                    "prediction request payload is unavailable"
                    if kind is TerminalFailureKind.RESERVATION_EXPIRED
                    else "prediction execution lease is unavailable"
                ),
                kind is TerminalFailureKind.RUNTIME_UNAVAILABLE,
                submitted_at,
                time(),
            ),
            ttl_seconds=status_ttl,
        )
        if isinstance(error, JobPayloadNotFound):
            redis.zrem(active_key, prediction_id)
        # An active conflicting lease means another execution may still own
        # CUDA. Preserve admission until its claimed deadline instead of
        # creating an overlap window.
        redis.hincrby(metrics_key, "failed_total", 1)
        raise
    if job.enqueued_at is not None:
        wait_ms = max(0.0, (started_at - job.enqueued_at.timestamp()) * 1_000.0)
        redis.hincrbyfloat(metrics_key, "queue_wait_ms_total", wait_ms)
        redis.hincrby(metrics_key, "queue_wait_count", 1)
        for upper_seconds in QUEUE_WAIT_BUCKET_SECONDS:
            if wait_ms <= upper_seconds * 1_000.0:
                redis.hincrby(
                    metrics_key,
                    f"queue_wait_le_{int(upper_seconds * 1_000.0)}",
                    1,
                )
        redis.hincrby(metrics_key, "queue_wait_le_inf", 1)
    global _executor_client
    if _executor_client is None:
        _executor_client = GpuExecutorClient(
            _executor_socket_path,
            timeout_seconds=_executor_timeout_seconds,
            task_timeout_seconds=_executor_task_timeout_seconds,
        )
    try:
        _executor_client.execute(PredictionId(prediction_id), locator)
    except Exception as error:
        redis.hincrby(metrics_key, "failed_total", 1)
        kind, detail, retryable = _terminal_failure(error)
        store_terminal_marker(
            redis,
            key_prefix=key_prefix,
            prediction_id=prediction_id,
            marker=TerminalFailureMarker(
                kind,
                detail,
                retryable,
                submitted_at,
                time(),
            ),
            ttl_seconds=status_ttl,
        )
        raise
    else:
        redis.hincrby(metrics_key, "succeeded_total", 1)
    finally:
        store.release_lease(locator)
        redis.zrem(active_key, prediction_id)


def _claim_reservation(
    redis: object,
    *,
    active_key: str,
    terminal_marker_key: str,
    prediction_id: str,
    now: float,
    execution_expires_at: float,
) -> bool:
    from redis.exceptions import WatchError

    while True:
        with redis.pipeline() as transaction:
            try:
                transaction.watch(active_key, terminal_marker_key)
                score = transaction.zscore(active_key, prediction_id)
                terminal = transaction.exists(terminal_marker_key)
                if score is None or float(score) <= now or terminal:
                    transaction.unwatch()
                    return False
                transaction.multi()
                transaction.zadd(
                    active_key,
                    {prediction_id: execution_expires_at},
                    xx=True,
                )
                updated = transaction.execute()
                return bool(updated and int(updated[0]) == 0)
            except WatchError:
                continue


def _terminal_failure(
    error: Exception,
) -> tuple[TerminalFailureKind, str, bool]:
    if isinstance(error, GpuExecutorCaseFailed) or is_case_input_failure(error):
        return (
            TerminalFailureKind.CASE_FAILED,
            "the submitted case could not be processed",
            False,
        )
    if isinstance(error, (GpuExecutorUnavailable, GpuExecutorBusy)):
        return (
            TerminalFailureKind.RUNTIME_UNAVAILABLE,
            "prediction runtime was unavailable",
            True,
        )
    if type(error).__name__ == "JobTimeoutException":
        return TerminalFailureKind.TIMEOUT, "prediction execution timed out", False
    return TerminalFailureKind.EXECUTION_FAILED, "prediction execution failed", False
