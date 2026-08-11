"""RQ-backed prediction gateway with private filesystem payloads."""

from __future__ import annotations

import json
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from time import monotonic, sleep, time

from vision_model_serving.pipeline.contracts import PredictionResult

from .contracts import (
    GatewayObservations,
    GatewayUnavailable,
    IdempotencyConflict,
    PredictionFailed,
    PredictionFailure,
    PredictionHandle,
    PredictionId,
    PredictionJobState,
    PredictionNotFound,
    PredictionRequest,
    PredictionRuntimeUnavailable,
    PredictionStatus,
    PredictionTimedOut,
    QueueSaturated,
    ResultExpired,
    ResultNotReady,
)
from .storage import EphemeralJobStore, JobResultNotFound

RQ_TASK_PATH = "vision_model_serving.execution.rq_worker.execute_prediction_job"


@dataclass(frozen=True, slots=True)
class RqExecutionConfig:
    capacity: int
    reservation_ttl_seconds: int
    job_timeout_seconds: int
    result_ttl_seconds: int
    status_ttl_seconds: int
    queue_name: str = "gpu-inference"
    key_prefix: str = "vision-model-serving:predictions"

    def __post_init__(self) -> None:
        if (
            isinstance(self.capacity, bool)
            or not isinstance(self.capacity, int)
            or self.capacity < 1
        ):
            raise ValueError("capacity must be a positive integer")
        for name in (
            "reservation_ttl_seconds",
            "job_timeout_seconds",
            "result_ttl_seconds",
            "status_ttl_seconds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.status_ttl_seconds <= self.result_ttl_seconds:
            raise ValueError("status TTL must exceed the result TTL")
        if not self.queue_name or not self.key_prefix:
            raise ValueError("queue name and key prefix must not be empty")


class RqGpuExecutionGateway:
    """Present the prediction interface while RQ owns the job lifecycle."""

    def __init__(
        self,
        *,
        redis_client: object,
        job_root: Path,
        config: RqExecutionConfig,
        clock: Callable[[], float] = time,
        monotonic_clock: Callable[[], float] = monotonic,
        sleeper: Callable[[float], None] = sleep,
        wait_poll_seconds: float = 0.05,
    ):
        from rq import Queue
        from rq.serializers import JSONSerializer

        if wait_poll_seconds <= 0:
            raise ValueError("wait polling interval must be positive")
        self._redis = redis_client
        self._store = EphemeralJobStore(job_root, clock=clock)
        self._store.cleanup_expired()
        self._config = config
        self._clock = clock
        self._monotonic_clock = monotonic_clock
        self._sleeper = sleeper
        self._wait_poll_seconds = wait_poll_seconds
        self._queue = Queue(
            config.queue_name,
            connection=redis_client,
            default_timeout=config.job_timeout_seconds,
            serializer=JSONSerializer,
        )

    def submit(self, request: PredictionRequest) -> PredictionHandle:
        if not isinstance(request, PredictionRequest):
            raise TypeError("request must be a PredictionRequest")
        prediction_id = PredictionId(secrets.token_urlsafe(24))
        stored = self._store.store_request(
            prediction_id,
            request,
            ttl_seconds=self._config.reservation_ttl_seconds,
        )
        now = self._clock()
        idempotency_key = None
        if request.idempotency_key is not None:
            digest = sha256(request.idempotency_key.encode("utf-8")).hexdigest()
            idempotency_key = f"{self._config.key_prefix}:idempotency:{digest}"
        try:
            while True:
                replay_id = self._admit_and_enqueue(
                    prediction_id=prediction_id,
                    locator=stored.locator,
                    request_fingerprint=stored.request_fingerprint,
                    idempotency_key=idempotency_key,
                    now=now,
                )
                if replay_id is None:
                    break
                try:
                    replay_status = self.status(replay_id)
                except PredictionNotFound:
                    replay_status = None
                if replay_status is None or (
                    replay_status.state is PredictionJobState.EXPIRED
                ):
                    if idempotency_key is None:
                        raise GatewayUnavailable(
                            "prediction idempotency state is unavailable"
                        )
                    self._forget_idempotency(
                        idempotency_key,
                        stored.request_fingerprint,
                        replay_id,
                    )
                    continue
                self._store.discard_job(stored.locator)
                return PredictionHandle(
                    prediction_id=replay_id,
                    state=replay_status.state,
                    submitted_at=replay_status.submitted_at,
                    expires_at=replay_status.expires_at,
                    idempotent_replay=True,
                )
        except (IdempotencyConflict, QueueSaturated):
            self._store.discard_job(stored.locator)
            raise
        except Exception:  # noqa: BLE001 - sanitize the Redis/RQ failure boundary
            self._store.discard_job(stored.locator)
            raise GatewayUnavailable("prediction dispatch is unavailable") from None
        return PredictionHandle(
            prediction_id=prediction_id,
            state=PredictionJobState.QUEUED,
            submitted_at=now,
            expires_at=stored.expires_at,
        )

    def status(self, prediction_id: PredictionId) -> PredictionStatus:
        self._store.maybe_cleanup()
        try:
            job = self._job(prediction_id)
        except PredictionNotFound:
            marker = self._terminal_marker(prediction_id)
            if marker is None:
                raise
            return self._status_from_marker(prediction_id, marker)
        try:
            status = _text(job.get_status(refresh=True))
        except Exception:  # noqa: BLE001 - sanitize the RQ status boundary
            raise GatewayUnavailable("prediction status is unavailable") from None
        now = self._clock()
        submitted_at = _timestamp(job.enqueued_at) or _timestamp(job.created_at)
        if submitted_at is None:
            submitted_at = now
        started_at = _timestamp(job.started_at)
        completed_at = _timestamp(job.ended_at)
        failure = None
        if status in {"queued", "deferred", "scheduled", "rate_limited"}:
            state = PredictionJobState.QUEUED
            expires_at = submitted_at + self._config.reservation_ttl_seconds
            if now >= expires_at:
                state = PredictionJobState.FAILED
                failure = PredictionFailure(
                    "prediction_reservation_expired",
                    "prediction reservation expired",
                )
                self._abandon_expired_reservation(
                    job,
                    prediction_id,
                    submitted_at=submitted_at,
                    now=now,
                )
        elif status == "started":
            state = PredictionJobState.RUNNING
            expires_at = (started_at or now) + self._config.job_timeout_seconds
        elif status == "finished":
            expires_at = (completed_at or now) + self._config.result_ttl_seconds
            state = (
                PredictionJobState.EXPIRED
                if now >= expires_at
                else PredictionJobState.SUCCEEDED
            )
        elif status in {"failed", "stopped", "canceled"}:
            state = PredictionJobState.FAILED
            expires_at = (completed_at or now) + self._config.status_ttl_seconds
            failure = self._classify_failure(job, prediction_id)
        else:
            raise PredictionNotFound("prediction job is unavailable")
        queue_wait_ms = None
        if started_at is not None:
            queue_wait_ms = max(0.0, (started_at - submitted_at) * 1_000.0)
        return PredictionStatus(
            prediction_id=prediction_id,
            state=state,
            submitted_at=submitted_at,
            started_at=started_at,
            completed_at=completed_at,
            expires_at=expires_at,
            failure=failure,
            queue_wait_ms=queue_wait_ms,
        )

    def _classify_failure(
        self,
        job: object,
        prediction_id: PredictionId,
    ) -> PredictionFailure:
        marker = self._terminal_marker(prediction_id)
        if marker is not None:
            return PredictionFailure(
                marker["code"],
                marker["detail"],
                retryable=marker["retryable"],
            )
        try:
            latest_result = job.latest_result()
        except Exception:  # noqa: BLE001 - sanitize the RQ result boundary
            raise GatewayUnavailable("prediction status is unavailable") from None
        failure_text = getattr(latest_result, "exc_string", "") or ""
        timed_out = any(
            marker_text in failure_text
            for marker_text in (
                "JobTimeoutException",
                "maximum timeout value",
            )
        )
        runtime_unavailable = (
            "GpuExecutorUnavailable" in failure_text
            or "GpuExecutorBusy" in failure_text
        )
        worker_lost = any(
            marker_text in failure_text
            for marker_text in (
                "AbandonedJobError",
                "Work-horse terminated unexpectedly",
            )
        )
        if (
            "GpuExecutorCaseFailed" in failure_text
            or "StoredPredictionCaseError" in failure_text
        ):
            return PredictionFailure(
                "prediction_case_failed",
                "the submitted case could not be processed",
            )
        if timed_out:
            return PredictionFailure(
                "prediction_timeout",
                "prediction execution timed out",
            )
        if runtime_unavailable or worker_lost:
            return PredictionFailure(
                "prediction_runtime_unavailable",
                "prediction runtime was unavailable",
                retryable=True,
            )
        return PredictionFailure(
            "prediction_execution_failed",
            "prediction execution failed",
        )

    def _terminal_key(self, prediction_id: PredictionId) -> str:
        return f"{self._config.key_prefix}:terminal:{prediction_id}"

    def _terminal_marker(self, prediction_id: PredictionId) -> dict[str, object] | None:
        try:
            raw = self._redis.get(self._terminal_key(prediction_id))
        except Exception:  # noqa: BLE001 - marker lookup is best-effort
            return None
        if raw is None:
            return None
        try:
            value = json.loads(_text(raw))
            return {
                "code": str(value["code"]),
                "detail": str(value["detail"]),
                "retryable": bool(value.get("retryable", False)),
                "submitted_at": float(value["submitted_at"]),
                "completed_at": float(value["completed_at"]),
            }
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None

    def _status_from_marker(
        self,
        prediction_id: PredictionId,
        marker: dict[str, object],
    ) -> PredictionStatus:
        completed_at = float(marker["completed_at"])
        return PredictionStatus(
            prediction_id=prediction_id,
            state=PredictionJobState.FAILED,
            submitted_at=float(marker["submitted_at"]),
            started_at=None,
            completed_at=completed_at,
            expires_at=completed_at + self._config.status_ttl_seconds,
            failure=PredictionFailure(
                str(marker["code"]),
                str(marker["detail"]),
                retryable=bool(marker["retryable"]),
            ),
            queue_wait_ms=None,
        )

    def _abandon_expired_reservation(
        self,
        job: object,
        prediction_id: PredictionId,
        *,
        submitted_at: float,
        now: float,
    ) -> None:
        """Convert a lapsed reservation into a terminal state and reclaim it."""

        try:
            self._redis.set(
                self._terminal_key(prediction_id),
                json.dumps(
                    {
                        "code": "prediction_reservation_expired",
                        "detail": "prediction reservation expired",
                        "retryable": False,
                        "submitted_at": submitted_at,
                        "completed_at": now,
                    },
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                nx=True,
                ex=self._config.status_ttl_seconds,
            )
            self._redis.zrem(
                f"{self._config.key_prefix}:active",
                str(prediction_id),
            )
            try:
                locator = self._locator(job)
            except GatewayUnavailable:
                locator = None
            job.cancel()
            if locator is not None:
                self._store.discard_job(locator)
        except Exception:  # noqa: BLE001, S110 - cleanup is best-effort; the
            # synthesized FAILED status is already correct without it
            pass

    def result(self, prediction_id: PredictionId) -> PredictionResult:
        status = self.status(prediction_id)
        if status.state is PredictionJobState.EXPIRED:
            raise ResultExpired("prediction result has expired")
        if status.state is PredictionJobState.FAILED:
            failure = status.failure or PredictionFailure(
                "prediction_failed",
                "prediction failed",
            )
            if failure.code == PredictionTimedOut.code:
                raise PredictionTimedOut(failure.detail)
            if failure.code == PredictionRuntimeUnavailable.code:
                raise PredictionRuntimeUnavailable(failure.detail)
            raise PredictionFailed(failure.detail)
        if status.state is not PredictionJobState.SUCCEEDED:
            raise ResultNotReady("prediction has not completed successfully")
        try:
            return self._store.load_result(self._locator(self._job(prediction_id)))
        except JobResultNotFound:
            raise ResultExpired("prediction result has expired") from None

    def wait(
        self,
        prediction_id: PredictionId,
        *,
        timeout_seconds: float,
    ) -> PredictionResult | PredictionHandle:
        if timeout_seconds < 0:
            raise ValueError("wait timeout must not be negative")
        deadline = self._monotonic_clock() + timeout_seconds
        while True:
            status = self.status(prediction_id)
            if status.state in {
                PredictionJobState.SUCCEEDED,
                PredictionJobState.FAILED,
                PredictionJobState.EXPIRED,
            }:
                return self.result(prediction_id)
            remaining = deadline - self._monotonic_clock()
            if remaining <= 0:
                return PredictionHandle(
                    prediction_id=prediction_id,
                    state=status.state,
                    submitted_at=status.submitted_at,
                    expires_at=status.expires_at,
                )
            self._sleeper(min(self._wait_poll_seconds, remaining))

    def observations(self) -> GatewayObservations:
        try:
            active_key = f"{self._config.key_prefix}:active"
            metrics = {
                _text(key): float(_text(value))
                for key, value in self._redis.hgetall(
                    f"{self._config.key_prefix}:metrics"
                ).items()
            }
            return GatewayObservations(
                active_jobs=int(self._redis.zcount(active_key, self._clock(), "+inf")),
                queued_jobs=int(self._queue.count),
                running_jobs=int(self._queue.started_job_registry.count),
                admitted_total=int(metrics.get("admitted_total", 0)),
                rejected_total=int(metrics.get("rejected_total", 0)),
                succeeded_total=int(metrics.get("succeeded_total", 0)),
                failed_total=int(metrics.get("failed_total", 0)),
                worker_lost_total=int(metrics.get("worker_lost_total", 0)),
                queue_wait_ms_total=metrics.get("queue_wait_ms_total", 0.0),
            )
        except Exception:  # noqa: BLE001 - sanitize the Redis metrics boundary
            raise GatewayUnavailable(
                "prediction observations are unavailable"
            ) from None

    def _job(self, prediction_id: PredictionId):
        from rq.exceptions import NoSuchJobError
        from rq.job import Job
        from rq.serializers import JSONSerializer

        try:
            return Job.fetch(
                str(prediction_id),
                connection=self._redis,
                serializer=JSONSerializer,
            )
        except NoSuchJobError:
            raise PredictionNotFound("prediction job was not found") from None
        except Exception:  # noqa: BLE001 - sanitize the Redis lookup boundary
            raise GatewayUnavailable("prediction status is unavailable") from None

    @staticmethod
    def _locator(job: object) -> str:
        try:
            locator = job.args[1]
        except (AttributeError, IndexError, TypeError):
            raise GatewayUnavailable("prediction job metadata is unavailable") from None
        if not isinstance(locator, str):
            raise GatewayUnavailable("prediction job metadata is unavailable")
        return locator

    def _admit_and_enqueue(
        self,
        *,
        prediction_id: PredictionId,
        locator: str,
        request_fingerprint: str,
        idempotency_key: str | None,
        now: float,
    ) -> PredictionId | None:
        from redis.exceptions import WatchError

        active_key = f"{self._config.key_prefix}:active"
        metrics_key = f"{self._config.key_prefix}:metrics"
        watched_keys = (
            (active_key, idempotency_key)
            if idempotency_key is not None
            else (active_key,)
        )
        while True:
            self._redis.zremrangebyscore(active_key, "-inf", now)
            with self._redis.pipeline() as transaction:
                try:
                    transaction.watch(*watched_keys)
                    if idempotency_key is not None:
                        existing = transaction.get(idempotency_key)
                        if existing is not None:
                            transaction.unwatch()
                            fingerprint, existing_id = _text(existing).split(":", 1)
                            if fingerprint != request_fingerprint:
                                raise IdempotencyConflict(
                                    "idempotency key was already used for another request"
                                )
                            return PredictionId(existing_id)
                    if int(transaction.zcard(active_key)) >= self._config.capacity:
                        transaction.multi()
                        transaction.hincrby(metrics_key, "rejected_total", 1)
                        transaction.execute()
                        raise QueueSaturated("prediction queue is full")
                    transaction.multi()
                    transaction.zadd(
                        active_key,
                        {
                            str(prediction_id): (
                                now + self._config.reservation_ttl_seconds
                            )
                        },
                    )
                    if idempotency_key is not None:
                        transaction.set(
                            idempotency_key,
                            f"{request_fingerprint}:{prediction_id}",
                            ex=(
                                self._config.reservation_ttl_seconds
                                + self._config.job_timeout_seconds
                                + self._config.status_ttl_seconds
                            ),
                        )
                    transaction.hincrby(metrics_key, "admitted_total", 1)
                    self._queue.enqueue_call(
                        func=RQ_TASK_PATH,
                        args=(str(prediction_id), locator),
                        job_id=str(prediction_id),
                        timeout=self._config.job_timeout_seconds,
                        ttl=self._config.reservation_ttl_seconds,
                        result_ttl=self._config.status_ttl_seconds,
                        failure_ttl=self._config.status_ttl_seconds,
                        retry=None,
                        meta={
                            "result_ttl_seconds": self._config.result_ttl_seconds,
                            "status_ttl_seconds": self._config.status_ttl_seconds,
                            "key_prefix": self._config.key_prefix,
                        },
                        pipeline=transaction,
                    )
                    transaction.execute()
                    return None
                except WatchError:
                    continue

    def _forget_idempotency(
        self,
        key: str,
        request_fingerprint: str,
        prediction_id: PredictionId,
    ) -> None:
        from redis.exceptions import WatchError

        expected = f"{request_fingerprint}:{prediction_id}"
        while True:
            with self._redis.pipeline() as transaction:
                try:
                    transaction.watch(key)
                    if _text(transaction.get(key)) != expected:
                        transaction.unwatch()
                        return
                    transaction.multi()
                    transaction.delete(key)
                    transaction.execute()
                    return
                except WatchError:
                    continue


def _text(value: object) -> str:
    if hasattr(value, "value"):
        value = value.value
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _timestamp(value: object) -> float | None:
    return float(value.timestamp()) if hasattr(value, "timestamp") else None
