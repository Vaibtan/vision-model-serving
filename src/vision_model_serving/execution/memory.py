"""In-memory prediction execution adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
import secrets
from threading import Event, RLock
from time import time
from typing import Callable, Protocol

from vision_model_serving.pipeline.contracts import CaseInput, PredictionResult

from .contracts import (
    IdempotencyConflict,
    PredictionFailed,
    PredictionFailure,
    PredictionHandle,
    PredictionId,
    PredictionJobState,
    PredictionNotFound,
    PredictionRequest,
    PredictionStatus,
    QueueSaturated,
    ResultExpired,
    ResultNotReady,
)
from .fingerprinting import request_fingerprint


class _Executor(Protocol):
    def submit(self, function: object, *args: object) -> object: ...


class _Pipeline(Protocol):
    def infer(self, case: CaseInput, mode: object) -> PredictionResult: ...


@dataclass(slots=True)
class _Job:
    request: PredictionRequest | None
    state: PredictionJobState
    submitted_at: float
    expires_at: float
    started_at: float | None = None
    completed_at: float | None = None
    result: PredictionResult | None = None
    failure: PredictionFailure | None = None
    idempotency_digest: str | None = None
    done: Event = field(default_factory=Event, repr=False)


class InMemoryGpuExecutionGateway:
    """Bound admission and job state behind the prediction gateway interface."""

    def __init__(
        self,
        *,
        pipeline: _Pipeline,
        capacity: int,
        executor: _Executor | None = None,
        reservation_ttl_seconds: float = 120.0,
        worker_loss_ttl_seconds: float = 900.0,
        result_ttl_seconds: float = 300.0,
        clock: Callable[[], float] = time,
    ):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        if (
            reservation_ttl_seconds <= 0
            or worker_loss_ttl_seconds <= 0
            or result_ttl_seconds <= 0
        ):
            raise ValueError(
                "reservation, worker-loss, and result TTLs must be positive"
            )
        self._pipeline = pipeline
        self._capacity = capacity
        self._executor = executor or ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="prediction-gateway",
        )
        self._owns_executor = executor is None
        self._reservation_ttl_seconds = float(reservation_ttl_seconds)
        self._worker_loss_ttl_seconds = float(worker_loss_ttl_seconds)
        self._result_ttl_seconds = float(result_ttl_seconds)
        self._clock = clock
        self._lock = RLock()
        self._jobs: dict[PredictionId, _Job] = {}
        self._idempotency: dict[str, tuple[PredictionId, str]] = {}

    def submit(self, request: PredictionRequest) -> PredictionHandle:
        if not isinstance(request, PredictionRequest):
            raise TypeError("request must be a PredictionRequest")
        payload = request.case.dicom_stream.read()
        if not isinstance(payload, bytes):
            raise TypeError("DICOM stream must return bytes")
        copied = PredictionRequest(
            case=CaseInput(BytesIO(payload), request.case.clinical_history),
            mode=request.mode,
            idempotency_key=request.idempotency_key,
        )
        fingerprint = request_fingerprint(copied, payload)
        now = self._clock()
        with self._lock:
            self._expire_jobs_locked(now)
            if request.idempotency_key is not None:
                idempotency_digest = sha256(
                    request.idempotency_key.encode("utf-8")
                ).hexdigest()
                existing = self._idempotency.get(idempotency_digest)
                if existing is not None and existing[1] == fingerprint:
                    prediction_id, _ = existing
                    return self._handle(
                        prediction_id,
                        self._jobs[prediction_id],
                        idempotent_replay=True,
                    )
                if existing is not None:
                    raise IdempotencyConflict(
                        "idempotency key is already bound to different input"
                    )
            active = sum(
                job.state in {PredictionJobState.QUEUED, PredictionJobState.RUNNING}
                for job in self._jobs.values()
            )
            if active >= self._capacity:
                raise QueueSaturated("prediction queue capacity is exhausted")
            prediction_id = PredictionId(secrets.token_urlsafe(24))
            job = _Job(
                request=copied,
                state=PredictionJobState.QUEUED,
                submitted_at=now,
                expires_at=now + self._reservation_ttl_seconds,
                idempotency_digest=(
                    idempotency_digest
                    if request.idempotency_key is not None
                    else None
                ),
            )
            self._jobs[prediction_id] = job
            if request.idempotency_key is not None:
                self._idempotency[idempotency_digest] = (
                    prediction_id,
                    fingerprint,
                )
        handle = self._handle(prediction_id, job)
        self._executor.submit(self._execute, prediction_id)
        return handle

    def status(self, prediction_id: PredictionId) -> PredictionStatus:
        with self._lock:
            self._expire_jobs_locked(self._clock())
            job = self._job_locked(prediction_id)
            return PredictionStatus(
                prediction_id=prediction_id,
                state=job.state,
                submitted_at=job.submitted_at,
                started_at=job.started_at,
                completed_at=job.completed_at,
                expires_at=job.expires_at,
                failure=job.failure,
            )

    def result(self, prediction_id: PredictionId) -> PredictionResult:
        with self._lock:
            self._expire_jobs_locked(self._clock())
            job = self._job_locked(prediction_id)
            if job.state is PredictionJobState.EXPIRED:
                raise ResultExpired("prediction result has expired")
            if job.state is PredictionJobState.FAILED:
                failure = job.failure or PredictionFailure(
                    "prediction_execution_failed",
                    "prediction execution failed",
                )
                raise PredictionFailed(failure.detail)
            if job.state is not PredictionJobState.SUCCEEDED:
                raise ResultNotReady("prediction has not completed successfully")
            if job.result is None:
                raise ResultNotReady("prediction result is unavailable")
            return job.result

    def wait(
        self,
        prediction_id: PredictionId,
        *,
        timeout_seconds: float,
    ) -> PredictionResult | PredictionHandle:
        if timeout_seconds < 0:
            raise ValueError("wait timeout must not be negative")
        with self._lock:
            self._expire_jobs_locked(self._clock())
            job = self._job_locked(prediction_id)
            done = job.done
        if done.wait(timeout_seconds):
            return self.result(prediction_id)
        with self._lock:
            self._expire_jobs_locked(self._clock())
            job = self._jobs[prediction_id]
            if job.done.is_set():
                return self.result(prediction_id)
            return self._handle(prediction_id, job)

    def close(self, *, wait: bool = True) -> None:
        if self._owns_executor:
            executor = self._executor
            if isinstance(executor, ThreadPoolExecutor):
                executor.shutdown(wait=wait, cancel_futures=False)

    def __enter__(self) -> InMemoryGpuExecutionGateway:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _execute(self, prediction_id: PredictionId) -> None:
        with self._lock:
            job = self._jobs[prediction_id]
            if job.state is not PredictionJobState.QUEUED or job.request is None:
                return
            job.state = PredictionJobState.RUNNING
            job.started_at = self._clock()
            job.expires_at = job.started_at + self._worker_loss_ttl_seconds
            request = job.request
        try:
            result = self._pipeline.infer(request.case, request.mode)
        except Exception:
            completed_at = self._clock()
            with self._lock:
                job = self._jobs[prediction_id]
                if job.state is not PredictionJobState.RUNNING:
                    return
                job.request = None
                job.state = PredictionJobState.FAILED
                job.completed_at = completed_at
                job.expires_at = completed_at + self._result_ttl_seconds
                job.failure = PredictionFailure(
                    code="prediction_execution_failed",
                    detail="prediction execution failed",
                )
                job.done.set()
            return
        completed_at = self._clock()
        with self._lock:
            job = self._jobs[prediction_id]
            if job.state is not PredictionJobState.RUNNING:
                return
            job.result = result
            job.request = None
            job.state = PredictionJobState.SUCCEEDED
            job.completed_at = completed_at
            job.expires_at = completed_at + self._result_ttl_seconds
            job.done.set()

    def _expire_jobs_locked(self, now: float) -> None:
        for prediction_id, job in self._jobs.items():
            if job.state is PredictionJobState.QUEUED and now >= job.expires_at:
                job.state = PredictionJobState.FAILED
                job.request = None
                job.completed_at = now
                job.expires_at = now + self._result_ttl_seconds
                job.failure = PredictionFailure(
                    code="prediction_reservation_expired",
                    detail="prediction queue reservation expired",
                )
                job.done.set()
            if job.state is PredictionJobState.RUNNING and now >= job.expires_at:
                job.state = PredictionJobState.FAILED
                job.request = None
                job.completed_at = now
                job.expires_at = now + self._result_ttl_seconds
                job.failure = PredictionFailure(
                    code="prediction_worker_lost",
                    detail="prediction worker was lost",
                )
                job.done.set()
            if (
                job.state
                in {PredictionJobState.SUCCEEDED, PredictionJobState.FAILED}
                and now >= job.expires_at
            ):
                job.state = PredictionJobState.EXPIRED
                job.request = None
                job.result = None
                job.failure = None
                if job.idempotency_digest is not None:
                    existing = self._idempotency.get(job.idempotency_digest)
                    if existing is not None and existing[0] == prediction_id:
                        del self._idempotency[job.idempotency_digest]

    def _job_locked(self, prediction_id: PredictionId) -> _Job:
        try:
            return self._jobs[prediction_id]
        except KeyError:
            raise PredictionNotFound(
                "prediction ID is unknown or no longer retained"
            ) from None

    @staticmethod
    def _handle(
        prediction_id: PredictionId,
        job: _Job,
        *,
        idempotent_replay: bool = False,
    ) -> PredictionHandle:
        return PredictionHandle(
            prediction_id=prediction_id,
            state=job.state,
            submitted_at=job.submitted_at,
            expires_at=job.expires_at,
            idempotent_replay=idempotent_replay,
        )
