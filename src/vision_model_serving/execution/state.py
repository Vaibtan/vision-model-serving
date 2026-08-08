"""Prediction job state repository seam and in-memory adapter."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from time import time
from typing import Callable, Protocol

from .contracts import (
    IdempotencyConflict,
    GatewayObservations,
    PredictionFailure,
    PredictionHandle,
    PredictionId,
    PredictionJobState,
    PredictionNotFound,
    PredictionStatus,
    QueueSaturated,
)


@dataclass(frozen=True, slots=True)
class Admission:
    handle: PredictionHandle
    locator: str


@dataclass(slots=True)
class JobStateRecord:
    prediction_id: PredictionId
    locator: str
    request_fingerprint: str
    idempotency_digest: str | None
    state: PredictionJobState
    submitted_at: float
    lease_expires_at: float
    result_expires_at: float | None = None
    started_at: float | None = None
    completed_at: float | None = None
    failure: PredictionFailure | None = None
    queue_wait_ms: float | None = None

    def status(self) -> PredictionStatus:
        return PredictionStatus(
            prediction_id=self.prediction_id,
            state=self.state,
            submitted_at=self.submitted_at,
            started_at=self.started_at,
            completed_at=self.completed_at,
            expires_at=self.result_expires_at or self.lease_expires_at,
            failure=self.failure,
            queue_wait_ms=self.queue_wait_ms,
        )


class PredictionStateRepository(Protocol):
    def admit(self, **values: object) -> Admission: ...

    def status(self, prediction_id: PredictionId) -> PredictionStatus: ...

    def record(self, prediction_id: PredictionId) -> JobStateRecord: ...

    def mark_running(
        self,
        prediction_id: PredictionId,
        locator: str,
        **values: object,
    ) -> bool: ...

    def mark_succeeded(
        self,
        prediction_id: PredictionId,
        locator: str,
        **values: object,
    ) -> bool: ...

    def mark_failed(
        self,
        prediction_id: PredictionId,
        locator: str,
        **values: object,
    ) -> bool: ...

    def cancel_admission(self, prediction_id: PredictionId, locator: str) -> bool: ...

    def observations(self) -> GatewayObservations: ...


class InMemoryPredictionStateRepository:
    """Atomic state repository adapter for tests and local execution."""

    def __init__(
        self,
        *,
        capacity: int,
        result_ttl_seconds: float,
        clock: Callable[[], float] = time,
    ):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        if result_ttl_seconds <= 0:
            raise ValueError("result TTL must be positive")
        self._capacity = capacity
        self._result_ttl_seconds = float(result_ttl_seconds)
        self._clock = clock
        self._lock = RLock()
        self._records: dict[PredictionId, JobStateRecord] = {}
        self._idempotency: dict[str, PredictionId] = {}
        self._metrics: dict[str, float] = {
            "admitted_total": 0,
            "rejected_total": 0,
            "succeeded_total": 0,
            "failed_total": 0,
            "worker_lost_total": 0,
            "queue_wait_ms_total": 0.0,
        }

    def admit(
        self,
        *,
        prediction_id: PredictionId,
        locator: str,
        request_fingerprint: str,
        idempotency_digest: str | None,
        submitted_at: float,
        reservation_expires_at: float,
    ) -> Admission:
        with self._lock:
            self._refresh_locked(self._clock())
            if idempotency_digest is not None:
                existing_id = self._idempotency.get(idempotency_digest)
                if existing_id is not None:
                    existing = self._records[existing_id]
                    if existing.request_fingerprint != request_fingerprint:
                        raise IdempotencyConflict(
                            "idempotency key is already bound to different input"
                        )
                    handle = _handle(existing, idempotent_replay=True)
                    return Admission(handle, existing.locator)
            active = sum(
                record.state
                in {PredictionJobState.QUEUED, PredictionJobState.RUNNING}
                for record in self._records.values()
            )
            if active >= self._capacity:
                self._metrics["rejected_total"] += 1
                raise QueueSaturated("prediction queue capacity is exhausted")
            record = JobStateRecord(
                prediction_id=prediction_id,
                locator=locator,
                request_fingerprint=request_fingerprint,
                idempotency_digest=idempotency_digest,
                state=PredictionJobState.QUEUED,
                submitted_at=submitted_at,
                lease_expires_at=reservation_expires_at,
            )
            self._records[prediction_id] = record
            if idempotency_digest is not None:
                self._idempotency[idempotency_digest] = prediction_id
            self._metrics["admitted_total"] += 1
            return Admission(_handle(record), locator)

    def status(self, prediction_id: PredictionId) -> PredictionStatus:
        with self._lock:
            self._refresh_locked(self._clock())
            return self._record_locked(prediction_id).status()

    def record(self, prediction_id: PredictionId) -> JobStateRecord:
        with self._lock:
            self._refresh_locked(self._clock())
            return self._record_locked(prediction_id)

    def mark_running(
        self,
        prediction_id: PredictionId,
        locator: str,
        *,
        started_at: float,
        lease_expires_at: float,
    ) -> bool:
        with self._lock:
            self._refresh_locked(self._clock())
            record = self._records[prediction_id]
            if record.locator != locator:
                return False
            if record.state is PredictionJobState.RUNNING:
                return True
            if record.state is not PredictionJobState.QUEUED:
                return False
            record.state = PredictionJobState.RUNNING
            record.started_at = started_at
            record.lease_expires_at = lease_expires_at
            record.queue_wait_ms = max(
                0.0,
                (started_at - record.submitted_at) * 1000.0,
            )
            self._metrics["queue_wait_ms_total"] += record.queue_wait_ms
            return True

    def cancel_admission(self, prediction_id: PredictionId, locator: str) -> bool:
        with self._lock:
            record = self._records.get(prediction_id)
            if (
                record is None
                or record.locator != locator
                or record.state is not PredictionJobState.QUEUED
            ):
                return False
            del self._records[prediction_id]
            if record.idempotency_digest is not None:
                existing = self._idempotency.get(record.idempotency_digest)
                if existing == prediction_id:
                    del self._idempotency[record.idempotency_digest]
            return True

    def mark_succeeded(
        self,
        prediction_id: PredictionId,
        locator: str,
        *,
        completed_at: float,
        result_expires_at: float,
    ) -> bool:
        with self._lock:
            record = self._records[prediction_id]
            if record.locator != locator:
                return False
            if record.state is PredictionJobState.SUCCEEDED:
                return True
            if record.state is not PredictionJobState.RUNNING:
                return False
            record.state = PredictionJobState.SUCCEEDED
            record.completed_at = completed_at
            record.result_expires_at = result_expires_at
            record.failure = None
            self._metrics["succeeded_total"] += 1
            return True

    def mark_failed(
        self,
        prediction_id: PredictionId,
        locator: str,
        *,
        completed_at: float,
        failure: PredictionFailure,
    ) -> bool:
        with self._lock:
            record = self._records[prediction_id]
            if record.locator != locator:
                return False
            if record.state in {
                PredictionJobState.SUCCEEDED,
                PredictionJobState.EXPIRED,
            }:
                return False
            record.state = PredictionJobState.FAILED
            record.completed_at = completed_at
            record.result_expires_at = completed_at + self._result_ttl_seconds
            record.failure = failure
            self._metrics["failed_total"] += 1
            return True

    def observations(self) -> GatewayObservations:
        with self._lock:
            self._refresh_locked(self._clock())
            queued = sum(
                record.state is PredictionJobState.QUEUED
                for record in self._records.values()
            )
            running = sum(
                record.state is PredictionJobState.RUNNING
                for record in self._records.values()
            )
            return GatewayObservations(
                active_jobs=queued + running,
                queued_jobs=queued,
                running_jobs=running,
                admitted_total=int(self._metrics["admitted_total"]),
                rejected_total=int(self._metrics["rejected_total"]),
                succeeded_total=int(self._metrics["succeeded_total"]),
                failed_total=int(self._metrics["failed_total"]),
                worker_lost_total=int(self._metrics["worker_lost_total"]),
                queue_wait_ms_total=self._metrics["queue_wait_ms_total"],
            )

    def _refresh_locked(self, now: float) -> None:
        for prediction_id, record in self._records.items():
            if (
                record.state in {PredictionJobState.QUEUED, PredictionJobState.RUNNING}
                and now >= record.lease_expires_at
            ):
                was_queued = record.state is PredictionJobState.QUEUED
                record.state = PredictionJobState.FAILED
                record.completed_at = now
                record.result_expires_at = now + self._result_ttl_seconds
                record.failure = PredictionFailure(
                    code=(
                        "prediction_reservation_expired"
                        if was_queued
                        else "prediction_worker_lost"
                    ),
                    detail=(
                        "prediction queue reservation expired"
                        if was_queued
                        else "prediction worker was lost"
                    ),
                )
                self._metrics["failed_total"] += 1
                if not was_queued:
                    self._metrics["worker_lost_total"] += 1
            if (
                record.state
                in {PredictionJobState.SUCCEEDED, PredictionJobState.FAILED}
                and record.result_expires_at is not None
                and now >= record.result_expires_at
            ):
                record.state = PredictionJobState.EXPIRED
                record.failure = None
                if record.idempotency_digest is not None:
                    existing = self._idempotency.get(record.idempotency_digest)
                    if existing == prediction_id:
                        del self._idempotency[record.idempotency_digest]

    def _record_locked(self, prediction_id: PredictionId) -> JobStateRecord:
        try:
            return self._records[prediction_id]
        except KeyError:
            raise PredictionNotFound(
                "prediction ID is unknown or no longer retained"
            ) from None


def _handle(
    record: JobStateRecord,
    *,
    idempotent_replay: bool = False,
) -> PredictionHandle:
    return PredictionHandle(
        prediction_id=record.prediction_id,
        state=record.state,
        submitted_at=record.submitted_at,
        expires_at=record.result_expires_at or record.lease_expires_at,
        idempotent_replay=idempotent_replay,
    )
