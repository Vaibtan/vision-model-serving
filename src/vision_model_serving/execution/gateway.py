"""Cross-process gateway over state, payload-store, and queue adapters."""

from __future__ import annotations

from hashlib import sha256
import secrets
from time import monotonic, sleep, time
from typing import Callable, Protocol

from vision_model_serving.pipeline.contracts import PredictionResult

from .contracts import (
    GatewayUnavailable,
    GatewayObservations,
    PredictionFailed,
    PredictionFailure,
    PredictionHandle,
    PredictionId,
    PredictionJobState,
    PredictionRequest,
    PredictionStatus,
    ResultExpired,
    ResultNotReady,
)
from .state import PredictionStateRepository
from .storage import EphemeralJobStore


class _Dispatcher(Protocol):
    def dispatch(self, prediction_id: PredictionId, locator: str) -> None: ...


class StoredGpuExecutionGateway:
    """Own admission and dispatch while private inputs remain in job storage."""

    def __init__(
        self,
        *,
        state: PredictionStateRepository,
        store: EphemeralJobStore,
        dispatcher: _Dispatcher,
        reservation_ttl_seconds: float,
        clock: Callable[[], float] = time,
        monotonic_clock: Callable[[], float] = monotonic,
        sleeper: Callable[[float], None] = sleep,
        wait_poll_seconds: float = 0.05,
    ):
        if reservation_ttl_seconds <= 0 or wait_poll_seconds <= 0:
            raise ValueError(
                "reservation TTL and wait polling interval must be positive"
            )
        self._state = state
        self._store = store
        self._dispatcher = dispatcher
        self._reservation_ttl_seconds = float(reservation_ttl_seconds)
        self._clock = clock
        self._monotonic_clock = monotonic_clock
        self._sleeper = sleeper
        self._wait_poll_seconds = float(wait_poll_seconds)

    def submit(self, request: PredictionRequest) -> PredictionHandle:
        if not isinstance(request, PredictionRequest):
            raise TypeError("request must be a PredictionRequest")
        now = self._clock()
        prediction_id = PredictionId(secrets.token_urlsafe(24))
        stored = self._store.store_request(
            prediction_id,
            request,
            ttl_seconds=self._reservation_ttl_seconds,
        )
        idempotency_digest = (
            sha256(request.idempotency_key.encode("utf-8")).hexdigest()
            if request.idempotency_key is not None
            else None
        )
        try:
            admission = self._state.admit(
                prediction_id=prediction_id,
                locator=stored.locator,
                request_fingerprint=stored.request_fingerprint,
                idempotency_digest=idempotency_digest,
                submitted_at=now,
                reservation_expires_at=stored.expires_at,
            )
        except Exception:
            self._store.discard_job(stored.locator)
            raise
        if admission.handle.idempotent_replay:
            self._store.discard_job(stored.locator)
            return admission.handle
        try:
            self._dispatcher.dispatch(prediction_id, stored.locator)
        except Exception:
            self._state.cancel_admission(prediction_id, stored.locator)
            self._store.discard_job(stored.locator)
            raise GatewayUnavailable("prediction dispatch is unavailable") from None
        return admission.handle

    def status(self, prediction_id: PredictionId) -> PredictionStatus:
        return self._state.status(prediction_id)

    def result(self, prediction_id: PredictionId) -> PredictionResult:
        status = self._state.status(prediction_id)
        if status.state is PredictionJobState.EXPIRED:
            raise ResultExpired("prediction result has expired")
        if status.state is PredictionJobState.FAILED:
            failure = status.failure or PredictionFailure(
                "prediction_execution_failed",
                "prediction execution failed",
            )
            raise PredictionFailed(failure.detail)
        if status.state is not PredictionJobState.SUCCEEDED:
            raise ResultNotReady("prediction has not completed successfully")
        return self._store.load_result(self._state.record(prediction_id).locator)

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
                    prediction_id=status.prediction_id,
                    state=status.state,
                    submitted_at=status.submitted_at,
                    expires_at=status.expires_at,
                )
            self._sleeper(min(self._wait_poll_seconds, remaining))

    def observations(self) -> GatewayObservations:
        return self._state.observations()
