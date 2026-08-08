"""Dedicated prediction worker orchestration."""

from __future__ import annotations

from time import time
from typing import Callable, Protocol

from vision_model_serving.pipeline.contracts import (
    CaseInput,
    PredictionMode,
    PredictionResult,
)

from .contracts import PredictionFailure, PredictionId, PredictionJobState
from .state import PredictionStateRepository
from .storage import EphemeralJobStore, JobResultNotFound


class _Pipeline(Protocol):
    def infer(self, case: CaseInput, mode: PredictionMode) -> PredictionResult: ...


class PredictionJobWorker:
    """Own one job's transition through the GPU prediction pipeline."""

    def __init__(
        self,
        *,
        state: PredictionStateRepository,
        store: EphemeralJobStore,
        pipeline: _Pipeline,
        worker_loss_ttl_seconds: float,
        result_ttl_seconds: float,
        clock: Callable[[], float] = time,
    ):
        if worker_loss_ttl_seconds <= 0 or result_ttl_seconds <= 0:
            raise ValueError("worker-loss and result TTLs must be positive")
        self._state = state
        self._store = store
        self._pipeline = pipeline
        self._worker_loss_ttl_seconds = float(worker_loss_ttl_seconds)
        self._result_ttl_seconds = float(result_ttl_seconds)
        self._clock = clock

    def execute(self, prediction_id: PredictionId, locator: str) -> None:
        existing = self._state.status(prediction_id)
        if existing.state is PredictionJobState.SUCCEEDED:
            return
        now = self._clock()
        if not self._state.mark_running(
            prediction_id,
            locator,
            started_at=now,
            lease_expires_at=now + self._worker_loss_ttl_seconds,
        ):
            return
        try:
            try:
                stored_result = self._store.load_result(locator)
            except JobResultNotFound:
                stored_result = None
            if stored_result is None:
                request = self._store.load_request(prediction_id, locator)
                result = self._pipeline.infer(request.case, request.mode)
                self._store.store_result(
                    locator,
                    result,
                    ttl_seconds=self._result_ttl_seconds,
                )
            completed_at = self._clock()
            self._state.mark_succeeded(
                prediction_id,
                locator,
                completed_at=completed_at,
                result_expires_at=completed_at + self._result_ttl_seconds,
            )
        except Exception:
            completed_at = self._clock()
            self._state.mark_failed(
                prediction_id,
                locator,
                completed_at=completed_at,
                failure=PredictionFailure(
                    code="prediction_execution_failed",
                    detail="prediction execution failed",
                ),
            )
        finally:
            self._store.purge_request(locator)
