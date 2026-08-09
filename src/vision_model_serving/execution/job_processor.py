"""Private stored-request processor owned by the persistent GPU executor."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from time import time
from typing import Protocol

from vision_model_serving.observability import (
    record_prediction_failure,
    record_prediction_success,
)
from vision_model_serving.pipeline.contracts import (
    CaseInput,
    PredictionMode,
    PredictionResult,
)

from .contracts import PredictionId
from .storage import EphemeralJobStore, JobResultNotFound


class _Pipeline(Protocol):
    def infer(self, case: CaseInput, mode: PredictionMode) -> PredictionResult: ...


class StoredPredictionProcessorError(RuntimeError):
    pass


class StoredPredictionProcessor:
    """Execute a private stored request inside the long-lived GPU owner."""

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
        mode: PredictionMode | None = None
        try:
            try:
                self._store.load_result(locator)
                return
            except JobResultNotFound:
                pass
            request = self._store.load_request(prediction_id, locator)
            mode = request.mode
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
            try:
                record_prediction_success(result)
            except Exception:  # noqa: BLE001, S110 - telemetry is non-authoritative
                pass
        except Exception as error:  # noqa: BLE001 - never persist private pipeline errors
            try:
                record_prediction_failure(mode, error)
            except Exception:  # noqa: BLE001, S110 - preserve the prediction seam
                pass
            raise StoredPredictionProcessorError("prediction execution failed") from None
        finally:
            self._store.purge_request(locator)

    def status(self) -> object:
        status = getattr(self._pipeline, "status", None)
        if not callable(status):
            raise StoredPredictionProcessorError("prediction runtime status is unavailable")
        return status()


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
