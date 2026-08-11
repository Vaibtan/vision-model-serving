"""Public contracts for bounded prediction execution."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import NewType, Protocol

from vision_model_serving.model_ids import MODEL_IDS
from vision_model_serving.pipeline.contracts import (
    CaseInput,
    PredictionMode,
    PredictionResult,
)

PredictionId = NewType("PredictionId", str)
QUEUE_WAIT_BUCKET_SECONDS = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0)


class PredictionGatewayError(RuntimeError):
    code = "prediction_gateway_failed"

    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(f"{self.code}: {detail}")


class IdempotencyConflict(PredictionGatewayError):
    code = "prediction_idempotency_conflict"


class QueueSaturated(PredictionGatewayError):
    code = "prediction_queue_full"


class GatewayUnavailable(PredictionGatewayError):
    code = "prediction_gateway_unavailable"


class PredictionNotFound(PredictionGatewayError):
    code = "prediction_not_found"


class ResultNotReady(PredictionGatewayError):
    code = "prediction_result_not_ready"


class ResultExpired(PredictionGatewayError):
    code = "prediction_result_expired"


class PredictionFailed(PredictionGatewayError):
    code = "prediction_failed"


class PredictionRuntimeUnavailable(PredictionFailed):
    code = "prediction_runtime_unavailable"


class PredictionTimedOut(PredictionFailed):
    code = "prediction_timeout"


@dataclass(frozen=True, slots=True)
class PredictionFailure:
    code: str
    detail: str
    retryable: bool = False


class PredictionJobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass(frozen=True, slots=True)
class PredictionRequest:
    case: CaseInput = field(repr=False)
    mode: PredictionMode
    idempotency_key: str | None = field(default=None, repr=False)
    detector_score_threshold: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.case, CaseInput):
            raise TypeError("case must be a CaseInput")
        if not isinstance(self.mode, PredictionMode):
            raise TypeError("mode must be a PredictionMode")
        if self.idempotency_key is not None and (
            not isinstance(self.idempotency_key, str)
            or not self.idempotency_key
            or len(self.idempotency_key) > 200
        ):
            raise ValueError("idempotency key must contain 1 to 200 characters")
        if self.detector_score_threshold is not None and (
            isinstance(self.detector_score_threshold, bool)
            or not isinstance(self.detector_score_threshold, (int, float))
            or not math.isfinite(float(self.detector_score_threshold))
            or not 0.0 <= float(self.detector_score_threshold) <= 1.0
        ):
            raise ValueError("detector score threshold must be between zero and one")
        if self.detector_score_threshold is not None:
            object.__setattr__(
                self,
                "detector_score_threshold",
                float(self.detector_score_threshold),
            )


@dataclass(frozen=True, slots=True)
class PredictionHandle:
    prediction_id: PredictionId
    state: PredictionJobState
    submitted_at: float
    expires_at: float
    idempotent_replay: bool = False


@dataclass(frozen=True, slots=True)
class PredictionStatus:
    prediction_id: PredictionId
    state: PredictionJobState
    submitted_at: float
    started_at: float | None
    completed_at: float | None
    expires_at: float
    failure: PredictionFailure | None = None
    queue_wait_ms: float | None = None


@dataclass(frozen=True, slots=True)
class GatewayObservations:
    active_jobs: int
    queued_jobs: int
    running_jobs: int
    admitted_total: int
    rejected_total: int
    succeeded_total: int
    failed_total: int
    worker_lost_total: int
    queue_wait_count: int
    queue_wait_ms_total: float
    queue_wait_buckets: tuple[tuple[float, int], ...]


@dataclass(frozen=True, slots=True)
class ExecutorStartupTimings:
    artifact_verification_ms: float
    runtime_initialization_ms: float
    process_start_to_artifact_ready_ms: float

    def __post_init__(self) -> None:
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (
                self.artifact_verification_ms,
                self.runtime_initialization_ms,
                self.process_start_to_artifact_ready_ms,
            )
        ):
            raise ValueError("executor startup timings must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class GpuExecutorStatus:
    verified_artifacts: bool
    runtime_initialized: bool
    device_available: bool
    native_operator_available: bool
    runtime_state: str
    active_model: str | None
    resident_models: tuple[str, ...]
    device_name: str
    last_error: str | None
    active_task: bool = False
    active_task_age_ms: float | None = None
    deadline_remaining_ms: float | None = None
    startup: ExecutorStartupTimings = field(
        default_factory=lambda: ExecutorStartupTimings(0.0, 0.0, 0.0)
    )

    def __post_init__(self) -> None:
        booleans = (
            self.verified_artifacts,
            self.runtime_initialized,
            self.device_available,
            self.native_operator_available,
        )
        if not all(isinstance(value, bool) for value in booleans):
            raise TypeError("executor readiness facts must be boolean")
        if self.runtime_state not in {
            "unloaded",
            "loading",
            "ready",
            "draining",
            "unloading",
            "failed",
        }:
            raise ValueError("executor runtime state is invalid")
        if not isinstance(self.device_name, str) or not self.device_name:
            raise ValueError("executor device name must not be empty")
        if self.last_error is not None and (
            not isinstance(self.last_error, str) or not self.last_error
        ):
            raise ValueError("executor failure code is invalid")
        for name in ("active_task_age_ms", "deadline_remaining_ms"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0.0
            ):
                raise ValueError(f"executor {name} must be finite and non-negative")
        if not isinstance(self.active_task, bool):
            raise TypeError("executor active_task must be boolean")
        if self.active_task != (self.active_task_age_ms is not None):
            raise ValueError("executor active-task age does not match active state")
        if self.active_task != (self.deadline_remaining_ms is not None):
            raise ValueError("executor deadline does not match active state")
        if (
            not isinstance(self.resident_models, tuple)
            or len(self.resident_models) > 1
            or len(set(self.resident_models)) != len(self.resident_models)
            or any(model not in MODEL_IDS for model in self.resident_models)
        ):
            raise ValueError("executor may report at most one known resident model")
        if self.active_model is not None and self.active_model not in MODEL_IDS:
            raise ValueError("executor active model is invalid")
        expected = () if self.active_model is None else (self.active_model,)
        if self.resident_models != expected:
            raise ValueError("executor active and resident model must be identical")
        if self.runtime_state == "ready" and not self.resident_models:
            raise ValueError("ready executor must have one resident model")
        if self.runtime_state in {"unloaded", "loading", "failed"} and (
            self.active_model is not None or self.resident_models
        ):
            raise ValueError(f"{self.runtime_state} executor cannot retain a model")

    @property
    def artifact_ready(self) -> bool:
        return (
            self.verified_artifacts
            and self.runtime_initialized
            and self.device_available
            and self.native_operator_available
            and self.runtime_state != "failed"
        )

    @property
    def inference_warm(self) -> bool:
        return self.runtime_state == "ready" and bool(self.resident_models)

    @property
    def warm_model(self) -> str | None:
        return self.active_model if self.inference_warm else None


class GpuExecutionGateway(Protocol):
    def submit(self, request: PredictionRequest) -> PredictionHandle: ...

    def status(self, prediction_id: PredictionId) -> PredictionStatus: ...

    def result(self, prediction_id: PredictionId) -> PredictionResult: ...

    def wait(
        self,
        prediction_id: PredictionId,
        *,
        timeout_seconds: float,
    ) -> PredictionResult | PredictionHandle: ...
