"""Public contracts for bounded prediction execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import NewType, Protocol

from vision_model_serving.pipeline.contracts import (
    CaseInput,
    PredictionMode,
    PredictionResult,
)

PredictionId = NewType("PredictionId", str)


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
    queue_wait_ms_total: float


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
