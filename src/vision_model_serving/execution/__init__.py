"""Bounded GPU prediction execution gateway."""

from .contracts import (
    GatewayUnavailable,
    GatewayObservations,
    GpuExecutionGateway,
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
from .memory import InMemoryGpuExecutionGateway
from .gateway import StoredGpuExecutionGateway
from .state import (
    Admission,
    InMemoryPredictionStateRepository,
    JobStateRecord,
    PredictionStateRepository,
)
from .worker import PredictionJobWorker
from .celery import (
    GPU_TASK_NAME,
    CeleryTaskDispatcher,
    configure_gpu_worker,
    register_prediction_task,
)
from .redis_state import RedisPredictionStateRepository
from .production import (
    CeleryRedisGpuExecutionGateway,
    GpuExecutionConfig,
    build_prediction_worker_factory,
)
from .storage import (
    EphemeralJobStore,
    JobPayloadNotFound,
    JobResultConflict,
    JobResultNotFound,
    StoredJobPayload,
)

__all__ = [
    "CeleryTaskDispatcher",
    "CeleryRedisGpuExecutionGateway",
    "IdempotencyConflict",
    "GatewayUnavailable",
    "GatewayObservations",
    "GPU_TASK_NAME",
    "GpuExecutionConfig",
    "GpuExecutionGateway",
    "Admission",
    "EphemeralJobStore",
    "InMemoryGpuExecutionGateway",
    "InMemoryPredictionStateRepository",
    "JobPayloadNotFound",
    "JobResultConflict",
    "JobResultNotFound",
    "JobStateRecord",
    "PredictionHandle",
    "PredictionFailed",
    "PredictionFailure",
    "PredictionId",
    "PredictionJobState",
    "PredictionJobWorker",
    "PredictionNotFound",
    "PredictionRequest",
    "PredictionStatus",
    "PredictionStateRepository",
    "QueueSaturated",
    "RedisPredictionStateRepository",
    "ResultExpired",
    "ResultNotReady",
    "StoredJobPayload",
    "StoredGpuExecutionGateway",
    "configure_gpu_worker",
    "build_prediction_worker_factory",
    "register_prediction_task",
]
