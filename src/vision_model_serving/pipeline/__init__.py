"""Deep end-to-end prediction pipeline interface."""

from .contracts import (
    ArtifactProvenance,
    AttentionInspection,
    CaseInput,
    ClassificationPrediction,
    ClassifierAdapterTimings,
    ClassifierInputSummary,
    ClassifierStageTimings,
    Detection,
    DetectorAdapterTimings,
    DetectorInputSummary,
    DetectorPrediction,
    DetectorStageTimings,
    GeometrySummary,
    InputSummary,
    MemorySummary,
    NumericTensor,
    PredictionMode,
    PredictionProvenance,
    PredictionResult,
    PredictionTimings,
    PredictionWarning,
    RuntimeExecutionSummary,
    TokenizerProvenance,
)
from .pipeline import (
    CLASSIFIER_MODEL_ID,
    DETECTOR_MODEL_ID,
    PredictionContractError,
    PredictionInputError,
    PredictionPipeline,
    PredictionPipelineError,
)
from .serialization import prediction_from_dict, prediction_to_dict


def __getattr__(name: str) -> object:
    if name in {"LocalCudaPipelineConfig", "build_local_cuda_pipeline"}:
        from .factory import LocalCudaPipelineConfig, build_local_cuda_pipeline

        return {
            "LocalCudaPipelineConfig": LocalCudaPipelineConfig,
            "build_local_cuda_pipeline": build_local_cuda_pipeline,
        }[name]
    raise AttributeError(name)

__all__ = [
    "ArtifactProvenance",
    "AttentionInspection",
    "CLASSIFIER_MODEL_ID",
    "CaseInput",
    "ClassificationPrediction",
    "ClassifierAdapterTimings",
    "ClassifierInputSummary",
    "ClassifierStageTimings",
    "DETECTOR_MODEL_ID",
    "Detection",
    "DetectorAdapterTimings",
    "DetectorInputSummary",
    "DetectorPrediction",
    "DetectorStageTimings",
    "GeometrySummary",
    "InputSummary",
    "MemorySummary",
    "LocalCudaPipelineConfig",
    "NumericTensor",
    "PredictionContractError",
    "PredictionInputError",
    "PredictionMode",
    "PredictionPipeline",
    "PredictionPipelineError",
    "PredictionProvenance",
    "PredictionResult",
    "PredictionTimings",
    "PredictionWarning",
    "RuntimeExecutionSummary",
    "TokenizerProvenance",
    "build_local_cuda_pipeline",
    "prediction_to_dict",
    "prediction_from_dict",
]
