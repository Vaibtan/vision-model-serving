"""Typed, serialization-safe prediction pipeline contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import BinaryIO


RESEARCH_USE_DISCLAIMER = "Research use only; not a medical diagnosis."


class PredictionMode(str, Enum):
    DETECTION = "detection"
    FULL = "full"


@dataclass(frozen=True, slots=True)
class CaseInput:
    """One caller-owned DICOM stream and optional clinical history."""

    dicom_stream: BinaryIO = field(repr=False)
    clinical_history: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not callable(getattr(self.dicom_stream, "read", None)):
            raise TypeError("DICOM input must be a readable binary stream")
        if self.clinical_history is not None and not isinstance(
            self.clinical_history,
            str,
        ):
            raise TypeError("clinical history must be text when supplied")


@dataclass(frozen=True, slots=True)
class NumericTensor:
    """A flat, shape-preserving numeric tensor safe for JSON conversion."""

    shape: tuple[int, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        expected = math.prod(self.shape)
        if (
            not self.shape
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size <= 0
                for size in self.shape
            )
            or expected != len(self.values)
            or any(not math.isfinite(value) for value in self.values)
        ):
            raise ValueError("numeric tensor shape or values are invalid")


@dataclass(frozen=True, slots=True)
class InputSummary:
    source_sha256: str
    rows: int
    columns: int
    frames: int
    modality: str | None
    photometric_interpretation: str
    presentation_lut_shape: str
    transfer_syntax_uid: str
    transfer_syntax_name: str
    compressed: bool
    sop_class_uid: str | None
    bits_allocated: int | None
    bits_stored: int | None
    pixel_representation: int | None
    modality_transform_applied: bool
    voi_transform_applied: bool
    voi_index: int


@dataclass(frozen=True, slots=True)
class GeometrySummary:
    original_width: int
    original_height: int
    crop_box: tuple[int, int, int, int]
    canonical_width: int
    canonical_height: int
    scale_x: float
    scale_y: float


@dataclass(frozen=True, slots=True)
class PredictionWarning:
    stage: str
    code: str
    detail: str


@dataclass(frozen=True, slots=True)
class ArtifactProvenance:
    id: str
    sha256: str
    repository_revision: str


@dataclass(frozen=True, slots=True)
class TokenizerProvenance:
    id: str
    revision: str
    file_sha256: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class PredictionProvenance:
    detector: ArtifactProvenance
    classifier: ArtifactProvenance | None
    tokenizer: TokenizerProvenance | None
    precision: str | None
    offline_assets_only: bool | None
    strict_checkpoint_load: bool | None


@dataclass(frozen=True, slots=True)
class MemorySummary:
    allocated_bytes: int
    reserved_bytes: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int


@dataclass(frozen=True, slots=True)
class RuntimeExecutionSummary:
    reused: bool
    load_ms: float
    inference_ms: float
    switch_ms: float


@dataclass(frozen=True, slots=True)
class DetectorAdapterTimings:
    load_ms: float
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float


@dataclass(frozen=True, slots=True)
class ClassifierAdapterTimings:
    load_ms: float
    crop_preprocess_ms: float
    tokenization_ms: float
    inference_ms: float
    result_ms: float


@dataclass(frozen=True, slots=True)
class DetectorStageTimings:
    runtime: RuntimeExecutionSummary
    adapter: DetectorAdapterTimings
    memory: MemorySummary


@dataclass(frozen=True, slots=True)
class ClassifierStageTimings:
    runtime: RuntimeExecutionSummary
    adapter: ClassifierAdapterTimings
    memory: MemorySummary


@dataclass(frozen=True, slots=True)
class PredictionTimings:
    decode_ms: float
    detector: DetectorStageTimings
    classifier: ClassifierStageTimings | None
    pipeline_ms: float


@dataclass(frozen=True, slots=True)
class DetectorInputSummary:
    source_size: tuple[int, int]
    resized_size: tuple[int, int]
    resize_short_edge: int
    resize_max_edge: int
    normalization_mean: tuple[float, float, float]
    normalization_std: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class Detection:
    rank: int
    query_index: int
    class_index: int
    raw_logit: float
    score: float
    normalized_cxcywh: tuple[float, float, float, float]
    normalized_xyxy: tuple[float, float, float, float]
    canonical_xyxy: tuple[float, float, float, float]
    original_xyxy: tuple[float, float, float, float]
    padded: bool
    duplicate_of_rank: int | None


@dataclass(frozen=True, slots=True)
class DetectorPrediction:
    input: DetectorInputSummary
    raw_logits: NumericTensor
    raw_scores: NumericTensor
    raw_boxes_cxcywh: NumericTensor
    prediction_sha256: str
    top_candidates: tuple[Detection, ...]
    post_nms: tuple[Detection, ...]
    classifier_rois: tuple[Detection, ...]


@dataclass(frozen=True, slots=True)
class ClassifierInputSummary:
    label_information_used: bool
    token_count: int
    clinical_text_truncated: bool
    crop_tensor_sha256: str
    input_ids_sha256: str
    attention_mask_sha256: str


@dataclass(frozen=True, slots=True)
class AttentionInspection:
    kind: str
    roi_weights: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ClassificationPrediction:
    input: ClassifierInputSummary
    class_indices: tuple[int, int]
    logits: tuple[float, float]
    probabilities: tuple[float, float]
    predicted_class_index: int
    attention: AttentionInspection
    prediction_sha256: str


@dataclass(frozen=True, slots=True)
class PredictionResult:
    mode: PredictionMode
    input: InputSummary
    geometry: GeometrySummary
    detector: DetectorPrediction
    classification: ClassificationPrediction | None
    provenance: PredictionProvenance
    timings: PredictionTimings
    warnings: tuple[PredictionWarning, ...]
    disclaimer: str = RESEARCH_USE_DISCLAIMER
    detector_score_threshold: float | None = None
