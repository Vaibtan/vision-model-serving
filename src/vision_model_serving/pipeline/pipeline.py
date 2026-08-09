"""End-to-end orchestration over the decoder and single-residency runtime."""

from __future__ import annotations

from time import perf_counter
from typing import Protocol

import numpy as np

from vision_model_serving.classifier import MmbcdResult
from vision_model_serving.detector import DetectorProposal, DetectorResult
from vision_model_serving.dicom import CanonicalMammogram
from vision_model_serving.model_ids import CLASSIFIER_MODEL_ID, DETECTOR_MODEL_ID
from vision_model_serving.residency import ModelOutputs

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


class PredictionPipelineError(RuntimeError):
    code = "prediction_pipeline_failed"

    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(f"{self.code}: {detail}")


class PredictionInputError(PredictionPipelineError):
    code = "prediction_input_invalid"


class PredictionContractError(PredictionPipelineError):
    code = "prediction_adapter_contract_invalid"


class _Decoder(Protocol):
    def decode(self, stream: object) -> CanonicalMammogram: ...


class _Runtime(Protocol):
    def execute(self, model_id: str, inputs: object) -> ModelOutputs: ...

    def status(self) -> object: ...


class PredictionPipeline:
    """Hide decoding, adapter orchestration, model switching, and result shaping."""

    def __init__(
        self,
        *,
        decoder: _Decoder,
        runtime: _Runtime,
    ):
        if not callable(getattr(decoder, "decode", None)):
            raise TypeError("decoder must implement decode(stream)")
        if not callable(getattr(runtime, "execute", None)):
            raise TypeError("runtime must implement execute(model_id, inputs)")
        self._decoder = decoder
        self._runtime = runtime

    def infer(self, case: CaseInput, mode: PredictionMode) -> PredictionResult:
        if not isinstance(case, CaseInput):
            raise PredictionInputError("case must be a CaseInput")
        if not isinstance(mode, PredictionMode):
            raise PredictionInputError("mode must be a PredictionMode")
        history = case.clinical_history or ""
        if mode is PredictionMode.FULL and not history.strip():
            raise PredictionInputError(
                "clinical history is required for full-pipeline mode"
            )

        total_started = perf_counter()
        decode_started = perf_counter()
        canonical = self._decoder.decode(case.dicom_stream)
        decode_ms = (perf_counter() - decode_started) * 1000.0
        if not isinstance(canonical, CanonicalMammogram):
            raise PredictionContractError("decoder returned an unexpected result type")

        detector_execution = self._runtime.execute(DETECTOR_MODEL_ID, canonical)
        detector_result = _detector_result(detector_execution)
        detector = _detector_prediction(detector_result)
        detector_artifact = _verified_artifact(
            detector_execution,
            detector_result.artifact,
        )
        classification = None
        classifier_artifact = None
        tokenizer = None
        precision = None
        offline_assets_only = None
        strict_checkpoint_load = None
        classifier_stage = None
        warnings = tuple(
            PredictionWarning("dicom", warning.code, warning.detail)
            for warning in canonical.warnings
        ) + tuple(
            PredictionWarning("detector", warning.code, warning.detail)
            for warning in detector_result.proposals.warnings
        )
        if mode is PredictionMode.FULL:
            classifier_execution = self._runtime.execute(
                CLASSIFIER_MODEL_ID,
                (canonical, detector_result.proposals.classifier_rois, history),
            )
            classifier_result = _classifier_result(classifier_execution)
            classification = _classification_prediction(classifier_result)
            classifier_artifact = _verified_artifact(
                classifier_execution,
                classifier_result.artifact,
            )
            classifier_provenance = classifier_result.provenance
            tokenizer = TokenizerProvenance(
                id=classifier_provenance.tokenizer.id,
                revision=classifier_provenance.tokenizer.revision,
                file_sha256=classifier_provenance.tokenizer.file_sha256,
            )
            precision = classifier_provenance.precision
            offline_assets_only = classifier_provenance.offline_assets_only
            strict_checkpoint_load = classifier_provenance.strict_checkpoint_load
            classifier_stage = _classifier_stage_timings(
                classifier_execution,
                classifier_result,
            )
            warnings += tuple(
                PredictionWarning(
                    "classifier",
                    warning,
                    warning.replace("_", " "),
                )
                for warning in classifier_result.warnings
            )
        total_ms = (perf_counter() - total_started) * 1000.0
        return PredictionResult(
            mode=mode,
            input=_input_summary(canonical),
            geometry=_geometry_summary(canonical),
            detector=detector,
            classification=classification,
            provenance=PredictionProvenance(
                detector=detector_artifact,
                classifier=classifier_artifact,
                tokenizer=tokenizer,
                precision=precision,
                offline_assets_only=offline_assets_only,
                strict_checkpoint_load=strict_checkpoint_load,
            ),
            timings=PredictionTimings(
                decode_ms=decode_ms,
                detector=_detector_stage_timings(
                    detector_execution,
                    detector_result,
                ),
                classifier=classifier_stage,
                total_ms=total_ms,
            ),
            warnings=warnings,
        )

    def close(self) -> None:
        close = getattr(self._runtime, "close", None)
        if callable(close):
            close()

    def status(self) -> object:
        return self._runtime.status()


def _detector_result(execution: ModelOutputs) -> DetectorResult:
    if execution.model_id != DETECTOR_MODEL_ID or not isinstance(
        execution.value,
        DetectorResult,
    ):
        raise PredictionContractError("runtime returned an unexpected detector result")
    return execution.value


def _classifier_result(execution: ModelOutputs) -> MmbcdResult:
    if execution.model_id != CLASSIFIER_MODEL_ID or not isinstance(
        execution.value,
        MmbcdResult,
    ):
        raise PredictionContractError(
            "runtime returned an unexpected classifier result"
        )
    return execution.value


def _verified_artifact(
    execution: ModelOutputs,
    adapter_artifact: object,
) -> ArtifactProvenance:
    runtime_identity = execution.artifact
    fields = (
        str(getattr(adapter_artifact, "id", "")),
        str(getattr(adapter_artifact, "sha256", "")),
        str(getattr(adapter_artifact, "repository_revision", "")),
    )
    if fields != (
        runtime_identity.id,
        runtime_identity.sha256,
        runtime_identity.repository_revision,
    ):
        raise PredictionContractError("runtime and adapter artifact identities differ")
    return ArtifactProvenance(*fields)


def _numeric_tensor(values: np.ndarray) -> NumericTensor:
    array = np.asarray(values)
    return NumericTensor(
        shape=tuple(int(size) for size in array.shape),
        values=tuple(float(value) for value in array.reshape(-1)),
    )


def _detection(proposal: DetectorProposal) -> Detection:
    return Detection(
        rank=proposal.rank,
        query_index=proposal.query_index,
        class_index=proposal.class_index,
        raw_logit=proposal.raw_logit,
        score=proposal.score,
        normalized_cxcywh=proposal.normalized_cxcywh,
        normalized_xyxy=proposal.normalized_xyxy,
        canonical_xyxy=proposal.canonical_xyxy,
        original_xyxy=proposal.original_xyxy,
        padded=proposal.padded,
        duplicate_of_rank=proposal.duplicate_of_rank,
    )


def _detector_prediction(result: DetectorResult) -> DetectorPrediction:
    detector_input = result.detector_input
    proposals = result.proposals
    return DetectorPrediction(
        input=DetectorInputSummary(
            source_size=detector_input.source_size,
            resized_size=detector_input.resized_size,
            resize_short_edge=detector_input.resize_short_edge,
            resize_max_edge=detector_input.resize_max_edge,
            normalization_mean=detector_input.normalization_mean,
            normalization_std=detector_input.normalization_std,
        ),
        raw_logits=_numeric_tensor(proposals.raw_logits),
        raw_scores=_numeric_tensor(proposals.raw_scores),
        raw_boxes_cxcywh=_numeric_tensor(proposals.raw_boxes_cxcywh),
        prediction_sha256=proposals.prediction_sha256,
        top_candidates=tuple(
            _detection(proposal) for proposal in proposals.top_candidates
        ),
        post_nms=tuple(_detection(proposal) for proposal in proposals.post_nms),
        classifier_rois=tuple(
            _detection(proposal) for proposal in proposals.classifier_rois
        ),
    )


def _classification_prediction(result: MmbcdResult) -> ClassificationPrediction:
    summary = result.input
    return ClassificationPrediction(
        input=ClassifierInputSummary(
            label_information_used=summary.label_information_used,
            token_count=summary.token_count,
            crop_tensor_sha256=summary.crop_tensor_sha256,
            input_ids_sha256=summary.input_ids_sha256,
            attention_mask_sha256=summary.attention_mask_sha256,
        ),
        class_indices=result.class_indices,
        logits=result.logits,
        probabilities=result.probabilities,
        predicted_class_index=result.predicted_class_index,
        attention=AttentionInspection(
            kind=result.attention.kind,
            roi_weights=result.attention.roi_weights,
        ),
        prediction_sha256=result.prediction_sha256,
    )


def _input_summary(canonical: CanonicalMammogram) -> InputSummary:
    metadata = canonical.metadata
    return InputSummary(
        source_sha256=canonical.source_sha256,
        rows=metadata.rows,
        columns=metadata.columns,
        frames=metadata.frames,
        modality=metadata.modality,
        photometric_interpretation=metadata.photometric_interpretation,
        presentation_lut_shape=metadata.presentation_lut_shape,
        transfer_syntax_uid=metadata.transfer_syntax_uid,
        transfer_syntax_name=metadata.transfer_syntax_name,
        compressed=metadata.compressed,
        sop_class_uid=metadata.sop_class_uid,
        bits_allocated=metadata.bits_allocated,
        bits_stored=metadata.bits_stored,
        pixel_representation=metadata.pixel_representation,
        modality_transform_applied=metadata.modality_transform_applied,
        voi_transform_applied=metadata.voi_transform_applied,
        voi_index=metadata.voi_index,
    )


def _geometry_summary(canonical: CanonicalMammogram) -> GeometrySummary:
    geometry = canonical.geometry
    return GeometrySummary(
        original_width=geometry.original_width,
        original_height=geometry.original_height,
        crop_box=geometry.crop_box,
        canonical_width=geometry.canonical_width,
        canonical_height=geometry.canonical_height,
        scale_x=geometry.scale_x,
        scale_y=geometry.scale_y,
    )


def _runtime_summary(execution: ModelOutputs) -> RuntimeExecutionSummary:
    return RuntimeExecutionSummary(
        reused=execution.reused,
        load_ms=execution.timings.load_ms,
        inference_ms=execution.timings.inference_ms,
        switch_ms=execution.timings.switch_ms,
    )


def _memory_summary(execution: ModelOutputs) -> MemorySummary:
    memory = execution.memory
    return MemorySummary(
        allocated_bytes=memory.allocated_bytes,
        reserved_bytes=memory.reserved_bytes,
        peak_allocated_bytes=memory.peak_allocated_bytes,
        peak_reserved_bytes=memory.peak_reserved_bytes,
    )


def _detector_stage_timings(
    execution: ModelOutputs,
    result: DetectorResult,
) -> DetectorStageTimings:
    timings = result.timings
    return DetectorStageTimings(
        runtime=_runtime_summary(execution),
        adapter=DetectorAdapterTimings(
            load_ms=timings.load_ms,
            preprocess_ms=timings.preprocess_ms,
            inference_ms=timings.inference_ms,
            postprocess_ms=timings.postprocess_ms,
        ),
        memory=_memory_summary(execution),
    )


def _classifier_stage_timings(
    execution: ModelOutputs,
    result: MmbcdResult,
) -> ClassifierStageTimings:
    timings = result.timings
    return ClassifierStageTimings(
        runtime=_runtime_summary(execution),
        adapter=ClassifierAdapterTimings(
            load_ms=timings.load_ms,
            crop_preprocess_ms=timings.crop_preprocess_ms,
            tokenization_ms=timings.tokenization_ms,
            inference_ms=timings.inference_ms,
            result_ms=timings.result_ms,
        ),
        memory=_memory_summary(execution),
    )
