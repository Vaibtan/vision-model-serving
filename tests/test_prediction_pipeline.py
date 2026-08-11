from __future__ import annotations

from contextlib import contextmanager
import gc
from io import BytesIO
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import weakref

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import vision_model_serving.pipeline as pipeline_api  # noqa: E402
from vision_model_serving.detector import (  # noqa: E402
    DetectorArtifactIdentity,
    DetectorInput,
    DetectorPostprocessor,
    DetectorResult,
    DetectorTimings,
)
from vision_model_serving.classifier import (  # noqa: E402
    AttentionInspection as AdapterAttentionInspection,
    ClassifierArtifactIdentity,
    ClassifierInputSummary as AdapterClassifierInputSummary,
    ClassifierProvenance,
    ClassifierTimings,
    LocalTokenizerIdentity,
    MmbcdResult,
)
from vision_model_serving.dicom import (  # noqa: E402
    CanonicalMammogram,
    DicomMetadata,
    DicomWarning,
    GeometryLedger,
)
from vision_model_serving.model_ids import (  # noqa: E402
    CLASSIFIER_MODEL_ID,
    DETECTOR_MODEL_ID,
)
from vision_model_serving.pipeline import (  # noqa: E402
    CaseInput,
    PredictionMode,
    PredictionInputError,
    PredictionPipeline,
    prediction_from_dict,
    prediction_to_dict,
)
from vision_model_serving.residency import (  # noqa: E402
    ExecutionTimings,
    MemorySnapshot,
    ModelBinding,
    ModelOutputs,
    RuntimeArtifactIdentity,
    RuntimeInferenceError,
    SingleResidencyRuntime,
)


DETECTOR_SHA256 = "a" * 64
DETECTOR_REVISION = "b" * 40
CLASSIFIER_SHA256 = "d" * 64
CLASSIFIER_REVISION = "e" * 40
TOKENIZER_REVISION = "f" * 40


def canonical_mammogram() -> CanonicalMammogram:
    pixels = np.zeros((1024, 1024), dtype=np.uint8)
    pixels.setflags(write=False)
    return CanonicalMammogram(
        pixels=pixels,
        geometry=GeometryLedger(
            original_width=400,
            original_height=200,
            crop_box=(40, 20, 360, 180),
        ),
        metadata=DicomMetadata(
            rows=200,
            columns=400,
            frames=1,
            modality="MG",
            photometric_interpretation="MONOCHROME2",
            presentation_lut_shape="IDENTITY",
            transfer_syntax_uid="1.2.840.10008.1.2.1",
            transfer_syntax_name="Explicit VR Little Endian",
            compressed=False,
            sop_class_uid="1.2.3",
            bits_allocated=16,
            bits_stored=12,
            pixel_representation=0,
            modality_transform_applied=True,
            voi_transform_applied=True,
            voi_index=0,
        ),
        warnings=(DicomWarning("pixel_padding_excluded", "padding excluded"),),
        source_sha256="c" * 64,
    )


def detector_result(mammogram: CanonicalMammogram) -> DetectorResult:
    logits = np.arange(8, 0, -1, dtype=np.float32).reshape(1, 8, 1)
    boxes = np.array(
        [
            [0.10, 0.10, 0.08, 0.08],
            [0.30, 0.10, 0.08, 0.08],
            [0.50, 0.10, 0.08, 0.08],
            [0.70, 0.10, 0.08, 0.08],
            [0.10, 0.50, 0.08, 0.08],
            [0.30, 0.50, 0.08, 0.08],
            [0.50, 0.50, 0.08, 0.08],
            [0.70, 0.50, 0.08, 0.08],
        ],
        dtype=np.float32,
    ).reshape(1, 8, 4)
    detector_tensor = np.zeros((3, 800, 800), dtype=np.float32)
    detector_tensor.setflags(write=False)
    return DetectorResult(
        artifact=DetectorArtifactIdentity(
            id=DETECTOR_MODEL_ID,
            sha256=DETECTOR_SHA256,
            repository_revision=DETECTOR_REVISION,
        ),
        detector_input=DetectorInput(
            tensor=detector_tensor,
            source_size=(1024, 1024),
            resized_size=(800, 800),
            resize_short_edge=800,
            resize_max_edge=1333,
            normalization_mean=(0.485, 0.456, 0.406),
            normalization_std=(0.229, 0.224, 0.225),
        ),
        proposals=DetectorPostprocessor(num_select=8).process(
            logits,
            boxes,
            mammogram.geometry,
        ),
        timings=DetectorTimings(
            load_ms=11.0,
            preprocess_ms=12.0,
            inference_ms=13.0,
            postprocess_ms=14.0,
        ),
    )


class DecoderFake:
    def __init__(self, result: CanonicalMammogram):
        self._result = result

    def decode(self, stream: object) -> CanonicalMammogram:
        return self._result


class RuntimeFake:
    def __init__(self, detector: DetectorResult):
        self._detector = detector

    def execute(self, model_id: str, inputs: object) -> ModelOutputs:
        if model_id != DETECTOR_MODEL_ID:
            raise AssertionError("detection mode requested an unexpected model")
        return ModelOutputs(
            model_id=model_id,
            value=self._detector,
            artifact=RuntimeArtifactIdentity(
                id=model_id,
                sha256=DETECTOR_SHA256,
                repository_revision=DETECTOR_REVISION,
            ),
            reused=False,
            timings=ExecutionTimings(
                load_ms=15.0,
                inference_ms=16.0,
                switch_ms=0.0,
            ),
            memory=MemorySnapshot(
                allocated_bytes=100,
                reserved_bytes=200,
                peak_allocated_bytes=300,
                peak_reserved_bytes=400,
            ),
        )


def classifier_result() -> MmbcdResult:
    embeddings = np.zeros((1, 768), dtype=np.float32)
    embeddings.setflags(write=False)
    tokenizer_files = tuple(
        (name, str(index) * 64)
        for index, name in enumerate(
            (
                "config.json",
                "merges.txt",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
            ),
            start=1,
        )
    )
    return MmbcdResult(
        input=AdapterClassifierInputSummary(
            prompt="Indication: prior surgery",
            label_information_used=False,
            token_count=5,
            clinical_text_truncated=False,
            crop_tensor_sha256="1" * 64,
            input_ids_sha256="2" * 64,
            attention_mask_sha256="3" * 64,
        ),
        class_indices=(0, 1),
        logits=(-1.5, 2.5),
        probabilities=(0.01798621, 0.98201379),
        predicted_class_index=1,
        fused_embeddings=embeddings,
        attention=AdapterAttentionInspection(
            kind="model_inspection_not_causal_or_clinical_evidence",
            roi_weights=(0.125,) * 8,
        ),
        prediction_sha256="4" * 64,
        provenance=ClassifierProvenance(
            artifact=ClassifierArtifactIdentity(
                id=CLASSIFIER_MODEL_ID,
                sha256=CLASSIFIER_SHA256,
                repository_revision=CLASSIFIER_REVISION,
            ),
            tokenizer=LocalTokenizerIdentity(
                id="roberta-base",
                revision=TOKENIZER_REVISION,
                file_sha256=tokenizer_files,
            ),
            precision="float32",
            offline_assets_only=True,
            strict_checkpoint_load=True,
        ),
        timings=ClassifierTimings(
            load_ms=21.0,
            crop_preprocess_ms=22.0,
            tokenization_ms=23.0,
            inference_ms=24.0,
            result_ms=25.0,
        ),
        warnings=(
            "class_semantics_and_decision_threshold_unverified",
            "attention_is_inspection_not_causal_or_clinical_evidence",
        ),
    )


class FullRuntimeFake(RuntimeFake):
    def __init__(
        self,
        detector: DetectorResult,
        classifier: MmbcdResult,
        mammogram: CanonicalMammogram,
    ):
        super().__init__(detector)
        self._classifier = classifier
        self._mammogram = mammogram

    def execute(self, model_id: str, inputs: object) -> ModelOutputs:
        if model_id == DETECTOR_MODEL_ID:
            return super().execute(model_id, inputs)
        if model_id != CLASSIFIER_MODEL_ID:
            raise AssertionError("full mode requested an unexpected model")
        mammogram, rois, history = inputs
        if mammogram is not self._mammogram or len(rois) != 8:
            raise AssertionError("classifier did not receive detector-owned inputs")
        if history != "  prior   surgery  ":
            raise AssertionError("classifier history changed before adapter normalization")
        return ModelOutputs(
            model_id=model_id,
            value=self._classifier,
            artifact=RuntimeArtifactIdentity(
                id=model_id,
                sha256=CLASSIFIER_SHA256,
                repository_revision=CLASSIFIER_REVISION,
            ),
            reused=False,
            timings=ExecutionTimings(
                load_ms=26.0,
                inference_ms=27.0,
                switch_ms=28.0,
            ),
            memory=MemorySnapshot(
                allocated_bytes=500,
                reserved_bytes=600,
                peak_allocated_bytes=700,
                peak_reserved_bytes=800,
            ),
        )


class AcceleratorFake:
    def prepare(self) -> None:
        pass

    def synchronize(self) -> None:
        pass

    def reset_peak_memory_stats(self) -> None:
        pass

    def empty_cache(self) -> None:
        pass

    @contextmanager
    def inference_mode(self):
        yield

    def memory_snapshot(self) -> MemorySnapshot:
        return MemorySnapshot(10, 20, 30, 40)


class DetectorResidentFake:
    artifact = SimpleNamespace(
        id=DETECTOR_MODEL_ID,
        sha256=DETECTOR_SHA256,
        repository_revision=DETECTOR_REVISION,
    )

    def warmup(self, _inputs: object) -> None:
        pass

    def execute(self, inputs: object) -> object:
        return detector_result(inputs)


class ClassifierResidentFake:
    artifact = SimpleNamespace(
        id=CLASSIFIER_MODEL_ID,
        sha256=CLASSIFIER_SHA256,
        repository_revision=CLASSIFIER_REVISION,
    )

    def warmup(self, _inputs: object) -> None:
        pass

    def execute(self, inputs: object) -> object:
        return classifier_result()


def single_residency_runtime() -> SingleResidencyRuntime:
    return SingleResidencyRuntime(
        bindings=(
            ModelBinding(
                model_id=DETECTOR_MODEL_ID,
                load=DetectorResidentFake,
                failure_token=lambda: "detector-v1",
            ),
            ModelBinding(
                model_id=CLASSIFIER_MODEL_ID,
                load=ClassifierResidentFake,
                failure_token=lambda: "classifier-v1",
            ),
        ),
        accelerator=AcceleratorFake(),
    )


class EphemeralDecoderFake:
    def __init__(self):
        self.pixel_reference: weakref.ReferenceType[np.ndarray] | None = None

    def decode(self, stream: object) -> CanonicalMammogram:
        result = canonical_mammogram()
        self.pixel_reference = weakref.ref(result.pixels)
        return result


class StatelessDetectorRuntimeFake:
    def execute(self, model_id: str, inputs: object) -> ModelOutputs:
        result = detector_result(inputs)
        return ModelOutputs(
            model_id=model_id,
            value=result,
            artifact=RuntimeArtifactIdentity(
                id=model_id,
                sha256=DETECTOR_SHA256,
                repository_revision=DETECTOR_REVISION,
            ),
            reused=False,
            timings=ExecutionTimings(0.0, 0.0, 0.0),
            memory=MemorySnapshot(0, 0, 0, 0),
        )


class FailingRuntimeFake:
    def execute(self, model_id: str, inputs: object) -> ModelOutputs:
        raise RuntimeInferenceError("sanitized boundary failure")


class PredictionPipelineDetectionTests(unittest.TestCase):
    def test_detection_mode_returns_the_complete_typed_result(self) -> None:
        mammogram = canonical_mammogram()
        pipeline = PredictionPipeline(
            decoder=DecoderFake(mammogram),
            runtime=RuntimeFake(detector_result(mammogram)),
        )

        result = pipeline.infer(
            CaseInput(dicom_stream=BytesIO(b"not decoded by this boundary fake")),
            PredictionMode.DETECTION,
        )

        self.assertIs(result.mode, PredictionMode.DETECTION)
        self.assertEqual(result.input.source_sha256, "c" * 64)
        self.assertEqual((result.input.rows, result.input.columns), (200, 400))
        self.assertEqual(result.geometry.crop_box, (40, 20, 360, 180))
        self.assertEqual(result.detector.raw_logits.shape, (1, 8, 1))
        self.assertEqual(result.detector.raw_logits.values, tuple(range(8, 0, -1)))
        self.assertEqual(result.detector.raw_boxes_cxcywh.shape, (1, 8, 4))
        self.assertEqual(len(result.detector.top_candidates), 8)
        self.assertEqual(len(result.detector.post_nms), 8)
        self.assertEqual(len(result.detector.classifier_rois), 8)
        self.assertEqual(
            result.detector.classifier_rois[0].canonical_xyxy,
            result.detector.top_candidates[0].canonical_xyxy,
        )
        self.assertNotEqual(
            result.detector.classifier_rois[0].canonical_xyxy,
            result.detector.classifier_rois[0].original_xyxy,
        )
        self.assertIsNone(result.classification)
        self.assertEqual(result.provenance.detector.id, DETECTOR_MODEL_ID)
        self.assertEqual(result.provenance.detector.sha256, DETECTOR_SHA256)
        self.assertIsNone(result.provenance.classifier)
        self.assertIsNone(result.provenance.tokenizer)
        self.assertEqual(result.timings.detector.runtime.inference_ms, 16.0)
        self.assertEqual(result.timings.detector.adapter.postprocess_ms, 14.0)
        self.assertGreaterEqual(result.timings.decode_ms, 0.0)
        self.assertGreaterEqual(result.timings.total_ms, 0.0)
        self.assertEqual(result.timings.detector.memory.peak_reserved_bytes, 400)
        self.assertEqual(
            {(warning.stage, warning.code) for warning in result.warnings},
            {("dicom", "pixel_padding_excluded")},
        )
        self.assertEqual(
            result.disclaimer,
            "Research use only; not a medical diagnosis.",
        )
        self.assertFalse(hasattr(result.classification, "label"))
        self.assertFalse(hasattr(result.classification, "threshold"))


class PredictionPipelineFullTests(unittest.TestCase):
    def test_full_mode_returns_numbers_and_provenance_without_prompt(self) -> None:
        mammogram = canonical_mammogram()
        pipeline = PredictionPipeline(
            decoder=DecoderFake(mammogram),
            runtime=FullRuntimeFake(
                detector_result(mammogram),
                classifier_result(),
                mammogram,
            ),
        )

        result = pipeline.infer(
            CaseInput(
                dicom_stream=BytesIO(b"not decoded by this boundary fake"),
                clinical_history="  prior   surgery  ",
            ),
            PredictionMode.FULL,
        )

        self.assertIs(result.mode, PredictionMode.FULL)
        self.assertIsNotNone(result.classification)
        classification = result.classification
        self.assertEqual(classification.class_indices, (0, 1))
        self.assertEqual(classification.logits, (-1.5, 2.5))
        self.assertEqual(classification.probabilities, (0.01798621, 0.98201379))
        self.assertEqual(classification.predicted_class_index, 1)
        self.assertEqual(classification.attention.roi_weights, (0.125,) * 8)
        self.assertEqual(classification.prediction_sha256, "4" * 64)
        self.assertFalse(hasattr(classification.input, "prompt"))
        self.assertEqual(classification.input.token_count, 5)
        self.assertFalse(classification.input.label_information_used)
        self.assertEqual(result.provenance.classifier.id, CLASSIFIER_MODEL_ID)
        self.assertEqual(result.provenance.classifier.sha256, CLASSIFIER_SHA256)
        self.assertEqual(result.provenance.tokenizer.id, "roberta-base")
        self.assertEqual(result.provenance.tokenizer.revision, TOKENIZER_REVISION)
        self.assertEqual(result.provenance.precision, "float32")
        self.assertTrue(result.provenance.offline_assets_only)
        self.assertTrue(result.provenance.strict_checkpoint_load)
        self.assertEqual(result.timings.classifier.runtime.switch_ms, 28.0)
        self.assertEqual(result.timings.classifier.adapter.tokenization_ms, 23.0)
        self.assertEqual(result.timings.classifier.memory.peak_reserved_bytes, 800)
        self.assertIn(
            ("classifier", "class_semantics_and_decision_threshold_unverified"),
            {(warning.stage, warning.code) for warning in result.warnings},
        )
        self.assertFalse(hasattr(classification, "label"))
        self.assertFalse(hasattr(classification, "threshold"))

    def test_result_converts_to_a_json_safe_payload_without_clinical_text(self) -> None:
        mammogram = canonical_mammogram()
        pipeline = PredictionPipeline(
            decoder=DecoderFake(mammogram),
            runtime=FullRuntimeFake(
                detector_result(mammogram),
                classifier_result(),
                mammogram,
            ),
        )
        result = pipeline.infer(
            CaseInput(
                dicom_stream=BytesIO(b"not decoded by this boundary fake"),
                clinical_history="  prior   surgery  ",
            ),
            PredictionMode.FULL,
        )

        payload = prediction_to_dict(result)
        encoded = json.dumps(payload, allow_nan=False, sort_keys=True)

        self.assertEqual(payload["mode"], "full")
        self.assertEqual(payload["detector"]["raw_logits"]["shape"], [1, 8, 1])
        self.assertEqual(payload["classification"]["logits"], [-1.5, 2.5])
        self.assertNotIn("prior surgery", encoded)
        self.assertNotIn("Indication", encoded)
        self.assertEqual(prediction_from_dict(payload), result)

    def test_full_mode_drives_the_single_residency_switch(self) -> None:
        mammogram = canonical_mammogram()
        runtime = single_residency_runtime()
        pipeline = PredictionPipeline(
            decoder=DecoderFake(mammogram),
            runtime=runtime,
        )

        result = pipeline.infer(
            CaseInput(BytesIO(b"fake"), clinical_history="history"),
            PredictionMode.FULL,
        )
        status = runtime.status()

        self.assertIsNotNone(result.classification)
        self.assertEqual(status.active_model, CLASSIFIER_MODEL_ID)
        self.assertEqual(status.metrics.load_count, 2)
        self.assertEqual(status.metrics.switch_count, 1)
        self.assertEqual(status.metrics.unload_count, 1)
        self.assertEqual(status.metrics.failure_count, 0)


class PredictionPipelineFailureAndCleanupTests(unittest.TestCase):
    def test_public_package_does_not_expose_executor_composition(self) -> None:
        self.assertNotIn("ExecutorPipelineConfig", pipeline_api.__all__)
        self.assertNotIn("build_executor_pipeline", pipeline_api.__all__)
        self.assertFalse(hasattr(pipeline_api, "LocalCudaPipelineConfig"))
        self.assertFalse(hasattr(pipeline_api, "build_local_cuda_pipeline"))

    def test_full_mode_rejects_blank_history_before_decoding(self) -> None:
        pipeline = PredictionPipeline(
            decoder=DecoderFake(canonical_mammogram()),
            runtime=RuntimeFake(detector_result(canonical_mammogram())),
        )

        with self.assertRaisesRegex(
            PredictionInputError,
            "clinical history is required",
        ):
            pipeline.infer(
                CaseInput(BytesIO(b"fake"), clinical_history="   "),
                PredictionMode.FULL,
            )

    def test_success_releases_pixels_and_keeps_caller_stream_open(self) -> None:
        decoder = EphemeralDecoderFake()
        pipeline = PredictionPipeline(
            decoder=decoder,
            runtime=StatelessDetectorRuntimeFake(),
        )
        stream = BytesIO(b"caller owned")

        result = pipeline.infer(CaseInput(stream), PredictionMode.DETECTION)
        gc.collect()

        self.assertIsNotNone(result.detector)
        self.assertFalse(stream.closed)
        self.assertIsNone(decoder.pixel_reference())

    def test_failure_does_not_retain_decoded_pixels_and_keeps_typed_error(self) -> None:
        decoder = EphemeralDecoderFake()
        pipeline = PredictionPipeline(
            decoder=decoder,
            runtime=FailingRuntimeFake(),
        )

        with self.assertRaises(RuntimeInferenceError):
            pipeline.infer(CaseInput(BytesIO(b"fake")), PredictionMode.DETECTION)
        gc.collect()

        self.assertIsNone(decoder.pixel_reference())


if __name__ == "__main__":
    unittest.main()
