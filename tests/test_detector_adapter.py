from __future__ import annotations

import hashlib
from contextlib import contextmanager, nullcontext
from io import BytesIO
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from vision_model_serving.detector import (  # noqa: E402
    DetectorInferenceError,
    DetectorLoadError,
    DetectorOutputError,
    DetectorPostprocessor,
    DetectorPreprocessor,
    FocalNetDinoAdapter,
    NoValidProposalsError,
    prediction_sha256,
    strict_nms,
)
from vision_model_serving.artifacts import ArtifactChangedError  # noqa: E402
from vision_model_serving.dicom import GeometryLedger  # noqa: E402


def geometry() -> GeometryLedger:
    return GeometryLedger(
        original_width=400,
        original_height=200,
        crop_box=(40, 20, 360, 180),
        canonical_width=1024,
        canonical_height=1024,
    )


class DetectorPreprocessorTests(unittest.TestCase):
    def test_official_square_transform_is_rgb_chw_and_imagenet_normalized(self) -> None:
        result = DetectorPreprocessor().prepare(
            np.zeros((1024, 1024), dtype=np.uint8)
        )

        self.assertEqual(result.tensor.shape, (3, 800, 800))
        self.assertEqual(result.tensor.dtype, np.float32)
        self.assertFalse(result.tensor.flags.writeable)
        self.assertEqual(result.source_size, (1024, 1024))
        self.assertEqual(result.resized_size, (800, 800))
        np.testing.assert_allclose(
            result.tensor[:, 0, 0],
            np.array(
                [
                    -0.485 / 0.229,
                    -0.456 / 0.224,
                    -0.406 / 0.225,
                ],
                dtype=np.float32,
            ),
            rtol=0,
            atol=1e-6,
        )

    def test_long_edge_cap_preserves_the_official_integer_resize_rule(self) -> None:
        result = DetectorPreprocessor().prepare(
            np.zeros((512, 1024), dtype=np.uint8)
        )

        self.assertEqual(result.resized_size, (1332, 666))
        self.assertEqual(result.tensor.shape, (3, 666, 1332))

    def test_preprocessor_rejects_noncanonical_arrays(self) -> None:
        with self.assertRaises(ValueError):
            DetectorPreprocessor().prepare(np.zeros((4, 4), dtype=np.float32))
        with self.assertRaises(ValueError):
            DetectorPreprocessor().prepare(np.zeros((4, 4, 1), dtype=np.uint8))


class DetectorPostprocessorTests(unittest.TestCase):
    def test_conversion_clipping_and_geometry_mapping_are_explicit(self) -> None:
        logits = np.array([[4.0], [3.0], [2.0]], dtype=np.float32)
        boxes = np.array(
            [
                [0.5, 0.5, 0.4, 0.2],
                [0.0, 0.5, 0.4, 0.2],
                [1.5, 0.5, 0.2, 0.2],
            ],
            dtype=np.float32,
        )

        result = DetectorPostprocessor(
            num_select=3,
            roi_count=2,
        ).process(logits, boxes, geometry())

        first, clipped = result.top_candidates
        np.testing.assert_allclose(
            first.normalized_xyxy,
            (0.3, 0.4, 0.7, 0.6),
            rtol=0,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            first.canonical_xyxy,
            (307.2, 409.6, 716.8, 614.4),
            rtol=0,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            first.original_xyxy,
            (136.0, 84.0, 264.0, 116.0),
            rtol=0,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            clipped.normalized_xyxy,
            (0.0, 0.4, 0.2, 0.6),
            rtol=0,
            atol=1e-7,
        )
        self.assertEqual(
            {warning.code for warning in result.warnings},
            {"degenerate_proposals_rejected"},
        )

    def test_nms_retains_iou_equal_to_threshold_and_suppresses_greater(self) -> None:
        boxes = np.array(
            [
                [0.0, 0.0, 0.5, 0.5],
                [0.4090909, 0.0, 0.9090909, 0.5],
            ],
            dtype=np.float64,
        )
        intersection = 0.5 - boxes[1, 0]
        exact_iou = (intersection * 0.5) / (0.5 - intersection * 0.5)

        self.assertEqual(strict_nms(boxes, exact_iou), (0, 1))
        self.assertEqual(
            strict_nms(boxes, np.nextafter(exact_iou, 0.0)),
            (0,),
        )

    def test_ties_are_stable_and_top_select_is_applied_before_nms(self) -> None:
        logits = np.array([[2.0], [2.0], [1.0], [0.0]], dtype=np.float32)
        boxes = np.array(
            [
                [0.1, 0.1, 0.1, 0.1],
                [0.3, 0.3, 0.1, 0.1],
                [0.5, 0.5, 0.1, 0.1],
                [0.7, 0.7, 0.1, 0.1],
            ],
            dtype=np.float32,
        )

        result = DetectorPostprocessor(
            num_select=3,
            roi_count=2,
        ).process(logits, boxes, geometry())

        self.assertEqual(
            [proposal.query_index for proposal in result.top_candidates],
            [0, 1, 2],
        )
        self.assertEqual(len(result.top_candidates), 3)
        self.assertEqual(len(result.classifier_rois), 2)

    def test_one_to_seven_proposals_are_padded_deterministically(self) -> None:
        all_boxes = np.array(
            [
                [0.1, 0.1, 0.1, 0.1],
                [0.4, 0.1, 0.1, 0.1],
                [0.7, 0.1, 0.1, 0.1],
                [0.1, 0.5, 0.1, 0.1],
                [0.4, 0.5, 0.1, 0.1],
                [0.7, 0.5, 0.1, 0.1],
                [0.9, 0.9, 0.1, 0.1],
            ],
            dtype=np.float32,
        )
        for proposal_count in range(1, 8):
            with self.subTest(proposal_count=proposal_count):
                logits = np.arange(
                    proposal_count,
                    0,
                    -1,
                    dtype=np.float32,
                )[:, None]
                result = DetectorPostprocessor(
                    num_select=300,
                    roi_count=8,
                ).process(logits, all_boxes[:proposal_count], geometry())

                self.assertEqual(
                    [proposal.query_index for proposal in result.classifier_rois],
                    [index % proposal_count for index in range(8)],
                )
                self.assertEqual(
                    [proposal.padded for proposal in result.classifier_rois],
                    [False] * proposal_count + [True] * (8 - proposal_count),
                )
                self.assertIn(
                    "classifier_rois_padded",
                    {warning.code for warning in result.warnings},
                )

    def test_zero_valid_proposals_fail_closed(self) -> None:
        logits = np.array([[3.0], [2.0]], dtype=np.float32)
        boxes = np.array(
            [
                [0.5, 0.5, 0.0, 0.2],
                [2.0, 2.0, 0.1, 0.1],
            ],
            dtype=np.float32,
        )

        with self.assertRaises(NoValidProposalsError) as raised:
            DetectorPostprocessor().process(logits, boxes, geometry())

        self.assertEqual(raised.exception.code, "detector_no_valid_proposals")

    def test_nonfinite_or_malformed_model_outputs_fail_as_typed_errors(self) -> None:
        with self.assertRaises(DetectorOutputError):
            DetectorPostprocessor().process(
                np.array([[float("nan")]], dtype=np.float32),
                np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32),
                geometry(),
            )

    def test_sigmoid_scores_remain_finite_without_clipping_valid_logits(self) -> None:
        result = DetectorPostprocessor(roi_count=1).process(
            np.array([[-100.0], [100.0]], dtype=np.float32),
            np.array(
                [[0.2, 0.2, 0.1, 0.1], [0.8, 0.8, 0.1, 0.1]],
                dtype=np.float32,
            ),
            geometry(),
        )

        self.assertTrue(np.isfinite(result.raw_scores).all())
        self.assertLess(float(result.raw_scores[0, 0, 0]), 1e-42)
        self.assertEqual(float(result.raw_scores[0, 1, 0]), 1.0)
        with self.assertRaises(DetectorOutputError):
            DetectorPostprocessor().process(
                np.zeros((2, 1), dtype=np.float32),
                np.zeros((3, 4), dtype=np.float32),
                geometry(),
            )

    def test_display_filtering_cannot_mutate_the_classifier_roi_contract(self) -> None:
        logits = np.array([[4.0], [0.0]], dtype=np.float32)
        boxes = np.array(
            [[0.2, 0.2, 0.1, 0.1], [0.8, 0.8, 0.1, 0.1]],
            dtype=np.float32,
        )
        result = DetectorPostprocessor(roi_count=8).process(
            logits,
            boxes,
            geometry(),
        )
        before = result.classifier_rois

        displayed = result.display_proposals(min_score=0.9)

        self.assertEqual(len(displayed), 1)
        self.assertIs(result.classifier_rois, before)
        self.assertEqual(len(result.classifier_rois), 8)

    def test_prediction_hash_is_little_endian_float32_and_inputs_are_immutable(self) -> None:
        logits = np.array([[[1.0], [2.0]]], dtype=">f4")
        boxes = np.array(
            [[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]],
            dtype=">f4",
        )
        digest = hashlib.sha256()
        digest.update(logits.astype("<f4").tobytes())
        digest.update(boxes.astype("<f4").tobytes())

        self.assertEqual(prediction_sha256(logits, boxes), digest.hexdigest())

        result = DetectorPostprocessor(roi_count=1).process(
            logits,
            boxes,
            geometry(),
        )
        self.assertFalse(result.raw_logits.flags.writeable)
        self.assertEqual(result.raw_scores.shape, result.raw_logits.shape)
        self.assertFalse(result.raw_scores.flags.writeable)
        np.testing.assert_allclose(
            result.raw_scores,
            1.0 / (1.0 + np.exp(-result.raw_logits)),
            rtol=0,
            atol=1e-7,
        )
        self.assertFalse(result.raw_boxes_cxcywh.flags.writeable)


class TensorStub:
    def __init__(self, values: np.ndarray):
        self.values = np.asarray(values)
        self.to_calls: list[dict[str, object]] = []

    def to(self, **kwargs: object) -> TensorStub:
        self.to_calls.append(kwargs)
        return self

    def detach(self) -> TensorStub:
        return self

    def cpu(self) -> TensorStub:
        return self

    def contiguous(self) -> TensorStub:
        return self

    def numpy(self) -> np.ndarray:
        return self.values


class ModelStub:
    def __init__(
        self,
        *,
        missing_keys: tuple[str, ...] = (),
        unexpected_keys: tuple[str, ...] = (),
    ):
        self.missing_keys = missing_keys
        self.unexpected_keys = unexpected_keys
        self.load_calls: list[tuple[object, bool]] = []
        self.to_calls: list[dict[str, object]] = []
        self.eval_called = False
        self.inputs: list[object] | None = None

    def load_state_dict(self, state_dict: object, *, strict: bool) -> object:
        self.load_calls.append((state_dict, strict))
        return SimpleNamespace(
            missing_keys=list(self.missing_keys),
            unexpected_keys=list(self.unexpected_keys),
        )

    def to(self, **kwargs: object) -> ModelStub:
        self.to_calls.append(kwargs)
        return self

    def eval(self) -> ModelStub:
        self.eval_called = True
        return self

    def __call__(self, inputs: list[object]) -> dict[str, TensorStub]:
        self.inputs = inputs
        logits = np.zeros((1, 900, 1), dtype=np.float32)
        boxes = np.zeros((1, 900, 4), dtype=np.float32)
        logits[0, :2, 0] = (3.0, 2.0)
        boxes[0, :2] = (
            (0.2, 0.2, 0.1, 0.1),
            (0.8, 0.8, 0.1, 0.1),
        )
        return {
            "pred_logits": TensorStub(logits),
            "pred_boxes": TensorStub(boxes),
        }


class MalformedOutputModelStub(ModelStub):
    def __call__(self, inputs: list[object]) -> dict[str, TensorStub]:
        self.inputs = inputs
        return {"pred_logits": TensorStub(np.zeros((1, 1, 1), dtype=np.float32))}


class WrongShapeModelStub(ModelStub):
    def __call__(self, inputs: list[object]) -> dict[str, TensorStub]:
        self.inputs = inputs
        return {
            "pred_logits": TensorStub(np.zeros((1, 2, 1), dtype=np.float32)),
            "pred_boxes": TensorStub(np.zeros((1, 2, 4), dtype=np.float32)),
        }


class ArtifactStub:
    id = "focalnet-dino-detector"
    role = "detector"
    sha256 = "a" * 64
    repository_revision = "b" * 40
    strict_load_verified = True

    @contextmanager
    def open_checkpoint(self):
        yield BytesIO(b"verified checkpoint bytes")


class ChangedArtifactStub(ArtifactStub):
    @contextmanager
    def open_checkpoint(self):
        raise ArtifactChangedError(self.id, "artifact checksum changed")
        yield BytesIO()  # pragma: no cover


def torch_stub(checkpoint: object) -> tuple[ModuleType, list[tuple[str, object]]]:
    module = ModuleType("torch")
    events: list[tuple[str, object]] = []
    module.float32 = "float32"
    module.serialization = SimpleNamespace(
        safe_globals=lambda values: nullcontext(events.append(("safe_globals", values)))
    )

    def load(stream: object, **kwargs: object) -> object:
        events.append(("load", kwargs))
        self_position = stream.tell()
        events.append(("stream_position", self_position))
        return checkpoint

    module.load = load
    module.from_numpy = lambda values: TensorStub(np.asarray(values))

    @contextmanager
    def inference_mode():
        events.append(("inference_mode_enter", True))
        yield
        events.append(("inference_mode_exit", True))

    module.inference_mode = inference_mode
    module.cuda = SimpleNamespace(
        synchronize=lambda device: events.append(("synchronize", device))
    )
    return module, events


class DetectorRuntimeTests(unittest.TestCase):
    def test_artifact_reverification_failure_propagates_before_state_load(self) -> None:
        model = ModelStub()
        fake_torch, _ = torch_stub({"model": {"weight": object()}})
        with patch.dict(sys.modules, {"torch": fake_torch}):
            with self.assertRaises(ArtifactChangedError) as raised:
                FocalNetDinoAdapter.from_artifact(
                    ChangedArtifactStub(),
                    model_factory=lambda: model,
                )

        self.assertEqual(raised.exception.code, "artifact_changed")
        self.assertEqual(model.load_calls, [])

    def test_adapter_strict_loads_and_keeps_torch_objects_private(self) -> None:
        model = ModelStub()
        fake_torch, events = torch_stub({"model": {"weight": object()}})
        with patch.dict(sys.modules, {"torch": fake_torch}):
            adapter = FocalNetDinoAdapter.from_artifact(
                ArtifactStub(),
                model_factory=lambda: model,
                device="cuda:0",
            )
            mammogram = SimpleNamespace(
                pixels=np.zeros((1024, 1024), dtype=np.uint8),
                geometry=geometry(),
            )
            result = adapter.predict(mammogram)

        self.assertEqual(model.load_calls[0][1], True)
        self.assertEqual(model.to_calls, [{"device": "cuda:0", "dtype": "float32"}])
        self.assertTrue(model.eval_called)
        self.assertIsNotNone(model.inputs)
        self.assertFalse(hasattr(adapter, "model"))
        self.assertEqual(result.artifact.id, ArtifactStub.id)
        self.assertEqual(result.artifact.sha256, ArtifactStub.sha256)
        self.assertEqual(len(result.proposals.classifier_rois), 8)
        self.assertGreaterEqual(result.timings.load_ms, 0.0)
        self.assertGreaterEqual(result.timings.preprocess_ms, 0.0)
        self.assertGreaterEqual(result.timings.inference_ms, 0.0)
        self.assertGreaterEqual(result.timings.postprocess_ms, 0.0)
        load_event = next(value for name, value in events if name == "load")
        self.assertEqual(
            load_event,
            {"map_location": "cpu", "weights_only": True},
        )
        self.assertEqual(
            [name for name, _ in events].count("synchronize"),
            2,
        )
        self.assertIn("inference_mode_enter", [name for name, _ in events])

    def test_missing_or_unexpected_state_keys_fail_closed(self) -> None:
        for model in (
            ModelStub(missing_keys=("patient_secret",)),
            ModelStub(unexpected_keys=("clinical_secret",)),
        ):
            fake_torch, _ = torch_stub({"model": {"weight": object()}})
            with patch.dict(sys.modules, {"torch": fake_torch}):
                with self.assertRaises(DetectorLoadError) as raised:
                    FocalNetDinoAdapter.from_artifact(
                        ArtifactStub(),
                        model_factory=lambda model=model: model,
                    )
            self.assertEqual(raised.exception.code, "detector_load_failed")
            self.assertNotIn("secret", str(raised.exception))

    def test_malformed_checkpoint_and_model_output_have_sanitized_errors(self) -> None:
        fake_torch, _ = torch_stub({"not-model": {}})
        with patch.dict(sys.modules, {"torch": fake_torch}):
            with self.assertRaises(DetectorLoadError) as raised:
                FocalNetDinoAdapter.from_artifact(
                    ArtifactStub(),
                    model_factory=ModelStub,
                )
        self.assertEqual(raised.exception.code, "detector_load_failed")
        self.assertIsNone(raised.exception.__cause__)

        model = MalformedOutputModelStub()
        fake_torch, _ = torch_stub({"model": {"weight": object()}})
        with patch.dict(sys.modules, {"torch": fake_torch}):
            adapter = FocalNetDinoAdapter.from_artifact(
                ArtifactStub(),
                model_factory=lambda: model,
            )
            with self.assertRaises(DetectorInferenceError) as inference_error:
                adapter.predict(
                    SimpleNamespace(
                        pixels=np.zeros((1024, 1024), dtype=np.uint8),
                        geometry=geometry(),
                    )
                )
        self.assertEqual(
            inference_error.exception.code,
            "detector_inference_failed",
        )

        wrong_shape_model = WrongShapeModelStub()
        fake_torch, _ = torch_stub({"model": {"weight": object()}})
        with patch.dict(sys.modules, {"torch": fake_torch}):
            adapter = FocalNetDinoAdapter.from_artifact(
                ArtifactStub(),
                model_factory=lambda: wrong_shape_model,
            )
            with self.assertRaises(DetectorInferenceError) as shape_error:
                adapter.predict(
                    SimpleNamespace(
                        pixels=np.zeros((1024, 1024), dtype=np.uint8),
                        geometry=geometry(),
                    )
                )
        self.assertEqual(shape_error.exception.code, "detector_inference_failed")

    def test_non_detector_or_unverified_artifact_is_rejected(self) -> None:
        for artifact in (
            SimpleNamespace(
                id="mmbcd-classifier",
                role="classifier",
                sha256="a" * 64,
                repository_revision="b" * 40,
                strict_load_verified=True,
            ),
            SimpleNamespace(
                id="focalnet-dino-detector",
                role="detector",
                sha256="a" * 64,
                repository_revision="b" * 40,
                strict_load_verified=False,
            ),
        ):
            with self.assertRaises(DetectorLoadError):
                FocalNetDinoAdapter.from_artifact(
                    artifact,
                    model_factory=ModelStub,
                )

        with self.assertRaises(DetectorLoadError) as invalid_device:
            FocalNetDinoAdapter.from_artifact(
                ArtifactStub(),
                model_factory=ModelStub,
                device="../gpu",
            )
        self.assertEqual(invalid_device.exception.detail, "detector device is invalid")


if __name__ == "__main__":
    unittest.main()
