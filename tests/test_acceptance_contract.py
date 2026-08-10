from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from vision_model_serving.model_ids import (  # noqa: E402
    CLASSIFIER_MODEL_ID,
    DETECTOR_MODEL_ID,
)
from vision_model_serving.pipeline import PredictionMode  # noqa: E402
from vision_model_serving.validation.acceptance_contract import (  # noqa: E402
    ARCHIVED_CLASSIFIER_OUTPUT_SHA256,
    PACKAGED_ACCEPTANCE,
    PACKAGED_MANIFEST_SHA256,
    PUBLIC_DICOM_SHA256,
    SERVED_CLASSIFIER_OUTPUT_SHA256,
    SERVED_DETECTOR_OUTPUT_SHA256,
)


def _detection() -> dict[str, object]:
    return {
        "score": 0.5,
        "normalized_xyxy": [0.1, 0.1, 0.2, 0.2],
        "canonical_xyxy": [10.0, 10.0, 20.0, 20.0],
        "original_xyxy": [10.0, 10.0, 20.0, 20.0],
    }


def _full_result() -> dict[str, object]:
    return {
        "mode": "full",
        "input": {
            "source_sha256": ("9f70081672a460f29231bb471e8a9e26dd3ed26a2ebbd91c064e575e7842a19c")
        },
        "disclaimer": "Research use only; not a medical diagnosis.",
        "geometry": {
            "canonical_width": 100,
            "canonical_height": 100,
            "original_width": 100,
            "original_height": 100,
        },
        "detector": {
            "prediction_sha256": (
                "4cdd09d986702e8839acff8d7517a63f263ca2a01b0607d78d6b2086c886a9a5"
            ),
            "top_candidates": [_detection()],
            "post_nms": [_detection()],
            "classifier_rois": [_detection() for _ in range(8)],
        },
        "classification": {
            "prediction_sha256": (
                "f994ccfad2e1894f95b487cf1068b5c0038b4bb12c7d49f5e0dc396afc83f1a3"
            ),
            "logits": [1.0, -1.0],
            "probabilities": [0.880797, 0.119203],
            "attention": {"roi_weights": [0.125] * 8},
        },
        "provenance": {
            "detector": {
                "id": DETECTOR_MODEL_ID,
                "sha256": ("67a7b0cd787a3aaba199cf1ff82ed2934c33ffe37544473379d7a837ab1637b4"),
                "repository_revision": ("23901e021dc6ec8f66bad47983f45a25574452cc"),
            },
            "classifier": {
                "id": CLASSIFIER_MODEL_ID,
                "sha256": ("2264351216f9fb4945af35e300459ff4ce2e7f5445519348024f3bf1eec721a4"),
                "repository_revision": ("14ac5e099c79253b01e0885d2ebefa6f86cfd8f0"),
            },
            "tokenizer": {"revision": "e2da8e2f811d1448a5b465c236feacd80ffbac7b"},
            "precision": "float32",
            "offline_assets_only": True,
            "strict_checkpoint_load": True,
        },
        "warnings": [
            {"code": "secondary_capture_storage"},
            {"code": "aspect_ratio_distorted"},
            {"code": "class_semantics_and_decision_threshold_unverified"},
            {"code": "attention_is_inspection_not_causal_or_clinical_evidence"},
        ],
    }


def _inventory() -> dict[str, object]:
    return {
        "manifest_id": "vision-model-serving-l4-fp32-20260807",
        "models": [
            {
                "id": DETECTOR_MODEL_ID,
                "role": "detector",
                "sha256": ("67a7b0cd787a3aaba199cf1ff82ed2934c33ffe37544473379d7a837ab1637b4"),
                "strict_load_verified": True,
            },
            {
                "id": CLASSIFIER_MODEL_ID,
                "role": "classifier",
                "sha256": ("2264351216f9fb4945af35e300459ff4ce2e7f5445519348024f3bf1eec721a4"),
                "strict_load_verified": True,
            },
        ],
        "runtime": {
            "state": "ready",
            "initialized": True,
            "artifact_ready": True,
            "inference_warm": True,
            "warm_model": DETECTOR_MODEL_ID,
            "active_model": DETECTOR_MODEL_ID,
            "resident_models": [DETECTOR_MODEL_ID],
            "last_error": None,
        },
    }


def _readiness() -> dict[str, object]:
    return {
        "schema_version": 2,
        "status": "ready",
        "readiness_scope": "artifact_ready",
        "checks": {
            "redis": True,
            "rq_worker": True,
            "executor_artifact_ready": True,
            "verified_artifacts": True,
            "runtime_initialized": True,
            "device_available": True,
            "native_operator_available": True,
            "manifest_available": True,
            "telemetry_available": True,
        },
        "reasons": [],
        "runtime": {
            "initialized": True,
            "state": "unloaded",
            "inference_warm": False,
            "warm_model": None,
        },
    }


class PackagedAcceptanceContractTests(unittest.TestCase):
    def test_benchmark_identity_is_exactly_bound_to_packaged_assets(self) -> None:
        manifest_path = REPOSITORY_ROOT / "config" / "model-artifacts.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        identity = {
            "manifest_id": manifest["manifest_id"],
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "models": manifest["artifacts"],
            "tokenizer": manifest["tokenizer"],
            "repository_assets": manifest["repository_assets"],
            "dicom_sha256": PUBLIC_DICOM_SHA256,
            "detector_prediction_sha256": SERVED_DETECTOR_OUTPUT_SHA256,
            "classifier_prediction_sha256": SERVED_CLASSIFIER_OUTPUT_SHA256,
        }

        self.assertEqual(identity["manifest_sha256"], PACKAGED_MANIFEST_SHA256)
        PACKAGED_ACCEPTANCE.validate_benchmark_identity(identity)
        altered_models = deepcopy(identity["models"])
        altered_models[0]["checkpoint"]["state_tensor_count"] += 1
        for field, value in (
            ("manifest_sha256", "0" * 64),
            ("dicom_sha256", "1" * 64),
            ("models", altered_models),
            ("repository_assets", [{"id": "unreviewed-source"}]),
            ("tokenizer", {"revision": manifest["tokenizer"]["revision"]}),
        ):
            tampered = deepcopy(identity)
            tampered[field] = value
            with self.subTest(field=field), self.assertRaises(AssertionError):
                PACKAGED_ACCEPTANCE.validate_benchmark_identity(tampered)

    def test_prediction_rejects_non_numeric_unbounded_or_degenerate_evidence(self) -> None:
        invalid_results: list[tuple[str, dict[str, object]]] = []

        string_vectors = _full_result()
        string_vectors["classification"]["logits"] = ["1.0", "-1.0"]
        invalid_results.append(("string logits", string_vectors))

        unbounded_probabilities = _full_result()
        unbounded_probabilities["classification"]["probabilities"] = [-1.0, 2.0]
        invalid_results.append(("unbounded probabilities", unbounded_probabilities))

        invalid_attention = _full_result()
        invalid_attention["classification"]["attention"]["roi_weights"] = [-99.0] * 8
        invalid_results.append(("invalid attention", invalid_attention))

        degenerate_box = _full_result()
        degenerate_box["detector"]["top_candidates"][0]["normalized_xyxy"] = [
            0.1,
            0.1,
            0.1,
            0.2,
        ]
        invalid_results.append(("degenerate box", degenerate_box))

        for name, result in invalid_results:
            with self.subTest(name=name), self.assertRaises(AssertionError):
                PACKAGED_ACCEPTANCE.validate_prediction(result, mode=PredictionMode.FULL)

    def test_readiness_requires_the_complete_named_check_contract(self) -> None:
        payload = _readiness()

        PACKAGED_ACCEPTANCE.validate_readiness(payload)
        for missing in tuple(payload["checks"]):
            with self.subTest(missing=missing):
                incomplete = deepcopy(payload)
                del incomplete["checks"][missing]
                with self.assertRaisesRegex(AssertionError, "not ready"):
                    PACKAGED_ACCEPTANCE.validate_readiness(incomplete)

    def test_readiness_rejects_stale_or_inconsistent_runtime_contracts(self) -> None:
        invalid_payloads = []

        stale = _readiness()
        stale["schema_version"] = 1
        invalid_payloads.append(stale)

        degraded = _readiness()
        degraded["reasons"] = ["runtime_unavailable"]
        invalid_payloads.append(degraded)

        incomplete = _readiness()
        del incomplete["runtime"]["warm_model"]
        invalid_payloads.append(incomplete)

        uninitialized = _readiness()
        uninitialized["runtime"]["initialized"] = False
        invalid_payloads.append(uninitialized)

        inconsistent_warm = _readiness()
        inconsistent_warm["runtime"] = {
            "initialized": True,
            "state": "ready",
            "inference_warm": False,
            "warm_model": None,
        }
        invalid_payloads.append(inconsistent_warm)

        inconsistent_cold = _readiness()
        inconsistent_cold["runtime"] = {
            "initialized": True,
            "state": "unloaded",
            "inference_warm": True,
            "warm_model": DETECTOR_MODEL_ID,
        }
        invalid_payloads.append(inconsistent_cold)

        malformed_state = _readiness()
        malformed_state["runtime"]["state"] = []
        invalid_payloads.append(malformed_state)

        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(AssertionError):
                    PACKAGED_ACCEPTANCE.validate_readiness(payload)

        warm = _readiness()
        warm["runtime"] = {
            "initialized": True,
            "state": "ready",
            "inference_warm": True,
            "warm_model": CLASSIFIER_MODEL_ID,
        }
        PACKAGED_ACCEPTANCE.validate_readiness(warm)

    def test_inventory_rejects_duplicates_extras_and_unverified_entries(self) -> None:
        invalid_inventories = []
        duplicate = _inventory()
        duplicate["models"].append(deepcopy(duplicate["models"][0]))
        invalid_inventories.append(duplicate)
        extra = _inventory()
        extra["models"].append(
            {
                "id": "unexpected-model",
                "role": "detector",
                "sha256": "0" * 64,
                "strict_load_verified": False,
            }
        )
        invalid_inventories.append(extra)
        unverified = _inventory()
        unverified["models"][0]["strict_load_verified"] = False
        invalid_inventories.append(unverified)

        for inventory in invalid_inventories:
            with self.subTest(models=inventory["models"]):
                with self.assertRaisesRegex(AssertionError, "inventory"):
                    PACKAGED_ACCEPTANCE.validate_inventory(inventory)

    def test_served_classifier_identity_remains_distinct_from_archive_identity(
        self,
    ) -> None:
        self.assertEqual(
            ARCHIVED_CLASSIFIER_OUTPUT_SHA256,
            "43ec1c4593c0549510098ea082ea7092c7fd5631c95d8b912ecf31633185899b",
        )
        self.assertEqual(
            SERVED_CLASSIFIER_OUTPUT_SHA256,
            "f994ccfad2e1894f95b487cf1068b5c0038b4bb12c7d49f5e0dc396afc83f1a3",
        )

        PACKAGED_ACCEPTANCE.validate_prediction(_full_result(), mode=PredictionMode.FULL)
        archived = _full_result()
        archived["classification"]["prediction_sha256"] = ARCHIVED_CLASSIFIER_OUTPUT_SHA256
        with self.assertRaisesRegex(
            AssertionError, "classifier output differs from the served golden"
        ):
            PACKAGED_ACCEPTANCE.validate_prediction(archived, mode=PredictionMode.FULL)

    def test_inventory_and_runtime_snapshot_share_one_model_identity_contract(
        self,
    ) -> None:
        inventory = _inventory()

        observed = PACKAGED_ACCEPTANCE.validate_inventory(inventory)
        PACKAGED_ACCEPTANCE.validate_runtime(observed["runtime"], active_model=DETECTOR_MODEL_ID)

        mismatched = deepcopy(observed["runtime"])
        mismatched["warm_model"] = CLASSIFIER_MODEL_ID
        with self.assertRaisesRegex(AssertionError, "unexpected executor lifecycle state"):
            PACKAGED_ACCEPTANCE.validate_runtime(mismatched, active_model=DETECTOR_MODEL_ID)


if __name__ == "__main__":
    unittest.main()
