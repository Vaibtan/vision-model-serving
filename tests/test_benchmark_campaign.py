from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from threading import Event, Lock
import unittest
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from vision_model_serving.model_ids import (  # noqa: E402
    CLASSIFIER_MODEL_ID,
    DETECTOR_MODEL_ID,
)
from vision_model_serving.pipeline import PredictionMode  # noqa: E402
from vision_model_serving.validation.benchmark import (  # noqa: E402
    validate_benchmark_record,
)
from vision_model_serving.validation.benchmark_campaign import (  # noqa: E402
    BenchmarkCampaignPlan,
    NvidiaSampler,
    run_benchmark_campaign,
)
from vision_model_serving.validation.benchmark_environment import (  # noqa: E402
    BENCHMARK_EXECUTOR_BASE_IMAGES,
    BENCHMARK_EXECUTOR_DOCKERFILE_SHA256,
    BENCHMARK_LANE_CONFIG_SHA256,
    BENCHMARK_LANE_HARDWARE,
    BENCHMARK_LANE_ID,
    BENCHMARK_LANE_NATIVE_OPERATOR,
    BENCHMARK_LANE_PACKAGES,
    BENCHMARK_LANE_SOFTWARE,
)
from vision_model_serving.validation.acceptance_contract import (  # noqa: E402
    CLASSIFIER_ARTIFACT,
    DETECTOR_ARTIFACT,
    PACKAGED_MANIFEST_ID,
    SERVED_CLASSIFIER_OUTPUT_SHA256,
    SERVED_DETECTOR_OUTPUT_SHA256,
    TOKENIZER_REVISION,
)
from vision_model_serving.validation.packaged_http import (  # noqa: E402
    PredictionObservation,
)


_TEST_DICOM = b"small deterministic benchmark fixture"
_TEST_DICOM_SHA256 = hashlib.sha256(_TEST_DICOM).hexdigest()


def _detection() -> dict[str, object]:
    return {
        "score": 0.5,
        "normalized_xyxy": [0.1, 0.1, 0.2, 0.2],
        "canonical_xyxy": [10.0, 10.0, 20.0, 20.0],
        "original_xyxy": [10.0, 10.0, 20.0, 20.0],
    }


def _campaign_plan() -> BenchmarkCampaignPlan:
    manifest_path = REPOSITORY_ROOT / "config" / "model-artifacts.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return BenchmarkCampaignPlan(
        dicom=_TEST_DICOM,
        runs=5,
        revision="a" * 40,
        script_sha256="b" * 64,
        environment={
            "lane": {
                "id": BENCHMARK_LANE_ID,
                "config_sha256": BENCHMARK_LANE_CONFIG_SHA256,
                "packages": dict(BENCHMARK_LANE_PACKAGES),
            },
            "hardware": {
                **BENCHMARK_LANE_HARDWARE,
                "total_memory_bytes": 24_000_000_000,
            },
            "software": {
                **BENCHMARK_LANE_SOFTWARE,
                "cudnn": "91002",
                "compiler": "gcc 13",
            },
            "native_operator": dict(BENCHMARK_LANE_NATIVE_OPERATOR),
            "container": {
                "executor_image_id": "sha256:" + "e" * 64,
                "executor_dockerfile_sha256": BENCHMARK_EXECUTOR_DOCKERFILE_SHA256,
                "pinned_base_images": list(BENCHMARK_EXECUTOR_BASE_IMAGES),
            },
        },
        identity={
            "manifest_id": manifest["manifest_id"],
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "models": manifest["artifacts"],
            "tokenizer": manifest["tokenizer"],
            "repository_assets": manifest["repository_assets"],
            "dicom_sha256": _TEST_DICOM_SHA256,
        },
        expected_detector_sha256=SERVED_DETECTOR_OUTPUT_SHA256,
        expected_classifier_sha256=SERVED_CLASSIFIER_OUTPUT_SHA256,
    )


class _InMemoryPackagedTarget:
    """HTTP-seam adapter with the strict residency behavior under measurement."""

    def __init__(
        self,
        *,
        pipeline_total_ms: float = 0.0,
        incomplete_prediction: bool = False,
    ) -> None:
        self._resident: str | None = None
        self._lock = Lock()
        self._pipeline_total_ms = pipeline_total_ms
        self._incomplete_prediction = incomplete_prediction

    def readiness(self) -> dict[str, object]:
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
                "state": "unloaded" if self._resident is None else "ready",
                "inference_warm": self._resident is not None,
                "warm_model": self._resident,
            },
        }

    def model_inventory(self) -> dict[str, object]:
        with self._lock:
            resident = self._resident
        return {
            "manifest_id": PACKAGED_MANIFEST_ID,
            "models": [
                {
                    "id": artifact.model_id,
                    "role": artifact.role,
                    "sha256": artifact.sha256,
                    "strict_load_verified": True,
                }
                for artifact in (DETECTOR_ARTIFACT, CLASSIFIER_ARTIFACT)
            ],
            "runtime": {
                "state": "unloaded" if resident is None else "ready",
                "initialized": True,
                "artifact_ready": True,
                "inference_warm": resident is not None,
                "warm_model": resident,
                "active_model": resident,
                "resident_models": [] if resident is None else [resident],
                "last_error": None,
            },
        }

    def operations(self) -> dict[str, object]:
        inventory = self.model_inventory()
        runtime = inventory["runtime"]
        return {
            "schema_version": 2,
            "status": "ready",
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
            "queue": {"available": True, "capacity": 4},
            "executor": {
                "available": True,
                "artifact_ready": True,
                "runtime_initialized": True,
                "device_available": True,
                "artifacts_verified": True,
                "native_operator_available": True,
                "failure_present": False,
                "precision": "float32",
                **runtime,
                "startup": {
                    "process_start_to_artifact_ready_seconds": 1.0,
                    "artifact_verification_seconds": 0.8,
                    "runtime_initialization_seconds": 0.1,
                },
            },
            "manifest_id": inventory["manifest_id"],
            "models": inventory["models"],
            "telemetry": {"events": {"cuda_oom_total": 0}},
        }

    def predict_observed(self, _dicom: bytes, *, mode: PredictionMode) -> PredictionObservation:
        with self._lock:
            detector_reused = self._resident == DETECTOR_MODEL_ID
            self._resident = (
                DETECTOR_MODEL_ID if mode is PredictionMode.DETECTION else CLASSIFIER_MODEL_ID
            )
        classifier_stage = None
        classification = None
        warnings = [
            {"code": "secondary_capture_storage"},
            {"code": "aspect_ratio_distorted"},
        ]
        provenance: dict[str, object] = {
            "detector": {
                "id": DETECTOR_ARTIFACT.model_id,
                "sha256": DETECTOR_ARTIFACT.sha256,
                "repository_revision": DETECTOR_ARTIFACT.repository_revision,
            },
            "classifier": None,
            "tokenizer": None,
            "precision": None,
            "offline_assets_only": None,
            "strict_checkpoint_load": None,
        }
        if mode is PredictionMode.FULL:
            classification = {
                "prediction_sha256": SERVED_CLASSIFIER_OUTPUT_SHA256,
                "logits": [1.0, -1.0],
                "probabilities": [0.880797, 0.119203],
                "attention": {"roi_weights": [0.125] * 8},
            }
            provenance.update(
                {
                    "classifier": {
                        "id": CLASSIFIER_ARTIFACT.model_id,
                        "sha256": CLASSIFIER_ARTIFACT.sha256,
                        "repository_revision": CLASSIFIER_ARTIFACT.repository_revision,
                    },
                    "tokenizer": {"revision": TOKENIZER_REVISION},
                    "precision": "float32",
                    "offline_assets_only": True,
                    "strict_checkpoint_load": True,
                }
            )
            warnings.extend(
                [
                    {"code": "class_semantics_and_decision_threshold_unverified"},
                    {"code": ("attention_is_inspection_not_causal_or_clinical_evidence")},
                ]
            )
            classifier_stage = {
                "runtime": {
                    "reused": False,
                    "load_ms": 0.0,
                    "inference_ms": 0.0,
                    "switch_ms": 0.0,
                },
                "adapter": {
                    "load_ms": 0.0,
                    "crop_preprocess_ms": 0.0,
                    "tokenization_ms": 0.0,
                    "inference_ms": 0.0,
                    "result_ms": 0.0,
                },
                "memory": {"peak_reserved_bytes": 200},
            }
        result = {
            "mode": mode.value,
            "input": {"source_sha256": _TEST_DICOM_SHA256},
            "disclaimer": "Research use only; not a medical diagnosis.",
            "geometry": {
                "canonical_width": 100,
                "canonical_height": 100,
                "original_width": 100,
                "original_height": 100,
            },
            "detector": {
                "prediction_sha256": SERVED_DETECTOR_OUTPUT_SHA256,
                "top_candidates": [_detection()],
                "post_nms": [_detection()],
                "classifier_rois": [_detection() for _ in range(8)],
            },
            "classification": classification,
            "provenance": provenance,
            "warnings": warnings,
            "timings": {
                "decode_ms": 0.0,
                "total_ms": self._pipeline_total_ms,
                "detector": {
                    "runtime": {
                        "reused": detector_reused,
                        "load_ms": 0.0,
                        "inference_ms": 0.0,
                        "switch_ms": 0.0,
                    },
                    "adapter": {
                        "load_ms": 0.0,
                        "preprocess_ms": 0.0,
                        "inference_ms": 0.0,
                        "postprocess_ms": 0.0,
                    },
                    "memory": {"peak_reserved_bytes": 100},
                },
                "classifier": classifier_stage,
            },
        }
        if self._incomplete_prediction:
            result.pop("warnings")
        return PredictionObservation(
            result=result,
            wall_seconds=0.0,
            queue_wait_seconds=0.0,
            states=("queued", "running", "succeeded"),
        )


class _CompletedResourceSampler:
    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def as_dict(self) -> dict[str, object]:
        distribution = {
            "count": 1,
            "min": 1.0,
            "p50": 1.0,
            "p95": 1.0,
            "p99": 1.0,
            "max": 1.0,
            "mean": 1.0,
            "population_stddev": 0.0,
        }
        return {
            "available": True,
            "interval_ms": 200,
            "duration_seconds": 0.1,
            "attempt_count": 1,
            "success_count": 1,
            "failure_count": 0,
            "sample_count": 1,
            "gpu_utilization_percent": dict(distribution),
            "memory_used_mib": dict(distribution),
            "power_watts": dict(distribution),
            "temperature_c": dict(distribution),
        }


class BenchmarkCampaignTests(unittest.TestCase):
    def test_nvidia_sampler_serializes_exact_attempt_accounting(self) -> None:
        for output, available in (("1, 2, 3, 4", True), ("invalid", False)):
            sampled = Event()

            def command(_command: list[str]) -> str:
                sampled.set()
                return output

            sampler = NvidiaSampler(50)
            with (
                self.subTest(output=output),
                patch(
                    "vision_model_serving.validation._benchmark_sampling._command",
                    side_effect=command,
                ),
            ):
                sampler.start()
                self.assertTrue(sampled.wait(1.0))
                sampler.stop()

            evidence = sampler.as_dict()
            self.assertEqual(evidence["available"], available)
            self.assertGreater(evidence["duration_seconds"], 0.0)
            self.assertGreaterEqual(evidence["attempt_count"], 1)
            self.assertEqual(
                evidence["attempt_count"],
                evidence["success_count"] + evidence["failure_count"],
            )
            self.assertEqual(evidence["sample_count"], evidence["success_count"])
            if available:
                self.assertEqual(evidence["failure_count"], 0)
            else:
                self.assertEqual(evidence["success_count"], 0)

    def test_campaign_plan_rejects_any_nonpackaged_dicom_by_default(self) -> None:
        with self.assertRaisesRegex(ValueError, "packaged acceptance"):
            _campaign_plan()

    @patch(
        "vision_model_serving.validation.acceptance_contract.PUBLIC_DICOM_SHA256",
        _TEST_DICOM_SHA256,
    )
    def test_campaign_owns_lifecycle_throughput_resources_and_promotion(self) -> None:
        plan = _campaign_plan()

        record = run_benchmark_campaign(
            _InMemoryPackagedTarget(),
            plan,
            _CompletedResourceSampler(),
        )

        validate_benchmark_record(record)
        self.assertEqual(record["schema_version"], 4)
        self.assertEqual(record["outcome"], "passed")
        self.assertEqual(
            len(record["measurements"]["lifecycle"]["warm_detection"]["aggregate"]["samples"]),
            5,
        )
        self.assertEqual(
            {
                concurrency: measurement["attempt_count"]
                for concurrency, measurement in record["measurements"]["throughput"].items()
            },
            {"1": 5, "2": 10, "4": 20},
        )
        self.assertEqual(
            record["measurements"]["lifecycle"]["cold_full"]["sample"][
                "detector_prediction_sha256"
            ],
            SERVED_DETECTOR_OUTPUT_SHA256,
        )
        self.assertEqual(
            record["measurements"]["lifecycle"]["cold_full"]["sample"][
                "classifier_prediction_sha256"
            ],
            SERVED_CLASSIFIER_OUTPUT_SHA256,
        )
        for measurement in record["measurements"]["throughput"].values():
            self.assertEqual(
                measurement["observed_prediction_sha256_counts"],
                {
                    "detector": {SERVED_DETECTOR_OUTPUT_SHA256: measurement["success_count"]},
                    "classifier": {},
                },
            )

    @patch(
        "vision_model_serving.validation.acceptance_contract.PUBLIC_DICOM_SHA256",
        _TEST_DICOM_SHA256,
    )
    def test_campaign_fails_closed_when_pipeline_time_exceeds_wall_time(self) -> None:
        target = _InMemoryPackagedTarget(pipeline_total_ms=500.0)
        plan = _campaign_plan()

        with self.assertRaisesRegex(ValueError, "pipeline timing exceeds"):
            run_benchmark_campaign(target, plan, _CompletedResourceSampler())

    @patch(
        "vision_model_serving.validation.acceptance_contract.PUBLIC_DICOM_SHA256",
        _TEST_DICOM_SHA256,
    )
    def test_campaign_rejects_sparse_hash_only_prediction_evidence(self) -> None:
        with self.assertRaisesRegex(ValueError, "packaged acceptance"):
            run_benchmark_campaign(
                _InMemoryPackagedTarget(incomplete_prediction=True),
                _campaign_plan(),
                _CompletedResourceSampler(),
            )


if __name__ == "__main__":
    unittest.main()
