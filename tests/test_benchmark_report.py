from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from vision_model_serving.validation.benchmark import (  # noqa: E402
    BenchmarkContractError,
    ThroughputMeasurement,
    assert_single_residency_snapshot,
    latency_distribution,
    render_benchmark_markdown,
    validate_benchmark_record,
)
from scripts import benchmark_api  # noqa: E402


def valid_record() -> dict[str, object]:
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
    throughput = {
        str(concurrency): {
            "concurrency": concurrency,
            "duration_seconds": 1.0,
            "attempt_count": 1,
            "success_count": 1,
            "failure_count": 0,
            "failure_codes": {},
            "successful_requests_per_second": 1.0,
            "attempted_requests_per_second": 1.0,
            "latency_seconds": dict(distribution),
            "queue_wait_seconds": dict(distribution),
        }
        for concurrency in (1, 2, 4)
    }
    return {
        "schema_version": 3,
        "measured_at": "2026-08-10T00:00:00+00:00",
        "harness": {
            "revision": "a" * 40,
            "revision_clean": True,
            "script_sha256": "b" * 64,
            "percentile_method": "nearest_rank",
            "stddev_method": "population",
        },
        "environment": {
            "hardware": {"gpu_name": "NVIDIA L4"},
            "software": {"torch": "2.8.0+cu128"},
            "native_operator": {"module": "MultiScaleDeformableAttention"},
            "container": {"executor_image_id": "sha256:" + "c" * 64},
        },
        "identity": {
            "manifest_id": "manifest",
            "manifest_sha256": "d" * 64,
            "models": [],
            "dicom_sha256": "e" * 64,
        },
        "policy": {
            "backend": "pytorch-eager",
            "precision": "float32",
            "tf32": False,
            "executor_concurrency": 1,
            "offered_concurrency": [1, 2, 4],
        },
        "measurements": {
            "startup": {},
            "lifecycle": {},
            "throughput": throughput,
            "resources": {"cuda_oom_total": 0},
        },
        "gates": {
            "environment_complete": True,
            "revision_exact_clean": True,
            "identity_exact": True,
            "artifact_ready": True,
            "single_residency_all_snapshots": True,
            "golden_outputs_all_successes": True,
            "measurement_matrix_complete": True,
            "failure_accounting_complete": True,
            "concurrency_1_no_failures": True,
            "each_concurrency_has_success": True,
            "no_cuda_oom": True,
            "resource_sampling_complete": True,
        },
        "outcome": "passed",
        "validation_boundary": "One pinned public DICOM on one NVIDIA L4.",
    }


class BenchmarkStatisticsTests(unittest.TestCase):
    def test_sample_preserves_nested_runtime_and_adapter_timings(self) -> None:
        result = {
            "detector": {"prediction_sha256": "d" * 64},
            "classification": {"prediction_sha256": "c" * 64},
            "timings": {
                "decode_ms": 1.0,
                "total_ms": 100.0,
                "detector": {
                    "runtime": {
                        "reused": False,
                        "load_ms": 2.0,
                        "inference_ms": 30.0,
                        "switch_ms": 0.0,
                    },
                    "adapter": {
                        "load_ms": 3.0,
                        "preprocess_ms": 4.0,
                        "inference_ms": 5.0,
                        "postprocess_ms": 6.0,
                    },
                    "memory": {"peak_reserved_bytes": 100},
                },
                "classifier": {
                    "runtime": {
                        "reused": False,
                        "load_ms": 7.0,
                        "inference_ms": 40.0,
                        "switch_ms": 8.0,
                    },
                    "adapter": {
                        "load_ms": 9.0,
                        "crop_preprocess_ms": 10.0,
                        "tokenization_ms": 11.0,
                        "inference_ms": 12.0,
                        "result_ms": 13.0,
                    },
                    "memory": {"peak_reserved_bytes": 200},
                },
            },
        }
        client = SimpleNamespace(
            predict_observed=lambda *_args, **_kwargs: SimpleNamespace(
                result=result,
                wall_seconds=0.2,
                queue_wait_seconds=0.01,
                states=("queued", "started", "succeeded"),
            )
        )

        observed = benchmark_api._run_sample(
            client,
            b"dicom",
            mode=benchmark_api.PredictionMode.FULL,
            expected_detector_sha256="d" * 64,
            expected_classifier_sha256="c" * 64,
        )

        stages = observed["stages_seconds"]
        self.assertEqual(stages["detector_runtime_execute"], 0.03)
        self.assertEqual(stages["detector_inference"], 0.005)
        self.assertEqual(stages["classifier_runtime_execute"], 0.04)
        self.assertEqual(stages["classifier_inference"], 0.012)
        self.assertEqual(stages["classifier_adapter_load"], 0.009)
        self.assertEqual(observed["peak_reserved_bytes"], 200)

    def test_environment_identity_accepts_the_real_packages_mapping(self) -> None:
        evidence = {
            "status": "passed",
            "gpu_gate": "passed",
            "snapshot": {
                "collection_errors": [],
                "python": "3.12.11",
                "packages": {
                    "torch": "2.8.0+cu128",
                    "torchvision": "0.23.0+cu128",
                },
                "torch_cuda": "12.8",
                "device_name": "NVIDIA L4",
                "compute_capability": "8.9",
                "total_device_memory_bytes": 23_659_151_360,
                "cudnn_version": "91002",
                "nvcc_release": "12.8",
                "compiler": "c++ 13.3.0",
                "driver": "580.173.02",
            },
        }
        with TemporaryDirectory() as directory:
            evidence_path = Path(directory) / "environment.json"
            evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
            observed = benchmark_api._environment_identity(
                REPOSITORY_ROOT,
                evidence_path,
                executor_image_id="sha256:" + "a" * 64,
            )

        self.assertEqual(observed["software"]["torch"], "2.8.0+cu128")
        self.assertEqual(
            observed["container"]["executor_image_id"],
            "sha256:" + "a" * 64,
        )

    def test_distribution_includes_every_required_statistic(self) -> None:
        observed = latency_distribution([1.0, 2.0, 3.0, 4.0, 5.0])

        self.assertEqual(observed["count"], 5)
        self.assertEqual(observed["min"], 1.0)
        self.assertEqual(observed["p50"], 3.0)
        self.assertEqual(observed["p95"], 5.0)
        self.assertEqual(observed["p99"], 5.0)
        self.assertEqual(observed["max"], 5.0)
        self.assertEqual(observed["mean"], 3.0)
        self.assertTrue(
            math.isclose(observed["population_stddev"], math.sqrt(2.0))
        )

    def test_throughput_requires_complete_failure_accounting(self) -> None:
        with self.assertRaises(BenchmarkContractError):
            ThroughputMeasurement(
                concurrency=2,
                duration_seconds=1.0,
                attempts=4,
                successful_latencies=(0.2, 0.3),
                queue_waits=(0.1, 0.1),
                failure_codes={"queue_full": 1},
            )

        observed = ThroughputMeasurement(
            concurrency=2,
            duration_seconds=1.0,
            attempts=4,
            successful_latencies=(0.2, 0.3),
            queue_waits=(0.1, 0.1),
            failure_codes={"queue_full": 2},
        ).as_dict()

        self.assertEqual(observed["attempt_count"], 4)
        self.assertEqual(observed["success_count"], 2)
        self.assertEqual(observed["failure_count"], 2)
        self.assertEqual(observed["successful_requests_per_second"], 2.0)

    def test_throughput_preserves_all_failed_offered_load(self) -> None:
        observed = ThroughputMeasurement(
            concurrency=4,
            duration_seconds=2.0,
            attempts=4,
            successful_latencies=(),
            queue_waits=(),
            failure_codes={"prediction_queue_full": 4},
        ).as_dict()

        self.assertEqual(observed["success_count"], 0)
        self.assertEqual(observed["failure_count"], 4)
        self.assertIsNone(observed["latency_seconds"])
        self.assertIsNone(observed["queue_wait_seconds"])


class BenchmarkContractTests(unittest.TestCase):
    def test_residency_snapshots_reject_dual_or_mismatched_models(self) -> None:
        assert_single_residency_snapshot(
            {
                "state": "ready",
                "active_model": "detector",
                "resident_models": ["detector"],
            }
        )
        for value in (
            {
                "state": "ready",
                "active_model": "detector",
                "resident_models": ["detector", "classifier"],
            },
            {
                "state": "ready",
                "active_model": "detector",
                "resident_models": ["classifier"],
            },
        ):
            with self.subTest(value=value), self.assertRaises(
                BenchmarkContractError
            ):
                assert_single_residency_snapshot(value)

    def test_schema_v3_requires_complete_concurrency_and_environment(self) -> None:
        record = valid_record()
        validate_benchmark_record(record)

        missing_concurrency = valid_record()
        del missing_concurrency["measurements"]["throughput"]["4"]
        with self.assertRaises(BenchmarkContractError):
            validate_benchmark_record(missing_concurrency)

        dirty = valid_record()
        dirty["harness"]["revision_clean"] = False
        with self.assertRaises(BenchmarkContractError):
            validate_benchmark_record(dirty)

    def test_markdown_reports_p99_failures_and_all_concurrencies(self) -> None:
        rendered = render_benchmark_markdown(valid_record())

        self.assertIn("p99", rendered)
        self.assertIn("Concurrency 1", rendered)
        self.assertIn("Concurrency 2", rendered)
        self.assertIn("Concurrency 4", rendered)
        self.assertIn("failure", rendered.lower())
        self.assertIn("single residency", rendered.lower())


if __name__ == "__main__":
    unittest.main()
