from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from vision_model_serving.validation.benchmark import (  # noqa: E402
    BenchmarkContractError,
    ThroughputMeasurement,
    assert_single_residency_snapshot,
    benchmark_promotion_gates,
    latency_distribution,
    render_benchmark_markdown,
    validate_benchmark_record,
)
from vision_model_serving.validation.acceptance_contract import (  # noqa: E402
    CLASSIFIER_ARTIFACT,
    DETECTOR_ARTIFACT,
    PACKAGED_MANIFEST_ID,
    PUBLIC_DICOM_SHA256,
    SERVED_CLASSIFIER_OUTPUT_SHA256,
    SERVED_DETECTOR_OUTPUT_SHA256,
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


def valid_record() -> dict[str, object]:
    manifest_path = REPOSITORY_ROOT / "config" / "model-artifacts.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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
            "attempt_count": 5 * concurrency,
            "success_count": 5 * concurrency,
            "failure_count": 0,
            "failure_codes": {},
            "successful_requests_per_second": float(5 * concurrency),
            "attempted_requests_per_second": float(5 * concurrency),
            "latency_seconds": {**distribution, "count": 5 * concurrency},
            "queue_wait_seconds": {**distribution, "count": 5 * concurrency},
            "observed_prediction_sha256_counts": {
                "detector": {SERVED_DETECTOR_OUTPUT_SHA256: 5 * concurrency},
                "classifier": {},
            },
        }
        for concurrency in (1, 2, 4)
    }
    detector_runtime = {
        "state": "ready",
        "active_model": "focalnet-dino-detector",
        "resident_models": ["focalnet-dino-detector"],
    }
    classifier_runtime = {
        "state": "ready",
        "active_model": "mmbcd-classifier",
        "resident_models": ["mmbcd-classifier"],
    }

    models = [
        {
            "id": artifact.model_id,
            "role": artifact.role,
            "sha256": artifact.sha256,
            "strict_load_verified": True,
        }
        for artifact in (DETECTOR_ARTIFACT, CLASSIFIER_ARTIFACT)
    ]

    def inventory(runtime: dict[str, object]) -> dict[str, object]:
        return {
            "manifest_id": PACKAGED_MANIFEST_ID,
            "models": deepcopy(models),
            "runtime": runtime,
        }

    def operations(runtime: dict[str, object]) -> dict[str, object]:
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
            },
            "manifest_id": PACKAGED_MANIFEST_ID,
            "models": deepcopy(models),
            "telemetry": {"events": {"cuda_oom_total": 0}},
        }

    def sample(
        mode: str,
        *,
        detector_reused: bool,
        classifier_reused: bool | None,
    ) -> dict[str, object]:
        return {
            "mode": mode,
            "wall_seconds": 1.0,
            "queue_wait_seconds": 0.1,
            "states": ["queued", "succeeded"],
            "stages_seconds": {"pipeline_total": 0.8},
            "http_queue_overhead_seconds": 0.2,
            "lifecycle": {
                "detector_reused": detector_reused,
                "classifier_reused": classifier_reused,
            },
            "peak_reserved_bytes": 100,
            "detector_prediction_sha256": SERVED_DETECTOR_OUTPUT_SHA256,
            "classifier_prediction_sha256": (
                SERVED_CLASSIFIER_OUTPUT_SHA256 if mode == "full" else None
            ),
        }

    def aggregate(value: dict[str, object]) -> dict[str, object]:
        return {
            "samples": [deepcopy(value) for _ in range(5)],
            "wall_seconds": {**distribution, "count": 5},
            "queue_wait_seconds": {
                **distribution,
                "count": 5,
                "min": 0.1,
                "p50": 0.1,
                "p95": 0.1,
                "p99": 0.1,
                "max": 0.1,
                "mean": 0.1,
            },
            "stages_seconds": {
                "pipeline_total": {
                    **distribution,
                    "count": 5,
                    "min": 0.8,
                    "p50": 0.8,
                    "p95": 0.8,
                    "p99": 0.8,
                    "max": 0.8,
                    "mean": 0.8,
                }
            },
            "maximum_peak_reserved_bytes": 100,
        }

    warm_detection = sample("detection", detector_reused=True, classifier_reused=None)
    repeated_full = sample("full", detector_reused=False, classifier_reused=False)
    lifecycle = {
        "cold_full": {
            "sample": sample("full", detector_reused=False, classifier_reused=False),
            "inventory": inventory(classifier_runtime),
        },
        "switch_to_detection": {
            "sample": sample("detection", detector_reused=False, classifier_reused=None),
            "inventory": inventory(detector_runtime),
        },
        "warm_detection": {
            "aggregate": aggregate(warm_detection),
            "inventory": inventory(detector_runtime),
        },
        "full_after_detection": {
            "sample": sample("full", detector_reused=True, classifier_reused=False),
            "inventory": inventory(classifier_runtime),
        },
        "repeated_full": {
            "aggregate": aggregate(repeated_full),
            "inventory": inventory(classifier_runtime),
        },
        "final_full": {
            "sample": sample("full", detector_reused=True, classifier_reused=False),
            "inventory": inventory(classifier_runtime),
        },
    }
    resource_distribution = dict(distribution)
    nvidia_smi = {
        "available": True,
        "interval_ms": 200,
        "duration_seconds": 0.1,
        "attempt_count": 1,
        "success_count": 1,
        "failure_count": 0,
        "sample_count": 1,
        "gpu_utilization_percent": dict(resource_distribution),
        "memory_used_mib": dict(resource_distribution),
        "power_watts": dict(resource_distribution),
        "temperature_c": dict(resource_distribution),
    }
    record = {
        "schema_version": 4,
        "measured_at": "2026-08-10T00:01:00+00:00",
        "harness": {
            "revision": "a" * 40,
            "revision_clean": True,
            "script_sha256": "b" * 64,
            "started_at": "2026-08-10T00:00:00+00:00",
            "completed_at": "2026-08-10T00:01:00+00:00",
            "warmup_count": 1,
            "measured_runs_per_lifecycle_phase": 5,
            "percentile_method": "nearest_rank",
            "stddev_method": "population",
        },
        "environment": {
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
                "executor_image_id": "sha256:" + "c" * 64,
                "executor_dockerfile_sha256": BENCHMARK_EXECUTOR_DOCKERFILE_SHA256,
                "pinned_base_images": list(BENCHMARK_EXECUTOR_BASE_IMAGES),
            },
        },
        "identity": {
            "manifest_id": PACKAGED_MANIFEST_ID,
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "models": deepcopy(manifest["artifacts"]),
            "tokenizer": deepcopy(manifest["tokenizer"]),
            "repository_assets": deepcopy(manifest["repository_assets"]),
            "dicom_sha256": PUBLIC_DICOM_SHA256,
            "detector_prediction_sha256": SERVED_DETECTOR_OUTPUT_SHA256,
            "classifier_prediction_sha256": SERVED_CLASSIFIER_OUTPUT_SHA256,
        },
        "policy": {
            "backend": "pytorch-eager",
            "precision": "float32",
            "tf32": False,
            "executor_concurrency": 1,
            "queue_capacity_requirement": 4,
            "offered_concurrency": [1, 2, 4],
            "lifecycle_sequence": [
                "cold_full",
                "switch_to_detection",
                "consecutive_warm_detection",
                "full_after_detection",
                "repeated_switch_bound_full",
                "throughput_detection",
                "final_full",
            ],
        },
        "measurements": {
            "startup": {
                "process_start_to_artifact_ready_seconds": 1.0,
                "artifact_verification_seconds": 0.8,
                "runtime_initialization_seconds": 0.1,
            },
            "lifecycle": lifecycle,
            "throughput": throughput,
            "resources": {
                "before_operations": operations(
                    {
                        "state": "unloaded",
                        "active_model": None,
                        "resident_models": [],
                    }
                ),
                "after_operations": operations(classifier_runtime),
                "cuda_oom_total": 0,
                "nvidia_smi": nvidia_smi,
            },
        },
        "gates": {},
        "outcome": "passed",
        "validation_boundary": "One pinned public DICOM on one NVIDIA L4.",
    }
    record["gates"] = benchmark_promotion_gates(record)
    return record


class BenchmarkStatisticsTests(unittest.TestCase):
    def test_distribution_includes_every_required_statistic(self) -> None:
        observed = latency_distribution([1.0, 2.0, 3.0, 4.0, 5.0])

        self.assertEqual(observed["count"], 5)
        self.assertEqual(observed["min"], 1.0)
        self.assertEqual(observed["p50"], 3.0)
        self.assertEqual(observed["p95"], 5.0)
        self.assertEqual(observed["p99"], 5.0)
        self.assertEqual(observed["max"], 5.0)
        self.assertEqual(observed["mean"], 3.0)
        self.assertTrue(math.isclose(observed["population_stddev"], math.sqrt(2.0)))

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
            with self.subTest(value=value), self.assertRaises(BenchmarkContractError):
                assert_single_residency_snapshot(value)

    def test_schema_v4_requires_complete_concurrency_and_environment(self) -> None:
        record = valid_record()
        validate_benchmark_record(record)

        missing_lane = valid_record()
        del missing_lane["environment"]["lane"]
        self.assertFalse(benchmark_promotion_gates(missing_lane)["environment_complete"])

        tampered_environments = []
        tampered_config = valid_record()
        tampered_config["environment"]["lane"]["config_sha256"] = "0" * 64
        tampered_environments.append(tampered_config)

        tampered_package = valid_record()
        tampered_package["environment"]["lane"]["packages"]["pillow"] = "0.0.0"
        tampered_environments.append(tampered_package)

        tampered_software = valid_record()
        tampered_software["environment"]["software"]["torch"] = "2.9.0"
        tampered_environments.append(tampered_software)

        tampered_dockerfile = valid_record()
        tampered_dockerfile["environment"]["container"]["executor_dockerfile_sha256"] = "0" * 64
        tampered_environments.append(tampered_dockerfile)

        for record in tampered_environments:
            with self.subTest(environment=record["environment"]):
                self.assertFalse(benchmark_promotion_gates(record)["environment_complete"])

        missing_concurrency = valid_record()
        del missing_concurrency["measurements"]["throughput"]["4"]
        with self.assertRaises(BenchmarkContractError):
            validate_benchmark_record(missing_concurrency)

        dirty = valid_record()
        dirty["harness"]["revision_clean"] = False
        with self.assertRaises(BenchmarkContractError):
            validate_benchmark_record(dirty)

    def test_schema_v4_rejects_impossible_throughput_semantics(self) -> None:
        invalid_records = []

        negative_duration = valid_record()
        negative_duration["measurements"]["throughput"]["1"]["duration_seconds"] = -1.0
        invalid_records.append(negative_duration)

        inconsistent_rate = valid_record()
        inconsistent_rate["measurements"]["throughput"]["2"]["successful_requests_per_second"] = (
            999.0
        )
        invalid_records.append(inconsistent_rate)

        impossible_distribution = valid_record()
        impossible_distribution["measurements"]["throughput"]["4"]["latency_seconds"]["count"] = 999
        invalid_records.append(impossible_distribution)

        invalid_failure_code = valid_record()
        invalid_failure_code["measurements"]["throughput"]["2"]["failure_codes"] = {1: 0}
        invalid_records.append(invalid_failure_code)

        for record in invalid_records:
            with self.subTest(record=record), self.assertRaises(BenchmarkContractError):
                validate_benchmark_record(record)

        queue_exceeds_latency = valid_record()
        queue_distribution = queue_exceeds_latency["measurements"]["throughput"]["1"][
            "queue_wait_seconds"
        ]
        for name in ("min", "p50", "p95", "p99", "max", "mean"):
            queue_distribution[name] = 2.0
        with self.assertRaisesRegex(BenchmarkContractError, "queue wait"):
            validate_benchmark_record(queue_exceeds_latency)

        latency_exceeds_campaign = valid_record()
        latency_distribution_payload = latency_exceeds_campaign["measurements"]["throughput"]["1"][
            "latency_seconds"
        ]
        for name in ("min", "p50", "p95", "p99", "max", "mean"):
            latency_distribution_payload[name] = 100.0
        with self.assertRaisesRegex(BenchmarkContractError, "duration"):
            validate_benchmark_record(latency_exceeds_campaign)

    def test_schema_v4_requires_semantic_startup_lifecycle_and_resources(self) -> None:
        for section in ("startup", "lifecycle", "resources"):
            record = valid_record()
            record["measurements"][section] = {}
            with self.subTest(section=section), self.assertRaises(BenchmarkContractError):
                validate_benchmark_record(record)

        invalid_sampler = valid_record()
        invalid_sampler["measurements"]["resources"]["nvidia_smi"]["sample_count"] = 2
        with self.assertRaises(BenchmarkContractError):
            validate_benchmark_record(invalid_sampler)

        invalid_lifecycle_distribution = valid_record()
        invalid_lifecycle_distribution["measurements"]["lifecycle"]["warm_detection"]["aggregate"][
            "wall_seconds"
        ]["population_stddev"] = 0.5
        with self.assertRaises(BenchmarkContractError):
            validate_benchmark_record(invalid_lifecycle_distribution)

        impossible_startup = valid_record()
        impossible_startup["measurements"]["startup"].update(
            {
                "process_start_to_artifact_ready_seconds": 1.0,
                "artifact_verification_seconds": 0.8,
                "runtime_initialization_seconds": 0.4,
            }
        )
        with self.assertRaisesRegex(BenchmarkContractError, "startup phase"):
            validate_benchmark_record(impossible_startup)

        impossible_queue_wait = valid_record()
        impossible_queue_wait["measurements"]["lifecycle"]["cold_full"]["sample"][
            "queue_wait_seconds"
        ] = 100.0
        with self.assertRaisesRegex(BenchmarkContractError, "queue wait"):
            validate_benchmark_record(impossible_queue_wait)

        impossible_stage = valid_record()
        impossible_stage["measurements"]["lifecycle"]["cold_full"]["sample"]["stages_seconds"][
            "detector_inference"
        ] = 999.0
        with self.assertRaisesRegex(BenchmarkContractError, "pipeline total"):
            validate_benchmark_record(impossible_stage)

    def test_schema_v4_binds_serialized_output_digests_to_identity(self) -> None:
        lifecycle_tamper = valid_record()
        lifecycle_tamper["measurements"]["lifecycle"]["cold_full"]["sample"][
            "detector_prediction_sha256"
        ] = "0" * 64
        self.assertFalse(
            benchmark_promotion_gates(lifecycle_tamper)["golden_outputs_all_successes"]
        )

        throughput_tamper = valid_record()
        counts = throughput_tamper["measurements"]["throughput"]["2"][
            "observed_prediction_sha256_counts"
        ]["detector"]
        counts["0" * 64] = counts.pop(SERVED_DETECTOR_OUTPUT_SHA256)
        self.assertFalse(
            benchmark_promotion_gates(throughput_tamper)["golden_outputs_all_successes"]
        )

        impossible_counts = valid_record()
        impossible_counts["measurements"]["throughput"]["4"]["observed_prediction_sha256_counts"][
            "detector"
        ][SERVED_DETECTOR_OUTPUT_SHA256] = 2
        with self.assertRaisesRegex(BenchmarkContractError, "digest counts"):
            validate_benchmark_record(impossible_counts)

    def test_schema_v4_requires_resource_sampling_accounting_and_coverage(self) -> None:
        failed_sample = valid_record()
        sampler = failed_sample["measurements"]["resources"]["nvidia_smi"]
        sampler["attempt_count"] = 2
        sampler["failure_count"] = 1
        self.assertFalse(benchmark_promotion_gates(failed_sample)["resource_sampling_complete"])

        sparse_sample = valid_record()
        sparse_sample["measurements"]["resources"]["nvidia_smi"]["duration_seconds"] = 100.0
        self.assertFalse(benchmark_promotion_gates(sparse_sample)["resource_sampling_complete"])

        impossible_accounting = valid_record()
        impossible_accounting["measurements"]["resources"]["nvidia_smi"]["attempt_count"] = 2
        with self.assertRaisesRegex(BenchmarkContractError, "sampler accounting"):
            validate_benchmark_record(impossible_accounting)

        impossible_utilization = valid_record()
        impossible_utilization["measurements"]["resources"]["nvidia_smi"][
            "gpu_utilization_percent"
        ]["max"] = 999.0
        with self.assertRaisesRegex(BenchmarkContractError, "utilization"):
            validate_benchmark_record(impossible_utilization)

        zero_series = valid_record()
        sampler = zero_series["measurements"]["resources"]["nvidia_smi"]
        for distribution_name in ("memory_used_mib", "power_watts", "temperature_c"):
            distribution = sampler[distribution_name]
            for statistic in ("min", "p50", "p95", "p99", "max", "mean"):
                distribution[statistic] = 0.0
        with self.assertRaisesRegex(BenchmarkContractError, "positive"):
            validate_benchmark_record(zero_series)

        impossible_memory = valid_record()
        memory = impossible_memory["measurements"]["resources"]["nvidia_smi"]["memory_used_mib"]
        for statistic in ("min", "p50", "p95", "p99", "max", "mean"):
            memory[statistic] = 1_000_000_000_000.0
        self.assertFalse(benchmark_promotion_gates(impossible_memory)["resource_sampling_complete"])

        contradictory_oom = valid_record()
        contradictory_oom["measurements"]["resources"]["after_operations"]["telemetry"]["events"][
            "cuda_oom_total"
        ] = 5
        self.assertFalse(benchmark_promotion_gates(contradictory_oom)["no_cuda_oom"])

    def test_schema_v4_binds_measurement_counts_to_campaign_plan(self) -> None:
        sparse_lifecycle = valid_record()
        aggregate = sparse_lifecycle["measurements"]["lifecycle"]["warm_detection"]["aggregate"]
        aggregate["samples"] = aggregate["samples"][:1]
        aggregate["wall_seconds"]["count"] = 1
        aggregate["queue_wait_seconds"]["count"] = 1
        aggregate["stages_seconds"]["pipeline_total"]["count"] = 1
        self.assertFalse(benchmark_promotion_gates(sparse_lifecycle)["measurement_matrix_complete"])

        sparse_throughput = valid_record()
        payload = sparse_throughput["measurements"]["throughput"]["4"]
        payload.update(
            {
                "attempt_count": 1,
                "success_count": 1,
                "successful_requests_per_second": 1.0,
                "attempted_requests_per_second": 1.0,
            }
        )
        payload["latency_seconds"]["count"] = 1
        payload["queue_wait_seconds"]["count"] = 1
        payload["observed_prediction_sha256_counts"]["detector"][SERVED_DETECTOR_OUTPUT_SHA256] = 1
        self.assertFalse(
            benchmark_promotion_gates(sparse_throughput)["measurement_matrix_complete"]
        )

    def test_promotion_gates_bind_identity_and_policy_to_observations(self) -> None:
        for field, value in (
            ("manifest_id", "arbitrary-manifest"),
            ("manifest_sha256", "0" * 64),
            ("dicom_sha256", "1" * 64),
            ("repository_assets", [{"id": "arbitrary-source"}]),
        ):
            tampered_identity = valid_record()
            tampered_identity["identity"][field] = value
            with self.subTest(field=field):
                gates = benchmark_promotion_gates(tampered_identity)
                self.assertFalse(gates["identity_exact"])
                self.assertFalse(gates["golden_outputs_all_successes"])

        tampered_capacity = valid_record()
        tampered_capacity["measurements"]["resources"]["before_operations"]["queue"]["capacity"] = (
            99
        )
        self.assertFalse(benchmark_promotion_gates(tampered_capacity)["artifact_ready"])

        tampered_precision = valid_record()
        tampered_precision["measurements"]["resources"]["after_operations"]["executor"][
            "precision"
        ] = "float16"
        self.assertFalse(benchmark_promotion_gates(tampered_precision)["artifact_ready"])

        incomplete_operations = valid_record()
        for name in ("before_operations", "after_operations"):
            del incomplete_operations["measurements"]["resources"][name]["checks"][
                "manifest_available"
            ]
        self.assertFalse(benchmark_promotion_gates(incomplete_operations)["artifact_ready"])

        failed_offered_load = valid_record()
        throughput = failed_offered_load["measurements"]["throughput"]["2"]
        throughput.update(
            {
                "success_count": 0,
                "failure_count": 10,
                "failure_codes": {"prediction_queue_full": 10},
                "successful_requests_per_second": 0.0,
                "latency_seconds": None,
                "queue_wait_seconds": None,
                "observed_prediction_sha256_counts": {
                    "detector": {},
                    "classifier": {},
                },
            }
        )
        self.assertFalse(
            benchmark_promotion_gates(failed_offered_load)["golden_outputs_all_successes"]
        )

    def test_sample_overhead_must_equal_wall_minus_pipeline_with_tolerance(self) -> None:
        record = valid_record()
        sample = record["measurements"]["lifecycle"]["cold_full"]["sample"]
        sample["http_queue_overhead_seconds"] = 0.0

        with self.assertRaisesRegex(BenchmarkContractError, "overhead"):
            validate_benchmark_record(record)

    def test_schema_v4_requires_ordered_timezone_aware_campaign_timestamps(self) -> None:
        invalid_records = []

        missing_measured_at = valid_record()
        missing_measured_at["measured_at"] = None
        invalid_records.append(missing_measured_at)

        naive_timestamp = valid_record()
        naive_timestamp["harness"]["started_at"] = "2026-08-10T00:00:00"
        invalid_records.append(naive_timestamp)

        reversed_timestamps = valid_record()
        reversed_timestamps["harness"]["started_at"] = "2099-01-01T00:00:00+00:00"
        reversed_timestamps["harness"]["completed_at"] = "1900-01-01T00:00:00+00:00"
        reversed_timestamps["measured_at"] = "1900-01-01T00:00:00+00:00"
        invalid_records.append(reversed_timestamps)

        mismatched_completion = valid_record()
        mismatched_completion["measured_at"] = "2026-08-10T00:00:59+00:00"
        invalid_records.append(mismatched_completion)

        for record in invalid_records:
            with (
                self.subTest(record=record),
                self.assertRaisesRegex(
                    BenchmarkContractError,
                    "timestamp",
                ),
            ):
                validate_benchmark_record(record)

    def test_committed_schema_v3_evidence_remains_valid(self) -> None:
        record = json.loads(
            (REPOSITORY_ROOT / "docs" / "validation" / "benchmark-l4-20260810.json").read_text(
                encoding="utf-8"
            )
        )

        validate_benchmark_record(deepcopy(record))

    def test_schema_v3_rejects_synthetic_or_tampered_evidence(self) -> None:
        synthetic = valid_record()
        synthetic["schema_version"] = 3

        historical = json.loads(
            (REPOSITORY_ROOT / "docs" / "validation" / "benchmark-l4-20260810.json").read_text(
                encoding="utf-8"
            )
        )
        historical["measured_at"] = "2026-08-10T00:00:01+00:00"

        for record in (synthetic, historical):
            with (
                self.subTest(record=record),
                self.assertRaisesRegex(
                    BenchmarkContractError,
                    "historical schema-v3",
                ),
            ):
                validate_benchmark_record(record)

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
