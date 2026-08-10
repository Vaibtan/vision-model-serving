from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from vision_model_serving.validation.optimization import (  # noqa: E402
    DetectorParityRecord,
    CandidateMeasurement,
    OptimizationExperiment,
    OptimizationContractError,
    OptimizationPolicy,
    candidate_promotion,
    compare_detector_records,
    validate_optimization_report,
)


NAMES = ("fp32", "tf32", "fp16", "bf16", "compile")
PRECISIONS = {
    "fp32": "float32",
    "tf32": "float32",
    "fp16": "float16",
    "bf16": "bfloat16",
    "compile": "float32",
}


def candidate(
    name: str,
    *,
    detector: bool,
    p50: float = 100.0,
    parity: bool = True,
) -> dict[str, object]:
    parity_evidence = (
        {
            "passed": parity,
            "method": "raw_post_nms_selected_roi_score_class_iou",
            "score_tolerance": 0.001,
            "logit_tolerance": 0.001,
            "box_tolerance": 0.001,
            "minimum_iou": 0.99,
            "raw_output_shapes": {
                "pred_logits": [1, 900, 1],
                "pred_boxes": [1, 900, 4],
            },
            "stages": {
                "raw": detector_stage(count=900, parity=parity),
                "post_nms": detector_stage(parity=parity),
                "selected_rois": detector_stage(count=8, parity=parity),
            },
            "output_sha256": {
                "pred_logits": "d" * 64,
                "pred_boxes": "e" * 64,
            },
        }
        if detector
        else {
            "passed": parity,
            "method": "elementwise_absolute",
            "absolute_tolerance": 0.001,
            "max_absolute_difference": {
                "logits": 0.0,
                "fused_embeddings": 0.0,
                "roi_attention": 0.0,
            },
            "output_sha256": {
                "logits": "f" * 64,
                "fused_embeddings": "1" * 64,
                "roi_attention": "2" * 64,
            },
        }
    )
    return {
        "name": name,
        "status": "passed" if parity else "rejected",
        "load_ms": 1.0,
        "parity": parity_evidence,
        "performance": {
            "latency_ms": {
                "count": 10,
                "min": p50 * 0.9,
                "p50": p50,
                "p95": p50 * 1.1,
                "p99": p50 * 1.1,
                "max": p50 * 1.1,
                "mean": p50,
                "population_stddev": p50 * 0.05,
            },
            "throughput_per_second": 1000.0 / p50,
            "peak_allocated_bytes": 90,
            "peak_reserved_bytes": 100,
        },
        "reliability": {
            "cuda_oom": False,
            "release": {
                "status": "measured",
                "model_reference_alive_after_release": False,
                "allocated_after_release_bytes": 0,
                "reserved_after_release_bytes": 0,
                "method": "weakref_and_cuda_allocator_after_release",
            },
        },
        "compile": {
            "enabled": name == "compile",
            "compilation_ms": 1.0 if name == "compile" else None,
            "recompilation_count": 0,
            "graph_break_count": 0,
            "fullgraph_required": name == "compile",
        },
        "precision": PRECISIONS[name],
        "tf32": name == "tf32",
        "promotion": {
            "accepted": False,
            "reasons": ["baseline_reference" if name == "fp32" else "material_improvement_missing"],
        },
    }


def detector_stage(*, count: int = 1, parity: bool = True) -> dict[str, object]:
    return {
        "baseline_count": count,
        "candidate_count": count,
        "matched_pairs": [[index, index] for index in range(count) if parity or index != 0],
        "unmatched_baseline": [] if parity else [0],
        "unmatched_candidate": [] if parity else [0],
        "max_score_difference": 0.0,
        "max_logit_difference": 0.0,
        "max_box_coordinate_difference": 0.0,
        "minimum_observed_iou": 1.0,
    }


def switch_reliability(*, measured: bool = True) -> dict[str, object]:
    if not measured:
        return {
            "status": "unmeasured",
            "method": None,
            "cycle_count": 0,
            "transitions": [],
            "maximum_same_model_reserved_growth_percent": None,
            "leak_observed": None,
            "failure_code": "candidate_switch_failed",
            "failure_detail": "candidate could not complete the repeated switch",
        }
    transitions = []
    for cycle in (1, 2):
        for model_id, allocated, reserved in (
            ("focalnet-dino-detector", 90, 100),
            ("mmbcd-classifier", 70, 80),
        ):
            transitions.append(
                {
                    "cycle": cycle,
                    "model_id": model_id,
                    "resident_before_load": [],
                    "resident_after_load": [model_id],
                    "resident_after_release": [],
                    "model_reference_alive_after_release": False,
                    "memory_after_inference": {
                        "allocated_bytes": allocated,
                        "reserved_bytes": reserved,
                        "peak_allocated_bytes": allocated,
                        "peak_reserved_bytes": reserved,
                    },
                    "memory_after_release": {
                        "allocated_bytes": 0,
                        "reserved_bytes": 0,
                    },
                }
            )
    return {
        "status": "measured",
        "method": "alternating_detector_classifier_release_v1",
        "cycle_count": 2,
        "transitions": transitions,
        "maximum_same_model_reserved_growth_percent": 0.0,
        "leak_observed": False,
        "failure_code": None,
        "failure_detail": None,
    }


def valid_report() -> dict[str, object]:
    return {
        "schema_version": 3,
        "revision": "a" * 40,
        "environment": {
            "measured_at": "2026-08-10T00:00:00+00:00",
            "device": "NVIDIA L4",
            "compute_capability": "8.9",
            "driver": "580.173.02",
            "torch": "2.8.0+cu128",
            "cuda": "12.8",
            "cudnn": "91002",
            "script_sha256": "b" * 64,
            "measurement_module_sha256": "3" * 64,
            "single_residency_evidence_sha256": "c" * 64,
        },
        "policy": {
            "warmup_runs": 3,
            "measured_runs": 10,
            "retention_threshold_percent": 15,
            "memory_retention_threshold_percent": 20,
            "candidate_order": list(NAMES),
            "one_change_at_a_time": True,
        },
        "switch_reliability": {name: switch_reliability() for name in NAMES},
        "models": {
            model_id: {
                "baseline": "fp32",
                "candidates": [
                    candidate(
                        name,
                        detector=model_id == "focalnet-dino-detector",
                    )
                    for name in NAMES
                ],
                "selected": "fp32",
            }
            for model_id in (
                "focalnet-dino-detector",
                "mmbcd-classifier",
            )
        },
        "final_runtime": {
            "backend": "pytorch-eager",
            "precision": "float32",
            "tf32": False,
        },
        "validation_boundary": "one checksum-pinned fixture on one NVIDIA L4",
    }


class OptimizationReportTests(unittest.TestCase):
    def test_experiment_aborts_before_loading_after_an_unclean_release(self) -> None:
        calls: list[str] = []

        def measure(policy: OptimizationPolicy) -> CandidateMeasurement:
            calls.append(policy.name)
            evidence = candidate(policy.name, detector=False)
            evidence.pop("parity")
            evidence.pop("promotion")
            evidence["reliability"]["release"]["model_reference_alive_after_release"] = True
            return CandidateMeasurement(
                evidence=evidence,
                outputs={"logits": np.array([1.0], dtype=np.float32)},
            )

        experiment = OptimizationExperiment(
            model_id="mmbcd-classifier",
            tolerances={name: 0.001 for name in NAMES},
            verify_reference=lambda outputs: None,
            compare=lambda baseline, observed, tolerance: {"passed": True},
        )

        with self.assertRaisesRegex(
            OptimizationContractError,
            "fp32 release did not pass",
        ):
            experiment.run(
                measure,
                switch_reliability={name: switch_reliability() for name in NAMES},
            )

        self.assertEqual(calls, ["fp32"])

    def test_experiment_owns_order_parity_and_fail_closed_promotion(self) -> None:
        measurements: dict[str, CandidateMeasurement] = {}
        for name in NAMES:
            evidence = candidate(name, detector=False, p50=50.0)
            evidence.pop("parity")
            evidence.pop("promotion")
            if name == "fp32":
                evidence["performance"]["latency_ms"] = {
                    "count": 10,
                    "min": 90.0,
                    "p50": 100.0,
                    "p95": 110.0,
                    "p99": 110.0,
                    "max": 110.0,
                    "mean": 100.0,
                    "population_stddev": 5.0,
                }
                evidence["performance"]["throughput_per_second"] = 10.0
            measurements[name] = CandidateMeasurement(
                evidence=evidence,
                outputs={"logits": np.array([1.0], dtype=np.float32)},
            )

        experiment = OptimizationExperiment(
            model_id="mmbcd-classifier",
            tolerances={name: 0.001 for name in NAMES},
            verify_reference=lambda outputs: None,
            compare=lambda baseline, observed, tolerance: {
                "passed": True,
                "method": "elementwise_absolute",
                "absolute_tolerance": tolerance,
                "max_absolute_difference": {"logits": 0.0},
                "output_sha256": {"logits": "f" * 64},
            },
        )

        switch_evidence = {name: switch_reliability() for name in NAMES}
        switch_evidence["compile"] = switch_reliability(measured=False)
        matrix = experiment.run(
            lambda policy: measurements[policy.name],
            switch_reliability=switch_evidence,
        )

        self.assertEqual(
            [value["name"] for value in matrix["candidates"]],
            list(NAMES),
        )
        compiled = matrix["candidates"][-1]
        self.assertFalse(compiled["promotion"]["accepted"])
        self.assertIn("repeated_switch_unmeasured", compiled["promotion"]["reasons"])
        self.assertEqual(matrix["selected"], "fp32")

    def test_candidate_requires_parity_reliability_and_material_gain(self) -> None:
        common = {
            "baseline_p50_ms": 100.0,
            "baseline_throughput": 10.0,
            "baseline_peak_bytes": 1000,
            "candidate_peak_bytes": 990,
            "cuda_oom": False,
            "release_measured": True,
            "release_leak": False,
            "repeated_switch_measured": True,
            "repeated_switch_leak": False,
            "graph_break_count": 0,
            "recompilation_count": 0,
        }
        accepted = candidate_promotion(
            **common,
            candidate_p50_ms=84.0,
            candidate_throughput=10.1,
            parity_passed=True,
        )
        rejected = candidate_promotion(
            **common,
            candidate_p50_ms=86.0,
            candidate_throughput=11.4,
            parity_passed=True,
        )
        unsafe = candidate_promotion(
            **common,
            candidate_p50_ms=50.0,
            candidate_throughput=20.0,
            parity_passed=False,
        )

        self.assertTrue(accepted.accepted)
        self.assertIn("warm_p50_improved_at_least_15_percent", accepted.reasons)
        self.assertFalse(rejected.accepted)
        self.assertIn("material_improvement_missing", rejected.reasons)
        self.assertFalse(unsafe.accepted)
        self.assertIn("parity_failed", unsafe.reasons)

        unmeasured = dict(common)
        unmeasured.update(
            repeated_switch_measured=False,
            repeated_switch_leak=None,
        )
        decision = candidate_promotion(
            **unmeasured,
            candidate_p50_ms=50.0,
            candidate_throughput=20.0,
            parity_passed=True,
        )
        self.assertFalse(decision.accepted)
        self.assertIn("repeated_switch_unmeasured", decision.reasons)

    def test_standalone_release_cannot_substitute_for_repeated_switches(self) -> None:
        decision = candidate_promotion(
            baseline_p50_ms=100.0,
            baseline_throughput=10.0,
            baseline_peak_bytes=1000,
            candidate_p50_ms=50.0,
            candidate_throughput=20.0,
            candidate_peak_bytes=500,
            parity_passed=True,
            cuda_oom=False,
            release_measured=True,
            release_leak=False,
            repeated_switch_measured=False,
            repeated_switch_leak=None,
            graph_break_count=0,
            recompilation_count=0,
        )

        self.assertFalse(decision.accepted)
        self.assertIn("repeated_switch_unmeasured", decision.reasons)

    def test_report_requires_full_semantic_matrix_for_both_models(self) -> None:
        report = valid_report()
        validate_optimization_report(report)

        report["models"]["mmbcd-classifier"]["candidates"].pop()
        with self.assertRaises(OptimizationContractError):
            validate_optimization_report(report)

    def test_report_rejects_unsupported_final_runtime_precision(self) -> None:
        report = valid_report()
        report["final_runtime"] = {
            "backend": "pytorch-eager",
            "precision": "float16",
            "tf32": True,
        }

        with self.assertRaisesRegex(
            OptimizationContractError,
            "eager FP32 with TF32 disabled",
        ):
            validate_optimization_report(report)

    def test_report_rejects_empty_identity_policy_or_boundary(self) -> None:
        for field, value, message in (
            ("environment", {}, "environment"),
            ("policy", {}, "policy"),
            ("validation_boundary", 1, "validation boundary"),
        ):
            with self.subTest(field=field):
                report = valid_report()
                report[field] = value
                with self.assertRaisesRegex(OptimizationContractError, message):
                    validate_optimization_report(report)

    def test_report_rejects_impossible_performance_statistics(self) -> None:
        report = valid_report()
        latency = report["models"]["focalnet-dino-detector"]["candidates"][0]["performance"][
            "latency_ms"
        ]
        latency["count"] = 999
        latency["p50"] = -1.0

        with self.assertRaisesRegex(OptimizationContractError, "latency"):
            validate_optimization_report(report)

    def test_report_rejects_parity_flags_that_contradict_measurements(self) -> None:
        report = valid_report()
        detector = report["models"]["focalnet-dino-detector"]["candidates"][0]
        detector["parity"]["stages"]["raw"]["unmatched_candidate"] = [0]
        with self.assertRaisesRegex(OptimizationContractError, "detector parity"):
            validate_optimization_report(report)

        report = valid_report()
        classifier = report["models"]["mmbcd-classifier"]["candidates"][0]
        classifier["parity"]["max_absolute_difference"]["logits"] = 1.0
        with self.assertRaisesRegex(OptimizationContractError, "classifier parity"):
            validate_optimization_report(report)

    def test_detector_parity_requires_raw_post_nms_and_selected_roi_semantics(self) -> None:
        for stage in ("raw", "post_nms", "selected_rois"):
            with self.subTest(stage=stage):
                report = valid_report()
                parity = report["models"]["focalnet-dino-detector"]["candidates"][0]["parity"]
                parity["stages"][stage]["max_logit_difference"] = 0.01
                with self.assertRaisesRegex(
                    OptimizationContractError,
                    "detector parity",
                ):
                    validate_optimization_report(report)

    def test_report_rejects_switch_claims_without_alternation_or_vram_cleanup(self) -> None:
        report = valid_report()
        transition = report["switch_reliability"]["tf32"]["transitions"][1]
        transition["resident_before_load"] = ["focalnet-dino-detector"]
        with self.assertRaisesRegex(OptimizationContractError, "co-residency"):
            validate_optimization_report(report)

        report = valid_report()
        transition = report["switch_reliability"]["tf32"]["transitions"][1]
        transition["memory_after_release"]["reserved_bytes"] = 1
        with self.assertRaisesRegex(OptimizationContractError, "VRAM"):
            validate_optimization_report(report)

    def test_report_rejects_switch_summary_that_hides_memory_growth(self) -> None:
        report = valid_report()
        transition = report["switch_reliability"]["tf32"]["transitions"][2]
        transition["memory_after_inference"]["reserved_bytes"] = 130
        transition["memory_after_inference"]["peak_reserved_bytes"] = 130

        with self.assertRaisesRegex(OptimizationContractError, "growth"):
            validate_optimization_report(report)

    def test_report_cannot_select_a_rejected_candidate(self) -> None:
        report = valid_report()
        report["models"]["focalnet-dino-detector"]["selected"] = "tf32"

        with self.assertRaises(OptimizationContractError):
            validate_optimization_report(report)

    def test_detector_comparison_matches_permuted_score_class_and_iou(self) -> None:
        expected = (
            DetectorParityRecord(
                0,
                2.0,
                0.9,
                (0.25, 0.25, 0.3, 0.3),
                (0.1, 0.1, 0.4, 0.4),
            ),
            DetectorParityRecord(
                1,
                1.5,
                0.8,
                (0.7, 0.7, 0.4, 0.4),
                (0.5, 0.5, 0.9, 0.9),
            ),
        )
        observed = (
            DetectorParityRecord(
                1,
                1.50001,
                0.80001,
                (0.7, 0.7, 0.4, 0.4),
                (0.5, 0.5, 0.9, 0.9),
            ),
            DetectorParityRecord(
                0,
                2.00001,
                0.90001,
                (0.2501, 0.25, 0.3, 0.3),
                (0.1001, 0.1, 0.4001, 0.4),
            ),
        )

        result = compare_detector_records(
            expected,
            observed,
            score_tolerance=0.001,
            logit_tolerance=0.001,
            box_tolerance=0.001,
            minimum_iou=0.99,
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.matched_pairs, ((0, 1), (1, 0)))
        self.assertGreaterEqual(result.minimum_observed_iou, 0.99)

    def test_detector_comparison_rejects_class_score_or_iou_mismatch(self) -> None:
        expected = (
            DetectorParityRecord(
                0,
                2.0,
                0.9,
                (0.25, 0.25, 0.3, 0.3),
                (0.1, 0.1, 0.4, 0.4),
            ),
        )
        mismatches = (
            DetectorParityRecord(1, 2.0, 0.9, (0.25, 0.25, 0.3, 0.3), (0.1, 0.1, 0.4, 0.4)),
            DetectorParityRecord(0, 2.0, 0.7, (0.25, 0.25, 0.3, 0.3), (0.1, 0.1, 0.4, 0.4)),
            DetectorParityRecord(0, 2.0, 0.9, (0.75, 0.75, 0.3, 0.3), (0.6, 0.6, 0.9, 0.9)),
            DetectorParityRecord(0, 2.1, 0.9, (0.25, 0.25, 0.3, 0.3), (0.1, 0.1, 0.4, 0.4)),
            DetectorParityRecord(0, 2.0, 0.9, (0.251, 0.25, 0.3, 0.3), (0.1, 0.1, 0.4, 0.4)),
        )
        for observed in mismatches:
            with self.subTest(observed=observed):
                result = compare_detector_records(
                    expected,
                    (observed,),
                    score_tolerance=0.001,
                    logit_tolerance=0.001,
                    box_tolerance=0.0001,
                    minimum_iou=0.99,
                )
                self.assertFalse(result.passed)


if __name__ == "__main__":
    unittest.main()
