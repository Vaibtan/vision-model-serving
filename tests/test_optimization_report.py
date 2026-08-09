from __future__ import annotations

from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from vision_model_serving.validation.optimization import (  # noqa: E402
    OptimizationContractError,
    candidate_promotion,
    validate_optimization_report,
)


def candidate(name: str, *, p50: float, parity: bool = True) -> dict[str, object]:
    return {
        "name": name,
        "status": "passed" if parity else "rejected",
        "parity": {"passed": parity},
        "performance": {
            "latency_ms": {"p50": p50},
            "throughput_per_second": 1000.0 / p50,
            "peak_reserved_bytes": 100,
        },
        "reliability": {"cuda_oom": False, "model_switch_leak": False},
        "compile": {
            "enabled": name == "compile",
            "compilation_ms": 1.0 if name == "compile" else None,
            "recompilation_count": 0,
            "graph_break_count": 0,
        },
        "promotion": {"accepted": name == "tf32", "reasons": []},
    }


class OptimizationReportTests(unittest.TestCase):
    def test_candidate_requires_parity_reliability_and_material_gain(self) -> None:
        accepted = candidate_promotion(
            baseline_p50_ms=100.0,
            baseline_throughput=10.0,
            baseline_peak_bytes=1000,
            candidate_p50_ms=84.0,
            candidate_throughput=10.1,
            candidate_peak_bytes=990,
            parity_passed=True,
            cuda_oom=False,
            model_switch_leak=False,
            graph_break_count=0,
            recompilation_count=0,
        )
        rejected = candidate_promotion(
            baseline_p50_ms=100.0,
            baseline_throughput=10.0,
            baseline_peak_bytes=1000,
            candidate_p50_ms=86.0,
            candidate_throughput=11.4,
            candidate_peak_bytes=810,
            parity_passed=True,
            cuda_oom=False,
            model_switch_leak=False,
            graph_break_count=0,
            recompilation_count=0,
        )
        unsafe = candidate_promotion(
            baseline_p50_ms=100.0,
            baseline_throughput=10.0,
            baseline_peak_bytes=1000,
            candidate_p50_ms=50.0,
            candidate_throughput=20.0,
            candidate_peak_bytes=500,
            parity_passed=False,
            cuda_oom=False,
            model_switch_leak=False,
            graph_break_count=0,
            recompilation_count=0,
        )

        self.assertTrue(accepted.accepted)
        self.assertIn("warm_p50_improved_at_least_15_percent", accepted.reasons)
        self.assertFalse(rejected.accepted)
        self.assertIn("material_improvement_missing", rejected.reasons)
        self.assertFalse(unsafe.accepted)
        self.assertIn("parity_failed", unsafe.reasons)

        unstable_compile = candidate_promotion(
            baseline_p50_ms=100.0,
            baseline_throughput=10.0,
            baseline_peak_bytes=1000,
            candidate_p50_ms=50.0,
            candidate_throughput=20.0,
            candidate_peak_bytes=500,
            parity_passed=True,
            cuda_oom=False,
            model_switch_leak=False,
            graph_break_count=0,
            recompilation_count=1,
        )
        self.assertFalse(unstable_compile.accepted)
        self.assertIn("compile_recompilation_observed", unstable_compile.reasons)

    def test_report_requires_full_matrix_for_both_models(self) -> None:
        names = ("fp32", "tf32", "fp16", "bf16", "compile")
        report = {
            "schema_version": 1,
            "revision": "a" * 40,
            "environment": {"device": "NVIDIA L4", "torch": "2.8.0+cu128"},
            "policy": {"retention_threshold_percent": 15},
            "models": {
                model_id: {
                    "baseline": "fp32",
                    "candidates": [candidate(name, p50=100.0) for name in names],
                    "selected": "tf32",
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
            "validation_boundary": "one fixture",
        }

        validate_optimization_report(report)
        report["models"]["mmbcd-classifier"]["candidates"].pop()
        with self.assertRaises(OptimizationContractError):
            validate_optimization_report(report)

    def test_report_cannot_select_a_rejected_candidate(self) -> None:
        names = ("fp32", "tf32", "fp16", "bf16", "compile")
        candidates = [candidate(name, p50=100.0) for name in names]
        candidates[1]["promotion"] = {
            "accepted": False,
            "reasons": ["material_improvement_missing"],
        }
        report = {
            "schema_version": 1,
            "revision": "a" * 40,
            "environment": {"device": "NVIDIA L4", "torch": "2.8.0+cu128"},
            "policy": {"retention_threshold_percent": 15},
            "models": {
                model_id: {
                    "baseline": "fp32",
                    "candidates": candidates,
                    "selected": "tf32",
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
            "validation_boundary": "one fixture",
        }

        with self.assertRaises(OptimizationContractError):
            validate_optimization_report(report)


if __name__ == "__main__":
    unittest.main()
