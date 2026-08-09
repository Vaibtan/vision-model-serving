"""Contracts for evidence-gated PyTorch optimization experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Mapping

from vision_model_serving.model_ids import CLASSIFIER_MODEL_ID, DETECTOR_MODEL_ID


_COMMIT = re.compile(r"[0-9a-f]{40}")
_CANDIDATES = ("fp32", "tf32", "fp16", "bf16", "compile")


class OptimizationContractError(ValueError):
    """Raised when an optimization report could overstate measured evidence."""


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    accepted: bool
    reasons: tuple[str, ...]


def candidate_promotion(
    *,
    baseline_p50_ms: float,
    baseline_throughput: float,
    baseline_peak_bytes: int,
    candidate_p50_ms: float,
    candidate_throughput: float,
    candidate_peak_bytes: int,
    parity_passed: bool,
    cuda_oom: bool,
    model_switch_leak: bool,
    graph_break_count: int,
    recompilation_count: int,
) -> PromotionDecision:
    """Apply the Spec's parity, reliability, and material-gain gates."""

    values = (
        baseline_p50_ms,
        baseline_throughput,
        candidate_p50_ms,
        candidate_throughput,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise OptimizationContractError("latency and throughput must be positive")
    if baseline_peak_bytes <= 0 or candidate_peak_bytes <= 0:
        raise OptimizationContractError("peak memory must be positive")
    reasons: list[str] = []
    if not parity_passed:
        reasons.append("parity_failed")
    if cuda_oom:
        reasons.append("cuda_oom_observed")
    if model_switch_leak:
        reasons.append("model_switch_leak_observed")
    if graph_break_count < 0 or recompilation_count < 0:
        raise OptimizationContractError("compile counters cannot be negative")
    if graph_break_count:
        reasons.append("compile_graph_break_observed")
    if recompilation_count:
        reasons.append("compile_recompilation_observed")
    improvements: list[str] = []
    if candidate_p50_ms <= baseline_p50_ms * 0.85:
        improvements.append("warm_p50_improved_at_least_15_percent")
    if candidate_throughput >= baseline_throughput * 1.15:
        improvements.append("throughput_improved_at_least_15_percent")
    if candidate_peak_bytes <= baseline_peak_bytes * 0.80:
        improvements.append("peak_memory_improved_at_least_20_percent")
    if not improvements:
        reasons.append("material_improvement_missing")
    reasons.extend(improvements)
    accepted = (
        parity_passed
        and not cuda_oom
        and not model_switch_leak
        and graph_break_count == 0
        and recompilation_count == 0
        and bool(improvements)
    )
    return PromotionDecision(accepted=accepted, reasons=tuple(reasons))


def validate_optimization_report(report: Mapping[str, object]) -> None:
    """Reject incomplete matrices and unsupported runtime selections."""

    required = {
        "schema_version",
        "revision",
        "environment",
        "policy",
        "models",
        "final_runtime",
        "validation_boundary",
    }
    if set(report) != required or report.get("schema_version") != 1:
        raise OptimizationContractError("optimization report schema is invalid")
    if _COMMIT.fullmatch(str(report.get("revision", ""))) is None:
        raise OptimizationContractError("optimization revision must be exact")
    models = _mapping(report.get("models"), "models")
    if set(models) != {DETECTOR_MODEL_ID, CLASSIFIER_MODEL_ID}:
        raise OptimizationContractError("both model matrices are required")
    for model_id, raw_matrix in models.items():
        matrix = _mapping(raw_matrix, f"models.{model_id}")
        if set(matrix) != {"baseline", "candidates", "selected"}:
            raise OptimizationContractError("model matrix schema is invalid")
        if matrix["baseline"] != "fp32":
            raise OptimizationContractError("FP32 must remain the baseline")
        candidates = matrix["candidates"]
        if not isinstance(candidates, list) or len(candidates) != len(_CANDIDATES):
            raise OptimizationContractError("optimization candidate matrix is incomplete")
        by_name: dict[str, Mapping[str, object]] = {}
        for candidate in candidates:
            value = _mapping(candidate, "candidate")
            name = value.get("name")
            if not isinstance(name, str) or name in by_name:
                raise OptimizationContractError("candidate identity is invalid")
            by_name[name] = value
        if tuple(by_name) != _CANDIDATES:
            raise OptimizationContractError("candidate order or identity differs")
        selected = matrix["selected"]
        if selected not in by_name:
            raise OptimizationContractError("selected candidate is unavailable")
        if selected != "fp32":
            promotion = _mapping(by_name[str(selected)].get("promotion"), "promotion")
            if promotion.get("accepted") is not True:
                raise OptimizationContractError("selected candidate did not pass gates")
    runtime = _mapping(report.get("final_runtime"), "final_runtime")
    if set(runtime) != {"backend", "precision", "tf32"}:
        raise OptimizationContractError("final runtime selection is incomplete")
    if runtime.get("backend") != "pytorch-eager":
        raise OptimizationContractError("unvalidated production backend selected")


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise OptimizationContractError(f"{path} must be an object")
    return value
