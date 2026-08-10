"""Public interface for evidence-gated acceleration experiments."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math

from vision_model_serving.model_ids import CLASSIFIER_MODEL_ID, DETECTOR_MODEL_ID

from ._optimization_contract import (
    CANDIDATE_NAMES,
    OptimizationContractError,
    PromotionDecision,
    candidate_promotion,
    release_leak,
    release_passed,
    validate_optimization_report,
)
from ._optimization_parity import (
    DetectorParityComparison,
    DetectorParityRecord,
    compare_detector_records,
)


__all__ = [
    "CandidateMeasurement",
    "DetectorParityComparison",
    "DetectorParityRecord",
    "OPTIMIZATION_POLICIES",
    "OptimizationContractError",
    "OptimizationExperiment",
    "OptimizationPolicy",
    "PromotionDecision",
    "candidate_promotion",
    "compare_detector_records",
    "validate_optimization_report",
]


@dataclass(frozen=True, slots=True)
class OptimizationPolicy:
    name: str
    precision: str = "float32"
    tf32: bool = False
    compile: bool = False


OPTIMIZATION_POLICIES = (
    OptimizationPolicy("fp32"),
    OptimizationPolicy("tf32", tf32=True),
    OptimizationPolicy("fp16", precision="float16"),
    OptimizationPolicy("bf16", precision="bfloat16"),
    OptimizationPolicy("compile", compile=True),
)


@dataclass(frozen=True, slots=True)
class CandidateMeasurement:
    """GPU-adapter result before parity and promotion are attached."""

    evidence: Mapping[str, object]
    outputs: Mapping[str, object] | None


@dataclass(frozen=True, slots=True)
class OptimizationExperiment:
    """Own candidate order, parity, reliability, and promotion lifecycle."""

    model_id: str
    tolerances: Mapping[str, float]
    verify_reference: Callable[[Mapping[str, object]], None]
    compare: Callable[
        [Mapping[str, object], Mapping[str, object], float],
        Mapping[str, object],
    ]

    def __post_init__(self) -> None:
        if self.model_id not in {DETECTOR_MODEL_ID, CLASSIFIER_MODEL_ID}:
            raise ValueError("optimization model id is invalid")
        if tuple(self.tolerances) != CANDIDATE_NAMES or any(
            not math.isfinite(value) or value < 0.0 for value in self.tolerances.values()
        ):
            raise ValueError("optimization tolerances are invalid")

    def run(
        self,
        measure: Callable[[OptimizationPolicy], CandidateMeasurement],
        *,
        switch_reliability: Mapping[str, Mapping[str, object]],
    ) -> dict[str, object]:
        """Evaluate the complete matrix through one GPU measurement adapter."""

        if tuple(switch_reliability) != CANDIDATE_NAMES:
            raise OptimizationContractError("switch-reliability candidate order differs")
        baseline_outputs: Mapping[str, object] | None = None
        candidates: list[dict[str, object]] = []
        for policy_index, policy in enumerate(OPTIMIZATION_POLICIES):
            observation = measure(policy)
            candidate = dict(observation.evidence)
            if candidate.get("name") != policy.name:
                raise OptimizationContractError("measurement candidate order differs")
            if candidate.get("status") == "failed":
                if observation.outputs is not None:
                    raise OptimizationContractError("failed measurement cannot publish outputs")
                candidate["parity"] = {"passed": False}
            else:
                if observation.outputs is None:
                    raise OptimizationContractError(
                        "successful measurement did not publish outputs"
                    )
                if baseline_outputs is None:
                    baseline_outputs = observation.outputs
                    self.verify_reference(baseline_outputs)
                parity = dict(
                    self.compare(
                        baseline_outputs,
                        observation.outputs,
                        self.tolerances[policy.name],
                    )
                )
                if not isinstance(parity.get("passed"), bool):
                    raise OptimizationContractError("parity result is invalid")
                candidate["parity"] = parity
                candidate["status"] = "passed" if parity["passed"] is True else "rejected"
            candidates.append(candidate)
            if policy_index < len(OPTIMIZATION_POLICIES) - 1:
                self._require_clean_release(candidate)

        baseline = candidates[0]
        if baseline.get("status") != "passed":
            failure_code = baseline.get("failure_code", "parity_failed")
            failure_detail = baseline.get("failure_detail", "")
            raise OptimizationContractError(
                f"FP32 optimization baseline did not pass: {failure_code}: {failure_detail}"
            )
        self._attach_promotions(candidates, switch_reliability)
        return {
            "baseline": "fp32",
            "candidates": candidates,
            "selected": "fp32",
        }

    @staticmethod
    def _require_clean_release(candidate: Mapping[str, object]) -> None:
        name = _required_string(candidate.get("name"), "candidate.name")
        reliability = _mapping(
            candidate.get("reliability"),
            f"{name}.reliability",
        )
        release = _mapping(
            reliability.get("release"),
            f"{name}.reliability.release",
        )
        if not release_passed(release):
            raise OptimizationContractError(f"{name} release did not pass")

    @staticmethod
    def _attach_promotions(
        candidates: list[dict[str, object]],
        switch_reliability: Mapping[str, Mapping[str, object]],
    ) -> None:
        baseline_performance = _mapping(
            candidates[0].get("performance"),
            "baseline.performance",
        )
        for candidate in candidates:
            if candidate["name"] == "fp32":
                candidate["promotion"] = {
                    "accepted": False,
                    "reasons": ["baseline_reference"],
                }
                continue
            if candidate["status"] == "failed":
                candidate["promotion"] = {
                    "accepted": False,
                    "reasons": [str(candidate["failure_code"])],
                }
                continue
            performance = _mapping(candidate.get("performance"), "performance")
            reliability = _mapping(candidate.get("reliability"), "reliability")
            release = _mapping(
                reliability.get("release"),
                "reliability.release",
            )
            switch = _mapping(
                switch_reliability.get(str(candidate["name"])),
                f"switch_reliability.{candidate['name']}",
            )
            compile_evidence = _mapping(candidate.get("compile"), "compile")
            decision = candidate_promotion(
                baseline_p50_ms=_p50(baseline_performance),
                baseline_throughput=_positive_number(
                    baseline_performance.get("throughput_per_second"),
                    "baseline throughput",
                ),
                baseline_peak_bytes=_positive_integer(
                    baseline_performance.get("peak_reserved_bytes"),
                    "baseline peak memory",
                ),
                candidate_p50_ms=_p50(performance),
                candidate_throughput=_positive_number(
                    performance.get("throughput_per_second"),
                    "candidate throughput",
                ),
                candidate_peak_bytes=_positive_integer(
                    performance.get("peak_reserved_bytes"),
                    "candidate peak memory",
                ),
                parity_passed=_mapping(candidate.get("parity"), "parity").get("passed") is True,
                cuda_oom=reliability.get("cuda_oom") is True,
                release_measured=release.get("status") == "measured",
                release_leak=release_leak(release),
                repeated_switch_measured=switch.get("status") == "measured",
                repeated_switch_leak=switch.get("leak_observed"),
                graph_break_count=_nonnegative_integer(
                    compile_evidence.get("graph_break_count"),
                    "compile graph breaks",
                ),
                recompilation_count=_nonnegative_integer(
                    compile_evidence.get("recompilation_count"),
                    "compile recompilations",
                ),
            )
            candidate["promotion"] = {
                "accepted": decision.accepted,
                "reasons": list(decision.reasons),
            }


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise OptimizationContractError(f"{path} must be an object")
    return value


def _required_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OptimizationContractError(f"{path} must be a non-empty string")
    return value


def _positive_number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OptimizationContractError(f"{path} must be positive")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise OptimizationContractError(f"{path} must be positive")
    return result


def _positive_integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise OptimizationContractError(f"{path} must be a positive integer")
    return value


def _nonnegative_integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OptimizationContractError(f"{path} must be a non-negative integer")
    return value


def _p50(performance: Mapping[str, object]) -> float:
    return _positive_number(
        _mapping(performance.get("latency_ms"), "latency").get("p50"),
        "latency p50",
    )
