"""Private repeated-switch and VRAM evidence contract."""

from __future__ import annotations

from collections.abc import Mapping
import math

from vision_model_serving.model_ids import CLASSIFIER_MODEL_ID, DETECTOR_MODEL_ID

from ._optimization_errors import OptimizationContractError


def validate_switch_reliability(
    matrix: Mapping[str, object],
    *,
    candidate_names: tuple[str, ...],
    memory_growth_threshold_percent: float,
) -> None:
    """Validate actual alternating loads, release, and VRAM stability."""

    if tuple(matrix) != candidate_names:
        raise OptimizationContractError("switch-reliability candidate matrix differs")
    for policy_name, raw_evidence in matrix.items():
        evidence = _mapping(
            raw_evidence,
            f"switch_reliability.{policy_name}",
        )
        required = {
            "status",
            "method",
            "cycle_count",
            "transitions",
            "maximum_same_model_reserved_growth_percent",
            "leak_observed",
            "failure_code",
            "failure_detail",
        }
        if set(evidence) != required:
            raise OptimizationContractError("switch-reliability evidence is incomplete")
        status = evidence.get("status")
        if status == "unmeasured":
            _validate_unmeasured(evidence)
            continue
        if status != "measured":
            raise OptimizationContractError("repeated-switch status is invalid")
        _validate_measured(
            evidence,
            policy_name=policy_name,
            memory_growth_threshold_percent=memory_growth_threshold_percent,
        )

    baseline = _mapping(matrix.get("fp32"), "switch_reliability.fp32")
    if baseline.get("status") != "measured" or baseline.get("leak_observed") is not False:
        raise OptimizationContractError("FP32 repeated-switch reliability did not pass")


def _validate_unmeasured(evidence: Mapping[str, object]) -> None:
    if (
        evidence.get("method") is not None
        or evidence.get("cycle_count") != 0
        or evidence.get("transitions") != []
        or evidence.get("maximum_same_model_reserved_growth_percent") is not None
        or evidence.get("leak_observed") is not None
    ):
        raise OptimizationContractError("unmeasured repeated-switch evidence is invalid")
    _required_string(evidence.get("failure_code"), "repeated-switch failure code")
    _required_string(evidence.get("failure_detail"), "repeated-switch failure detail")


def _validate_measured(
    evidence: Mapping[str, object],
    *,
    policy_name: str,
    memory_growth_threshold_percent: float,
) -> None:
    if (
        evidence.get("method") != "alternating_detector_classifier_release_v1"
        or evidence.get("failure_code") is not None
        or evidence.get("failure_detail") is not None
    ):
        raise OptimizationContractError("measured repeated-switch method is invalid")
    cycle_count = _positive_integer(
        evidence.get("cycle_count"),
        "repeated-switch cycle count",
    )
    if cycle_count < 2:
        raise OptimizationContractError("repeated-switch evidence needs two cycles")
    transitions = evidence.get("transitions")
    if not isinstance(transitions, list) or len(transitions) != cycle_count * 2:
        raise OptimizationContractError("repeated-switch transition count differs")

    first_reserved: dict[str, int] = {}
    maximum_growth = 0.0
    residency_leak = False
    reference_leak = False
    vram_leak = False
    for index, raw_transition in enumerate(transitions):
        cycle = index // 2 + 1
        model_id = (DETECTOR_MODEL_ID, CLASSIFIER_MODEL_ID)[index % 2]
        transition = _mapping(
            raw_transition,
            f"switch_reliability.{policy_name}.transitions[{index}]",
        )
        expected = {
            "cycle",
            "model_id",
            "resident_before_load",
            "resident_after_load",
            "resident_after_release",
            "model_reference_alive_after_release",
            "memory_after_inference",
            "memory_after_release",
        }
        if set(transition) != expected:
            raise OptimizationContractError("repeated-switch transition is incomplete")
        if transition.get("cycle") != cycle or transition.get("model_id") != model_id:
            raise OptimizationContractError("repeated-switch alternation differs")
        residency_leak |= transition.get("resident_before_load") != []
        residency_leak |= transition.get("resident_after_load") != [model_id]
        residency_leak |= transition.get("resident_after_release") != []
        reference_alive = transition.get("model_reference_alive_after_release")
        if not isinstance(reference_alive, bool):
            raise OptimizationContractError("repeated-switch reference result is invalid")
        reference_leak |= reference_alive

        _, reserved = _validate_loaded_memory(
            _mapping(
                transition.get("memory_after_inference"),
                "switch memory after inference",
            )
        )
        released_memory = _validate_released_memory(
            _mapping(
                transition.get("memory_after_release"),
                "switch memory after release",
            )
        )
        vram_leak |= any(released_memory)
        initial = first_reserved.setdefault(model_id, reserved)
        maximum_growth = max(
            maximum_growth,
            max(0.0, (reserved - initial) * 100.0 / initial),
        )

    observed_growth = _nonnegative_number(
        evidence.get("maximum_same_model_reserved_growth_percent"),
        "repeated-switch VRAM growth",
    )
    if not math.isclose(observed_growth, maximum_growth, rel_tol=1e-9, abs_tol=1e-9):
        raise OptimizationContractError("repeated-switch VRAM growth summary differs")
    growth_leak = maximum_growth > memory_growth_threshold_percent
    measured_leak = residency_leak or reference_leak or vram_leak or growth_leak
    if evidence.get("leak_observed") is not measured_leak:
        if residency_leak:
            raise OptimizationContractError("repeated-switch co-residency claim differs")
        if vram_leak:
            raise OptimizationContractError("repeated-switch VRAM cleanup claim differs")
        if growth_leak:
            raise OptimizationContractError("repeated-switch VRAM growth claim differs")
        raise OptimizationContractError("repeated-switch leak claim differs")


def _validate_loaded_memory(memory: Mapping[str, object]) -> tuple[int, int]:
    required = {
        "allocated_bytes",
        "reserved_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    }
    if set(memory) != required:
        raise OptimizationContractError("switch loaded VRAM evidence is incomplete")
    allocated = _positive_integer(memory.get("allocated_bytes"), "switch allocated VRAM")
    reserved = _positive_integer(memory.get("reserved_bytes"), "switch reserved VRAM")
    peak_allocated = _positive_integer(
        memory.get("peak_allocated_bytes"),
        "switch peak allocated VRAM",
    )
    peak_reserved = _positive_integer(
        memory.get("peak_reserved_bytes"),
        "switch peak reserved VRAM",
    )
    if allocated > reserved or allocated > peak_allocated or reserved > peak_reserved:
        raise OptimizationContractError("switch loaded VRAM evidence is impossible")
    if peak_allocated > peak_reserved:
        raise OptimizationContractError("switch peak VRAM evidence is impossible")
    return allocated, reserved


def _validate_released_memory(memory: Mapping[str, object]) -> tuple[int, int]:
    if set(memory) != {"allocated_bytes", "reserved_bytes"}:
        raise OptimizationContractError("switch released VRAM evidence is incomplete")
    allocated = _nonnegative_integer(
        memory.get("allocated_bytes"),
        "switch released allocated VRAM",
    )
    reserved = _nonnegative_integer(
        memory.get("reserved_bytes"),
        "switch released reserved VRAM",
    )
    if allocated > reserved:
        raise OptimizationContractError("switch released VRAM evidence is impossible")
    return allocated, reserved


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise OptimizationContractError(f"{path} must be an object")
    return value


def _required_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OptimizationContractError(f"{path} must be a non-empty string")
    return value


def _positive_integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise OptimizationContractError(f"{path} must be a positive integer")
    return value


def _nonnegative_integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OptimizationContractError(f"{path} must be a non-negative integer")
    return value


def _nonnegative_number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OptimizationContractError(f"{path} must be non-negative")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise OptimizationContractError(f"{path} must be non-negative")
    return result
