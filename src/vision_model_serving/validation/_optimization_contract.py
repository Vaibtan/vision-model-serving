"""Private promotion and schema contract for optimization evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
import math
import re

from vision_model_serving.model_ids import CLASSIFIER_MODEL_ID, DETECTOR_MODEL_ID

from ._optimization_errors import OptimizationContractError
from ._optimization_switch import validate_switch_reliability


_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMPUTE_CAPABILITY = re.compile(r"[0-9]+\.[0-9]+")
CANDIDATE_NAMES = ("fp32", "tf32", "fp16", "bf16", "compile")
_PRECISIONS = {
    "fp32": "float32",
    "tf32": "float32",
    "fp16": "float16",
    "bf16": "bfloat16",
    "compile": "float32",
}
_ENVIRONMENT_KEYS = {
    "measured_at",
    "device",
    "compute_capability",
    "driver",
    "torch",
    "cuda",
    "cudnn",
    "script_sha256",
    "measurement_module_sha256",
    "single_residency_evidence_sha256",
}
_POLICY_KEYS = {
    "warmup_runs",
    "measured_runs",
    "retention_threshold_percent",
    "memory_retention_threshold_percent",
    "candidate_order",
    "one_change_at_a_time",
}
_LATENCY_KEYS = {
    "count",
    "min",
    "p50",
    "p95",
    "p99",
    "max",
    "mean",
    "population_stddev",
}


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
    release_measured: bool,
    release_leak: bool | None,
    repeated_switch_measured: bool,
    repeated_switch_leak: bool | None,
    graph_break_count: int,
    recompilation_count: int,
) -> PromotionDecision:
    """Apply parity, release, repeated-switch, and material-gain gates."""

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
    for label, measured, leak in (
        ("candidate release", release_measured, release_leak),
        ("repeated switch", repeated_switch_measured, repeated_switch_leak),
    ):
        if not isinstance(measured, bool):
            raise OptimizationContractError(f"{label} measurement state is invalid")
        if measured:
            if not isinstance(leak, bool):
                raise OptimizationContractError(f"measured {label} result is invalid")
        elif leak is not None:
            raise OptimizationContractError(f"unmeasured {label} result must be null")

    reasons: list[str] = []
    if not parity_passed:
        reasons.append("parity_failed")
    if cuda_oom:
        reasons.append("cuda_oom_observed")
    if not release_measured:
        reasons.append("candidate_release_unmeasured")
    elif release_leak:
        reasons.append("candidate_release_leak_observed")
    if not repeated_switch_measured:
        reasons.append("repeated_switch_unmeasured")
    elif repeated_switch_leak:
        reasons.append("repeated_switch_reliability_failed")
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
        and release_measured
        and release_leak is False
        and repeated_switch_measured
        and repeated_switch_leak is False
        and graph_break_count == 0
        and recompilation_count == 0
        and bool(improvements)
    )
    return PromotionDecision(accepted=accepted, reasons=tuple(reasons))


def validate_optimization_report(report: Mapping[str, object]) -> None:
    """Validate the complete semantic contract for a publishable report."""

    required = {
        "schema_version",
        "revision",
        "environment",
        "policy",
        "switch_reliability",
        "models",
        "final_runtime",
        "validation_boundary",
    }
    if set(report) != required or report.get("schema_version") != 3:
        raise OptimizationContractError("optimization report schema is invalid")
    if _COMMIT.fullmatch(str(report.get("revision", ""))) is None:
        raise OptimizationContractError("optimization revision must be exact")
    _validate_environment(_mapping(report.get("environment"), "environment"))
    policy = _mapping(report.get("policy"), "policy")
    measured_runs = _validate_policy(policy)
    switch_reliability = _mapping(
        report.get("switch_reliability"),
        "switch_reliability",
    )
    validate_switch_reliability(
        switch_reliability,
        candidate_names=CANDIDATE_NAMES,
        memory_growth_threshold_percent=float(policy["memory_retention_threshold_percent"]),
    )
    boundary = report.get("validation_boundary")
    if not isinstance(boundary, str) or not boundary.strip():
        raise OptimizationContractError("validation boundary must be explicit")

    models = _mapping(report.get("models"), "models")
    if set(models) != {DETECTOR_MODEL_ID, CLASSIFIER_MODEL_ID}:
        raise OptimizationContractError("both model matrices are required")
    for model_id, raw_matrix in models.items():
        _validate_model_matrix(
            model_id,
            _mapping(raw_matrix, f"models.{model_id}"),
            measured_runs,
            switch_reliability,
        )

    runtime = _mapping(report.get("final_runtime"), "final_runtime")
    expected_runtime = {
        "backend": "pytorch-eager",
        "precision": "float32",
        "tf32": False,
    }
    if dict(runtime) != expected_runtime:
        raise OptimizationContractError("final runtime must remain eager FP32 with TF32 disabled")


def _validate_environment(environment: Mapping[str, object]) -> None:
    if set(environment) != _ENVIRONMENT_KEYS:
        raise OptimizationContractError("optimization environment is incomplete")
    measured_at = environment.get("measured_at")
    if not isinstance(measured_at, str):
        raise OptimizationContractError("environment measured_at is invalid")
    try:
        parsed = datetime.fromisoformat(measured_at)
    except ValueError:
        raise OptimizationContractError("environment measured_at is invalid") from None
    if parsed.tzinfo is None:
        raise OptimizationContractError("environment measured_at must include a timezone")
    for field in ("device", "driver", "torch", "cuda", "cudnn"):
        _required_string(environment.get(field), f"environment.{field}")
    if _COMPUTE_CAPABILITY.fullmatch(str(environment.get("compute_capability", ""))) is None:
        raise OptimizationContractError("environment compute capability is invalid")
    for field in (
        "script_sha256",
        "measurement_module_sha256",
        "single_residency_evidence_sha256",
    ):
        if _SHA256.fullmatch(str(environment.get(field, ""))) is None:
            raise OptimizationContractError(f"environment.{field} is invalid")


def _validate_policy(policy: Mapping[str, object]) -> int:
    if set(policy) != _POLICY_KEYS:
        raise OptimizationContractError("optimization policy is incomplete")
    warmup_runs = _positive_integer(policy.get("warmup_runs"), "policy.warmup_runs")
    measured_runs = _positive_integer(
        policy.get("measured_runs"),
        "policy.measured_runs",
    )
    if warmup_runs < 1 or measured_runs < 5:
        raise OptimizationContractError("optimization policy run counts are too low")
    if policy.get("retention_threshold_percent") != 15:
        raise OptimizationContractError("optimization latency policy differs")
    if policy.get("memory_retention_threshold_percent") != 20:
        raise OptimizationContractError("optimization memory policy differs")
    if policy.get("candidate_order") != list(CANDIDATE_NAMES):
        raise OptimizationContractError("optimization candidate policy differs")
    if policy.get("one_change_at_a_time") is not True:
        raise OptimizationContractError("optimization isolation policy differs")
    return measured_runs


def _validate_model_matrix(
    model_id: str,
    matrix: Mapping[str, object],
    measured_runs: int,
    switch_reliability: Mapping[str, object],
) -> None:
    if set(matrix) != {"baseline", "candidates", "selected"}:
        raise OptimizationContractError("model matrix schema is invalid")
    if matrix.get("baseline") != "fp32":
        raise OptimizationContractError("FP32 must remain the baseline")
    candidates = matrix.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != len(CANDIDATE_NAMES):
        raise OptimizationContractError("optimization candidate matrix is incomplete")
    by_name: dict[str, Mapping[str, object]] = {}
    for raw_candidate in candidates:
        value = _mapping(raw_candidate, f"models.{model_id}.candidate")
        name = value.get("name")
        if not isinstance(name, str) or name in by_name:
            raise OptimizationContractError("candidate identity is invalid")
        by_name[name] = value
        _validate_candidate(model_id, value, measured_runs)
    if tuple(by_name) != CANDIDATE_NAMES:
        raise OptimizationContractError("candidate order or identity differs")
    for name in CANDIDATE_NAMES[:-1]:
        if not release_passed(_release_evidence(by_name[name])):
            raise OptimizationContractError(
                f"candidate {name} was not released before the next load"
            )
    if by_name["fp32"].get("status") != "passed":
        raise OptimizationContractError("FP32 baseline did not pass")
    baseline_release = _release_evidence(by_name["fp32"])
    if not release_passed(baseline_release):
        raise OptimizationContractError("FP32 baseline release did not pass")
    if dict(_mapping(by_name["fp32"].get("promotion"), "baseline.promotion")) != {
        "accepted": False,
        "reasons": ["baseline_reference"],
    }:
        raise OptimizationContractError("FP32 baseline promotion is invalid")
    if matrix.get("selected") != "fp32":
        raise OptimizationContractError("unpackaged optimization was selected")

    baseline_performance = _mapping(
        by_name["fp32"].get("performance"),
        "baseline.performance",
    )
    for name in CANDIDATE_NAMES[1:]:
        candidate = by_name[name]
        promotion = _mapping(candidate.get("promotion"), f"{name}.promotion")
        if candidate.get("status") == "failed":
            expected = {
                "accepted": False,
                "reasons": [str(candidate.get("failure_code"))],
            }
            if dict(promotion) != expected:
                raise OptimizationContractError("failed candidate promotion differs")
            continue
        reliability = _mapping(candidate.get("reliability"), f"{name}.reliability")
        release = _mapping(reliability.get("release"), f"{name}.release")
        switch = _mapping(switch_reliability.get(name), f"switch_reliability.{name}")
        compile_evidence = _mapping(candidate.get("compile"), f"{name}.compile")
        performance = _mapping(candidate.get("performance"), f"{name}.performance")
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
        if dict(promotion) != {
            "accepted": decision.accepted,
            "reasons": list(decision.reasons),
        }:
            raise OptimizationContractError("candidate promotion contradicts gates")


def _validate_candidate(
    model_id: str,
    candidate: Mapping[str, object],
    measured_runs: int,
) -> None:
    name = str(candidate.get("name", ""))
    status = candidate.get("status")
    common = {
        "name",
        "status",
        "parity",
        "performance",
        "reliability",
        "compile",
        "precision",
        "tf32",
        "promotion",
    }
    expected = (
        common | {"failure_code", "failure_detail"} if status == "failed" else common | {"load_ms"}
    )
    if set(candidate) != expected or status not in {"passed", "rejected", "failed"}:
        raise OptimizationContractError(f"candidate {name} schema is invalid")
    if candidate.get("precision") != _PRECISIONS.get(name):
        raise OptimizationContractError(f"candidate {name} precision differs")
    if candidate.get("tf32") is not (name == "tf32"):
        raise OptimizationContractError(f"candidate {name} TF32 policy differs")

    parity = _mapping(candidate.get("parity"), f"candidate {name}.parity")
    if status == "failed":
        if dict(parity) != {"passed": False}:
            raise OptimizationContractError(f"failed candidate {name} parity is invalid")
        _required_string(candidate.get("failure_code"), f"candidate {name} failure")
        _required_string(candidate.get("failure_detail"), f"candidate {name} detail")
    else:
        _positive_number(candidate.get("load_ms"), f"candidate {name} load time")
        _validate_parity(model_id, name, parity, status)
    _validate_performance(candidate, measured_runs)
    _validate_reliability(candidate)
    _validate_compile(candidate)
    promotion = _mapping(candidate.get("promotion"), f"candidate {name}.promotion")
    if set(promotion) != {"accepted", "reasons"} or not isinstance(
        promotion.get("accepted"),
        bool,
    ):
        raise OptimizationContractError(f"candidate {name} promotion is invalid")
    reasons = promotion.get("reasons")
    if (
        not isinstance(reasons, list)
        or not reasons
        or any(not isinstance(reason, str) or not reason for reason in reasons)
    ):
        raise OptimizationContractError(f"candidate {name} promotion reasons are invalid")
    if status == "failed" and promotion.get("accepted") is not False:
        raise OptimizationContractError("failed candidate cannot be promoted")


def _validate_parity(
    model_id: str,
    name: str,
    parity: Mapping[str, object],
    status: object,
) -> None:
    if parity.get("passed") is not (status == "passed"):
        raise OptimizationContractError(f"candidate {name} status and parity differ")
    if model_id == DETECTOR_MODEL_ID:
        required = {
            "passed",
            "method",
            "score_tolerance",
            "logit_tolerance",
            "box_tolerance",
            "minimum_iou",
            "raw_output_shapes",
            "stages",
            "output_sha256",
        }
        if (
            set(parity) != required
            or parity.get("method") != "raw_post_nms_selected_roi_score_class_iou"
        ):
            raise OptimizationContractError("detector parity method is invalid")
        score_tolerance = _nonnegative_number(
            parity.get("score_tolerance"),
            "detector score tolerance",
        )
        logit_tolerance = _nonnegative_number(
            parity.get("logit_tolerance"),
            "detector logit tolerance",
        )
        box_tolerance = _nonnegative_number(
            parity.get("box_tolerance"),
            "detector box tolerance",
        )
        minimum_iou = _nonnegative_number(parity.get("minimum_iou"), "detector IoU")
        if minimum_iou > 1.0:
            raise OptimizationContractError("detector IoU is invalid")
        shapes = _mapping(parity.get("raw_output_shapes"), "detector raw shapes")
        if set(shapes) != {"pred_logits", "pred_boxes"}:
            raise OptimizationContractError("detector raw output shapes are incomplete")
        logits_shape = _shape(shapes.get("pred_logits"), "detector pred_logits shape")
        boxes_shape = _shape(shapes.get("pred_boxes"), "detector pred_boxes shape")
        if logits_shape != [1, 900, 1] or boxes_shape != [1, 900, 4]:
            raise OptimizationContractError("detector raw output shapes differ")
        stages = _mapping(parity.get("stages"), "detector parity stages")
        if set(stages) != {"raw", "post_nms", "selected_rois"}:
            raise OptimizationContractError("detector parity stages are incomplete")
        stage_results = {
            stage: _validate_detector_stage(
                _mapping(stages.get(stage), f"detector parity {stage}"),
                score_tolerance=score_tolerance,
                logit_tolerance=logit_tolerance,
                box_tolerance=box_tolerance,
                minimum_iou=minimum_iou,
            )
            for stage in ("raw", "post_nms", "selected_rois")
        }
        expected_raw_count = logits_shape[1] * logits_shape[2]
        raw_stage = _mapping(stages.get("raw"), "detector raw parity")
        if (
            raw_stage.get("baseline_count") != expected_raw_count
            or raw_stage.get("candidate_count") != expected_raw_count
        ):
            raise OptimizationContractError("detector raw parity count differs")
        selected_stage = _mapping(
            stages.get("selected_rois"),
            "detector selected ROI parity",
        )
        if selected_stage.get("baseline_count") != 8 or selected_stage.get("candidate_count") != 8:
            raise OptimizationContractError("detector selected ROI count differs")
        measured_passed = all(stage_results.values())
        if parity.get("passed") is not measured_passed:
            raise OptimizationContractError("detector parity contradicts measurements")
    else:
        required = {
            "passed",
            "method",
            "absolute_tolerance",
            "max_absolute_difference",
            "output_sha256",
        }
        if set(parity) != required or parity.get("method") != "elementwise_absolute":
            raise OptimizationContractError("classifier parity method is invalid")
        _nonnegative_number(parity.get("absolute_tolerance"), "classifier tolerance")
        differences = _mapping(
            parity.get("max_absolute_difference"),
            "classifier differences",
        )
        if set(differences) != {"logits", "fused_embeddings", "roi_attention"}:
            raise OptimizationContractError("classifier differences are incomplete")
        difference_values = [
            _nonnegative_number(value, "classifier difference") for value in differences.values()
        ]
        measured_passed = all(value <= parity["absolute_tolerance"] for value in difference_values)
        if parity.get("passed") is not measured_passed:
            raise OptimizationContractError("classifier parity contradicts measurements")
    hashes = _mapping(parity.get("output_sha256"), "output hashes")
    expected_hashes = (
        {"pred_logits", "pred_boxes"}
        if model_id == DETECTOR_MODEL_ID
        else {"logits", "fused_embeddings", "roi_attention"}
    )
    if set(hashes) != expected_hashes or any(
        _SHA256.fullmatch(str(value)) is None for value in hashes.values()
    ):
        raise OptimizationContractError("candidate output identity is invalid")


def _validate_detector_stage(
    stage: Mapping[str, object],
    *,
    score_tolerance: float,
    logit_tolerance: float,
    box_tolerance: float,
    minimum_iou: float,
) -> bool:
    required = {
        "baseline_count",
        "candidate_count",
        "matched_pairs",
        "unmatched_baseline",
        "unmatched_candidate",
        "max_score_difference",
        "max_logit_difference",
        "max_box_coordinate_difference",
        "minimum_observed_iou",
    }
    if set(stage) != required:
        raise OptimizationContractError("detector parity stage is incomplete")
    baseline_count = _positive_integer(
        stage.get("baseline_count"),
        "detector baseline count",
    )
    candidate_count = _positive_integer(
        stage.get("candidate_count"),
        "detector candidate count",
    )
    matched_pairs = _index_pairs(
        stage.get("matched_pairs"),
        "detector matched pairs",
    )
    unmatched_baseline = _indexes(
        stage.get("unmatched_baseline"),
        "detector unmatched baseline",
    )
    unmatched_candidate = _indexes(
        stage.get("unmatched_candidate"),
        "detector unmatched candidate",
    )
    baseline_indexes = [left for left, _ in matched_pairs] + unmatched_baseline
    candidate_indexes = [right for _, right in matched_pairs] + unmatched_candidate
    if (
        len(set(baseline_indexes)) != len(baseline_indexes)
        or len(set(candidate_indexes)) != len(candidate_indexes)
        or sorted(baseline_indexes) != list(range(baseline_count))
        or sorted(candidate_indexes) != list(range(candidate_count))
    ):
        raise OptimizationContractError("detector parity matching is inconsistent")
    max_score_difference = _nonnegative_number(
        stage.get("max_score_difference"),
        "detector score difference",
    )
    max_logit_difference = _nonnegative_number(
        stage.get("max_logit_difference"),
        "detector logit difference",
    )
    max_box_difference = _nonnegative_number(
        stage.get("max_box_coordinate_difference"),
        "detector box difference",
    )
    observed_iou = _nonnegative_number(
        stage.get("minimum_observed_iou"),
        "detector observed IoU",
    )
    if observed_iou > 1.0:
        raise OptimizationContractError("detector observed IoU is invalid")
    return (
        not unmatched_baseline
        and not unmatched_candidate
        and max_score_difference <= score_tolerance
        and max_logit_difference <= logit_tolerance
        and max_box_difference <= box_tolerance
        and observed_iou >= minimum_iou
    )


def _validate_performance(candidate: Mapping[str, object], measured_runs: int) -> None:
    name = str(candidate.get("name", ""))
    performance = _mapping(candidate.get("performance"), f"candidate {name}.performance")
    required = {
        "latency_ms",
        "throughput_per_second",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    }
    if set(performance) != required:
        raise OptimizationContractError(f"candidate {name} performance is invalid")
    if candidate.get("status") == "failed":
        if any(value is not None for value in performance.values()):
            raise OptimizationContractError("failed candidate has performance claims")
        return
    latency = _mapping(performance.get("latency_ms"), f"candidate {name}.latency")
    if set(latency) != _LATENCY_KEYS:
        raise OptimizationContractError(f"candidate {name} latency is incomplete")
    if latency.get("count") != measured_runs:
        raise OptimizationContractError(f"candidate {name} latency count differs")
    ordered = [
        _positive_number(latency.get(field), f"candidate {name} latency {field}")
        for field in ("min", "p50", "p95", "p99", "max")
    ]
    if ordered != sorted(ordered):
        raise OptimizationContractError(f"candidate {name} latency order is invalid")
    _positive_number(latency.get("mean"), f"candidate {name} latency mean")
    _nonnegative_number(
        latency.get("population_stddev"),
        f"candidate {name} latency standard deviation",
    )
    _positive_number(
        performance.get("throughput_per_second"),
        f"candidate {name} throughput",
    )
    throughput = float(performance["throughput_per_second"])
    mean = float(latency["mean"])
    if not math.isclose(throughput, 1_000.0 / mean, rel_tol=1e-6):
        raise OptimizationContractError(f"candidate {name} throughput is inconsistent")
    allocated = _positive_integer(
        performance.get("peak_allocated_bytes"),
        f"candidate {name} allocated memory",
    )
    reserved = _positive_integer(
        performance.get("peak_reserved_bytes"),
        f"candidate {name} reserved memory",
    )
    if allocated > reserved:
        raise OptimizationContractError(f"candidate {name} memory is impossible")


def _validate_reliability(candidate: Mapping[str, object]) -> None:
    name = str(candidate.get("name", ""))
    reliability = _mapping(candidate.get("reliability"), f"candidate {name}.reliability")
    if set(reliability) != {"cuda_oom", "release"} or not isinstance(
        reliability.get("cuda_oom"),
        bool,
    ):
        raise OptimizationContractError(f"candidate {name} reliability is invalid")
    if candidate.get("status") != "failed" and reliability.get("cuda_oom") is not False:
        raise OptimizationContractError("successful candidate cannot claim CUDA OOM")
    release = _mapping(
        reliability.get("release"),
        f"candidate {name}.release",
    )
    required = {
        "status",
        "model_reference_alive_after_release",
        "allocated_after_release_bytes",
        "reserved_after_release_bytes",
        "method",
    }
    if set(release) != required:
        raise OptimizationContractError(f"candidate {name} release evidence is invalid")
    if release.get("status") == "measured":
        if not isinstance(release.get("model_reference_alive_after_release"), bool):
            raise OptimizationContractError("measured release reference result is invalid")
        allocated = _nonnegative_integer(
            release.get("allocated_after_release_bytes"),
            "release allocated VRAM",
        )
        reserved = _nonnegative_integer(
            release.get("reserved_after_release_bytes"),
            "release reserved VRAM",
        )
        if allocated > reserved:
            raise OptimizationContractError("release VRAM evidence is impossible")
        if release.get("method") != "weakref_and_cuda_allocator_after_release":
            raise OptimizationContractError("measured release method is invalid")
    elif release.get("status") == "unmeasured":
        if any(release.get(field) is not None for field in required - {"status"}):
            raise OptimizationContractError("unmeasured release result must be null")
    else:
        raise OptimizationContractError("candidate release measurement status is invalid")
    if candidate.get("status") != "failed" and not release_passed(release):
        promotion = _mapping(candidate.get("promotion"), "promotion")
        reasons = promotion.get("reasons")
        expected_reason = (
            "candidate_release_unmeasured"
            if release.get("status") != "measured"
            else "candidate_release_leak_observed"
        )
        if promotion.get("accepted") is not False or expected_reason not in reasons:
            raise OptimizationContractError("unclean release candidate did not fail closed")


def _validate_compile(candidate: Mapping[str, object]) -> None:
    name = str(candidate.get("name", ""))
    compile_evidence = _mapping(candidate.get("compile"), f"candidate {name}.compile")
    required = {
        "enabled",
        "compilation_ms",
        "recompilation_count",
        "graph_break_count",
        "fullgraph_required",
    }
    if set(compile_evidence) != required:
        raise OptimizationContractError(f"candidate {name} compile evidence is invalid")
    enabled = name == "compile"
    if (
        compile_evidence.get("enabled") is not enabled
        or compile_evidence.get("fullgraph_required") is not enabled
    ):
        raise OptimizationContractError(f"candidate {name} compile policy differs")
    if candidate.get("status") == "failed":
        if any(
            compile_evidence.get(field) is not None
            for field in ("compilation_ms", "recompilation_count", "graph_break_count")
        ):
            raise OptimizationContractError("failed candidate has compile measurements")
        return
    if enabled:
        _positive_number(compile_evidence.get("compilation_ms"), "compilation time")
    elif compile_evidence.get("compilation_ms") is not None:
        raise OptimizationContractError("non-compile candidate has compilation time")
    _nonnegative_integer(compile_evidence.get("recompilation_count"), "recompilations")
    _nonnegative_integer(compile_evidence.get("graph_break_count"), "graph breaks")


def _release_evidence(candidate: Mapping[str, object]) -> Mapping[str, object]:
    reliability = _mapping(candidate.get("reliability"), "candidate.reliability")
    return _mapping(reliability.get("release"), "candidate.reliability.release")


def release_leak(release: Mapping[str, object]) -> bool | None:
    if release.get("status") != "measured":
        return None
    return bool(
        release.get("model_reference_alive_after_release") is True
        or release.get("allocated_after_release_bytes") != 0
        or release.get("reserved_after_release_bytes") != 0
    )


def release_passed(release: Mapping[str, object]) -> bool:
    return release.get("status") == "measured" and release_leak(release) is False


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


def _nonnegative_number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OptimizationContractError(f"{path} must be non-negative")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise OptimizationContractError(f"{path} must be non-negative")
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


def _indexes(value: object, path: str) -> list[int]:
    if not isinstance(value, list) or any(
        isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in value
    ):
        raise OptimizationContractError(f"{path} is invalid")
    return value


def _index_pairs(value: object, path: str) -> list[tuple[int, int]]:
    if not isinstance(value, list) or any(
        not isinstance(pair, list)
        or len(pair) != 2
        or any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in pair)
        for pair in value
    ):
        raise OptimizationContractError(f"{path} is invalid")
    return [(pair[0], pair[1]) for pair in value]


def _shape(value: object, path: str) -> list[int]:
    if not isinstance(value, list) or any(
        isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in value
    ):
        raise OptimizationContractError(f"{path} is invalid")
    return value
