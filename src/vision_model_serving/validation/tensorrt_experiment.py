"""Orchestrate and conclude fail-closed TensorRT validation experiments."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
from pathlib import Path
import re
from typing import Any, Protocol

from vision_model_serving.model_ids import CLASSIFIER_MODEL_ID, DETECTOR_MODEL_ID
from vision_model_serving.validation._tensorrt_measurements import (
    TorchTensorRtMeasurements,
    load_classifier_inputs,
)
from vision_model_serving.validation.evidence import sanitize_error_detail


TENSORRT_REPORT_SCHEMA_VERSION = 2
TENSORRT_VALIDATION_BOUNDARY = (
    "FP32/TF32-disabled TensorRT feasibility with a static token-width-5 "
    "diagnostic engine and a separately gated 2..90 production profile "
    "on one NVIDIA L4 and one public DICOM. It is not clinical or "
    "cross-hardware evidence."
)

_COMMIT = re.compile(r"[0-9a-f]{40}")

TensorRtEnvironment = Mapping[str, object] | Callable[[], Mapping[str, object]]


class TensorRtMeasurements(Protocol):
    """External GPU measurements required by the experiment orchestrator."""

    def measure_detector(self) -> Mapping[str, object]: ...

    def release_detector(self) -> None: ...

    def measure_classifier(self) -> Mapping[str, object]: ...


@dataclass(frozen=True, slots=True)
class TensorRtReportContext:
    """Immutable identity and environment attached to one experiment report."""

    revision: str
    environment: TensorRtEnvironment
    measured_at: str | None = None
    validation_boundary: str = TENSORRT_VALIDATION_BOUNDARY

    def __post_init__(self) -> None:
        if _COMMIT.fullmatch(self.revision) is None:
            raise ValueError("TensorRT report revision must be a full lowercase commit")
        if self.measured_at is not None and not self.measured_at:
            raise ValueError("TensorRT report measured_at cannot be blank")
        if isinstance(self.environment, Mapping):
            if not self.environment:
                raise ValueError("TensorRT report environment is required")
        elif not callable(self.environment):
            raise ValueError("TensorRT report environment must be a mapping or supplier")
        if not self.validation_boundary:
            raise ValueError("TensorRT validation boundary is required")


@dataclass(frozen=True, slots=True)
class TensorRtExperimentResult:
    """Machine-readable and human-readable conclusions from one experiment."""

    report: dict[str, Any]
    markdown: str


def run_tensorrt_experiment(
    context: TensorRtReportContext,
    measurements: TensorRtMeasurements,
    *,
    output_dir: Path,
    failure_roots: Iterable[Path],
) -> TensorRtExperimentResult:
    """Measure detector then classifier and return one fail-closed conclusion.

    The classifier measurement cannot begin until detector residency is released.
    Its diagnostic plan is deleted on both success and failure; no production
    TensorRT selection or eager fallback is represented by this interface.
    """

    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    sanitization_roots = (destination, *tuple(failure_roots))
    candidate_plan = destination / "mmbcd-fp32.candidate.plan"
    try:
        candidate_plan.unlink(missing_ok=True)
        detector = dict(measurements.measure_detector())
        measurements.release_detector()
        try:
            classifier = dict(measurements.measure_classifier())
        except Exception as error:
            failure_path = destination / "mmbcd-tensorrt-failure.txt"
            failure_path.write_text(
                sanitize_error_detail(error, sanitization_roots),
                encoding="utf-8",
            )
            classifier = _classifier_failure(error, failure_path)
    finally:
        candidate_plan.unlink(missing_ok=True)
    return _conclude_tensorrt_experiment(context, detector, classifier)


def _conclude_tensorrt_experiment(
    context: TensorRtReportContext,
    detector: Mapping[str, object],
    classifier: Mapping[str, object],
) -> TensorRtExperimentResult:
    """Build the schema-v2 report and Markdown from measured model outcomes."""

    detector_result = dict(detector)
    classifier_result = dict(classifier)
    detector_eligible = _detector_eligible(detector_result)
    classifier_eligible = _classifier_eligible(classifier_result)
    detector_result["decision"] = "go" if detector_eligible else "stop"
    classifier_result["decision"] = "go" if classifier_eligible else "stop"
    decision = _experiment_decision(detector_eligible, classifier_eligible)
    report: dict[str, Any] = {
        "schema_version": TENSORRT_REPORT_SCHEMA_VERSION,
        "revision": context.revision,
        "measured_at": context.measured_at or datetime.now(UTC).isoformat(),
        "environment": _resolve_environment(context.environment),
        "models": {
            DETECTOR_MODEL_ID: detector_result,
            CLASSIFIER_MODEL_ID: classifier_result,
        },
        "decision": decision,
        "production_selection": {
            "backend": "pytorch-eager",
            "reason": _production_reason(
                decision,
                detector_result,
                classifier_result,
            ),
        },
        "validation_boundary": context.validation_boundary,
    }
    return TensorRtExperimentResult(
        report=report,
        markdown=_render_markdown(report),
    )


def _experiment_decision(detector_eligible: bool, classifier_eligible: bool) -> str:
    if classifier_eligible and not detector_eligible:
        return "partial"
    if classifier_eligible and detector_eligible:
        return "go"
    return "stop"


def _resolve_environment(environment: TensorRtEnvironment) -> dict[str, object]:
    resolved = environment() if callable(environment) else environment
    if not isinstance(resolved, Mapping) or not resolved:
        raise ValueError("TensorRT report environment supplier must return a non-empty mapping")
    return dict(resolved)


def _detector_eligible(result: Mapping[str, object]) -> bool:
    return (
        _full_compilation_passed(result)
        and result.get("strict_export") is True
        and result.get("engine_built") is True
        and result.get("tensorrt_only_runtime_passed") is True
        and result.get("parity_passed") is True
        and result.get("performance_threshold_passed") is True
    )


def _classifier_eligible(result: Mapping[str, object]) -> bool:
    parity = result.get("parity")
    performance = result.get("performance")
    return (
        _full_compilation_passed(result)
        and result.get("strict_export") is True
        and result.get("engine_built") is True
        and result.get("tensorrt_only_runtime_passed") is True
        and isinstance(parity, Mapping)
        and parity.get("passed") is True
        and isinstance(performance, Mapping)
        and performance.get("promotion_threshold_passed") is True
        and _classifier_profile_passed(result.get("shape_coverage"))
    )


def _full_compilation_passed(result: Mapping[str, object]) -> bool:
    return (
        result.get("dryrun_completed") is True
        and result.get("require_full_compilation") is True
        and result.get("pytorch_partition_count") == 0
        and result.get("unsupported_operators") == []
    )


def _classifier_profile_passed(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    required_profile = value.get("required_profile")
    dynamic_export = value.get("dynamic_strict_export")
    return (
        value.get("passed") is True
        and required_profile
        == {
            "min_token_width": 2,
            "opt_token_width": 5,
            "max_token_width": 90,
        }
        and value.get("static_fixture_token_width") == 5
        and isinstance(dynamic_export, Mapping)
        and dynamic_export.get("passed") is True
        and "failure_code" in value
        and value.get("failure_code") is None
    )


def _production_reason(
    decision: str,
    detector: Mapping[str, object],
    classifier: Mapping[str, object],
) -> str:
    if decision == "go":
        return (
            "Both strict production profiles passed the isolated TensorRT gates. "
            "Production remains eager FP32 until TensorRT is deliberately selected "
            "and the packaged single-residency, restart, and benchmark gates pass."
        )
    if decision == "partial":
        return (
            "The classifier production profile passed, but detector full coverage "
            "did not. This is a PARTIAL result, not whole-pipeline TensorRT; "
            "production remains eager FP32 and no fallback is enabled."
        )
    classifier_shape_coverage = classifier.get("shape_coverage")
    classifier_profile_failed = (
        isinstance(classifier_shape_coverage, Mapping)
        and classifier_shape_coverage.get("passed") is False
    )
    classifier_reason = (
        "the classifier's required token-width 2..90 profile did not pass"
        if classifier_profile_failed
        else "the classifier's full-engine gates did not all pass"
    )
    detector_reason = (
        "detector full coverage did not pass"
        if detector.get("decision") != "go"
        else "detector full coverage passed"
    )
    return (
        f"TensorRT was not promoted because {classifier_reason} and "
        f"{detector_reason}. Production remains eager FP32 and no fallback is "
        "enabled."
    )


def _classifier_failure(error: Exception, failure_path: Path) -> dict[str, object]:
    return {
        "strict_export": False,
        "dryrun_completed": False,
        "require_full_compilation": True,
        "pytorch_partition_count": None,
        "unsupported_operators": [],
        "engine_built": False,
        "tensorrt_only_runtime_passed": False,
        "parity": {"passed": False},
        "performance": {"promotion_threshold_passed": False},
        "decision": "stop",
        "failure_code": f"classifier_tensorrt_failed:{type(error).__name__}",
        "failure_report_sha256": _sha256_file(failure_path),
    }


def _render_markdown(report: Mapping[str, Any]) -> str:
    models = report["models"]
    detector = models[DETECTOR_MODEL_ID]
    classifier = models[CLASSIFIER_MODEL_ID]
    detector_row = " | ".join(
        (
            f"| {DETECTOR_MODEL_ID}",
            str(detector["strict_export"]),
            str(detector["dryrun_completed"]),
            str(detector["engine_built"]),
            str(detector["parity_passed"]),
            str(detector["performance_threshold_passed"]),
            f"{detector['decision']} |",
        )
    )
    classifier_row = " | ".join(
        (
            f"| {CLASSIFIER_MODEL_ID}",
            str(classifier["strict_export"]),
            str(classifier["dryrun_completed"]),
            str(classifier["engine_built"]),
            str(classifier["parity"]["passed"]),
            str(classifier["performance"]["promotion_threshold_passed"]),
            f"{classifier['decision']} |",
        )
    )
    plugin_note = (
        "Detector strict export failed before Torch-TensorRT coverage analysis; "
        "plugin requirement was not reached."
        if detector["plugin_requirement_status"] == "not_reached"
        else (
            "Detector coverage measured a required TensorRT plugin."
            if detector["plugin_required"] is True
            else "Detector coverage did not establish a required TensorRT plugin."
        )
    )
    return "\n".join(
        (
            "# TensorRT spike",
            "",
            f"Revision: `{report['revision']}`",
            f"Decision: **{str(report['decision']).upper()}**",
            "",
            (
                "| Model | Strict export | Full compilation | Engine | Parity | "
                "Performance gate | Decision |"
            ),
            "| --- | --- | --- | --- | --- | --- | --- |",
            detector_row,
            classifier_row,
            "",
            plugin_note,
            "",
            str(report["production_selection"]["reason"]),
            "",
            str(report["validation_boundary"]),
            "",
        )
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "TENSORRT_REPORT_SCHEMA_VERSION",
    "TENSORRT_VALIDATION_BOUNDARY",
    "TensorRtExperimentResult",
    "TensorRtMeasurements",
    "TensorRtReportContext",
    "TorchTensorRtMeasurements",
    "load_classifier_inputs",
    "run_tensorrt_experiment",
]
