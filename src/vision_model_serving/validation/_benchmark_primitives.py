"""Shared primitives for benchmark evidence production and validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import statistics
from typing import Any

from vision_model_serving.model_ids import CLASSIFIER_MODEL_ID, DETECTOR_MODEL_ID


_DISTRIBUTION_FIELDS = {
    "count",
    "min",
    "p50",
    "p95",
    "p99",
    "max",
    "mean",
    "population_stddev",
}
_STARTUP_FIELDS = {
    "process_start_to_artifact_ready_seconds",
    "artifact_verification_seconds",
    "runtime_initialization_seconds",
}
_TIMING_TOLERANCE_SECONDS = 0.001
_LIFECYCLE_RULES = {
    "cold_full": ("sample", "full", False, False, CLASSIFIER_MODEL_ID),
    "switch_to_detection": (
        "sample",
        "detection",
        False,
        None,
        DETECTOR_MODEL_ID,
    ),
    "warm_detection": (
        "aggregate",
        "detection",
        True,
        None,
        DETECTOR_MODEL_ID,
    ),
    "full_after_detection": (
        "sample",
        "full",
        True,
        False,
        CLASSIFIER_MODEL_ID,
    ),
    "repeated_full": (
        "aggregate",
        "full",
        False,
        False,
        CLASSIFIER_MODEL_ID,
    ),
    "final_full": (
        "sample",
        "full",
        True,
        False,
        CLASSIFIER_MODEL_ID,
    ),
}


class BenchmarkContractError(ValueError):
    """Raised when benchmark evidence is incomplete or internally inconsistent."""


def latency_distribution(samples: Sequence[float]) -> dict[str, float | int]:
    """Return the Spec-mandated population statistics using nearest-rank percentiles."""

    values = tuple(float(value) for value in samples)
    if not values or any(not math.isfinite(value) or value < 0.0 for value in values):
        raise BenchmarkContractError("latency samples must be finite non-negative values")
    ordered = sorted(values)
    return {
        "count": len(values),
        "min": ordered[0],
        "p50": _nearest_rank(ordered, 0.50),
        "p95": _nearest_rank(ordered, 0.95),
        "p99": _nearest_rank(ordered, 0.99),
        "max": ordered[-1],
        "mean": statistics.fmean(values),
        "population_stddev": statistics.pstdev(values),
    }


def assert_single_residency_snapshot(runtime: Mapping[str, object]) -> None:
    """Fail when an inventory snapshot can describe overlapping model residency."""

    if not isinstance(runtime, Mapping):
        raise BenchmarkContractError("runtime snapshot must be an object")
    state = runtime.get("state")
    active = runtime.get("active_model")
    residents = runtime.get("resident_models")
    if state not in {
        "unloaded",
        "loading",
        "ready",
        "draining",
        "unloading",
        "failed",
    }:
        raise BenchmarkContractError("runtime snapshot state is invalid")
    if (
        not isinstance(residents, (list, tuple))
        or len(residents) > 1
        or any(not isinstance(model, str) or not model for model in residents)
    ):
        raise BenchmarkContractError("runtime snapshot exceeds single residency")
    expected = [] if active is None else [active]
    if list(residents) != expected:
        raise BenchmarkContractError("active and resident model differ")
    if state == "ready" and not residents:
        raise BenchmarkContractError("ready runtime has no resident model")
    if state in {"unloaded", "loading", "failed"} and residents:
        raise BenchmarkContractError(f"{state} runtime retained a model")


def _nearest_rank(ordered: Sequence[float], quantile: float) -> float:
    return ordered[max(0, math.ceil(len(ordered) * quantile) - 1)]


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BenchmarkContractError(f"{path} must be an object")
    return value


def _sequence(value: object, path: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise BenchmarkContractError(f"{path} must be an array")
    return value


def _positive_integer(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _non_negative_integer(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _finite_non_negative(value: object, path: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise BenchmarkContractError(f"{path} must be finite and non-negative")
    return float(value)
