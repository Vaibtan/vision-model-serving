"""Structural and arithmetic validation for benchmark evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import re
from typing import Any

from vision_model_serving.validation._benchmark_primitives import (
    _DISTRIBUTION_FIELDS,
    _LIFECYCLE_RULES,
    _STARTUP_FIELDS,
    _TIMING_TOLERANCE_SECONDS,
    BenchmarkContractError,
    _finite_non_negative,
    _mapping,
    _non_negative_integer,
    _positive_integer,
    _sequence,
    assert_single_residency_snapshot,
    latency_distribution,
)


_SHA256 = re.compile(r"[0-9a-f]{64}")


def validate_startup_payload(value: object) -> None:
    startup = _mapping(value, "measurements.startup")
    if set(startup) != _STARTUP_FIELDS:
        raise BenchmarkContractError("startup measurements are incomplete")
    values = {
        name: _finite_non_negative(startup.get(name), f"startup.{name}") for name in _STARTUP_FIELDS
    }
    phase_total = values["artifact_verification_seconds"] + values["runtime_initialization_seconds"]
    if values["process_start_to_artifact_ready_seconds"] + _TIMING_TOLERANCE_SECONDS < phase_total:
        raise BenchmarkContractError("startup total is shorter than its startup phases")


def validate_lifecycle_payload(value: object, *, schema_version: int) -> None:
    lifecycle = _mapping(value, "measurements.lifecycle")
    if set(lifecycle) != set(_LIFECYCLE_RULES):
        raise BenchmarkContractError("lifecycle measurements are incomplete")
    for phase, rule in _LIFECYCLE_RULES.items():
        measurement_kind, mode, detector_reused, classifier_reused, resident = rule
        payload = _mapping(lifecycle.get(phase), f"lifecycle.{phase}")
        if set(payload) != {measurement_kind, "inventory"}:
            raise BenchmarkContractError(f"lifecycle.{phase} is incomplete")
        inventory = _mapping(payload.get("inventory"), f"lifecycle.{phase}.inventory")
        if not isinstance(inventory.get("manifest_id"), str) or not inventory.get("manifest_id"):
            raise BenchmarkContractError(f"lifecycle.{phase}.inventory manifest is missing")
        if not _sequence(inventory.get("models"), f"lifecycle.{phase}.models"):
            raise BenchmarkContractError(f"lifecycle.{phase}.inventory models are missing")
        runtime = _mapping(inventory.get("runtime"), f"lifecycle.{phase}.inventory.runtime")
        assert_single_residency_snapshot(runtime)
        if runtime.get("state") != "ready" or runtime.get("resident_models") != [resident]:
            raise BenchmarkContractError(f"lifecycle.{phase} has the wrong resident model")
        if measurement_kind == "sample":
            _validate_sample(
                payload.get("sample"),
                f"lifecycle.{phase}.sample",
                mode=mode,
                detector_reused=detector_reused,
                classifier_reused=classifier_reused,
                schema_version=schema_version,
            )
        else:
            _validate_aggregate(
                payload.get("aggregate"),
                f"lifecycle.{phase}.aggregate",
                mode=mode,
                detector_reused=detector_reused,
                classifier_reused=classifier_reused,
                schema_version=schema_version,
            )


def validate_resources_payload(value: object, *, schema_version: int) -> None:
    resources = _mapping(value, "measurements.resources")
    if set(resources) != {
        "before_operations",
        "after_operations",
        "cuda_oom_total",
        "nvidia_smi",
    }:
        raise BenchmarkContractError("resource measurements are incomplete")
    for name in ("before_operations", "after_operations"):
        if not _mapping(resources.get(name), f"resources.{name}"):
            raise BenchmarkContractError(f"resources.{name} is empty")
    if not _non_negative_integer(resources.get("cuda_oom_total")):
        raise BenchmarkContractError("resources.cuda_oom_total is invalid")
    sampler = _mapping(resources.get("nvidia_smi"), "resources.nvidia_smi")
    available = sampler.get("available")
    if not isinstance(available, bool):
        raise BenchmarkContractError("resource sampler availability is invalid")
    if not _positive_integer(sampler.get("interval_ms")):
        raise BenchmarkContractError("resource sampler interval is invalid")
    sample_count = sampler.get("sample_count")
    if not _non_negative_integer(sample_count):
        raise BenchmarkContractError("resource sampler count is invalid")
    distribution_names = {
        "gpu_utilization_percent",
        "memory_used_mib",
        "power_watts",
        "temperature_c",
    }
    accounting_fields: set[str] = set()
    if schema_version == 4:
        accounting_fields = {
            "duration_seconds",
            "attempt_count",
            "success_count",
            "failure_count",
        }
        duration = _finite_non_negative(
            sampler.get("duration_seconds"),
            "resources.nvidia_smi.duration_seconds",
        )
        attempts = sampler.get("attempt_count")
        successes = sampler.get("success_count")
        failures = sampler.get("failure_count")
        if (
            not _non_negative_integer(attempts)
            or not _non_negative_integer(successes)
            or not _non_negative_integer(failures)
            or attempts != successes + failures
            or sample_count != successes
        ):
            raise BenchmarkContractError("resource sampler accounting is invalid")
        if attempts and duration == 0.0:
            raise BenchmarkContractError("resource sampler duration must be positive")
        if available is not (successes > 0):
            raise BenchmarkContractError("resource sampler availability differs from successes")
    if available:
        if (
            set(sampler)
            != {
                "available",
                "interval_ms",
                "sample_count",
                *accounting_fields,
                *distribution_names,
            }
            or sample_count == 0
        ):
            raise BenchmarkContractError("resource sampler evidence is incomplete")
        distributions = {
            name: _validate_distribution(
                sampler.get(name),
                f"resources.nvidia_smi.{name}",
                expected_count=sample_count,
            )
            for name in distribution_names
        }
        if schema_version == 4:
            if distributions["gpu_utilization_percent"]["max"] > 100.0:
                raise BenchmarkContractError("resource sampler GPU utilization exceeds 100 percent")
            if any(
                distributions[name]["max"] <= 0.0
                for name in ("memory_used_mib", "power_watts", "temperature_c")
            ):
                raise BenchmarkContractError(
                    "resource sampler memory, power, and temperature maxima must be positive"
                )
    elif (
        set(sampler)
        != {
            "available",
            "interval_ms",
            "sample_count",
            *accounting_fields,
        }
        or sample_count
    ):
        raise BenchmarkContractError("unavailable resource sampler has measurements")


def validate_throughput_payload(
    value: object,
    concurrency: int,
    *,
    schema_version: int,
) -> None:
    payload = _mapping(value, f"throughput.{concurrency}")
    required = {
        "concurrency",
        "duration_seconds",
        "attempt_count",
        "success_count",
        "failure_count",
        "failure_codes",
        "successful_requests_per_second",
        "attempted_requests_per_second",
        "latency_seconds",
        "queue_wait_seconds",
    }
    if schema_version == 4:
        required.add("observed_prediction_sha256_counts")
    if set(payload) != required or payload.get("concurrency") != concurrency:
        raise BenchmarkContractError(f"throughput.{concurrency} payload is incomplete")
    attempts = payload.get("attempt_count")
    successes = payload.get("success_count")
    failures = payload.get("failure_count")
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (attempts, successes, failures)
        )
        or attempts <= 0
        or attempts != successes + failures
    ):
        raise BenchmarkContractError(f"throughput.{concurrency} failure accounting is invalid")
    duration = _finite_non_negative(
        payload.get("duration_seconds"), f"throughput.{concurrency}.duration_seconds"
    )
    if duration == 0.0:
        raise BenchmarkContractError(f"throughput.{concurrency}.duration_seconds must be positive")
    failure_codes = _mapping(
        payload.get("failure_codes"), f"throughput.{concurrency}.failure_codes"
    )
    if any(
        not isinstance(code, str) or not code or not _positive_integer(count)
        for code, count in failure_codes.items()
    ):
        raise BenchmarkContractError(f"throughput.{concurrency} failure codes are invalid")
    if sum(failure_codes.values()) != failures:
        raise BenchmarkContractError(f"throughput.{concurrency} failure codes are incomplete")
    if schema_version == 4:
        _validate_prediction_digest_counts(
            payload.get("observed_prediction_sha256_counts"),
            f"throughput.{concurrency}.observed_prediction_sha256_counts",
            expected_detector_count=successes,
        )
    expected_rates = {
        "successful_requests_per_second": successes / duration,
        "attempted_requests_per_second": attempts / duration,
    }
    for name, expected in expected_rates.items():
        observed = _finite_non_negative(payload.get(name), f"throughput.{concurrency}.{name}")
        if not math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise BenchmarkContractError(
                f"throughput.{concurrency}.{name} differs from counts and duration"
            )
    distributions: dict[str, Mapping[str, Any]] = {}
    for name in ("latency_seconds", "queue_wait_seconds"):
        raw_distribution = payload.get(name)
        if successes == 0:
            if raw_distribution is not None:
                raise BenchmarkContractError(
                    f"throughput.{concurrency}.{name} must be null without successes"
                )
            continue
        distributions[name] = _validate_distribution(
            raw_distribution,
            f"throughput.{concurrency}.{name}",
            expected_count=successes,
        )
    if successes and schema_version == 4:
        latency = distributions["latency_seconds"]
        queue_wait = distributions["queue_wait_seconds"]
        if latency["max"] > duration + _TIMING_TOLERANCE_SECONDS:
            raise BenchmarkContractError(
                f"throughput.{concurrency} latency exceeds measurement duration"
            )
        for name in ("min", "p50", "p95", "p99", "max", "mean"):
            if queue_wait[name] > latency[name] + _TIMING_TOLERANCE_SECONDS:
                raise BenchmarkContractError(
                    f"throughput.{concurrency} queue wait exceeds request latency"
                )


def _validate_distribution(
    value: object,
    path: str,
    *,
    expected_count: int | None = None,
) -> Mapping[str, Any]:
    distribution = _mapping(value, path)
    if set(distribution) != _DISTRIBUTION_FIELDS:
        raise BenchmarkContractError(f"{path} is incomplete")
    count = distribution.get("count")
    if not _positive_integer(count):
        raise BenchmarkContractError(f"{path}.count must be positive")
    if expected_count is not None and count != expected_count:
        raise BenchmarkContractError(f"{path}.count differs from its samples")
    numbers = {
        name: _finite_non_negative(distribution.get(name), f"{path}.{name}")
        for name in _DISTRIBUTION_FIELDS - {"count"}
    }
    if not (numbers["min"] <= numbers["p50"] <= numbers["p95"] <= numbers["p99"] <= numbers["max"]):
        raise BenchmarkContractError(f"{path} percentiles are not monotonic")
    mean_below_min = numbers["mean"] < numbers["min"] and not math.isclose(
        numbers["mean"], numbers["min"], rel_tol=1e-12, abs_tol=1e-15
    )
    mean_above_max = numbers["mean"] > numbers["max"] and not math.isclose(
        numbers["mean"], numbers["max"], rel_tol=1e-12, abs_tol=1e-15
    )
    if mean_below_min or mean_above_max:
        raise BenchmarkContractError(f"{path}.mean is outside the observed range")
    return distribution


def _validate_distribution_matches_samples(
    value: object,
    path: str,
    samples: Sequence[float],
) -> None:
    distribution = _validate_distribution(
        value,
        path,
        expected_count=len(samples),
    )
    expected = latency_distribution(samples)
    for name, expected_value in expected.items():
        observed = distribution.get(name)
        if name == "count":
            matches = observed == expected_value
        else:
            matches = isinstance(observed, (int, float)) and math.isclose(
                float(observed),
                float(expected_value),
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        if not matches:
            raise BenchmarkContractError(f"{path}.{name} differs from its samples")


def _validate_sample(
    value: object,
    path: str,
    *,
    mode: str,
    detector_reused: bool,
    classifier_reused: bool | None,
    schema_version: int,
) -> Mapping[str, Any]:
    sample = _mapping(value, path)
    required = {
        "mode",
        "wall_seconds",
        "queue_wait_seconds",
        "states",
        "stages_seconds",
        "http_queue_overhead_seconds",
        "lifecycle",
        "peak_reserved_bytes",
    }
    if schema_version == 4:
        required.update(
            {
                "detector_prediction_sha256",
                "classifier_prediction_sha256",
            }
        )
    if set(sample) != required or sample.get("mode") != mode:
        raise BenchmarkContractError(f"{path} is incomplete")
    wall_seconds = _finite_non_negative(sample.get("wall_seconds"), f"{path}.wall_seconds")
    queue_wait_seconds = _finite_non_negative(
        sample.get("queue_wait_seconds"),
        f"{path}.queue_wait_seconds",
    )
    if schema_version == 4 and queue_wait_seconds > wall_seconds + _TIMING_TOLERANCE_SECONDS:
        raise BenchmarkContractError(f"{path} queue wait exceeds wall time")
    overhead_seconds = _finite_non_negative(
        sample.get("http_queue_overhead_seconds"),
        f"{path}.http_queue_overhead_seconds",
    )
    states = _sequence(sample.get("states"), f"{path}.states")
    if (
        not states
        or states[-1] != "succeeded"
        or any(state not in {"submitted", "queued", "running", "succeeded"} for state in states)
    ):
        raise BenchmarkContractError(f"{path}.states is not a successful lifecycle")
    stages = _mapping(sample.get("stages_seconds"), f"{path}.stages_seconds")
    if "pipeline_total" not in stages:
        raise BenchmarkContractError(f"{path}.stages_seconds is incomplete")
    pipeline_seconds = _finite_non_negative(
        stages.get("pipeline_total"),
        f"{path}.stages_seconds.pipeline_total",
    )
    for name, duration in stages.items():
        if not isinstance(name, str) or not name:
            raise BenchmarkContractError(f"{path}.stages_seconds name is invalid")
        if duration is not None:
            stage_seconds = _finite_non_negative(
                duration,
                f"{path}.stages_seconds.{name}",
            )
            if (
                schema_version == 4
                and name != "pipeline_total"
                and stage_seconds > pipeline_seconds + _TIMING_TOLERANCE_SECONDS
            ):
                raise BenchmarkContractError(
                    f"{path}.stages_seconds.{name} exceeds the pipeline total"
                )
    difference = wall_seconds - pipeline_seconds
    if difference < -_TIMING_TOLERANCE_SECONDS:
        raise BenchmarkContractError(
            f"{path} pipeline timing exceeds wall time beyond the 1 ms tolerance"
        )
    expected_overhead = max(0.0, difference)
    if not math.isclose(
        overhead_seconds,
        expected_overhead,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise BenchmarkContractError(
            f"{path}.http_queue_overhead_seconds differs from wall minus pipeline"
        )
    expected_lifecycle = {
        "detector_reused": detector_reused,
        "classifier_reused": classifier_reused,
    }
    if _mapping(sample.get("lifecycle"), f"{path}.lifecycle") != expected_lifecycle:
        raise BenchmarkContractError(f"{path}.lifecycle is invalid")
    if not _non_negative_integer(sample.get("peak_reserved_bytes")):
        raise BenchmarkContractError(f"{path}.peak_reserved_bytes is invalid")
    if schema_version == 4:
        if _SHA256.fullmatch(str(sample.get("detector_prediction_sha256", ""))) is None:
            raise BenchmarkContractError(f"{path}.detector_prediction_sha256 is invalid")
        classifier_digest = sample.get("classifier_prediction_sha256")
        if mode == "full":
            if _SHA256.fullmatch(str(classifier_digest or "")) is None:
                raise BenchmarkContractError(f"{path}.classifier_prediction_sha256 is invalid")
        elif classifier_digest is not None:
            raise BenchmarkContractError(
                f"{path}.classifier_prediction_sha256 must be null for detection"
            )
    return sample


def _validate_aggregate(
    value: object,
    path: str,
    *,
    mode: str,
    detector_reused: bool,
    classifier_reused: bool | None,
    schema_version: int,
) -> None:
    aggregate = _mapping(value, path)
    required = {
        "samples",
        "wall_seconds",
        "queue_wait_seconds",
        "stages_seconds",
        "maximum_peak_reserved_bytes",
    }
    if set(aggregate) != required:
        raise BenchmarkContractError(f"{path} is incomplete")
    raw_samples = _sequence(aggregate.get("samples"), f"{path}.samples")
    if not raw_samples:
        raise BenchmarkContractError(f"{path}.samples is empty")
    samples = [
        _validate_sample(
            sample,
            f"{path}.samples[{index}]",
            mode=mode,
            detector_reused=detector_reused,
            classifier_reused=classifier_reused,
            schema_version=schema_version,
        )
        for index, sample in enumerate(raw_samples)
    ]
    for name in ("wall_seconds", "queue_wait_seconds"):
        _validate_distribution_matches_samples(
            aggregate.get(name),
            f"{path}.{name}",
            [float(sample[name]) for sample in samples],
        )
    stage_distributions = _mapping(aggregate.get("stages_seconds"), f"{path}.stages_seconds")
    if not stage_distributions:
        raise BenchmarkContractError(f"{path}.stages_seconds is empty")
    expected_stage_names = {
        name
        for sample in samples
        for name, duration in _mapping(sample["stages_seconds"], "sample stages").items()
        if duration is not None
    }
    if set(stage_distributions) != expected_stage_names:
        raise BenchmarkContractError(f"{path}.stages_seconds differs from its samples")
    for name, distribution in stage_distributions.items():
        stage_samples: list[float] = []
        for sample in samples:
            value = _mapping(sample["stages_seconds"], "sample stages").get(name)
            if value is not None:
                stage_samples.append(float(value))
        _validate_distribution_matches_samples(
            distribution,
            f"{path}.stages_seconds.{name}",
            stage_samples,
        )
    observed_maximum = max(int(sample["peak_reserved_bytes"]) for sample in samples)
    if aggregate.get("maximum_peak_reserved_bytes") != observed_maximum:
        raise BenchmarkContractError(f"{path} peak memory differs from its samples")


def _validate_prediction_digest_counts(
    value: object,
    path: str,
    *,
    expected_detector_count: int,
) -> None:
    counts = _mapping(value, path)
    if set(counts) != {"detector", "classifier"}:
        raise BenchmarkContractError(f"{path} is incomplete")
    detector = _mapping(counts.get("detector"), f"{path}.detector")
    classifier = _mapping(counts.get("classifier"), f"{path}.classifier")
    if classifier:
        raise BenchmarkContractError(f"{path}.classifier must be empty for detection")
    if (
        any(
            _SHA256.fullmatch(str(digest)) is None or not _positive_integer(count)
            for digest, count in detector.items()
        )
        or sum(detector.values()) != expected_detector_count
    ):
        raise BenchmarkContractError(f"{path} digest counts differ from successes")
