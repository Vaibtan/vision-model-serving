"""Schema-v3 benchmark statistics, validation, and report rendering."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import re
import statistics
from typing import Any


_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
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
_GATES = {
    "environment_complete",
    "revision_exact_clean",
    "identity_exact",
    "artifact_ready",
    "single_residency_all_snapshots",
    "golden_outputs_all_successes",
    "measurement_matrix_complete",
    "failure_accounting_complete",
    "concurrency_1_no_failures",
    "each_concurrency_has_success",
    "no_cuda_oom",
    "resource_sampling_complete",
}


class BenchmarkContractError(ValueError):
    """Raised when benchmark evidence is incomplete or internally inconsistent."""


def latency_distribution(samples: Sequence[float]) -> dict[str, float | int]:
    """Return the Spec-mandated population statistics using nearest-rank percentiles."""

    values = tuple(float(value) for value in samples)
    if not values or any(not math.isfinite(value) or value < 0.0 for value in values):
        raise BenchmarkContractError(
            "latency samples must be finite non-negative values"
        )
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


@dataclass(frozen=True, slots=True)
class ThroughputMeasurement:
    concurrency: int
    duration_seconds: float
    attempts: int
    successful_latencies: tuple[float, ...]
    queue_waits: tuple[float, ...]
    failure_codes: Mapping[str, int]

    def __post_init__(self) -> None:
        if self.concurrency not in {1, 2, 4}:
            raise BenchmarkContractError("offered concurrency must be 1, 2, or 4")
        if not math.isfinite(self.duration_seconds) or self.duration_seconds <= 0:
            raise BenchmarkContractError("throughput duration must be positive")
        if (
            isinstance(self.attempts, bool)
            or not isinstance(self.attempts, int)
            or self.attempts <= 0
        ):
            raise BenchmarkContractError("throughput attempts must be positive")
        if len(self.successful_latencies) != len(self.queue_waits):
            raise BenchmarkContractError(
                "queue-wait samples must match successful latency samples"
            )
        if self.successful_latencies:
            latency_distribution(self.successful_latencies)
            latency_distribution(self.queue_waits)
        if not isinstance(self.failure_codes, Mapping) or any(
            not isinstance(code, str)
            or not code
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            for code, count in self.failure_codes.items()
        ):
            raise BenchmarkContractError("failure codes must contain positive counts")
        failures = sum(self.failure_codes.values())
        if len(self.successful_latencies) + failures != self.attempts:
            raise BenchmarkContractError(
                "attempts must equal successful requests plus accounted failures"
            )

    def as_dict(self) -> dict[str, object]:
        successes = len(self.successful_latencies)
        failures = sum(self.failure_codes.values())
        return {
            "concurrency": self.concurrency,
            "duration_seconds": self.duration_seconds,
            "attempt_count": self.attempts,
            "success_count": successes,
            "failure_count": failures,
            "failure_codes": dict(sorted(self.failure_codes.items())),
            "successful_requests_per_second": successes / self.duration_seconds,
            "attempted_requests_per_second": self.attempts / self.duration_seconds,
            "latency_seconds": (
                latency_distribution(self.successful_latencies)
                if self.successful_latencies
                else None
            ),
            "queue_wait_seconds": (
                latency_distribution(self.queue_waits)
                if self.queue_waits
                else None
            ),
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
    if not isinstance(residents, (list, tuple)) or len(residents) > 1 or any(
        not isinstance(model, str) or not model for model in residents
    ):
        raise BenchmarkContractError("runtime snapshot exceeds single residency")
    expected = [] if active is None else [active]
    if list(residents) != expected:
        raise BenchmarkContractError("active and resident model differ")
    if state == "ready" and not residents:
        raise BenchmarkContractError("ready runtime has no resident model")
    if state in {"unloaded", "loading", "failed"} and residents:
        raise BenchmarkContractError(f"{state} runtime retained a model")


def validate_benchmark_record(record: Mapping[str, object]) -> None:
    """Validate the committed schema-v3 evidence contract before publication."""

    required = {
        "schema_version",
        "measured_at",
        "harness",
        "environment",
        "identity",
        "policy",
        "measurements",
        "gates",
        "outcome",
        "validation_boundary",
    }
    if not isinstance(record, Mapping) or set(record) != required:
        raise BenchmarkContractError("benchmark record top-level schema is invalid")
    if record.get("schema_version") != 3:
        raise BenchmarkContractError("benchmark schema_version must be 3")
    harness = _mapping(record.get("harness"), "harness")
    if _COMMIT.fullmatch(str(harness.get("revision", ""))) is None:
        raise BenchmarkContractError("benchmark revision must be a full commit")
    if harness.get("revision_clean") is not True:
        raise BenchmarkContractError("benchmark revision must be clean")
    if _SHA256.fullmatch(str(harness.get("script_sha256", ""))) is None:
        raise BenchmarkContractError("benchmark script SHA-256 is invalid")
    if harness.get("percentile_method") != "nearest_rank":
        raise BenchmarkContractError("benchmark percentile method is invalid")
    if harness.get("stddev_method") != "population":
        raise BenchmarkContractError("benchmark stddev method is invalid")

    environment = _mapping(record.get("environment"), "environment")
    for name in ("hardware", "software", "native_operator", "container"):
        if not _mapping(environment.get(name), f"environment.{name}"):
            raise BenchmarkContractError(f"environment.{name} must not be empty")
    identity = _mapping(record.get("identity"), "identity")
    if _SHA256.fullmatch(str(identity.get("manifest_sha256", ""))) is None:
        raise BenchmarkContractError("manifest SHA-256 is invalid")
    if _SHA256.fullmatch(str(identity.get("dicom_sha256", ""))) is None:
        raise BenchmarkContractError("DICOM SHA-256 is invalid")

    policy = _mapping(record.get("policy"), "policy")
    if policy.get("offered_concurrency") != [1, 2, 4]:
        raise BenchmarkContractError("offered concurrency must be exactly 1, 2, 4")
    measurements = _mapping(record.get("measurements"), "measurements")
    if set(measurements) != {"startup", "lifecycle", "throughput", "resources"}:
        raise BenchmarkContractError("benchmark measurement matrix is incomplete")
    throughput = _mapping(measurements.get("throughput"), "throughput")
    if set(throughput) != {"1", "2", "4"}:
        raise BenchmarkContractError("throughput must cover concurrency 1, 2, and 4")
    for concurrency in (1, 2, 4):
        _validate_throughput_payload(throughput[str(concurrency)], concurrency)

    gates = _mapping(record.get("gates"), "gates")
    if set(gates) != _GATES or not all(
        isinstance(value, bool) for value in gates.values()
    ):
        raise BenchmarkContractError("benchmark gates are incomplete")
    outcome = record.get("outcome")
    if outcome not in {"passed", "failed"}:
        raise BenchmarkContractError("benchmark outcome is invalid")
    if (outcome == "passed") != all(gates.values()):
        raise BenchmarkContractError("benchmark outcome differs from promotion gates")
    if not isinstance(record.get("validation_boundary"), str) or not record.get(
        "validation_boundary"
    ):
        raise BenchmarkContractError("benchmark validation boundary is required")


def render_benchmark_markdown(record: Mapping[str, object]) -> str:
    """Render the validated record without inventing or dropping measurements."""

    validate_benchmark_record(record)
    throughput = record["measurements"]["throughput"]
    sections: list[str] = []
    for concurrency in (1, 2, 4):
        item = throughput[str(concurrency)]
        latency = item["latency_seconds"]
        sections.append(
            "| Concurrency {concurrency} | {attempts} | {successes} | "
            "{failures} | {p50:.6f} | {p95:.6f} | {p99:.6f} | "
            "{throughput:.6f} |".format(
                concurrency=concurrency,
                attempts=item["attempt_count"],
                successes=item["success_count"],
                failures=item["failure_count"],
                p50=(latency["p50"] if latency is not None else float("nan")),
                p95=(latency["p95"] if latency is not None else float("nan")),
                p99=(latency["p99"] if latency is not None else float("nan")),
                throughput=item["successful_requests_per_second"],
            )
        )
    gate_rows = "\n".join(
        f"| {name.replace('_', ' ')} | {'PASS' if value else 'FAIL'} |"
        for name, value in record["gates"].items()
    )
    return f"""# Packaged NVIDIA L4 benchmark

Revision: `{record['harness']['revision']}`<br>
Measured at: `{record['measured_at']}`<br>
Outcome: **{str(record['outcome']).upper()}**

## Throughput and failure accounting

| Offered load | Attempts | Successes | Failures | p50 s | p95 s | p99 s | successful requests/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(sections)}

## Promotion gates

| Gate | Result |
| --- | --- |
{gate_rows}

The `single residency all snapshots` gate proves that no recorded runtime
inventory contained more than one accelerator-resident model.

## Validation boundary

{record['validation_boundary']}
"""


def _nearest_rank(ordered: Sequence[float], quantile: float) -> float:
    return ordered[max(0, math.ceil(len(ordered) * quantile) - 1)]


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BenchmarkContractError(f"{path} must be an object")
    return value


def _validate_throughput_payload(value: object, concurrency: int) -> None:
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
    if set(payload) != required or payload.get("concurrency") != concurrency:
        raise BenchmarkContractError(
            f"throughput.{concurrency} payload is incomplete"
        )
    attempts = payload.get("attempt_count")
    successes = payload.get("success_count")
    failures = payload.get("failure_count")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (attempts, successes, failures)
    ) or attempts != successes + failures:
        raise BenchmarkContractError(
            f"throughput.{concurrency} failure accounting is invalid"
        )
    failure_codes = _mapping(
        payload.get("failure_codes"), f"throughput.{concurrency}.failure_codes"
    )
    if sum(failure_codes.values()) != failures:
        raise BenchmarkContractError(
            f"throughput.{concurrency} failure codes are incomplete"
        )
    for name in ("latency_seconds", "queue_wait_seconds"):
        raw_distribution = payload.get(name)
        if successes == 0:
            if raw_distribution is not None:
                raise BenchmarkContractError(
                    f"throughput.{concurrency}.{name} must be null without successes"
                )
            continue
        distribution = _mapping(
            raw_distribution, f"throughput.{concurrency}.{name}"
        )
        if set(distribution) != _DISTRIBUTION_FIELDS:
            raise BenchmarkContractError(
                f"throughput.{concurrency}.{name} is incomplete"
            )
