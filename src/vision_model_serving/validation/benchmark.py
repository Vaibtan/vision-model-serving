"""Public statistics and reporting seam for schema-versioned benchmark evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

from vision_model_serving.validation._benchmark_contract import (
    benchmark_promotion_gates,
    validate_benchmark_record,
)
from vision_model_serving.validation._benchmark_primitives import (
    BenchmarkContractError,
    assert_single_residency_snapshot,
    latency_distribution,
)


__all__ = [
    "BenchmarkContractError",
    "ThroughputMeasurement",
    "assert_single_residency_snapshot",
    "benchmark_promotion_gates",
    "latency_distribution",
    "render_benchmark_markdown",
    "validate_benchmark_record",
]


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
            raise BenchmarkContractError("queue-wait samples must match successful latency samples")
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
                latency_distribution(self.queue_waits) if self.queue_waits else None
            ),
        }


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

Revision: `{record["harness"]["revision"]}`<br>
Measured at: `{record["measured_at"]}`<br>
Outcome: **{str(record["outcome"]).upper()}**

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

{record["validation_boundary"]}
"""
