"""Cross-section evidence gates for schema-versioned benchmark reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import re
from typing import Any

from vision_model_serving.validation._benchmark_primitives import (
    _LIFECYCLE_RULES,
    _positive_integer,
)
from vision_model_serving.validation.benchmark_environment import (
    BENCHMARK_EXECUTOR_BASE_IMAGES,
    BENCHMARK_EXECUTOR_DOCKERFILE_SHA256,
    BENCHMARK_LANE_CONFIG_SHA256,
    BENCHMARK_LANE_HARDWARE,
    BENCHMARK_LANE_ID,
    BENCHMARK_LANE_NATIVE_OPERATOR,
    BENCHMARK_LANE_PACKAGES,
    BENCHMARK_LANE_SOFTWARE,
)


_SHA256 = re.compile(r"[0-9a-f]{64}")
# One sample may straddle either campaign boundary, and one additional interval
# is reserved for ordinary thread scheduling jitter.
_RESOURCE_SAMPLING_TOLERANCE_INTERVALS = 2


def environment_is_complete(
    environment: Mapping[str, Any],
    *,
    schema_version: int,
) -> bool:
    hardware = environment.get("hardware")
    software = environment.get("software")
    native_operator = environment.get("native_operator")
    container = environment.get("container")
    if (
        schema_version == 4
        and (
            not _environment_lane_is_complete(environment.get("lane"))
            or not _mapping_matches(hardware, BENCHMARK_LANE_HARDWARE)
            or not _mapping_matches(software, BENCHMARK_LANE_SOFTWARE)
            or not _mapping_matches(native_operator, BENCHMARK_LANE_NATIVE_OPERATOR)
        )
    ) or not all(
        isinstance(value, Mapping) for value in (hardware, software, native_operator, container)
    ):
        return False
    assert isinstance(hardware, Mapping)
    assert isinstance(software, Mapping)
    assert isinstance(native_operator, Mapping)
    assert isinstance(container, Mapping)
    if (
        hardware.get("gpu_name") != "NVIDIA L4"
        or not all(
            hardware.get(name) is not None and hardware.get(name) != ""
            for name in ("compute_capability", "driver")
        )
        or not _positive_integer(hardware.get("total_memory_bytes"))
        or not all(
            software.get(name) is not None and software.get(name) != ""
            for name in (
                "python",
                "torch",
                "torchvision",
                "cuda_runtime",
                "cuda_toolkit",
                "cudnn",
                "compiler",
            )
        )
        or native_operator.get("module") != "MultiScaleDeformableAttention"
        or native_operator.get("verification") != "functional_forward_parity"
        or native_operator.get("torch_arch_list") != "8.9"
        or _SHA256.fullmatch(str(native_operator.get("reference_sha256", ""))) is None
    ):
        return False
    image_id = str(container.get("executor_image_id", ""))
    dockerfile_digest = str(container.get("executor_dockerfile_sha256", ""))
    base_images = container.get("pinned_base_images")
    return (
        image_id.startswith("sha256:")
        and _SHA256.fullmatch(image_id.removeprefix("sha256:")) is not None
        and _SHA256.fullmatch(dockerfile_digest) is not None
        and (schema_version == 3 or dockerfile_digest == BENCHMARK_EXECUTOR_DOCKERFILE_SHA256)
        and isinstance(base_images, Sequence)
        and not isinstance(base_images, (str, bytes))
        and bool(base_images)
        and all(
            isinstance(image, str) and re.search(r"@sha256:[0-9a-f]{64}$", image) is not None
            for image in base_images
        )
        and (schema_version == 3 or tuple(base_images) == BENCHMARK_EXECUTOR_BASE_IMAGES)
    )


def resource_sampling_is_complete(
    sampler: Mapping[str, Any],
    *,
    environment: Mapping[str, Any],
    schema_version: int,
) -> bool:
    if schema_version == 3:
        return sampler.get("available") is True
    interval_ms = sampler.get("interval_ms")
    duration = sampler.get("duration_seconds")
    attempts = sampler.get("attempt_count")
    successes = sampler.get("success_count")
    failures = sampler.get("failure_count")
    sample_count = sampler.get("sample_count")
    if (
        sampler.get("available") is not True
        or not _positive_integer(interval_ms)
        or not isinstance(duration, (int, float))
        or isinstance(duration, bool)
        or duration <= 0.0
        or not _positive_integer(attempts)
        or successes != attempts
        or failures != 0
        or sample_count != successes
    ):
        return False
    interval_seconds = interval_ms / 1_000.0
    ideal_attempts = math.floor(duration / interval_seconds) + 1
    minimum_attempts = max(
        1,
        ideal_attempts - _RESOURCE_SAMPLING_TOLERANCE_INTERVALS,
    )
    maximum_attempts = (
        math.ceil(duration / interval_seconds) + 1 + _RESOURCE_SAMPLING_TOLERANCE_INTERVALS
    )
    hardware = environment.get("hardware")
    memory = sampler.get("memory_used_mib")
    if not isinstance(hardware, Mapping) or not isinstance(memory, Mapping):
        return False
    total_memory_bytes = hardware.get("total_memory_bytes")
    memory_max_mib = memory.get("max")
    return (
        minimum_attempts <= attempts <= maximum_attempts
        and _positive_integer(total_memory_bytes)
        and isinstance(memory_max_mib, (int, float))
        and not isinstance(memory_max_mib, bool)
        and memory_max_mib <= total_memory_bytes / (2**20)
    )


def measurement_counts_match_campaign(
    lifecycle: Mapping[str, Any],
    throughput: Mapping[str, Any],
    measured_runs: object,
) -> bool:
    if isinstance(measured_runs, bool) or not isinstance(measured_runs, int) or measured_runs < 5:
        return False
    for phase, rule in _LIFECYCLE_RULES.items():
        if rule[0] != "aggregate":
            continue
        payload = lifecycle.get(phase)
        if not isinstance(payload, Mapping):
            return False
        aggregate = payload.get("aggregate")
        if not isinstance(aggregate, Mapping):
            return False
        samples = aggregate.get("samples")
        if (
            not isinstance(samples, Sequence)
            or isinstance(samples, (str, bytes))
            or len(samples) != measured_runs
        ):
            return False
    return all(
        throughput[str(concurrency)].get("attempt_count") == measured_runs * concurrency
        for concurrency in (1, 2, 4)
    )


def _environment_lane_is_complete(value: object) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "id",
        "config_sha256",
        "packages",
    }:
        return False
    return (
        value.get("id") == BENCHMARK_LANE_ID
        and value.get("config_sha256") == BENCHMARK_LANE_CONFIG_SHA256
        and value.get("packages") == BENCHMARK_LANE_PACKAGES
    )


def _mapping_matches(
    observed: object,
    expected: Mapping[str, str],
) -> bool:
    return isinstance(observed, Mapping) and all(
        observed.get(name) == value for name, value in expected.items()
    )
