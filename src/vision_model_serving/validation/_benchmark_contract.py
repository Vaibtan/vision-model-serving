"""Promotion and evidence policy for benchmark records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
import hashlib
import json
import re
from typing import Any

from vision_model_serving.validation.acceptance_contract import (
    CLASSIFIER_ARTIFACT,
    DETECTOR_ARTIFACT,
    PACKAGED_ACCEPTANCE,
    PACKAGED_MANIFEST_ID,
)
from vision_model_serving.validation._benchmark_primitives import (
    _LIFECYCLE_RULES,
    BenchmarkContractError,
    _mapping,
    _positive_integer,
    _sequence,
)
from vision_model_serving.validation._benchmark_evidence import (
    environment_is_complete,
    measurement_counts_match_campaign,
    resource_sampling_is_complete,
)
from vision_model_serving.validation._benchmark_schema import (
    validate_lifecycle_payload,
    validate_resources_payload,
    validate_startup_payload,
    validate_throughput_payload,
)


_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_HISTORICAL_SCHEMA_V3_CANONICAL_SHA256 = (
    "8d84b8f7aec09a117d3b008b876f91f9885ef5dfde5eecf91783974ab02bab0b"
)
_OPERATIONAL_CORE_CHECKS = frozenset(
    {
        "redis",
        "rq_worker",
        "executor_artifact_ready",
        "verified_artifacts",
        "runtime_initialized",
        "device_available",
        "native_operator_available",
    }
)
_OPERATIONAL_CHECKS = _OPERATIONAL_CORE_CHECKS | frozenset(
    {"manifest_available", "telemetry_available"}
)


def validate_benchmark_record(record: Mapping[str, object]) -> None:
    """Validate the benchmark evidence contract before publication."""

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
    schema_version = _record_schema_version(record)
    harness = _mapping(record.get("harness"), "harness")
    if set(harness) != {
        "revision",
        "revision_clean",
        "script_sha256",
        "started_at",
        "completed_at",
        "warmup_count",
        "measured_runs_per_lifecycle_phase",
        "percentile_method",
        "stddev_method",
    }:
        raise BenchmarkContractError("benchmark harness is incomplete")
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
    for name in ("started_at", "completed_at"):
        if not isinstance(harness.get(name), str) or not harness.get(name):
            raise BenchmarkContractError(f"benchmark {name} is required")
    measured_runs = harness.get("measured_runs_per_lifecycle_phase")
    if (
        harness.get("warmup_count") != 1
        or not _positive_integer(measured_runs)
        or (schema_version == 4 and measured_runs < 5)
    ):
        raise BenchmarkContractError("benchmark run policy is invalid")
    if schema_version == 4:
        started_at = _aware_timestamp(harness.get("started_at"), "harness.started_at")
        completed_at = _aware_timestamp(
            harness.get("completed_at"),
            "harness.completed_at",
        )
        measured_at = _aware_timestamp(record.get("measured_at"), "measured_at")
        if started_at > completed_at or measured_at != completed_at:
            raise BenchmarkContractError("benchmark timestamp ordering is invalid")

    environment = _mapping(record.get("environment"), "environment")
    for name in ("hardware", "software", "native_operator", "container"):
        if not _mapping(environment.get(name), f"environment.{name}"):
            raise BenchmarkContractError(f"environment.{name} must not be empty")
    identity = _mapping(record.get("identity"), "identity")
    if set(identity) != {
        "manifest_id",
        "manifest_sha256",
        "models",
        "tokenizer",
        "repository_assets",
        "dicom_sha256",
        "detector_prediction_sha256",
        "classifier_prediction_sha256",
    }:
        raise BenchmarkContractError("benchmark identity is incomplete")
    if not isinstance(identity.get("manifest_id"), str) or not identity.get("manifest_id"):
        raise BenchmarkContractError("benchmark manifest identity is required")
    if _SHA256.fullmatch(str(identity.get("manifest_sha256", ""))) is None:
        raise BenchmarkContractError("manifest SHA-256 is invalid")
    if _SHA256.fullmatch(str(identity.get("dicom_sha256", ""))) is None:
        raise BenchmarkContractError("DICOM SHA-256 is invalid")
    for name in ("detector_prediction_sha256", "classifier_prediction_sha256"):
        if _SHA256.fullmatch(str(identity.get(name, ""))) is None:
            raise BenchmarkContractError(f"{name} is invalid")
    if not _sequence(identity.get("models"), "identity.models"):
        raise BenchmarkContractError("benchmark model identity is empty")
    if not _mapping(identity.get("tokenizer"), "identity.tokenizer"):
        raise BenchmarkContractError("benchmark tokenizer identity is empty")
    if not _sequence(identity.get("repository_assets"), "identity.repository_assets"):
        raise BenchmarkContractError("benchmark repository identity is empty")

    policy = _mapping(record.get("policy"), "policy")
    if set(policy) != {
        "backend",
        "precision",
        "tf32",
        "executor_concurrency",
        "queue_capacity_requirement",
        "offered_concurrency",
        "lifecycle_sequence",
    }:
        raise BenchmarkContractError("benchmark policy is incomplete")
    if (
        policy.get("backend") != "pytorch-eager"
        or policy.get("precision") != "float32"
        or policy.get("tf32") is not False
        or policy.get("executor_concurrency") != 1
        or policy.get("queue_capacity_requirement") != 4
    ):
        raise BenchmarkContractError("benchmark execution policy is invalid")
    if policy.get("offered_concurrency") != [1, 2, 4]:
        raise BenchmarkContractError("offered concurrency must be exactly 1, 2, 4")
    if policy.get("lifecycle_sequence") != [
        "cold_full",
        "switch_to_detection",
        "consecutive_warm_detection",
        "full_after_detection",
        "repeated_switch_bound_full",
        "throughput_detection",
        "final_full",
    ]:
        raise BenchmarkContractError("benchmark lifecycle sequence is invalid")
    expected_gates = benchmark_promotion_gates(record)

    gates = _mapping(record.get("gates"), "gates")
    if set(gates) != set(expected_gates) or not all(
        isinstance(value, bool) for value in gates.values()
    ):
        raise BenchmarkContractError("benchmark gates are incomplete")
    if dict(gates) != expected_gates:
        raise BenchmarkContractError("benchmark gates differ from measurements")
    outcome = record.get("outcome")
    if outcome not in {"passed", "failed"}:
        raise BenchmarkContractError("benchmark outcome is invalid")
    if (outcome == "passed") != all(gates.values()):
        raise BenchmarkContractError("benchmark outcome differs from promotion gates")
    if not isinstance(record.get("validation_boundary"), str) or not record.get(
        "validation_boundary"
    ):
        raise BenchmarkContractError("benchmark validation boundary is required")


def benchmark_promotion_gates(
    record: Mapping[str, object],
) -> dict[str, bool]:
    """Validate the measurement matrix and derive its one promotion decision."""

    schema_version = _record_schema_version(record)
    measurements = _mapping(record.get("measurements"), "measurements")
    if set(measurements) != {"startup", "lifecycle", "throughput", "resources"}:
        raise BenchmarkContractError("benchmark measurement matrix is incomplete")
    validate_startup_payload(measurements.get("startup"))
    validate_lifecycle_payload(
        measurements.get("lifecycle"),
        schema_version=schema_version,
    )
    throughput = _mapping(measurements.get("throughput"), "throughput")
    if set(throughput) != {"1", "2", "4"}:
        raise BenchmarkContractError("throughput must cover concurrency 1, 2, and 4")
    for concurrency in (1, 2, 4):
        validate_throughput_payload(
            throughput[str(concurrency)],
            concurrency,
            schema_version=schema_version,
        )
    validate_resources_payload(
        measurements.get("resources"),
        schema_version=schema_version,
    )
    harness = _mapping(record.get("harness"), "harness")
    environment = _mapping(record.get("environment"), "environment")
    identity = _mapping(record.get("identity"), "identity")
    policy = _mapping(record.get("policy"), "policy")
    lifecycle = _mapping(measurements.get("lifecycle"), "lifecycle")
    resources = _mapping(measurements.get("resources"), "resources")
    sampler = _mapping(resources.get("nvidia_smi"), "resources.nvidia_smi")
    operation_snapshots = (
        _mapping(resources.get("before_operations"), "resources.before_operations"),
        _mapping(resources.get("after_operations"), "resources.after_operations"),
    )
    before_cuda_oom = _operation_cuda_oom_total(operation_snapshots[0])
    after_cuda_oom = _operation_cuda_oom_total(operation_snapshots[1])
    throughput_complete = all(
        throughput[str(concurrency)]["attempt_count"]
        == throughput[str(concurrency)]["success_count"]
        + throughput[str(concurrency)]["failure_count"]
        == throughput[str(concurrency)]["success_count"]
        + sum(throughput[str(concurrency)]["failure_codes"].values())
        for concurrency in (1, 2, 4)
    )
    lifecycle_samples = tuple(_lifecycle_samples(lifecycle))
    identity_exact = _identity_matches_observations(
        identity,
        lifecycle=lifecycle,
        operation_snapshots=operation_snapshots,
    )
    offered_load_has_no_failures = all(
        throughput[str(concurrency)]["failure_count"] == 0 for concurrency in (1, 2, 4)
    )
    return {
        "environment_complete": environment_is_complete(
            environment,
            schema_version=schema_version,
        ),
        "revision_exact_clean": (
            _COMMIT.fullmatch(str(harness.get("revision", ""))) is not None
            and harness.get("revision_clean") is True
            and _SHA256.fullmatch(str(harness.get("script_sha256", ""))) is not None
        ),
        "identity_exact": identity_exact,
        "artifact_ready": all(
            _operation_matches_policy(
                snapshot,
                policy,
                schema_version=schema_version,
            )
            for snapshot in operation_snapshots
        ),
        "single_residency_all_snapshots": _all_snapshots_single_resident(lifecycle),
        "golden_outputs_all_successes": (
            identity_exact
            and bool(lifecycle_samples)
            and all(sample["states"][-1] == "succeeded" for sample in lifecycle_samples)
            and offered_load_has_no_failures
            and _serialized_outputs_match_identity(
                identity,
                lifecycle_samples=lifecycle_samples,
                throughput=throughput,
                schema_version=schema_version,
            )
        ),
        "measurement_matrix_complete": (
            set(measurements) == {"startup", "lifecycle", "throughput", "resources"}
            and set(lifecycle) == set(_LIFECYCLE_RULES)
            and set(throughput) == {"1", "2", "4"}
            and (
                schema_version == 3
                or measurement_counts_match_campaign(
                    lifecycle,
                    throughput,
                    harness.get("measured_runs_per_lifecycle_phase"),
                )
            )
        ),
        "failure_accounting_complete": throughput_complete,
        "concurrency_1_no_failures": throughput["1"]["failure_count"] == 0,
        "each_concurrency_has_success": all(
            throughput[str(concurrency)]["success_count"] > 0 for concurrency in (1, 2, 4)
        ),
        "no_cuda_oom": (
            resources["cuda_oom_total"] == 0
            if schema_version == 3
            else (
                before_cuda_oom is not None
                and after_cuda_oom is not None
                and before_cuda_oom <= after_cuda_oom
                and resources["cuda_oom_total"] == after_cuda_oom == 0
            )
        ),
        "resource_sampling_complete": resource_sampling_is_complete(
            sampler,
            environment=environment,
            schema_version=schema_version,
        ),
    }


def _record_schema_version(record: Mapping[str, object]) -> int:
    schema_version = record.get("schema_version")
    if schema_version not in {3, 4}:
        raise BenchmarkContractError("benchmark schema_version must be 3 or 4")
    if schema_version == 3 and _canonical_record_sha256(record) != (
        _HISTORICAL_SCHEMA_V3_CANONICAL_SHA256
    ):
        raise BenchmarkContractError(
            "historical schema-v3 evidence must match the canonical committed record"
        )
    return schema_version


def _canonical_record_sha256(record: Mapping[str, object]) -> str:
    try:
        payload = json.dumps(
            record,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise BenchmarkContractError("benchmark record is not canonical JSON") from error
    return hashlib.sha256(payload).hexdigest()


def _aware_timestamp(value: object, path: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise BenchmarkContractError(f"benchmark {path} timestamp is required")
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError:
        raise BenchmarkContractError(f"benchmark {path} timestamp is invalid") from None
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise BenchmarkContractError(f"benchmark {path} timestamp must be timezone-aware")
    return timestamp


def _model_projection(
    value: object,
    *,
    require_strict: bool,
) -> tuple[tuple[str, str, str], ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    projected: list[tuple[str, str, str]] = []
    for raw_model in value:
        if not isinstance(raw_model, Mapping):
            return None
        model_id = raw_model.get("id")
        role = raw_model.get("role")
        digest = raw_model.get("sha256")
        if (
            not isinstance(model_id, str)
            or not isinstance(role, str)
            or not isinstance(digest, str)
            or (require_strict and raw_model.get("strict_load_verified") is not True)
        ):
            return None
        projected.append((model_id, role, digest))
    if len(projected) != len(set(projected)):
        return None
    return tuple(sorted(projected))


def _expected_model_projection() -> tuple[tuple[str, str, str], ...]:
    return tuple(
        sorted(
            (artifact.model_id, artifact.role, artifact.sha256)
            for artifact in (DETECTOR_ARTIFACT, CLASSIFIER_ARTIFACT)
        )
    )


def _identity_matches_observations(
    identity: Mapping[str, Any],
    *,
    lifecycle: Mapping[str, Any],
    operation_snapshots: Sequence[Mapping[str, Any]],
) -> bool:
    try:
        PACKAGED_ACCEPTANCE.validate_benchmark_identity(identity)
    except AssertionError:
        return False
    inventories = [
        _mapping(payload.get("inventory"), f"lifecycle.{phase}.inventory")
        for phase, payload in lifecycle.items()
    ]
    inventories.extend(operation_snapshots)
    return all(
        inventory.get("manifest_id") == PACKAGED_MANIFEST_ID
        and _model_projection(inventory.get("models"), require_strict=True)
        == _expected_model_projection()
        for inventory in inventories
    )


def _serialized_outputs_match_identity(
    identity: Mapping[str, Any],
    *,
    lifecycle_samples: Sequence[Mapping[str, Any]],
    throughput: Mapping[str, Any],
    schema_version: int,
) -> bool:
    if schema_version == 3:
        return True
    detector_digest = identity.get("detector_prediction_sha256")
    classifier_digest = identity.get("classifier_prediction_sha256")
    if not all(
        sample.get("detector_prediction_sha256") == detector_digest
        and sample.get("classifier_prediction_sha256")
        == (classifier_digest if sample.get("mode") == "full" else None)
        for sample in lifecycle_samples
    ):
        return False
    for concurrency in (1, 2, 4):
        payload = throughput[str(concurrency)]
        counts = payload["observed_prediction_sha256_counts"]
        expected_detector_counts = (
            {detector_digest: payload["success_count"]} if payload["success_count"] > 0 else {}
        )
        if counts != {
            "detector": expected_detector_counts,
            "classifier": {},
        }:
            return False
    return True


def _operation_matches_policy(
    snapshot: Mapping[str, Any],
    policy: Mapping[str, Any],
    *,
    schema_version: int,
) -> bool:
    checks = snapshot.get("checks")
    queue = snapshot.get("queue")
    executor = snapshot.get("executor")
    telemetry = snapshot.get("telemetry")
    return (
        snapshot.get("schema_version") == 2
        and snapshot.get("status") == "ready"
        and snapshot.get("reasons") == []
        and isinstance(checks, Mapping)
        and (
            _OPERATIONAL_CORE_CHECKS <= set(checks)
            if schema_version == 3
            else set(checks) == _OPERATIONAL_CHECKS
        )
        and all(value is True for value in checks.values())
        and isinstance(queue, Mapping)
        and queue.get("available") is True
        and queue.get("capacity") == policy.get("queue_capacity_requirement")
        and isinstance(executor, Mapping)
        and executor.get("available") is True
        and executor.get("artifact_ready") is True
        and executor.get("runtime_initialized") is True
        and executor.get("device_available") is True
        and executor.get("artifacts_verified") is True
        and executor.get("native_operator_available") is True
        and executor.get("failure_present") is False
        and executor.get("precision") == policy.get("precision")
        and _runtime_snapshot_is_single(executor)
        and snapshot.get("manifest_id") == PACKAGED_MANIFEST_ID
        and _model_projection(snapshot.get("models"), require_strict=True)
        == _expected_model_projection()
        and isinstance(telemetry, Mapping)
        and bool(telemetry)
    )


def _operation_cuda_oom_total(snapshot: Mapping[str, Any]) -> int | None:
    telemetry = snapshot.get("telemetry")
    if not isinstance(telemetry, Mapping):
        return None
    events = telemetry.get("events")
    if not isinstance(events, Mapping):
        return None
    value = events.get("cuda_oom_total")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _lifecycle_samples(
    lifecycle: Mapping[str, Any],
) -> Sequence[Mapping[str, Any]]:
    samples: list[Mapping[str, Any]] = []
    for phase, rule in _LIFECYCLE_RULES.items():
        measurement_kind = rule[0]
        payload = _mapping(lifecycle.get(phase), f"lifecycle.{phase}")
        if measurement_kind == "sample":
            samples.append(_mapping(payload.get("sample"), f"lifecycle.{phase}.sample"))
        else:
            aggregate = _mapping(payload.get("aggregate"), f"lifecycle.{phase}.aggregate")
            samples.extend(
                _mapping(sample, f"lifecycle.{phase}.aggregate.sample")
                for sample in _sequence(
                    aggregate.get("samples"), f"lifecycle.{phase}.aggregate.samples"
                )
            )
    return tuple(samples)


def _all_snapshots_single_resident(lifecycle: Mapping[str, Any]) -> bool:
    for phase in _LIFECYCLE_RULES:
        payload = _mapping(lifecycle.get(phase), f"lifecycle.{phase}")
        inventory = _mapping(payload.get("inventory"), f"lifecycle.{phase}.inventory")
        runtime = _mapping(inventory.get("runtime"), f"lifecycle.{phase}.inventory.runtime")
        if not _runtime_snapshot_is_single(runtime):
            return False
    return True


def _runtime_snapshot_is_single(runtime: Mapping[str, Any]) -> bool:
    residents = runtime.get("resident_models")
    active = runtime.get("active_model")
    return (
        isinstance(residents, Sequence)
        and not isinstance(residents, (str, bytes))
        and len(residents) <= 1
        and list(residents) == ([] if active is None else [active])
    )
