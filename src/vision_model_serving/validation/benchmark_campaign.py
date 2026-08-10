"""One deep module for the packaged benchmark campaign and promotion decision."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
import math
import re
import time
from typing import Any, Protocol

from vision_model_serving.model_ids import CLASSIFIER_MODEL_ID, DETECTOR_MODEL_ID
from vision_model_serving.pipeline.contracts import PredictionMode
from vision_model_serving.validation.benchmark import (
    BenchmarkContractError,
    ThroughputMeasurement,
    assert_single_residency_snapshot,
    benchmark_promotion_gates,
    latency_distribution,
    validate_benchmark_record,
)
from vision_model_serving.validation.acceptance_contract import (
    PACKAGED_ACCEPTANCE,
)
from vision_model_serving.validation._benchmark_sampling import NvidiaSampler
from vision_model_serving.validation.packaged_http import (
    PackagedHttpError,
    PredictionObservation,
)


_SHA256 = re.compile(r"[0-9a-f]{64}")
_TIMING_TOLERANCE_SECONDS = 0.001

__all__ = ["BenchmarkCampaignPlan", "NvidiaSampler", "run_benchmark_campaign"]


class BenchmarkTarget(Protocol):
    """HTTP-visible operations required by the campaign."""

    def readiness(self) -> dict[str, Any]: ...

    def model_inventory(self) -> dict[str, Any]: ...

    def operations(self) -> dict[str, Any]: ...

    def predict_observed(self, dicom: bytes, *, mode: PredictionMode) -> PredictionObservation: ...


class ResourceSampler(Protocol):
    """Resource sampling adapter used for the duration of one campaign."""

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def as_dict(self) -> dict[str, object]: ...


@dataclass(frozen=True, slots=True)
class BenchmarkCampaignPlan:
    dicom: bytes
    runs: int
    revision: str
    script_sha256: str
    environment: Mapping[str, object]
    identity: Mapping[str, object]
    expected_detector_sha256: str
    expected_classifier_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.dicom, bytes) or not self.dicom:
            raise BenchmarkContractError("benchmark DICOM bytes are required")
        if isinstance(self.runs, bool) or not isinstance(self.runs, int) or self.runs < 5:
            raise BenchmarkContractError("benchmark requires at least five measured runs")
        if re.fullmatch(r"[0-9a-f]{40}", self.revision) is None:
            raise BenchmarkContractError("benchmark revision must be a full commit")
        for name, digest in (
            ("script", self.script_sha256),
            ("detector", self.expected_detector_sha256),
            ("classifier", self.expected_classifier_sha256),
        ):
            if _SHA256.fullmatch(digest) is None:
                raise BenchmarkContractError(f"benchmark {name} SHA-256 is invalid")
        if not self.environment or not self.identity:
            raise BenchmarkContractError("benchmark environment and identity are required")
        identity = {
            **self.identity,
            "detector_prediction_sha256": self.expected_detector_sha256,
            "classifier_prediction_sha256": self.expected_classifier_sha256,
        }
        try:
            PACKAGED_ACCEPTANCE.validate_benchmark_identity(identity, dicom=self.dicom)
        except AssertionError as error:
            raise BenchmarkContractError(
                "benchmark plan differs from packaged acceptance"
            ) from error


def run_benchmark_campaign(
    target: BenchmarkTarget,
    plan: BenchmarkCampaignPlan,
    sampler: ResourceSampler,
) -> dict[str, Any]:
    """Run, account, sample, and validate one fixed packaged benchmark campaign."""

    measured_started = datetime.now(UTC)
    sampler.start()
    try:
        record = _measure(
            target=target,
            plan=plan,
            measured_started=measured_started,
            sampler=sampler,
        )
    finally:
        sampler.stop()
    sampler_evidence = sampler.as_dict()
    resources = record["measurements"]["resources"]
    resources["nvidia_smi"] = sampler_evidence
    record["gates"] = benchmark_promotion_gates(record)
    record["outcome"] = "passed" if all(record["gates"].values()) else "failed"
    validate_benchmark_record(record)
    return record


def _measure(
    *,
    target: BenchmarkTarget,
    plan: BenchmarkCampaignPlan,
    measured_started: datetime,
    sampler: ResourceSampler,
) -> dict[str, Any]:
    readiness = target.readiness()
    try:
        PACKAGED_ACCEPTANCE.validate_readiness(readiness)
    except AssertionError as error:
        raise BenchmarkContractError("service packaged readiness contract did not pass") from error
    before = _inventory(target.model_inventory())
    assert_single_residency_snapshot(before["runtime"])
    if before["runtime"]["state"] != "unloaded":
        raise BenchmarkContractError("cold benchmark requires a freshly started unloaded executor")
    before_operations = target.operations()
    _assert_plan_identity(plan, before)
    _assert_observed_policy(before_operations)
    startup = _startup_evidence(before_operations)

    cold_full = _run_sample(target, plan, mode=PredictionMode.FULL)
    _assert_lifecycle(cold_full, detector_reused=False, classifier_reused=False)
    cold_inventory = _assert_resident(target, CLASSIFIER_MODEL_ID)

    switch_detection = _run_sample(target, plan, mode=PredictionMode.DETECTION)
    _assert_lifecycle(switch_detection, detector_reused=False, classifier_reused=None)
    detection_inventory = _assert_resident(target, DETECTOR_MODEL_ID)

    warm_detection = [
        _run_sample(target, plan, mode=PredictionMode.DETECTION) for _ in range(plan.runs)
    ]
    for sample in warm_detection:
        _assert_lifecycle(sample, detector_reused=True, classifier_reused=None)
    warm_detection_inventory = _assert_resident(target, DETECTOR_MODEL_ID)

    full_after_detection = _run_sample(target, plan, mode=PredictionMode.FULL)
    _assert_lifecycle(
        full_after_detection,
        detector_reused=True,
        classifier_reused=False,
    )
    full_after_detection_inventory = _assert_resident(target, CLASSIFIER_MODEL_ID)

    repeated_full = [_run_sample(target, plan, mode=PredictionMode.FULL) for _ in range(plan.runs)]
    for sample in repeated_full:
        _assert_lifecycle(sample, detector_reused=False, classifier_reused=False)
    repeated_full_inventory = _assert_resident(target, CLASSIFIER_MODEL_ID)

    throughput: dict[str, dict[str, object]] = {}
    for concurrency in (1, 2, 4):
        _run_sample(target, plan, mode=PredictionMode.DETECTION)
        _assert_resident(target, DETECTOR_MODEL_ID)
        throughput[str(concurrency)] = _throughput_measurement(
            target,
            plan,
            concurrency=concurrency,
            attempts=plan.runs * concurrency,
        )

    final_full = _run_sample(target, plan, mode=PredictionMode.FULL)
    _assert_lifecycle(final_full, detector_reused=True, classifier_reused=False)
    final_inventory = _assert_resident(target, CLASSIFIER_MODEL_ID)
    after_operations = target.operations()
    _assert_observed_policy(after_operations)
    before_operations_oom = _cuda_oom_total(before_operations, "before_operations")
    operations_oom = _cuda_oom_total(after_operations, "after_operations")
    if before_operations_oom > operations_oom:
        raise BenchmarkContractError("CUDA OOM counter decreased during the campaign")
    completed_at = datetime.now(UTC).isoformat()
    return {
        "schema_version": 4,
        "measured_at": completed_at,
        "harness": {
            "revision": plan.revision,
            "revision_clean": True,
            "script_sha256": plan.script_sha256,
            "started_at": measured_started.isoformat(),
            "completed_at": completed_at,
            "warmup_count": 1,
            "measured_runs_per_lifecycle_phase": plan.runs,
            "percentile_method": "nearest_rank",
            "stddev_method": "population",
        },
        "environment": dict(plan.environment),
        "identity": {
            **plan.identity,
            "detector_prediction_sha256": plan.expected_detector_sha256,
            "classifier_prediction_sha256": plan.expected_classifier_sha256,
        },
        "policy": {
            "backend": "pytorch-eager",
            "precision": "float32",
            "tf32": False,
            "executor_concurrency": 1,
            "queue_capacity_requirement": 4,
            "offered_concurrency": [1, 2, 4],
            "lifecycle_sequence": [
                "cold_full",
                "switch_to_detection",
                "consecutive_warm_detection",
                "full_after_detection",
                "repeated_switch_bound_full",
                "throughput_detection",
                "final_full",
            ],
        },
        "measurements": {
            "startup": startup,
            "lifecycle": {
                "cold_full": {"sample": cold_full, "inventory": cold_inventory},
                "switch_to_detection": {
                    "sample": switch_detection,
                    "inventory": detection_inventory,
                },
                "warm_detection": {
                    "aggregate": _aggregate(warm_detection),
                    "inventory": warm_detection_inventory,
                },
                "full_after_detection": {
                    "sample": full_after_detection,
                    "inventory": full_after_detection_inventory,
                },
                "repeated_full": {
                    "aggregate": _aggregate(repeated_full),
                    "inventory": repeated_full_inventory,
                },
                "final_full": {
                    "sample": final_full,
                    "inventory": final_inventory,
                },
            },
            "throughput": throughput,
            "resources": {
                "before_operations": before_operations,
                "after_operations": after_operations,
                "cuda_oom_total": operations_oom,
                "nvidia_smi": sampler.as_dict(),
            },
        },
        "gates": {},
        "outcome": "failed",
        "validation_boundary": (
            "One serialized NVIDIA L4 executor, one checksum-pinned public "
            "Secondary Capture DICOM, pinned FP32 artifacts, and offered HTTP "
            "concurrency 1/2/4. This is not evidence of accuracy, calibration, "
            "robustness, clinical performance, or multi-GPU scaling."
        ),
    }


def _run_sample(
    target: BenchmarkTarget,
    plan: BenchmarkCampaignPlan,
    *,
    mode: PredictionMode,
) -> dict[str, Any]:
    observation = target.predict_observed(plan.dicom, mode=mode)
    result = observation.result
    try:
        PACKAGED_ACCEPTANCE.validate_prediction(result, mode=mode)
    except AssertionError as error:
        raise BenchmarkContractError(
            "benchmark prediction differs from packaged acceptance"
        ) from error
    detector = result["detector"]
    classification = result["classification"]
    if detector["prediction_sha256"] != plan.expected_detector_sha256:
        raise BenchmarkContractError("detector output differs from the pinned golden")
    if mode is PredictionMode.FULL:
        if (
            classification is None
            or classification["prediction_sha256"] != plan.expected_classifier_sha256
        ):
            raise BenchmarkContractError("classifier output differs from the pinned golden")
    elif classification is not None:
        raise BenchmarkContractError("detection benchmark unexpectedly ran the classifier")
    timings = result["timings"]
    detector_stage = timings["detector"]
    detector_adapter = detector_stage["adapter"]
    classifier_stage = timings["classifier"]
    stages: dict[str, float | None] = {
        "dicom_decode": timings["decode_ms"] / 1_000.0,
        "pipeline_total": timings["total_ms"] / 1_000.0,
        "detector_adapter_load": detector_adapter["load_ms"] / 1_000.0,
        "detector_preprocess": detector_adapter["preprocess_ms"] / 1_000.0,
        "detector_load_warmup": detector_stage["runtime"]["load_ms"] / 1_000.0,
        "detector_runtime_execute": (detector_stage["runtime"]["inference_ms"] / 1_000.0),
        "detector_inference": detector_adapter["inference_ms"] / 1_000.0,
        "detector_postprocess": detector_adapter["postprocess_ms"] / 1_000.0,
        "detector_switch": detector_stage["runtime"]["switch_ms"] / 1_000.0,
        "classifier_adapter_load": None,
        "classifier_crop_preprocess": None,
        "classifier_tokenization": None,
        "classifier_load_warmup": None,
        "classifier_runtime_execute": None,
        "classifier_inference": None,
        "classifier_result": None,
        "classifier_switch": None,
    }
    if classifier_stage is not None:
        classifier_adapter = classifier_stage["adapter"]
        stages.update(
            {
                "classifier_adapter_load": classifier_adapter["load_ms"] / 1_000.0,
                "classifier_crop_preprocess": classifier_adapter["crop_preprocess_ms"] / 1_000.0,
                "classifier_tokenization": classifier_adapter["tokenization_ms"] / 1_000.0,
                "classifier_load_warmup": classifier_stage["runtime"]["load_ms"] / 1_000.0,
                "classifier_runtime_execute": classifier_stage["runtime"]["inference_ms"] / 1_000.0,
                "classifier_inference": classifier_adapter["inference_ms"] / 1_000.0,
                "classifier_result": classifier_adapter["result_ms"] / 1_000.0,
                "classifier_switch": classifier_stage["runtime"]["switch_ms"] / 1_000.0,
            }
        )
    peak_reserved = detector_stage["memory"]["peak_reserved_bytes"]
    if classifier_stage is not None:
        peak_reserved = max(
            peak_reserved,
            classifier_stage["memory"]["peak_reserved_bytes"],
        )
    pipeline_seconds = timings["total_ms"] / 1_000.0
    overhead_seconds = observation.wall_seconds - pipeline_seconds
    if overhead_seconds < -_TIMING_TOLERANCE_SECONDS:
        raise BenchmarkContractError(
            "pipeline timing exceeds HTTP wall time beyond the 1 ms tolerance"
        )
    return {
        "mode": mode.value,
        "wall_seconds": observation.wall_seconds,
        "queue_wait_seconds": observation.queue_wait_seconds,
        "states": list(observation.states),
        "stages_seconds": stages,
        "http_queue_overhead_seconds": max(0.0, overhead_seconds),
        "lifecycle": {
            "detector_reused": detector_stage["runtime"]["reused"],
            "classifier_reused": (
                None if classifier_stage is None else classifier_stage["runtime"]["reused"]
            ),
        },
        "peak_reserved_bytes": peak_reserved,
        "detector_prediction_sha256": detector["prediction_sha256"],
        "classifier_prediction_sha256": (
            None if classification is None else classification["prediction_sha256"]
        ),
    }


def _aggregate(samples: list[dict[str, Any]]) -> dict[str, object]:
    stage_names = tuple(samples[0]["stages_seconds"])
    return {
        "samples": samples,
        "wall_seconds": latency_distribution([sample["wall_seconds"] for sample in samples]),
        "queue_wait_seconds": latency_distribution(
            [sample["queue_wait_seconds"] for sample in samples]
        ),
        "stages_seconds": {
            name: latency_distribution(
                [
                    sample["stages_seconds"][name]
                    for sample in samples
                    if sample["stages_seconds"][name] is not None
                ]
            )
            for name in stage_names
            if any(sample["stages_seconds"][name] is not None for sample in samples)
        },
        "maximum_peak_reserved_bytes": max(sample["peak_reserved_bytes"] for sample in samples),
    }


def _throughput_measurement(
    target: BenchmarkTarget,
    plan: BenchmarkCampaignPlan,
    *,
    concurrency: int,
    attempts: int,
) -> dict[str, object]:
    latencies: list[float] = []
    queue_waits: list[float] = []
    failures: Counter[str] = Counter()
    detector_digests: Counter[str] = Counter()
    classifier_digests: Counter[str] = Counter()
    started_ns = time.perf_counter_ns()

    def invoke() -> tuple[float, float, str, str | None]:
        observation = target.predict_observed(plan.dicom, mode=PredictionMode.DETECTION)
        result = observation.result
        try:
            PACKAGED_ACCEPTANCE.validate_prediction(
                result,
                mode=PredictionMode.DETECTION,
            )
        except AssertionError as error:
            raise BenchmarkContractError(
                "throughput prediction differs from packaged acceptance"
            ) from error
        if (
            result["classification"] is not None
            or result["detector"]["prediction_sha256"] != plan.expected_detector_sha256
        ):
            raise BenchmarkContractError("throughput output parity failed")
        return (
            observation.wall_seconds,
            observation.queue_wait_seconds,
            result["detector"]["prediction_sha256"],
            None,
        )

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(invoke) for _ in range(attempts)]
        for future in as_completed(futures):
            try:
                latency, queue_wait, detector_digest, classifier_digest = future.result()
            except BenchmarkContractError:
                raise
            except PackagedHttpError as error:
                failures[error.code] += 1
            except Exception as error:
                failures[type(error).__name__] += 1
            else:
                latencies.append(latency)
                queue_waits.append(queue_wait)
                detector_digests[detector_digest] += 1
                if classifier_digest is not None:
                    classifier_digests[classifier_digest] += 1
    duration = (time.perf_counter_ns() - started_ns) / 1_000_000_000.0
    measurement = ThroughputMeasurement(
        concurrency=concurrency,
        duration_seconds=duration,
        attempts=attempts,
        successful_latencies=tuple(latencies),
        queue_waits=tuple(queue_waits),
        failure_codes=failures,
    ).as_dict()
    measurement["observed_prediction_sha256_counts"] = {
        "detector": dict(sorted(detector_digests.items())),
        "classifier": dict(sorted(classifier_digests.items())),
    }
    return measurement


def _assert_lifecycle(
    sample: Mapping[str, Any],
    *,
    detector_reused: bool,
    classifier_reused: bool | None,
) -> None:
    if sample["lifecycle"] != {
        "detector_reused": detector_reused,
        "classifier_reused": classifier_reused,
    }:
        raise BenchmarkContractError("runtime lifecycle differs from strict residency")


def _assert_resident(target: BenchmarkTarget, expected_model: str) -> dict[str, Any]:
    inventory = _inventory(target.model_inventory())
    runtime = inventory["runtime"]
    assert_single_residency_snapshot(runtime)
    if runtime["state"] != "ready" or runtime["resident_models"] != [expected_model]:
        raise BenchmarkContractError("runtime residency differs from expected model")
    try:
        PACKAGED_ACCEPTANCE.validate_runtime(runtime, active_model=expected_model)
    except AssertionError as error:
        raise BenchmarkContractError(
            "runtime differs from the packaged lifecycle contract"
        ) from error
    return inventory


def _inventory(payload: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return PACKAGED_ACCEPTANCE.validate_inventory(payload)
    except (AssertionError, KeyError) as error:
        raise BenchmarkContractError("model inventory differs from packaged acceptance") from error


def _assert_plan_identity(
    plan: BenchmarkCampaignPlan,
    inventory: Mapping[str, Any],
) -> None:
    def model_identity(
        raw_models: object,
    ) -> tuple[tuple[object, object, object], ...] | None:
        if not isinstance(raw_models, list):
            return None
        projected = tuple(
            (model.get("id"), model.get("role"), model.get("sha256"))
            for model in raw_models
            if isinstance(model, Mapping)
        )
        if (
            len(projected) != len(raw_models)
            or any(not all(isinstance(value, str) for value in item) for item in projected)
            or len(projected) != len(set(projected))
        ):
            return None
        return tuple(sorted(projected))

    if plan.identity.get("manifest_id") != inventory.get("manifest_id") or model_identity(
        plan.identity.get("models")
    ) != model_identity(inventory.get("models")):
        raise BenchmarkContractError("benchmark plan identity differs from the observed service")


def _assert_observed_policy(operations: Mapping[str, object]) -> None:
    queue = operations.get("queue")
    executor = operations.get("executor")
    if (
        not isinstance(queue, Mapping)
        or queue.get("available") is not True
        or queue.get("capacity") != 4
        or not isinstance(executor, Mapping)
        or executor.get("available") is not True
        or executor.get("precision") != "float32"
    ):
        raise BenchmarkContractError(
            "observed service policy differs from serialized FP32 capacity-four lane"
        )


def _startup_evidence(operations: Mapping[str, object]) -> dict[str, float]:
    executor = operations.get("executor")
    if not isinstance(executor, Mapping):
        raise BenchmarkContractError("executor startup evidence is unavailable")
    payload = executor.get("startup")
    if not isinstance(payload, Mapping):
        raise BenchmarkContractError("executor startup evidence is unavailable")
    required = {
        "process_start_to_artifact_ready_seconds",
        "artifact_verification_seconds",
        "runtime_initialization_seconds",
    }
    if set(payload) != required or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in payload.values()
    ):
        raise BenchmarkContractError("startup evidence is incomplete")
    return {name: float(value) for name, value in payload.items()}


def _cuda_oom_total(operations: Mapping[str, object], path: str) -> int:
    telemetry = operations.get("telemetry")
    if not isinstance(telemetry, Mapping):
        raise BenchmarkContractError(f"{path} CUDA OOM telemetry is unavailable")
    events = telemetry.get("events")
    if not isinstance(events, Mapping):
        raise BenchmarkContractError(f"{path} CUDA OOM telemetry is unavailable")
    value = events.get("cuda_oom_total")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BenchmarkContractError(f"{path} CUDA OOM telemetry is invalid")
    return value
