#!/usr/bin/env python3
"""Measure the packaged strict-residency L4 path and publish schema-v3 evidence."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from threading import Event, Thread
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vision_model_serving.model_ids import CLASSIFIER_MODEL_ID, DETECTOR_MODEL_ID
from vision_model_serving.pipeline.contracts import PredictionMode
from vision_model_serving.validation.benchmark import (
    BenchmarkContractError,
    ThroughputMeasurement,
    assert_single_residency_snapshot,
    latency_distribution,
    render_benchmark_markdown,
    validate_benchmark_record,
)
from vision_model_serving.validation.packaged_http import (
    PackagedHttpError,
    PackagedPredictionClient,
)
from vision_model_serving.validation.reporting import write_json_atomic
from vision_model_serving.validation.revision import (
    RevisionEvidenceError,
    require_clean_revision,
)


_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"(?:sha256:)?[0-9a-f]{64}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--dicom", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=float, default=180)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--environment-evidence", type=Path, required=True)
    parser.add_argument("--executor-image-id", required=True)
    parser.add_argument("--expected-detector-sha256", required=True)
    parser.add_argument("--expected-classifier-sha256", required=True)
    parser.add_argument("--resource-sample-interval-ms", type=int, default=200)
    args = parser.parse_args()
    if args.runs < 5:
        parser.error("--runs must be at least five")
    if args.resource_sample_interval_ms < 50:
        parser.error("--resource-sample-interval-ms must be at least 50")
    if _COMMIT.fullmatch(args.revision) is None:
        parser.error("--revision must be a full lowercase Git commit")
    for name in ("expected_detector_sha256", "expected_classifier_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", getattr(args, name)) is None:
            parser.error(f"--{name.replace('_', '-')} must be a SHA-256 digest")
    if _SHA256.fullmatch(args.executor_image_id) is None:
        parser.error("--executor-image-id must be a SHA-256 image id")

    project_root = PROJECT_ROOT
    _verify_clean_revision(project_root, args.revision)
    dicom = args.dicom.read_bytes()
    environment = _environment_identity(
        project_root,
        args.environment_evidence,
        executor_image_id=args.executor_image_id,
    )
    identity = _benchmark_identity(project_root, dicom)
    client = PackagedPredictionClient(
        args.base_url,
        timeout_seconds=args.timeout_seconds,
    )
    measured_started = datetime.now(UTC)
    sampler = NvidiaSampler(args.resource_sample_interval_ms)
    sampler.start()
    try:
        record = _measure(
            client=client,
            dicom=dicom,
            runs=args.runs,
            revision=args.revision,
            environment=environment,
            identity=identity,
            expected_detector_sha256=args.expected_detector_sha256,
            expected_classifier_sha256=args.expected_classifier_sha256,
            measured_started=measured_started,
            sampler=sampler,
        )
    finally:
        sampler.stop()
    record["measurements"]["resources"]["nvidia_smi"] = sampler.as_dict()
    record["gates"]["resource_sampling_complete"] = (
        record["measurements"]["resources"]["nvidia_smi"]["available"]
        and record["measurements"]["resources"]["after_operations"] is not None
    )
    record["outcome"] = (
        "passed" if all(record["gates"].values()) else "failed"
    )
    validate_benchmark_record(record)
    write_json_atomic(args.output, record)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(
        render_benchmark_markdown(record),
        encoding="utf-8",
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["outcome"] == "passed" else 1


def _measure(
    *,
    client: PackagedPredictionClient,
    dicom: bytes,
    runs: int,
    revision: str,
    environment: dict[str, object],
    identity: dict[str, object],
    expected_detector_sha256: str,
    expected_classifier_sha256: str,
    measured_started: datetime,
    sampler: NvidiaSampler,
) -> dict[str, object]:
    readiness = client.readiness()
    if (
        readiness.get("status") != "ready"
        or readiness.get("readiness_scope") != "artifact_ready"
        or not all(readiness.get("checks", {}).values())
    ):
        raise BenchmarkContractError("service artifact readiness did not pass")
    before = _inventory(client.model_inventory())
    assert_single_residency_snapshot(before["runtime"])
    if before["runtime"]["state"] != "unloaded":
        raise BenchmarkContractError(
            "cold benchmark requires a freshly started unloaded executor"
        )
    before_operations = client.operations()
    startup = _startup_evidence(before_operations)

    cold_full = _run_sample(
        client,
        dicom,
        mode=PredictionMode.FULL,
        expected_detector_sha256=expected_detector_sha256,
        expected_classifier_sha256=expected_classifier_sha256,
    )
    _assert_lifecycle(cold_full, detector_reused=False, classifier_reused=False)
    cold_inventory = _assert_resident(client, CLASSIFIER_MODEL_ID)

    switch_detection = _run_sample(
        client,
        dicom,
        mode=PredictionMode.DETECTION,
        expected_detector_sha256=expected_detector_sha256,
        expected_classifier_sha256=expected_classifier_sha256,
    )
    _assert_lifecycle(switch_detection, detector_reused=False, classifier_reused=None)
    detection_inventory = _assert_resident(client, DETECTOR_MODEL_ID)

    warm_detection = [
        _run_sample(
            client,
            dicom,
            mode=PredictionMode.DETECTION,
            expected_detector_sha256=expected_detector_sha256,
            expected_classifier_sha256=expected_classifier_sha256,
        )
        for _ in range(runs)
    ]
    for sample in warm_detection:
        _assert_lifecycle(sample, detector_reused=True, classifier_reused=None)
    warm_detection_inventory = _assert_resident(client, DETECTOR_MODEL_ID)

    full_after_detection = _run_sample(
        client,
        dicom,
        mode=PredictionMode.FULL,
        expected_detector_sha256=expected_detector_sha256,
        expected_classifier_sha256=expected_classifier_sha256,
    )
    _assert_lifecycle(
        full_after_detection,
        detector_reused=True,
        classifier_reused=False,
    )
    full_after_detection_inventory = _assert_resident(client, CLASSIFIER_MODEL_ID)

    repeated_full = [
        _run_sample(
            client,
            dicom,
            mode=PredictionMode.FULL,
            expected_detector_sha256=expected_detector_sha256,
            expected_classifier_sha256=expected_classifier_sha256,
        )
        for _ in range(runs)
    ]
    for sample in repeated_full:
        _assert_lifecycle(sample, detector_reused=False, classifier_reused=False)
    repeated_full_inventory = _assert_resident(client, CLASSIFIER_MODEL_ID)

    throughput: dict[str, object] = {}
    for concurrency in (1, 2, 4):
        _run_sample(
            client,
            dicom,
            mode=PredictionMode.DETECTION,
            expected_detector_sha256=expected_detector_sha256,
            expected_classifier_sha256=expected_classifier_sha256,
        )
        _assert_resident(client, DETECTOR_MODEL_ID)
        throughput[str(concurrency)] = _throughput_measurement(
            client,
            dicom,
            concurrency=concurrency,
            attempts=runs * concurrency,
            expected_detector_sha256=expected_detector_sha256,
        )

    final_full = _run_sample(
        client,
        dicom,
        mode=PredictionMode.FULL,
        expected_detector_sha256=expected_detector_sha256,
        expected_classifier_sha256=expected_classifier_sha256,
    )
    _assert_lifecycle(final_full, detector_reused=True, classifier_reused=False)
    final_inventory = _assert_resident(client, CLASSIFIER_MODEL_ID)
    after_operations = client.operations()
    operations_oom = int(
        after_operations.get("telemetry", {})
        .get("events", {})
        .get("cuda_oom_total", 0)
    )
    gates = {
        "environment_complete": True,
        "revision_exact_clean": True,
        "identity_exact": True,
        "artifact_ready": True,
        "single_residency_all_snapshots": True,
        "golden_outputs_all_successes": True,
        "measurement_matrix_complete": True,
        "failure_accounting_complete": all(
            value["attempt_count"]
            == value["success_count"] + value["failure_count"]
            for value in throughput.values()
        ),
        "concurrency_1_no_failures": throughput["1"]["failure_count"] == 0,
        "each_concurrency_has_success": all(
            value["success_count"] > 0 for value in throughput.values()
        ),
        "no_cuda_oom": operations_oom == 0,
        "resource_sampling_complete": False,
    }
    return {
        "schema_version": 3,
        "measured_at": datetime.now(UTC).isoformat(),
        "harness": {
            "revision": revision,
            "revision_clean": True,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "started_at": measured_started.isoformat(),
            "completed_at": datetime.now(UTC).isoformat(),
            "warmup_count": 1,
            "measured_runs_per_lifecycle_phase": runs,
            "percentile_method": "nearest_rank",
            "stddev_method": "population",
        },
        "environment": environment,
        "identity": {
            **identity,
            "detector_prediction_sha256": expected_detector_sha256,
            "classifier_prediction_sha256": expected_classifier_sha256,
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
                "final_full": {"sample": final_full, "inventory": final_inventory},
            },
            "throughput": throughput,
            "resources": {
                "before_operations": before_operations,
                "after_operations": after_operations,
                "cuda_oom_total": operations_oom,
                "nvidia_smi": sampler.as_dict(),
            },
        },
        "gates": gates,
        "outcome": "failed",
        "validation_boundary": (
            "One serialized NVIDIA L4 executor, one checksum-pinned public "
            "Secondary Capture DICOM, pinned FP32 artifacts, and offered HTTP "
            "concurrency 1/2/4. This is not evidence of accuracy, calibration, "
            "robustness, clinical performance, or multi-GPU scaling."
        ),
    }


def _run_sample(
    client: PackagedPredictionClient,
    dicom: bytes,
    *,
    mode: PredictionMode,
    expected_detector_sha256: str,
    expected_classifier_sha256: str,
) -> dict[str, Any]:
    observation = client.predict_observed(dicom, mode=mode)
    result = observation.result
    detector = result["detector"]
    classification = result["classification"]
    if detector["prediction_sha256"] != expected_detector_sha256:
        raise BenchmarkContractError("detector output differs from the pinned golden")
    if mode is PredictionMode.FULL:
        if (
            classification is None
            or classification["prediction_sha256"] != expected_classifier_sha256
        ):
            raise BenchmarkContractError(
                "classifier output differs from the pinned golden"
            )
    elif classification is not None:
        raise BenchmarkContractError(
            "detection benchmark unexpectedly ran the classifier"
        )
    timings = result["timings"]
    detector_stage = timings["detector"]
    classifier_stage = timings["classifier"]
    stages: dict[str, float | None] = {
        "dicom_decode": timings["decode_ms"] / 1_000.0,
        "pipeline_total": timings["total_ms"] / 1_000.0,
        "detector_preprocess": detector_stage["preprocess_ms"] / 1_000.0,
        "detector_load_warmup": detector_stage["runtime"]["load_ms"] / 1_000.0,
        "detector_inference": detector_stage["runtime"]["inference_ms"] / 1_000.0,
        "detector_postprocess": detector_stage["postprocess_ms"] / 1_000.0,
        "detector_switch": detector_stage["runtime"]["switch_ms"] / 1_000.0,
        "classifier_crop_preprocess": None,
        "classifier_tokenization": None,
        "classifier_load_warmup": None,
        "classifier_inference": None,
        "classifier_result": None,
        "classifier_switch": None,
    }
    if classifier_stage is not None:
        stages.update(
            {
                "classifier_crop_preprocess": classifier_stage[
                    "crop_preprocess_ms"
                ]
                / 1_000.0,
                "classifier_tokenization": classifier_stage["tokenization_ms"]
                / 1_000.0,
                "classifier_load_warmup": classifier_stage["runtime"]["load_ms"]
                / 1_000.0,
                "classifier_inference": classifier_stage["runtime"][
                    "inference_ms"
                ]
                / 1_000.0,
                "classifier_result": classifier_stage["result_ms"] / 1_000.0,
                "classifier_switch": classifier_stage["runtime"]["switch_ms"]
                / 1_000.0,
            }
        )
    peak_reserved = detector_stage["memory"]["peak_reserved_bytes"]
    if classifier_stage is not None:
        peak_reserved = max(
            peak_reserved,
            classifier_stage["memory"]["peak_reserved_bytes"],
        )
    return {
        "mode": mode.value,
        "wall_seconds": observation.wall_seconds,
        "queue_wait_seconds": observation.queue_wait_seconds,
        "states": list(observation.states),
        "stages_seconds": stages,
        "http_queue_overhead_seconds": max(
            0.0,
            observation.wall_seconds - timings["total_ms"] / 1_000.0,
        ),
        "lifecycle": {
            "detector_reused": detector_stage["runtime"]["reused"],
            "classifier_reused": (
                None
                if classifier_stage is None
                else classifier_stage["runtime"]["reused"]
            ),
        },
        "peak_reserved_bytes": peak_reserved,
    }


def _aggregate(samples: list[dict[str, Any]]) -> dict[str, object]:
    stage_names = tuple(samples[0]["stages_seconds"])
    return {
        "samples": samples,
        "wall_seconds": latency_distribution(
            [sample["wall_seconds"] for sample in samples]
        ),
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
        "maximum_peak_reserved_bytes": max(
            sample["peak_reserved_bytes"] for sample in samples
        ),
    }


def _throughput_measurement(
    client: PackagedPredictionClient,
    dicom: bytes,
    *,
    concurrency: int,
    attempts: int,
    expected_detector_sha256: str,
) -> dict[str, object]:
    latencies: list[float] = []
    queue_waits: list[float] = []
    failures: Counter[str] = Counter()
    started = time.monotonic()

    def invoke() -> tuple[float, float]:
        observation = client.predict_observed(dicom, mode=PredictionMode.DETECTION)
        result = observation.result
        if (
            result["classification"] is not None
            or result["detector"]["prediction_sha256"]
            != expected_detector_sha256
        ):
            raise BenchmarkContractError("throughput output parity failed")
        return observation.wall_seconds, observation.queue_wait_seconds

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(invoke) for _ in range(attempts)]
        for future in as_completed(futures):
            try:
                latency, queue_wait = future.result()
            except PackagedHttpError as error:
                failures[error.code] += 1
            except Exception as error:
                failures[type(error).__name__] += 1
            else:
                latencies.append(latency)
                queue_waits.append(queue_wait)
    duration = time.monotonic() - started
    return ThroughputMeasurement(
        concurrency=concurrency,
        duration_seconds=duration,
        attempts=attempts,
        successful_latencies=tuple(latencies),
        queue_waits=tuple(queue_waits),
        failure_codes=failures,
    ).as_dict()


def _assert_lifecycle(
    sample: dict[str, Any],
    *,
    detector_reused: bool,
    classifier_reused: bool | None,
) -> None:
    if sample["lifecycle"] != {
        "detector_reused": detector_reused,
        "classifier_reused": classifier_reused,
    }:
        raise BenchmarkContractError("runtime lifecycle differs from strict residency")


def _assert_resident(
    client: PackagedPredictionClient, expected_model: str
) -> dict[str, object]:
    inventory = _inventory(client.model_inventory())
    runtime = inventory["runtime"]
    assert_single_residency_snapshot(runtime)
    if runtime["state"] != "ready" or runtime["resident_models"] != [expected_model]:
        raise BenchmarkContractError("runtime residency differs from expected model")
    return inventory


def _inventory(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "manifest_id": payload["manifest_id"],
        "models": payload["models"],
        "runtime": payload["runtime"],
    }


def _verify_clean_revision(project_root: Path, revision: str) -> None:
    try:
        require_clean_revision(project_root, revision)
    except RevisionEvidenceError as error:
        raise BenchmarkContractError(
            f"benchmark must run from the exact clean committed revision: {error}"
        ) from None


def _environment_identity(
    project_root: Path,
    evidence_path: Path,
    *,
    executor_image_id: str,
) -> dict[str, object]:
    evidence = _json_object(evidence_path)
    if evidence.get("status") != "passed" or evidence.get("gpu_gate") != "passed":
        raise BenchmarkContractError("L4 environment evidence did not pass")
    snapshot = evidence.get("snapshot")
    if not isinstance(snapshot, dict) or snapshot.get("collection_errors") != []:
        raise BenchmarkContractError("L4 environment snapshot is incomplete")
    required = (
        "python",
        "packages",
        "torch_cuda",
        "device_name",
        "compute_capability",
        "total_device_memory_bytes",
        "cudnn_version",
        "nvcc_release",
        "compiler",
        "driver",
    )
    if any(snapshot.get(name) in {None, ""} for name in required):
        raise BenchmarkContractError("L4 runtime/compiler identity is incomplete")
    lane = _json_object(project_root / "config" / "l4-fp32-environment.json")
    dockerfile = project_root / "docker" / "executor.Dockerfile"
    return {
        "hardware": {
            "gpu_name": snapshot["device_name"],
            "total_memory_bytes": snapshot["total_device_memory_bytes"],
            "compute_capability": snapshot["compute_capability"],
            "driver": snapshot["driver"],
        },
        "software": {
            "python": snapshot["python"],
            "torch": snapshot["packages"].get("torch"),
            "torchvision": snapshot["packages"].get("torchvision"),
            "cuda_runtime": snapshot["torch_cuda"],
            "cuda_toolkit": snapshot["nvcc_release"],
            "cudnn": snapshot["cudnn_version"],
            "compiler": snapshot["compiler"],
        },
        "native_operator": {
            **lane["native_operator"],
            "torch_arch_list": lane["cuda"]["torch_arch_list"],
        },
        "container": {
            "executor_image_id": executor_image_id,
            "executor_dockerfile_sha256": hashlib.sha256(
                dockerfile.read_bytes()
            ).hexdigest(),
            "pinned_base_images": _docker_base_images(dockerfile),
        },
    }


def _benchmark_identity(project_root: Path, dicom: bytes) -> dict[str, object]:
    path = project_root / "config" / "model-artifacts.json"
    manifest = _json_object(path)
    return {
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "models": manifest["artifacts"],
        "tokenizer": manifest["tokenizer"],
        "repository_assets": manifest["repository_assets"],
        "dicom_sha256": hashlib.sha256(dicom).hexdigest(),
    }


def _startup_evidence(operations: dict[str, object]) -> dict[str, object]:
    executor = operations.get("executor")
    if not isinstance(executor, dict):
        raise BenchmarkContractError("executor startup evidence is unavailable")
    payload = executor.get("startup")
    if not isinstance(payload, dict):
        raise BenchmarkContractError("executor startup evidence is unavailable")
    required = {
        "process_start_to_artifact_ready_seconds",
        "artifact_verification_seconds",
        "runtime_initialization_seconds",
    }
    if set(payload) != required or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not float(value) >= 0.0
        for value in payload.values()
    ):
        raise BenchmarkContractError("startup evidence is incomplete")
    return {name: float(value) for name, value in payload.items()}


class NvidiaSampler:
    def __init__(self, interval_ms: int):
        self._interval_seconds = interval_ms / 1_000.0
        self._stop = Event()
        self._thread: Thread | None = None
        self._samples: list[tuple[float, float, float, float]] = []

    def start(self) -> None:
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(max(2.0, self._interval_seconds * 4))

    def as_dict(self) -> dict[str, object]:
        if not self._samples:
            return {
                "available": False,
                "interval_ms": int(self._interval_seconds * 1_000),
                "sample_count": 0,
            }
        columns = tuple(zip(*self._samples, strict=True))
        return {
            "available": True,
            "interval_ms": int(self._interval_seconds * 1_000),
            "sample_count": len(self._samples),
            "gpu_utilization_percent": latency_distribution(columns[0]),
            "memory_used_mib": latency_distribution(columns[1]),
            "power_watts": latency_distribution(columns[2]),
            "temperature_c": latency_distribution(columns[3]),
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            output = _command(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                required=False,
            )
            if output:
                try:
                    values = tuple(float(value.strip()) for value in output.split(","))
                    if len(values) == 4 and all(
                        value >= 0 and value == value for value in values
                    ):
                        self._samples.append(values)
                except ValueError:
                    pass
            self._stop.wait(self._interval_seconds)


def _docker_base_images(path: Path) -> list[str]:
    images: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("ARG ") and "IMAGE=" in line and "@sha256:" in line:
            images.append(line.split("=", 1)[1])
    if len(images) != 3:
        raise BenchmarkContractError("executor base-image identity is incomplete")
    return images


def _json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise BenchmarkContractError(f"evidence file is unavailable: {path.name}") from None
    if not isinstance(payload, dict):
        raise BenchmarkContractError(f"evidence file is not an object: {path.name}")
    return payload


def _command(
    command: list[str],
    *,
    cwd: Path | None = None,
    required: bool = True,
) -> str:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        if required:
            raise BenchmarkContractError(f"command is unavailable: {command[0]}") from None
        return ""
    output = completed.stdout.strip()
    if completed.returncode != 0:
        if required:
            raise BenchmarkContractError(f"command failed: {command[0]}")
        return ""
    return output


if __name__ == "__main__":
    raise SystemExit(main())
