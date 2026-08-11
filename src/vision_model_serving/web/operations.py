"""One bounded operational snapshot for readiness, monitoring, and metrics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import math
import threading
import time

from django.conf import settings
from prometheus_client import REGISTRY, CollectorRegistry, generate_latest, multiprocess
from prometheus_client.parser import text_string_to_metric_families
from redis import Redis
from rq import Queue, Worker

from vision_model_serving.artifacts.manifest import load_manifest

from .runtime import executor_client, prediction_gateway


@dataclass(frozen=True, slots=True)
class OperationalSnapshot:
    captured_at: str
    status: str
    checks: dict[str, bool]
    reasons: tuple[str, ...]
    queue: dict[str, object]
    executor: dict[str, object]
    manifest_id: str | None
    models: tuple[dict[str, object], ...]
    telemetry: dict[str, object]

    def readiness_dict(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "status": self.status,
            "readiness_scope": "artifact_ready",
            "checks": dict(self.checks),
            "reasons": list(self.reasons),
            "runtime": {
                "initialized": bool(self.executor["runtime_initialized"]),
                "state": self.executor["state"],
                "inference_warm": bool(self.executor["inference_warm"]),
                "warm_model": self.executor["warm_model"],
            },
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "captured_at": self.captured_at,
            "status": self.status,
            "checks": dict(self.checks),
            "reasons": list(self.reasons),
            "queue": {key: value for key, value in self.queue.items() if not key.startswith("_")},
            "executor": {
                **self.executor,
                "resident_models": list(self.executor["resident_models"]),
            },
            "manifest_id": self.manifest_id,
            "models": [dict(model) for model in self.models],
            "telemetry": self.telemetry,
            "validation_boundary": (
                "Operational telemetry only; not a clinical audit log and not "
                "evidence of model accuracy, calibration, or medical fitness."
            ),
        }


def read_operational_snapshot() -> OperationalSnapshot:
    redis_ready, worker_ready = _broker_readiness()
    observations = None
    if redis_ready:
        try:
            observations = prediction_gateway().observations()
        except Exception:  # noqa: BLE001 - operational state is fail-closed and bounded
            pass
    executor = None
    try:
        executor = executor_client().status()
    except Exception:  # noqa: BLE001 - operational state is fail-closed and bounded
        pass

    manifest_available, manifest_id, models = _manifest_payload()
    telemetry_available, telemetry = _telemetry_snapshot()
    if observations is not None:
        telemetry = _with_queue_wait(telemetry, observations)
    checks = {
        "redis": redis_ready,
        "rq_worker": worker_ready,
        "executor_artifact_ready": (executor.artifact_ready if executor is not None else False),
        "verified_artifacts": (executor.verified_artifacts if executor is not None else False),
        "runtime_initialized": (executor.runtime_initialized if executor is not None else False),
        "device_available": (executor.device_available if executor is not None else False),
        "native_operator_available": (
            executor.native_operator_available if executor is not None else False
        ),
        "manifest_available": manifest_available,
        "telemetry_available": telemetry_available,
    }
    # telemetry_available is informational only: a corrupt or full metrics store
    # must never gate readiness while inference still works.
    ready = all(value for name, value in checks.items() if name != "telemetry_available")
    return OperationalSnapshot(
        captured_at=datetime.now(UTC).isoformat(),
        status="ready" if ready else "not_ready",
        checks=checks,
        reasons=tuple(_unavailable_reason(name) for name, passed in checks.items() if not passed),
        queue=_queue_payload(observations),
        executor=_executor_payload(executor),
        manifest_id=manifest_id,
        models=models,
        telemetry=telemetry,
    )


def _manifest_payload() -> tuple[
    bool,
    str | None,
    tuple[dict[str, object], ...],
]:
    try:
        manifest = load_manifest(settings.BASE_DIR / "config" / "model-artifacts.json")
    except Exception:  # noqa: BLE001 - readiness exposes only a bounded reason
        return False, None, ()
    return (
        True,
        manifest.manifest_id,
        tuple(
            {
                "id": artifact.id,
                "role": artifact.role,
                "sha256": artifact.sha256,
                "strict_load_verified": artifact.strict_load_verified,
                "semantics_status": artifact.semantics_status,
                "class_names": (
                    list(artifact.class_names) if artifact.class_names is not None else None
                ),
                "decision_threshold": artifact.decision_threshold,
            }
            for artifact in manifest.artifacts
        ),
    )


# Healthcheck probes arrive every few seconds; the multiprocess collect, render,
# and reparse below is the expensive part of a snapshot, so reuse it briefly.
_TELEMETRY_CACHE_SECONDS = 2.0
_telemetry_cache_lock = threading.Lock()
_telemetry_cache: tuple[float, tuple[bool, dict[str, object]]] | None = None


def _telemetry_snapshot() -> tuple[bool, dict[str, object]]:
    global _telemetry_cache
    with _telemetry_cache_lock:
        now = time.monotonic()
        if _telemetry_cache is not None and now - _telemetry_cache[0] < _TELEMETRY_CACHE_SECONDS:
            return _telemetry_cache[1]
        try:
            snapshot = True, _telemetry_payload(_metric_samples(instrumented_metrics()))
        except Exception:  # noqa: BLE001 - readiness exposes only a bounded reason
            snapshot = False, _telemetry_payload(())
        _telemetry_cache = (now, snapshot)
        return snapshot


def _unavailable_reason(check_name: str) -> str:
    if check_name in {"manifest_available", "telemetry_available"}:
        return f"{check_name.removesuffix('_available')}_unavailable"
    return f"{check_name}_unavailable"


def instrumented_metrics() -> bytes:
    if settings.VMS_METRICS_DIR is None:
        return generate_latest(REGISTRY)
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(
        registry,
        path=str(settings.VMS_METRICS_DIR),
    )
    return generate_latest(registry)


def _broker_readiness() -> tuple[bool, bool]:
    try:
        redis = Redis.from_url(
            settings.VMS_REDIS_URL,
            socket_connect_timeout=settings.VMS_OPERATIONAL_PROBE_TIMEOUT_SECONDS,
            socket_timeout=settings.VMS_OPERATIONAL_PROBE_TIMEOUT_SECONDS,
        )
        redis_ready = bool(redis.ping())
        now = datetime.now(UTC)
        live_workers = [
            worker
            for worker in Worker.all(queue=Queue(settings.VMS_QUEUE_NAME, connection=redis))
            if _worker_heartbeat_age_seconds(worker, now)
            < settings.VMS_RQ_WORKER_HEARTBEAT_MAX_AGE_SECONDS
        ]
        worker_ready = len(live_workers) == 1
        return redis_ready, worker_ready
    except Exception:  # noqa: BLE001 - no exception detail crosses this seam
        return False, False


def _worker_heartbeat_age_seconds(worker: object, now: datetime) -> float:
    heartbeat = getattr(worker, "last_heartbeat", None)
    if not isinstance(heartbeat, datetime):
        return math.inf
    if heartbeat.tzinfo is None:
        heartbeat = heartbeat.replace(tzinfo=UTC)
    return max(0.0, (now - heartbeat).total_seconds())


def _queue_payload(observations: object | None) -> dict[str, object]:
    if observations is None:
        return {
            "available": False,
            "capacity": settings.VMS_QUEUE_CAPACITY,
            "active": 0,
            "queued": 0,
            "running": 0,
            "admitted_total": 0,
            "rejected_total": 0,
            "succeeded_total": 0,
            "failed_total": 0,
            "worker_lost_total": 0,
            "wait_count": 0,
            "wait_accumulated_seconds": 0.0,
            "_wait_buckets": (),
        }
    return {
        "available": True,
        "capacity": settings.VMS_QUEUE_CAPACITY,
        "active": int(getattr(observations, "active_jobs")),
        "queued": int(getattr(observations, "queued_jobs")),
        "running": int(getattr(observations, "running_jobs")),
        "admitted_total": int(getattr(observations, "admitted_total")),
        "rejected_total": int(getattr(observations, "rejected_total")),
        "succeeded_total": int(getattr(observations, "succeeded_total")),
        "failed_total": int(getattr(observations, "failed_total")),
        "worker_lost_total": int(getattr(observations, "worker_lost_total")),
        "wait_count": int(getattr(observations, "queue_wait_count")),
        "wait_accumulated_seconds": float(getattr(observations, "queue_wait_ms_total")) / 1_000.0,
        "_wait_buckets": tuple(getattr(observations, "queue_wait_buckets")),
    }


def _with_queue_wait(
    telemetry: dict[str, object],
    observations: object,
) -> dict[str, object]:
    count = int(getattr(observations, "queue_wait_count"))
    total = float(getattr(observations, "queue_wait_ms_total")) / 1_000.0
    buckets = dict(getattr(observations, "queue_wait_buckets"))
    latency = dict(telemetry["latency_seconds"])
    latency["queue_wait"] = {
        "count": count,
        "mean": None if count <= 0 else total / count,
        "p50": _bucket_quantile(buckets, count, 0.5),
        "p95": _bucket_quantile(buckets, count, 0.95),
    }
    return {**telemetry, "latency_seconds": latency}


def _executor_payload(executor: object | None) -> dict[str, object]:
    if executor is None:
        return {
            "available": False,
            "artifact_ready": False,
            "runtime_initialized": False,
            "inference_warm": False,
            "warm_model": None,
            "state": "unavailable",
            "active_model": None,
            "resident_models": (),
            "device": None,
            "device_available": False,
            "artifacts_verified": False,
            "native_operator_available": False,
            "startup": None,
            "failure_code": None,
            "failure_present": False,
            "active_task": False,
            "active_task_age_seconds": None,
            "deadline_remaining_seconds": None,
            "precision": "float32",
        }
    return {
        "available": True,
        "artifact_ready": bool(getattr(executor, "artifact_ready")),
        "runtime_initialized": bool(getattr(executor, "runtime_initialized")),
        "inference_warm": bool(getattr(executor, "inference_warm")),
        "warm_model": getattr(executor, "warm_model"),
        "state": str(getattr(executor, "runtime_state")),
        "active_model": getattr(executor, "active_model"),
        "resident_models": tuple(getattr(executor, "resident_models")),
        "device": str(getattr(executor, "device_name")),
        "device_available": bool(getattr(executor, "device_available")),
        "artifacts_verified": bool(getattr(executor, "verified_artifacts")),
        "native_operator_available": bool(getattr(executor, "native_operator_available")),
        "startup": {
            "artifact_verification_seconds": (
                getattr(executor, "startup").artifact_verification_ms / 1_000.0
            ),
            "runtime_initialization_seconds": (
                getattr(executor, "startup").runtime_initialization_ms / 1_000.0
            ),
            "process_start_to_artifact_ready_seconds": (
                getattr(executor, "startup").process_start_to_artifact_ready_ms / 1_000.0
            ),
        },
        "failure_code": getattr(executor, "last_error"),
        "failure_present": getattr(executor, "last_error") is not None,
        "active_task": bool(getattr(executor, "active_task", False)),
        "active_task_age_seconds": (
            None
            if getattr(executor, "active_task_age_ms", None) is None
            else float(getattr(executor, "active_task_age_ms")) / 1_000.0
        ),
        "deadline_remaining_seconds": (
            None
            if getattr(executor, "deadline_remaining_ms", None) is None
            else float(getattr(executor, "deadline_remaining_ms")) / 1_000.0
        ),
        "precision": "float32",
    }


def _metric_samples(payload: bytes) -> tuple[object, ...]:
    samples: list[object] = []
    for family in text_string_to_metric_families(payload.decode("utf-8")):
        samples.extend(family.samples)
    return tuple(samples)


def _telemetry_payload(samples: tuple[object, ...]) -> dict[str, object]:
    return {
        "scope": "cumulative_since_process_start",
        "traffic": {
            "http": {
                outcome: int(_sample_sum(samples, "vms_http_requests_total", {"outcome": outcome}))
                for outcome in ("success", "client_error", "server_error")
            },
            "predictions": {
                mode: {
                    outcome: int(
                        _sample_sum(
                            samples,
                            "vms_predictions_total",
                            {"mode": mode, "outcome": outcome},
                        )
                    )
                    for outcome in ("succeeded", "failed")
                }
                for mode in ("detection", "full")
            },
        },
        "latency_seconds": {
            "http": _histogram(samples, "vms_http_request_duration_seconds"),
            "queue_wait": _histogram(samples, "vms_queue_wait_seconds"),
            "pipeline_total": _histogram(
                samples,
                "vms_pipeline_stage_seconds",
                {"stage": "total"},
            ),
            "dicom_decode": _histogram(
                samples,
                "vms_pipeline_stage_seconds",
                {"stage": "decode"},
            ),
            "detector_inference": _histogram(
                samples,
                "vms_model_inference_seconds",
                {"model": "detector"},
            ),
            "classifier_inference": _histogram(
                samples,
                "vms_model_inference_seconds",
                {"model": "classifier"},
            ),
            "detector_load": _histogram(
                samples,
                "vms_model_load_seconds",
                {"model": "detector"},
            ),
            "classifier_load": _histogram(
                samples,
                "vms_model_load_seconds",
                {"model": "classifier"},
            ),
        },
        "memory_bytes": {
            "cuda": {
                model: {
                    kind: int(
                        _sample_value(
                            samples,
                            "vms_cuda_memory_bytes",
                            {"model": model, "kind": kind},
                        )
                    )
                    for kind in (
                        "allocated",
                        "reserved",
                        "peak_allocated",
                        "peak_reserved",
                    )
                }
                for model in ("detector", "classifier")
            },
            "process_rss": {
                process: int(
                    _sample_value(
                        samples,
                        "vms_process_rss_bytes",
                        {"process": process},
                    )
                )
                for process in ("web", "worker", "executor")
            },
        },
        "events": {
            "cuda_oom_total": int(_sample_sum(samples, "vms_cuda_oom_total")),
            "roi_fallbacks_total": int(_sample_sum(samples, "vms_roi_fallbacks_total")),
            "lifecycle": {
                model: {
                    event: int(
                        _sample_sum(
                            samples,
                            "vms_model_lifecycle_total",
                            {"model": model, "event": event},
                        )
                    )
                    for event in ("load", "reuse", "switch", "unload", "failure")
                }
                for model in ("detector", "classifier")
            },
        },
    }


def _histogram(
    samples: tuple[object, ...],
    name: str,
    labels: dict[str, str] | None = None,
) -> dict[str, float | int | None]:
    labels = labels or {}
    count = _sample_sum(samples, f"{name}_count", labels)
    total = _sample_sum(samples, f"{name}_sum", labels)
    buckets: dict[float, float] = {}
    for sample in samples:
        if getattr(sample, "name", None) != f"{name}_bucket":
            continue
        sample_labels = getattr(sample, "labels", {})
        if not _labels_match(sample_labels, labels):
            continue
        upper_text = str(sample_labels.get("le", "+Inf"))
        upper = math.inf if upper_text == "+Inf" else float(upper_text)
        buckets[upper] = buckets.get(upper, 0.0) + float(getattr(sample, "value"))
    return {
        "count": int(count),
        "mean": None if count <= 0 else total / count,
        "p50": _bucket_quantile(buckets, count, 0.5),
        "p95": _bucket_quantile(buckets, count, 0.95),
    }


def _bucket_quantile(buckets: dict[float, float], count: float, quantile: float) -> float | None:
    if count <= 0 or not buckets:
        return None
    target = count * quantile
    previous_upper = 0.0
    previous_count = 0.0
    for upper, cumulative in sorted(buckets.items()):
        if cumulative < target:
            if math.isfinite(upper):
                previous_upper = upper
            previous_count = cumulative
            continue
        if not math.isfinite(upper):
            return previous_upper
        bucket_count = cumulative - previous_count
        if bucket_count <= 0:
            return upper
        position = (target - previous_count) / bucket_count
        return previous_upper + (upper - previous_upper) * position
    return None


def _sample_sum(
    samples: tuple[object, ...], name: str, labels: dict[str, str] | None = None
) -> float:
    labels = labels or {}
    return sum(
        float(getattr(sample, "value"))
        for sample in samples
        if getattr(sample, "name", None) == name
        and _labels_match(getattr(sample, "labels", {}), labels)
    )


def _sample_value(
    samples: tuple[object, ...], name: str, labels: dict[str, str] | None = None
) -> float:
    # livemostrecent-mode gauges emit one sample per labelset, so read it directly.
    labels = labels or {}
    for sample in samples:
        if getattr(sample, "name", None) == name and _labels_match(
            getattr(sample, "labels", {}), labels
        ):
            return float(getattr(sample, "value"))
    return 0.0


def _labels_match(observed: dict[str, str], expected: dict[str, str]) -> bool:
    return all(observed.get(name) == value for name, value in expected.items())
