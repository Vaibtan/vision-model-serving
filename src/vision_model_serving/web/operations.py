"""One bounded operational snapshot for readiness, monitoring, and metrics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import math
from typing import Any

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
    manifest_id: str
    models: tuple[dict[str, object], ...]
    telemetry: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "captured_at": self.captured_at,
            "status": self.status,
            "checks": dict(self.checks),
            "reasons": list(self.reasons),
            "queue": dict(self.queue),
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

    checks = {
        "redis": redis_ready,
        "rq_worker": worker_ready,
        "executor": executor.ready if executor is not None else False,
        "verified_artifacts": (
            executor.verified_artifacts if executor is not None else False
        ),
        "device": executor.device if executor is not None else False,
        "native_operator": executor.native_operator if executor is not None else False,
    }
    ready = all(checks.values())
    manifest = load_manifest(settings.BASE_DIR / "config" / "model-artifacts.json")
    samples = _metric_samples(instrumented_metrics())
    return OperationalSnapshot(
        captured_at=datetime.now(UTC).isoformat(),
        status="ready" if ready else "not_ready",
        checks=checks,
        reasons=tuple(
            f"{name}_unavailable" for name, passed in checks.items() if not passed
        ),
        queue=_queue_payload(observations),
        executor=_executor_payload(executor),
        manifest_id=manifest.manifest_id,
        models=tuple(
            {
                "id": artifact.id,
                "role": artifact.role,
                "sha256": artifact.sha256,
                "strict_load_verified": artifact.strict_load_verified,
                "semantics_status": artifact.semantics_status,
                "class_names": (
                    list(artifact.class_names)
                    if artifact.class_names is not None
                    else None
                ),
                "decision_threshold": artifact.decision_threshold,
            }
            for artifact in manifest.artifacts
        ),
        telemetry=_telemetry_payload(samples),
    )


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
        worker_ready = bool(
            Worker.all(queue=Queue(settings.VMS_QUEUE_NAME, connection=redis))
        )
        return redis_ready, worker_ready
    except Exception:  # noqa: BLE001 - no exception detail crosses this seam
        return False, False


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
            "wait_accumulated_seconds": 0.0,
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
        "wait_accumulated_seconds": float(
            getattr(observations, "queue_wait_ms_total")
        )
        / 1_000.0,
    }


def _executor_payload(executor: object | None) -> dict[str, object]:
    if executor is None:
        return {
            "available": False,
            "ready": False,
            "state": "unavailable",
            "active_model": None,
            "resident_models": (),
            "device": None,
            "device_available": False,
            "artifacts_verified": False,
            "native_operator": False,
            "failure_code": None,
            "failure_present": False,
            "precision": "float32",
        }
    return {
        "available": True,
        "ready": bool(getattr(executor, "ready")),
        "state": str(getattr(executor, "runtime_state")),
        "active_model": getattr(executor, "active_model"),
        "resident_models": tuple(getattr(executor, "resident_models")),
        "device": str(getattr(executor, "device_name")),
        "device_available": bool(getattr(executor, "device")),
        "artifacts_verified": bool(getattr(executor, "verified_artifacts")),
        "native_operator": bool(getattr(executor, "native_operator")),
        "failure_code": getattr(executor, "last_error"),
        "failure_present": getattr(executor, "last_error") is not None,
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
                outcome: int(
                    _sample_sum(samples, "vms_http_requests_total", {"outcome": outcome})
                )
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
                        _sample_max(
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
                    _sample_max(
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
            "roi_fallbacks_total": int(
                _sample_sum(samples, "vms_roi_fallbacks_total")
            ),
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


def _bucket_quantile(
    buckets: dict[float, float], count: float, quantile: float
) -> float | None:
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


def _sample_max(
    samples: tuple[object, ...], name: str, labels: dict[str, str] | None = None
) -> float:
    labels = labels or {}
    values = [
        float(getattr(sample, "value"))
        for sample in samples
        if getattr(sample, "name", None) == name
        and _labels_match(getattr(sample, "labels", {}), labels)
    ]
    return max(values, default=0.0)


def _labels_match(observed: dict[str, str], expected: dict[str, str]) -> bool:
    return all(observed.get(name) == value for name, value in expected.items())
