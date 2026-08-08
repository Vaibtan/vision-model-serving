"""Operational HTTP views that never initialize CUDA or load model weights."""

from __future__ import annotations

from ipaddress import ip_address, ip_network

from django.conf import settings
from django.http import HttpResponse
from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import extend_schema
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    CollectorRegistry,
    generate_latest,
    multiprocess,
)
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily
from redis import Redis
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView
from rq import Queue, Worker

from vision_model_serving.artifacts.manifest import load_manifest
from vision_model_serving.execution import GatewayObservations, GpuExecutorStatus

from .errors import public_error
from .runtime import executor_client, prediction_gateway


class LivenessView(APIView):
    @extend_schema(responses={200: OpenApiTypes.OBJECT})
    def get(self, _request: Request) -> Response:
        return Response({"status": "alive"})


class ReadinessView(APIView):
    @extend_schema(responses={200: OpenApiTypes.OBJECT, 503: OpenApiTypes.OBJECT})
    def get(self, _request: Request) -> Response:
        redis_ready = False
        worker_ready = False
        try:
            redis = Redis.from_url(settings.VMS_REDIS_URL)
            redis_ready = bool(redis.ping())
            queue = Queue(settings.VMS_QUEUE_NAME, connection=redis)
            worker_ready = bool(Worker.all(queue=queue))
        except Exception:  # noqa: BLE001 - readiness is a sanitized process seam
            redis_ready = False
            worker_ready = False
        executor = _executor_status()
        checks = {
            "redis": redis_ready,
            "rq_worker": worker_ready,
            "executor": executor.ready if executor is not None else False,
            "verified_artifacts": (
                executor.verified_artifacts if executor is not None else False
            ),
            "device": executor.device if executor is not None else False,
            "native_operator": (
                executor.native_operator if executor is not None else False
            ),
        }
        ready = all(checks.values())
        reasons = [
            f"{name}_unavailable" for name, passed in checks.items() if not passed
        ]
        return Response(
            {
                "status": "ready" if ready else "not_ready",
                "checks": checks,
                "reasons": reasons,
            },
            status=200 if ready else 503,
        )


class ModelInventoryView(APIView):
    @extend_schema(responses={200: OpenApiTypes.OBJECT})
    def get(self, _request: Request) -> Response:
        manifest = load_manifest(settings.BASE_DIR / "config" / "model-artifacts.json")
        runtime = {
            "state": "unavailable",
            "active_model": None,
            "resident_models": [],
            "device": None,
            "last_error": None,
        }
        executor = _executor_status()
        if executor is not None:
            runtime = {
                "state": executor.runtime_state,
                "active_model": executor.active_model,
                "resident_models": list(executor.resident_models),
                "device": executor.device_name,
                "last_error": executor.last_error,
            }
        return Response(
            {
                "manifest_id": manifest.manifest_id,
                "models": [
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
                ],
                "runtime": runtime,
            }
        )


class MetricsIntegrationView(APIView):
    @extend_schema(
        responses={
            200: OpenApiTypes.STR,
            403: OpenApiTypes.OBJECT,
            503: OpenApiTypes.OBJECT,
        }
    )
    def get(self, request: Request) -> HttpResponse | Response:
        if not settings.VMS_METRICS_ENABLED:
            return public_error(
                request,
                "metrics_not_configured",
                "Metrics instrumentation is not configured.",
                503,
            )
        if not _metrics_request_is_trusted(request):
            return public_error(
                request,
                "metrics_forbidden",
                "Metrics are restricted to trusted networks.",
                403,
            )
        try:
            observations = prediction_gateway().observations()
            executor = executor_client().status()
            payload = _metrics_payload(observations, executor)
        except Exception:  # noqa: BLE001 - metrics are an internal process seam
            return public_error(
                request,
                "metrics_unavailable",
                "Metrics are temporarily unavailable.",
                503,
            )
        return HttpResponse(payload, content_type=CONTENT_TYPE_LATEST)


def _executor_status() -> GpuExecutorStatus | None:
    try:
        return executor_client().status()
    except Exception:  # noqa: BLE001 - operational diagnostics are sanitized
        return None


def _metrics_request_is_trusted(request: Request) -> bool:
    try:
        address = ip_address(str(request.META.get("REMOTE_ADDR", "")))
        networks = tuple(
            ip_network(value, strict=False)
            for value in settings.VMS_METRICS_ALLOWED_NETWORKS
        )
    except ValueError:
        return False
    return any(address in network for network in networks)


def _metrics_payload(
    observations: GatewayObservations,
    executor: GpuExecutorStatus,
) -> bytes:
    if settings.VMS_METRICS_DIR is None:
        instrumented = generate_latest(REGISTRY)
    else:
        registry = CollectorRegistry(support_collectors_without_names=True)
        multiprocess.MultiProcessCollector(
            registry,
            path=str(settings.VMS_METRICS_DIR),
        )
        instrumented = generate_latest(registry)
    snapshot = CollectorRegistry()
    snapshot.register(_SnapshotCollector(observations, executor))
    return instrumented + generate_latest(snapshot)


class _SnapshotCollector:
    def __init__(
        self,
        observations: GatewayObservations,
        executor: GpuExecutorStatus,
    ):
        self._observations = observations
        self._executor = executor

    def collect(self):
        for name, help_text, value in (
            ("vms_queue_active_jobs", "Reserved prediction jobs.", "active_jobs"),
            ("vms_queue_queued_jobs", "Queued prediction jobs.", "queued_jobs"),
            ("vms_queue_running_jobs", "Running prediction jobs.", "running_jobs"),
        ):
            yield GaugeMetricFamily(
                name,
                help_text,
                value=float(getattr(self._observations, value)),
            )
        for name, help_text, value in (
            ("vms_queue_admitted", "Admitted prediction jobs.", "admitted_total"),
            ("vms_queue_rejected", "Rejected prediction jobs.", "rejected_total"),
            ("vms_queue_succeeded", "Succeeded prediction jobs.", "succeeded_total"),
            ("vms_queue_failed", "Failed prediction jobs.", "failed_total"),
            (
                "vms_queue_worker_lost",
                "Jobs lost with an RQ work-horse.",
                "worker_lost_total",
            ),
        ):
            yield CounterMetricFamily(
                name,
                help_text,
                value=float(getattr(self._observations, value)),
            )
        yield CounterMetricFamily(
            "vms_queue_wait_accumulated_seconds",
            "Cumulative queue wait time.",
            value=float(self._observations.queue_wait_ms_total) / 1_000.0,
        )
        for name, help_text, value in (
            ("vms_executor_ready", "Executor readiness.", self._executor.ready),
            (
                "vms_executor_artifacts_verified",
                "Artifact verification state.",
                self._executor.verified_artifacts,
            ),
            (
                "vms_executor_device_available",
                "Configured CUDA device availability.",
                self._executor.device,
            ),
            (
                "vms_executor_native_operator_available",
                "Native detector operator availability.",
                self._executor.native_operator,
            ),
        ):
            yield GaugeMetricFamily(name, help_text, value=float(value))
        residents = GaugeMetricFamily(
            "vms_executor_model_resident",
            "Model residency by bounded model identity.",
            labels=("model",),
        )
        resident_models = set(self._executor.resident_models)
        for model in ("focalnet-dino-detector", "mmbcd-classifier"):
            residents.add_metric((model,), float(model in resident_models))
        yield residents
