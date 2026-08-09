"""Operational HTTP views that never initialize CUDA or load model weights."""

from __future__ import annotations

from ipaddress import ip_address, ip_network

from django.conf import settings
from django.http import HttpResponse
from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import extend_schema
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    generate_latest,
)
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from vision_model_serving.model_ids import MODEL_IDS

from .errors import public_error
from .operations import (
    OperationalSnapshot,
    instrumented_metrics,
    read_operational_snapshot,
)


class LivenessView(APIView):
    @extend_schema(responses={200: OpenApiTypes.OBJECT})
    def get(self, _request: Request) -> Response:
        return Response({"status": "alive"})


class ReadinessView(APIView):
    @extend_schema(responses={200: OpenApiTypes.OBJECT, 503: OpenApiTypes.OBJECT})
    def get(self, _request: Request) -> Response:
        snapshot = read_operational_snapshot()
        return Response(
            snapshot.readiness_dict(),
            status=200 if snapshot.status == "ready" else 503,
        )


class ModelInventoryView(APIView):
    @extend_schema(responses={200: OpenApiTypes.OBJECT})
    def get(self, _request: Request) -> Response:
        snapshot = read_operational_snapshot()
        executor = snapshot.executor
        return Response(
            {
                "manifest_id": snapshot.manifest_id,
                "models": [dict(model) for model in snapshot.models],
                "runtime": {
                    "state": executor["state"],
                    "initialized": executor["runtime_initialized"],
                    "artifact_ready": executor["artifact_ready"],
                    "inference_warm": executor["inference_warm"],
                    "warm_model": executor["warm_model"],
                    "startup": executor["startup"],
                    "active_model": executor["active_model"],
                    "resident_models": list(executor["resident_models"]),
                    "device": executor["device"],
                    "last_error": executor["failure_code"],
                },
            }
        )


class OperationsSnapshotView(APIView):
    @extend_schema(responses={200: OpenApiTypes.OBJECT})
    def get(self, _request: Request) -> Response:
        response = Response(read_operational_snapshot().as_dict())
        response["Cache-Control"] = "no-store"
        return response


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
            snapshot = read_operational_snapshot()
            if not snapshot.queue["available"] or not snapshot.executor["available"]:
                raise RuntimeError("operational snapshot is incomplete")
            payload = _metrics_payload(snapshot)
        except Exception:  # noqa: BLE001 - metrics are an internal process seam
            return public_error(
                request,
                "metrics_unavailable",
                "Metrics are temporarily unavailable.",
                503,
            )
        return HttpResponse(payload, content_type=CONTENT_TYPE_LATEST)


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


def _metrics_payload(operations: OperationalSnapshot) -> bytes:
    instrumented = instrumented_metrics()
    registry = CollectorRegistry()
    registry.register(_SnapshotCollector(operations))
    return instrumented + generate_latest(registry)


class _SnapshotCollector:
    def __init__(self, snapshot: OperationalSnapshot):
        self._queue = snapshot.queue
        self._executor = snapshot.executor

    def collect(self):
        for name, help_text, value in (
            ("vms_queue_active_jobs", "Reserved prediction jobs.", "active"),
            ("vms_queue_queued_jobs", "Queued prediction jobs.", "queued"),
            ("vms_queue_running_jobs", "Running prediction jobs.", "running"),
        ):
            yield GaugeMetricFamily(
                name,
                help_text,
                value=float(self._queue[value]),
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
                value=float(self._queue[value]),
            )
        yield CounterMetricFamily(
            "vms_queue_wait_accumulated_seconds",
            "Cumulative queue wait time.",
            value=float(self._queue["wait_accumulated_seconds"]),
        )
        for name, help_text, value in (
            (
                "vms_executor_artifact_ready",
                "Executor artifact readiness.",
                self._executor["artifact_ready"],
            ),
            (
                "vms_executor_runtime_initialized",
                "Executor runtime controller initialization state.",
                self._executor["runtime_initialized"],
            ),
            (
                "vms_executor_inference_warm",
                "Whether the active model is resident and ready.",
                self._executor["inference_warm"],
            ),
            (
                "vms_executor_artifacts_verified",
                "Artifact verification state.",
                self._executor["artifacts_verified"],
            ),
            (
                "vms_executor_device_available",
                "Configured CUDA device availability.",
                self._executor["device_available"],
            ),
            (
                "vms_executor_native_operator_available",
                "Native detector operator availability.",
                self._executor["native_operator_available"],
            ),
        ):
            yield GaugeMetricFamily(name, help_text, value=float(value))
        residents = GaugeMetricFamily(
            "vms_executor_model_resident",
            "Model residency by bounded model identity.",
            labels=("model",),
        )
        resident_models = set(self._executor["resident_models"])
        for model in MODEL_IDS:
            residents.add_metric((model,), float(model in resident_models))
        yield residents
