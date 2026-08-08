"""Operational HTTP views that never initialize CUDA or load model weights."""

from __future__ import annotations

from django.conf import settings
from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import extend_schema
from redis import Redis
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView
from rq import Worker

from vision_model_serving.artifacts.manifest import load_manifest

from .errors import public_error


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
            worker_ready = bool(Worker.all(connection=redis))
        except Exception:  # noqa: BLE001 - readiness is a sanitized process seam
            redis_ready = False
            worker_ready = False
        checks = {
            "redis": redis_ready,
            "rq_worker": worker_ready,
            "executor": False,
            "verified_artifacts": False,
            "device": False,
            "native_operator": False,
        }
        ready = all(checks.values())
        return Response(
            {
                "status": "ready" if ready else "not_ready",
                "checks": checks,
            },
            status=200 if ready else 503,
        )


class ModelInventoryView(APIView):
    @extend_schema(responses={200: OpenApiTypes.OBJECT})
    def get(self, _request: Request) -> Response:
        manifest = load_manifest(settings.BASE_DIR / "config" / "model-artifacts.json")
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
                "runtime": {
                    "state": "not_reported",
                    "active_model": None,
                    "resident_models": [],
                    "device": None,
                    "last_error": None,
                },
            }
        )


class MetricsIntegrationView(APIView):
    @extend_schema(responses={503: OpenApiTypes.OBJECT})
    def get(self, request: Request) -> Response:
        return public_error(
            request,
            "metrics_not_configured",
            "Metrics instrumentation is not configured.",
            503,
        )
