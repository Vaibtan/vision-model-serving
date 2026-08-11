from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import fakeredis


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "vision_model_serving.web.settings")

import django  # noqa: E402

django.setup()

from django.test import Client, override_settings  # noqa: E402

from vision_model_serving import observability  # noqa: E402
from vision_model_serving.web import operations  # noqa: E402
from vision_model_serving.web.runtime import prediction_gateway  # noqa: E402


def _reset_telemetry_cache() -> None:
    operations._telemetry_cache = None


def _ready_executor_status() -> SimpleNamespace:
    return SimpleNamespace(
        artifact_ready=True,
        verified_artifacts=True,
        runtime_initialized=True,
        device_available=True,
        native_operator_available=True,
        inference_warm=True,
        warm_model="detector",
        runtime_state="warm",
        active_model="detector",
        resident_models=("detector",),
        device_name="cuda:0",
        startup=SimpleNamespace(
            artifact_verification_ms=1.0,
            runtime_initialization_ms=1.0,
            process_start_to_artifact_ready_ms=1.0,
        ),
        last_error=None,
    )


class RequestPathRedisDeadlineTests(unittest.TestCase):
    def tearDown(self) -> None:
        prediction_gateway.cache_clear()

    def test_prediction_gateway_applies_connect_and_socket_deadlines(self) -> None:
        redis = fakeredis.FakeRedis()
        with TemporaryDirectory(prefix="vms-web-deadline-") as directory:
            with override_settings(
                VMS_JOB_ROOT=Path(directory),
                VMS_REDIS_CONNECT_TIMEOUT_SECONDS=1.25,
                VMS_REDIS_SOCKET_TIMEOUT_SECONDS=4.5,
            ):
                prediction_gateway.cache_clear()
                with patch(
                    "vision_model_serving.web.runtime.Redis.from_url",
                    return_value=redis,
                ) as from_url:
                    prediction_gateway()

        from_url.assert_called_once_with(
            "redis://127.0.0.1:6379/0",
            socket_connect_timeout=1.25,
            socket_timeout=4.5,
        )


class OperationalFailureTests(unittest.TestCase):
    client = Client()

    def setUp(self) -> None:
        # The snapshot cache would otherwise replay telemetry patched by an earlier test.
        _reset_telemetry_cache()

    def test_missing_manifest_is_a_bounded_not_ready_response(self) -> None:
        with (
            patch(
                "vision_model_serving.web.operations._broker_readiness",
                return_value=(False, False),
            ),
            patch(
                "vision_model_serving.web.operations.executor_client",
                side_effect=OSError("executor unavailable"),
            ),
            patch(
                "vision_model_serving.web.operations.load_manifest",
                side_effect=OSError("manifest unavailable"),
            ),
            patch(
                "vision_model_serving.web.operations.instrumented_metrics",
                return_value=b"",
            ),
        ):
            response = self.client.get("/readyz", HTTP_HOST="localhost")

        self.assertEqual(response.status_code, 503)
        payload = response.json()
        self.assertFalse(payload["checks"]["manifest_available"])
        self.assertTrue(payload["checks"]["telemetry_available"])
        self.assertIn("manifest_unavailable", payload["reasons"])

    def test_broken_telemetry_is_a_bounded_not_ready_response(self) -> None:
        # The 503 here comes from the broker and executor being down; telemetry
        # unavailability is reported as informational and no longer gates readiness.
        with (
            patch(
                "vision_model_serving.web.operations._broker_readiness",
                return_value=(False, False),
            ),
            patch(
                "vision_model_serving.web.operations.executor_client",
                side_effect=OSError("executor unavailable"),
            ),
            patch(
                "vision_model_serving.web.operations.instrumented_metrics",
                side_effect=RuntimeError("collector unavailable"),
            ),
        ):
            response = self.client.get("/readyz", HTTP_HOST="localhost")

        self.assertEqual(response.status_code, 503)
        payload = response.json()
        self.assertTrue(payload["checks"]["manifest_available"])
        self.assertFalse(payload["checks"]["telemetry_available"])
        self.assertIn("telemetry_unavailable", payload["reasons"])

    def test_readiness_does_not_gate_on_telemetry(self) -> None:
        executor_status = _ready_executor_status()
        with (
            patch(
                "vision_model_serving.web.operations._broker_readiness",
                return_value=(True, True),
            ),
            patch(
                "vision_model_serving.web.operations.prediction_gateway",
                side_effect=RuntimeError("gateway unavailable"),
            ),
            patch(
                "vision_model_serving.web.operations.executor_client",
                return_value=SimpleNamespace(status=lambda: executor_status),
            ),
            patch(
                "vision_model_serving.web.operations.instrumented_metrics",
                side_effect=RuntimeError("collector unavailable"),
            ),
        ):
            response = self.client.get("/readyz", HTTP_HOST="localhost")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ready")
        self.assertFalse(payload["checks"]["telemetry_available"])
        self.assertIn("telemetry_unavailable", payload["reasons"])


class TelemetrySnapshotCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        _reset_telemetry_cache()

    def tearDown(self) -> None:
        _reset_telemetry_cache()

    def test_immediate_snapshots_collect_metrics_once(self) -> None:
        with (
            patch(
                "vision_model_serving.web.operations._broker_readiness",
                return_value=(False, False),
            ),
            patch(
                "vision_model_serving.web.operations.executor_client",
                side_effect=OSError("executor unavailable"),
            ),
            patch(
                "vision_model_serving.web.operations.instrumented_metrics",
                return_value=b"",
            ) as collector,
        ):
            first = operations.read_operational_snapshot()
            second = operations.read_operational_snapshot()

        self.assertEqual(collector.call_count, 1)
        self.assertEqual(first.telemetry, second.telemetry)

    def test_expired_cache_collects_metrics_again(self) -> None:
        with (
            patch(
                "vision_model_serving.web.operations._broker_readiness",
                return_value=(False, False),
            ),
            patch(
                "vision_model_serving.web.operations.executor_client",
                side_effect=OSError("executor unavailable"),
            ),
            patch(
                "vision_model_serving.web.operations.instrumented_metrics",
                return_value=b"",
            ) as collector,
        ):
            operations.read_operational_snapshot()
            cached_at, cached_value = operations._telemetry_cache
            operations._telemetry_cache = (
                cached_at - operations._TELEMETRY_CACHE_SECONDS,
                cached_value,
            )
            operations.read_operational_snapshot()

        self.assertEqual(collector.call_count, 2)


class TelemetryWriteGuardTests(unittest.TestCase):
    def test_http_response_metric_failure_does_not_propagate(self) -> None:
        # Simulates ENOSPC on the multiprocess metrics tmpfs: the middleware calls
        # record_http_response unguarded, so it must swallow the failure itself.
        with patch.object(
            observability._HTTP_REQUESTS,
            "labels",
            side_effect=OSError(28, "No space left on device"),
        ):
            self.assertIsNone(
                observability.record_http_response(
                    SimpleNamespace(method="GET"),
                    status_code=200,
                    duration_seconds=0.01,
                )
            )

    def test_dicom_metric_failure_does_not_propagate(self) -> None:
        with patch.object(
            observability._DICOM,
            "labels",
            side_effect=OSError(28, "No space left on device"),
        ):
            self.assertIsNone(observability.record_dicom("accepted"))


class MediaNegotiationTests(unittest.TestCase):
    client = Client()

    def test_preview_accepts_png_even_when_validation_returns_json(self) -> None:
        response = self.client.post(
            "/api/v1/dicom-preview",
            data={},
            HTTP_ACCEPT="image/png",
            HTTP_HOST="localhost",
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response["Content-Type"], "application/json")
        self.assertEqual(response.json()["error"]["code"], "invalid_request")

    @override_settings(VMS_METRICS_ENABLED=False)
    def test_metrics_accepts_text_plain_even_when_unavailable_response_is_json(
        self,
    ) -> None:
        response = self.client.get(
            "/metrics",
            HTTP_ACCEPT="text/plain",
            HTTP_HOST="localhost",
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response["Content-Type"], "application/json")
        self.assertEqual(
            response.json()["error"]["code"],
            "metrics_not_configured",
        )


class OpenApiOperationalContractTests(unittest.TestCase):
    client = Client()

    def test_operational_responses_are_named_schemas(self) -> None:
        response = self.client.get(
            "/api/schema/?format=json",
            HTTP_HOST="localhost",
        )

        self.assertEqual(response.status_code, 200)
        schema = json.loads(response.content)
        cases = (
            ("/livez", "200"),
            ("/readyz", "200"),
            ("/readyz", "503"),
            ("/api/v1/models", "200"),
            ("/api/v1/operations", "200"),
        )
        for path, status_code in cases:
            with self.subTest(path=path, status_code=status_code):
                response_schema = schema["paths"][path]["get"]["responses"][status_code]["content"][
                    "application/json"
                ]["schema"]
                self.assertIn("$ref", response_schema)

    def test_binary_and_text_success_media_are_declared(self) -> None:
        response = self.client.get(
            "/api/schema/?format=json",
            HTTP_HOST="localhost",
        )

        schema = json.loads(response.content)
        preview_content = schema["paths"]["/api/v1/dicom-preview"]["post"]["responses"]["200"][
            "content"
        ]
        metrics_content = schema["paths"]["/metrics"]["get"]["responses"]["200"]["content"]
        self.assertIn("image/png", preview_content)
        self.assertIn("text/plain", metrics_content)


if __name__ == "__main__":
    unittest.main()
