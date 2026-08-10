from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import fakeredis


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "vision_model_serving.web.settings")

import django  # noqa: E402

django.setup()

from django.test import Client, override_settings  # noqa: E402

from vision_model_serving.web.runtime import prediction_gateway  # noqa: E402


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
