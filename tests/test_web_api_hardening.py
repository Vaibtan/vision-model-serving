from __future__ import annotations

from io import BytesIO
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import fakeredis
import numpy as np
import pydicom


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "vision_model_serving.web.settings")
os.environ.setdefault("VMS_CACHE_BACKEND", "django.core.cache.backends.locmem.LocMemCache")

import django  # noqa: E402

django.setup()

from django.core.cache import cache  # noqa: E402
from django.core.files.uploadedfile import SimpleUploadedFile  # noqa: E402
from django.test import Client, override_settings  # noqa: E402
from rest_framework.throttling import ScopedRateThrottle  # noqa: E402

from tests.test_dicom_canonicalization import dicom_bytes  # noqa: E402
from vision_model_serving.dicom import DicomCanonicalizer  # noqa: E402
from vision_model_serving.web.runtime import prediction_gateway  # noqa: E402


def mammogram_bytes() -> bytes:
    return dicom_bytes(np.full((64, 64), 512, dtype=np.uint16))


def upload(payload: bytes) -> SimpleUploadedFile:
    return SimpleUploadedFile("case.dcm", payload, content_type="application/dicom")


class HardeningTestCase(unittest.TestCase):
    client = Client()

    def setUp(self) -> None:
        # Throttle history and the cached gateway are process singletons.
        cache.clear()
        prediction_gateway.cache_clear()

    def tearDown(self) -> None:
        cache.clear()
        prediction_gateway.cache_clear()

    def submit_async(
        self,
        payload: bytes,
        *,
        threshold: str | None = None,
        **extra: str,
    ) -> object:
        data: dict[str, object] = {"dicom": upload(payload), "mode": "detection"}
        if threshold is not None:
            data["detector_score_threshold"] = threshold
        redis = fakeredis.FakeRedis()
        with TemporaryDirectory(prefix="vms-hardening-jobs-") as directory:
            with override_settings(VMS_JOB_ROOT=Path(directory)):
                prediction_gateway.cache_clear()
                with patch(
                    "vision_model_serving.web.runtime.Redis.from_url",
                    return_value=redis,
                ):
                    return self.client.post(
                        "/api/v1/predictions",
                        data=data,
                        HTTP_HOST="localhost",
                        HTTP_PREFER="respond-async",
                        **extra,
                    )


class BrowserOriginProtectionTests(HardeningTestCase):
    def test_cross_site_fetch_metadata_is_rejected(self) -> None:
        response = self.client.post(
            "/api/v1/predictions",
            data={},
            HTTP_HOST="localhost",
            HTTP_SEC_FETCH_SITE="cross-site",
        )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(
            response.json()["error"]["code"],
            "cross_site_request_rejected",
        )

    def test_foreign_origin_is_rejected_on_every_post_route(self) -> None:
        for url in ("/api/v1/predictions", "/api/v1/dicom-preview"):
            with self.subTest(url=url):
                response = self.client.post(
                    url,
                    data={},
                    HTTP_HOST="localhost",
                    HTTP_ORIGIN="https://attacker.example",
                )

                self.assertEqual(response.status_code, 403)
                self.assertEqual(
                    response.json()["error"]["code"],
                    "cross_site_request_rejected",
                )

    def test_same_origin_fetch_metadata_passes_the_guard(self) -> None:
        response = self.client.post(
            "/api/v1/predictions",
            data={},
            HTTP_HOST="localhost",
            HTTP_SEC_FETCH_SITE="same-origin",
        )

        # The guard passed; the empty submission then fails field validation.
        self.assertNotEqual(response.status_code, 403)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "invalid_request")

    def test_matching_origin_passes_the_guard(self) -> None:
        response = self.client.post(
            "/api/v1/predictions",
            data={},
            HTTP_HOST="localhost",
            HTTP_ORIGIN="http://localhost",
        )

        self.assertNotEqual(response.status_code, 403)
        self.assertEqual(response.status_code, 400)

    def test_matching_host_with_foreign_scheme_is_rejected(self) -> None:
        response = self.client.post(
            "/api/v1/predictions",
            data={},
            HTTP_HOST="localhost",
            HTTP_ORIGIN="https://localhost",
        )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(
            response.json()["error"]["code"],
            "cross_site_request_rejected",
        )

    def test_header_less_non_browser_clients_pass_the_guard(self) -> None:
        response = self.client.post(
            "/api/v1/predictions",
            data={},
            HTTP_HOST="localhost",
        )

        self.assertNotEqual(response.status_code, 403)
        self.assertEqual(response.status_code, 400)


class CacheHeaderTests(HardeningTestCase):
    def test_submission_error_response_is_no_store(self) -> None:
        response = self.client.post(
            "/api/v1/predictions",
            data={},
            HTTP_HOST="localhost",
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response["Cache-Control"], "no-store")

    def test_status_and_result_responses_are_no_store(self) -> None:
        redis = fakeredis.FakeRedis()
        with patch(
            "vision_model_serving.web.runtime.Redis.from_url",
            return_value=redis,
        ):
            status_response = self.client.get(
                "/api/v1/predictions/unknown",
                HTTP_HOST="localhost",
            )
            result_response = self.client.get(
                "/api/v1/predictions/unknown/result",
                HTTP_HOST="localhost",
            )

        for response in (status_response, result_response):
            with self.subTest(path=response.request["PATH_INFO"]):
                self.assertEqual(response.status_code, 404)
                self.assertEqual(response["Cache-Control"], "no-store")


class SubmissionAdmissionTests(HardeningTestCase):
    def test_admission_never_decodes_pixels_and_echoes_the_threshold(self) -> None:
        with patch.object(
            DicomCanonicalizer,
            "decode",
            side_effect=AssertionError("the web tier must not decode pixels"),
        ):
            response = self.submit_async(mammogram_bytes(), threshold="0.25")

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response["Cache-Control"], "no-store")
        body = response.json()
        self.assertEqual(body["request"], {"detector_score_threshold": 0.25})
        self.assertIn("status_url", body)
        self.assertIn("result_url", body)

    def test_omitted_threshold_is_echoed_as_null(self) -> None:
        response = self.submit_async(mammogram_bytes())

        self.assertEqual(response.status_code, 202)
        self.assertEqual(
            response.json()["request"],
            {"detector_score_threshold": None},
        )

    def test_structurally_invalid_dicom_is_rejected_before_admission(self) -> None:
        response = self.submit_async(b"never-a-dicom-object")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "dicom_invalid")
        self.assertEqual(response["Cache-Control"], "no-store")


class ModalityPolicyTests(HardeningTestCase):
    def test_disallowed_modality_is_rejected(self) -> None:
        dataset = pydicom.dcmread(BytesIO(mammogram_bytes()))
        dataset.Modality = "OT"
        stream = BytesIO()
        dataset.save_as(stream, enforce_file_format=True)

        with override_settings(VMS_ALLOWED_MODALITIES=("MG",)):
            response = self.client.post(
                "/api/v1/predictions",
                data={"dicom": upload(stream.getvalue()), "mode": "detection"},
                HTTP_HOST="localhost",
            )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "dicom_modality_rejected")
        self.assertEqual(response["Cache-Control"], "no-store")

    def test_allowed_modality_passes_the_policy(self) -> None:
        with override_settings(VMS_ALLOWED_MODALITIES=("MG",)):
            response = self.submit_async(mammogram_bytes())

        self.assertEqual(response.status_code, 202)

    def test_empty_allowlist_accepts_any_modality(self) -> None:
        dataset = pydicom.dcmread(BytesIO(mammogram_bytes()))
        dataset.Modality = "OT"
        stream = BytesIO()
        dataset.save_as(stream, enforce_file_format=True)

        with override_settings(VMS_ALLOWED_MODALITIES=()):
            response = self.submit_async(stream.getvalue())

        self.assertEqual(response.status_code, 202)


class ThrottleTests(HardeningTestCase):
    def test_preview_posts_beyond_the_rate_get_the_throttled_envelope(self) -> None:
        # SimpleRateThrottle binds THROTTLE_RATES at class definition, so the
        # rate must be patched on the class rather than via override_settings.
        with patch.dict(ScopedRateThrottle.THROTTLE_RATES, {"preview": "1/minute"}):
            first = self.client.post(
                "/api/v1/dicom-preview",
                data={},
                HTTP_HOST="localhost",
            )
            second = self.client.post(
                "/api/v1/dicom-preview",
                data={},
                HTTP_HOST="localhost",
            )

        # The first request consumes the 1/minute allowance (and fails
        # validation); the second is throttled before the view body runs.
        self.assertEqual(first.status_code, 400)
        self.assertEqual(second.status_code, 429)
        payload = second.json()
        self.assertEqual(payload["error"]["code"], "throttled")
        self.assertIn("message", payload["error"])
        self.assertIn("request_id", payload["error"])
        self.assertEqual(payload["error"]["details"], {})


if __name__ == "__main__":
    unittest.main()
