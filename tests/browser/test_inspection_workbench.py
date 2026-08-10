from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
from threading import Thread
from typing import Any
import unittest
from wsgiref.simple_server import WSGIServer, make_server

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "vision_model_serving.web.settings")
os.environ.setdefault("VMS_ALLOWED_HOSTS", "localhost,127.0.0.1")

import django  # noqa: E402

django.setup()

from django.core.wsgi import get_wsgi_application  # noqa: E402
from PIL import Image  # noqa: E402

BROWSER_REQUIRED = os.environ.get("VMS_BROWSER_REQUIRED", "").strip().lower() in {
    "1",
    "true",
    "yes",
}

try:  # Browser dependencies are an explicit CI/L4 lane.
    from playwright.sync_api import (  # noqa: E402
        Browser,
        Error as PlaywrightError,
        Page,
        Playwright,
        sync_playwright,
    )
except ImportError:  # pragma: no cover - depends on the selected dependency groups
    if BROWSER_REQUIRED:
        raise
    Browser = Page = Playwright = Any  # type: ignore[misc,assignment]
    PlaywrightError = RuntimeError  # type: ignore[assignment]
    sync_playwright = None

from tests.test_dicom_canonicalization import dicom_bytes  # noqa: E402


def prediction_result() -> dict[str, object]:
    rois = [
        {
            "canonical_xyxy": [
                40 + index * 30,
                60 + index * 25,
                180 + index * 30,
                210 + index * 25,
            ],
            "score": 0.95 - index * 0.04,
            "padded": False,
        }
        for index in range(8)
    ]
    runtime = {
        "reused": False,
        "load_ms": 100.0,
        "inference_ms": 20.0,
        "switch_ms": 10.0,
    }
    memory = {
        "allocated_bytes": 10_000,
        "reserved_bytes": 20_000,
        "peak_allocated_bytes": 30_000,
        "peak_reserved_bytes": 40_000,
    }
    return {
        "mode": "full",
        "geometry": {"canonical_width": 1024, "canonical_height": 1024},
        "detector": {
            "classifier_rois": rois,
            "prediction_sha256": "1" * 64,
        },
        "classification": {
            "predicted_class_index": 1,
            "probabilities": [0.25, 0.75],
            "prediction_sha256": "2" * 64,
            "attention": {"roi_weights": [0.125] * 8},
        },
        "timings": {
            "decode_ms": 5.0,
            "total_ms": 155.0,
            "detector": {"runtime": runtime, "memory": memory},
            "classifier": {"runtime": runtime, "memory": memory},
        },
        "provenance": {
            "precision": "float32",
            "detector": {"id": "focalnet-dino-detector", "sha256": "a" * 64},
            "classifier": {"id": "mmbcd-classifier", "sha256": "b" * 64},
        },
        "warnings": [],
        "disclaimer": "Research use only; semantics unverified.",
    }


def fulfill(route: object, payload: dict[str, object], *, status: int = 200) -> None:
    route.fulfill(
        status=status,
        content_type="application/json",
        body=json.dumps(payload),
    )


@unittest.skipUnless(sync_playwright is not None, "Playwright group is not installed")
class InspectionWorkbenchBrowserTests(unittest.TestCase):
    playwright: Playwright
    browser: Browser
    server: WSGIServer
    server_thread: Thread
    live_server_url: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = make_server("127.0.0.1", 0, get_wsgi_application())
        cls.live_server_url = f"http://127.0.0.1:{cls.server.server_port}"
        cls.server_thread = Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()
        cls.playwright = sync_playwright().start()
        try:
            cls.browser = cls.playwright.chromium.launch(headless=True)
        except PlaywrightError as error:
            cls.playwright.stop()
            cls.server.shutdown()
            cls.server.server_close()
            cls.server_thread.join(2)
            if BROWSER_REQUIRED:
                raise
            raise unittest.SkipTest("Playwright Chromium is not installed") from error

    @classmethod
    def tearDownClass(cls) -> None:
        cls.browser.close()
        cls.playwright.stop()
        cls.server.shutdown()
        cls.server.server_close()
        cls.server_thread.join(2)

    def setUp(self) -> None:
        self.context = self.browser.new_context(accept_downloads=True)
        self.page = self.context.new_page()
        self.browser_errors: list[str] = []
        self.page.on("console", self._record_console_error)
        self.page.on("pageerror", lambda error: self.browser_errors.append(str(error)))

    def tearDown(self) -> None:
        self.context.close()

    def test_full_workflow_renders_selects_and_exports_evidence(self) -> None:
        result = prediction_result()
        status_calls = 0
        idempotency_key = ""

        def status_route(route: object) -> None:
            nonlocal status_calls
            states = ("queued", "started", "succeeded")
            state = states[min(status_calls, len(states) - 1)]
            status_calls += 1
            fulfill(route, {"prediction_id": "browser-job", "state": state})

        self.page.route(
            "**/api/v1/predictions/browser-job/result",
            lambda route: fulfill(route, {"result": result}),
        )
        self.page.route("**/api/v1/predictions/browser-job", status_route)

        def submission(route: object) -> None:
            nonlocal idempotency_key
            idempotency_key = route.request.headers.get("idempotency-key", "")
            fulfill(
                route,
                {"prediction_id": "browser-job", "state": "queued"},
                status=202,
            )

        self.page.route("**/api/v1/predictions", submission)
        self.page.route(
            "**/api/v1/models",
            lambda route: fulfill(
                route,
                {
                    "runtime": {
                        "artifact_ready": True,
                        "initialized": True,
                        "inference_warm": True,
                        "warm_model": "mmbcd-classifier",
                        "active_model": "mmbcd-classifier",
                        "resident_models": ["mmbcd-classifier"],
                    }
                },
            ),
        )
        with TemporaryDirectory() as directory:
            dicom_path = Path(directory) / "synthetic.dcm"
            pixels = np.arange(1024 * 1024, dtype=np.uint16).reshape(1024, 1024)
            dicom_path.write_bytes(dicom_bytes(pixels))

            self.page.add_init_script(
                "Object.defineProperty(Crypto.prototype, 'randomUUID', "
                "{value: undefined, configurable: true});"
            )
            self.page.goto(self.live_server_url)
            self.page.locator("#dicom").set_input_files(dicom_path)
            self.page.locator("#preview-image").wait_for(state="visible")
            self.page.locator("#clinical-history").fill("Prior surgery")
            self.page.get_by_role("button", name="Run inference").click()
            self.page.locator("#results-panel.is-visible").wait_for()

            self.assertEqual(self.page.locator("#roi-overlay rect").count(), 8)
            self.assertEqual(self.page.locator(".roi-card canvas").count(), 8)
            self.page.locator(".roi-card").nth(2).click()
            self.assertTrue(
                self.page.locator(".roi-card")
                .nth(2)
                .evaluate("element => element.classList.contains('is-selected')")
            )
            self.assertIn("R3", self.page.locator("#selected-roi-detail").inner_text())
            self.assertIn("75.0%", self.page.locator("#summary-grid").inner_text())
            self.assertIn(
                "mmbcd-classifier",
                self.page.locator("#runtime-values").inner_text(),
            )

            with self.page.expect_download() as json_download:
                self.page.get_by_role("button", name="Download JSON").click()
            json_path = Path(directory) / "result.json"
            json_download.value.save_as(json_path)
            exported = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertNotIn("Prior surgery", json.dumps(exported))

            with self.page.expect_download() as png_download:
                self.page.get_by_role("button", name="Download overlay PNG").click()
            png_path = Path(directory) / "overlay.png"
            png_download.value.save_as(png_path)
            with Image.open(png_path) as image:
                self.assertEqual(image.size, (1024, 1024))
                colors = image.convert("RGB").getcolors(maxcolors=2_000_000)
                self.assertTrue(colors and len(colors) > 2)

        self.assertGreaterEqual(status_calls, 3)
        self.assertRegex(
            idempotency_key,
            re.compile(
                r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-"
                r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
            ),
        )
        self.assertEqual(self.browser_errors, [])

    def test_detection_mode_never_submits_hidden_history(self) -> None:
        observed_body = b""

        def submission(route: object) -> None:
            nonlocal observed_body
            observed_body = route.request.post_data_buffer or b""
            fulfill(
                route,
                {"prediction_id": "browser-job", "state": "failed"},
                status=202,
            )

        self.page.route(
            "**/api/v1/predictions/browser-job",
            lambda route: fulfill(
                route,
                {"prediction_id": "browser-job", "state": "failed"},
            ),
        )
        self.page.route("**/api/v1/predictions", submission)
        with TemporaryDirectory() as directory:
            dicom_path = Path(directory) / "synthetic.dcm"
            dicom_path.write_bytes(dicom_bytes(np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)))
            self.page.goto(self.live_server_url)
            self.page.locator("#dicom").set_input_files(dicom_path)
            self.page.locator("#preview-image").wait_for(state="visible")
            self.page.locator("#clinical-history").fill("must not leave browser")
            self.page.get_by_label("Detection", exact=True).check()
            self.page.get_by_role("button", name="Run inference").click()
            self.page.wait_for_timeout(100)

        self.assertNotIn(b"must not leave browser", observed_body)
        self.assertNotIn(b"clinical_history", observed_body)

    def _record_console_error(self, message: object) -> None:
        if message.type == "error":
            self.browser_errors.append(message.text)


if __name__ == "__main__":
    unittest.main()
