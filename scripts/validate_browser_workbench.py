#!/usr/bin/env python3
"""Drive the real packaged inspection workbench through a Chromium browser."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from playwright.sync_api import sync_playwright
from vision_model_serving.validation.acceptance_contract import (
    PACKAGED_ACCEPTANCE_HISTORY,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--dicom", type=Path, required=True)
    parser.add_argument("--expected-detector-sha256", required=True)
    parser.add_argument("--expected-classifier-sha256", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    args = parser.parse_args()
    timeout_ms = int(args.timeout_seconds * 1_000)
    browser_errors: list[str] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()
        page.on(
            "console",
            lambda message: (
                browser_errors.append(message.text)
                if message.type == "error"
                else None
            ),
        )
        page.on("pageerror", lambda error: browser_errors.append(str(error)))
        page.goto(args.base_url, wait_until="networkidle", timeout=timeout_ms)
        page.evaluate(
            """
            window.__vmsObservedStates = [];
            const target = document.querySelector('#prediction-status');
            new MutationObserver(() => {
              window.__vmsObservedStates.push(target.textContent.trim());
            }).observe(target, {childList: true, characterData: true, subtree: true});
            """
        )
        page.locator("#dicom").set_input_files(args.dicom)
        page.locator("#preview-image").wait_for(state="visible", timeout=timeout_ms)
        page.locator("#clinical-history").fill(PACKAGED_ACCEPTANCE_HISTORY)
        page.get_by_role("button", name="Run inference").click()
        page.locator("#results-panel.is-visible").wait_for(
            state="visible",
            timeout=timeout_ms,
        )
        if page.locator("#roi-overlay rect").count() != 8:
            raise RuntimeError("browser overlay did not render eight ROIs")
        if page.locator(".roi-card canvas").count() != 8:
            raise RuntimeError("browser ROI gallery did not render eight crops")
        page.locator(".roi-card").nth(0).click()
        if "detector score" not in page.locator("#selected-roi-detail").inner_text():
            raise RuntimeError("browser ROI selection did not expose score/attention")
        summary = page.locator("#summary-grid").inner_text()
        runtime = page.locator("#runtime-values").inner_text()
        if (
            "class probabilities" not in summary.casefold()
            or "mmbcd-classifier" not in runtime
        ):
            raise RuntimeError(
                "browser result evidence panels are incomplete: "
                f"summary={summary!r}, runtime={runtime!r}"
            )
        states = page.evaluate("window.__vmsObservedStates")
        if "queued" not in states or "succeeded" not in states:
            raise RuntimeError(f"browser did not observe queue completion: {states!r}")
        with TemporaryDirectory(prefix="vms-browser-") as directory:
            output = Path(directory)
            with page.expect_download(timeout=timeout_ms) as json_download:
                page.get_by_role("button", name="Download JSON").click()
            json_path = output / "prediction-result.json"
            json_download.value.save_as(json_path)
            result = json.loads(json_path.read_text(encoding="utf-8"))
            if result["detector"]["prediction_sha256"] != args.expected_detector_sha256:
                raise RuntimeError("browser detector output differs from the golden")
            if (
                result["classification"]["prediction_sha256"]
                != args.expected_classifier_sha256
            ):
                raise RuntimeError("browser classifier output differs from the golden")
            serialized = json.dumps(result, sort_keys=True)
            if PACKAGED_ACCEPTANCE_HISTORY in serialized:
                raise RuntimeError("browser JSON export retained clinical history")
            with page.expect_download(timeout=timeout_ms) as png_download:
                page.get_by_role("button", name="Download overlay PNG").click()
            png_path = output / "prediction-overlay.png"
            png_download.value.save_as(png_path)
            if not png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
                raise RuntimeError("browser overlay export is not a PNG")
        context.close()
        browser.close()
    if browser_errors:
        raise RuntimeError(f"browser console/page errors: {browser_errors!r}")
    print(
        json.dumps(
            {
                "workflow": "real_packaged_full_prediction",
                "upload": True,
                "polling": True,
                "overlay_rois": 8,
                "roi_attention_selection": True,
                "json_export": True,
                "png_export": True,
                "console_errors": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print("PACKAGED BROWSER WORKBENCH ACCEPTANCE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
