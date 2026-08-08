"""Real Django, Redis, RQ, executor, models, and public-DICOM acceptance."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from time import monotonic, sleep

EXPECTED_DICOM_SHA256 = (
    "9f70081672a460f29231bb471e8a9e26dd3ed26a2ebbd91c064e575e7842a19c"
)
EXPECTED_DETECTOR_SHA256 = (
    "4cdd09d986702e8839acff8d7517a63f263ca2a01b0607d78d6b2086c886a9a5"
)
EXPECTED_CLASSIFIER_SHA256 = (
    "f994ccfad2e1894f95b487cf1068b5c0038b4bb12c7d49f5e0dc396afc83f1a3"
)
CLINICAL_HISTORY = "real public mammogram acceptance."


def main() -> int:
    redis_url = _required_environment("VMS_TEST_REDIS_URL")
    job_root = Path(_required_environment("VMS_TEST_JOB_ROOT")).resolve()
    executor_socket = Path(_required_environment("VMS_TEST_EXECUTOR_SOCKET")).resolve()
    dicom = Path(_required_environment("VMS_TEST_DICOM_PATH")).read_bytes()
    observed_hash = hashlib.sha256(dicom).hexdigest()
    if observed_hash != EXPECTED_DICOM_SHA256:
        raise AssertionError("public DICOM fixture hash differs from the contract")

    os.environ["DJANGO_SETTINGS_MODULE"] = "vision_model_serving.web.settings"
    os.environ["VMS_REDIS_URL"] = redis_url
    os.environ["VMS_JOB_ROOT"] = str(job_root)
    os.environ["VMS_EXECUTOR_SOCKET"] = str(executor_socket)
    os.environ["VMS_ALLOWED_HOSTS"] = "testserver,localhost"
    os.environ["VMS_SYNC_WAIT_SECONDS"] = "5"

    import django

    django.setup()

    from django.test import Client
    from redis import Redis

    client = Client()
    redis = Redis.from_url(redis_url)
    if not redis.ping():
        raise AssertionError("real Redis did not answer ping")

    readiness = client.get("/readyz")
    _assert_status(readiness.status_code, 200, readiness.content)
    if not all(readiness.json()["checks"].values()):
        raise AssertionError(f"readiness check failed: {readiness.json()!r}")

    before = client.get("/api/v1/models")
    _assert_status(before.status_code, 200, before.content)
    if before.json()["runtime"]["state"] != "unloaded":
        raise AssertionError(f"executor did not start cold: {before.json()!r}")

    full = client.post(
        "/api/v1/predictions",
        {
            "dicom": _upload(dicom),
            "mode": "full",
            "clinical_history": CLINICAL_HISTORY,
            "detector_score_threshold": "1.0",
        },
        HTTP_PREFER="respond-async",
        HTTP_IDEMPOTENCY_KEY="real-l4-full-threshold",
    )
    _assert_status(full.status_code, 202, full.content)
    prediction_id = full.json()["prediction_id"]
    completed = _wait_for_success(client, prediction_id, timeout_seconds=90)
    result_response = client.get(completed["result_url"])
    _assert_status(result_response.status_code, 200, result_response.content)
    result = result_response.json()["result"]
    if result["detector"]["prediction_sha256"] != EXPECTED_DETECTOR_SHA256:
        raise AssertionError("HTTP detector result differs from the L4 golden")
    if result["classification"]["prediction_sha256"] != EXPECTED_CLASSIFIER_SHA256:
        raise AssertionError("HTTP classifier result differs from the L4 golden")
    if result["detector"]["top_candidates"] or result["detector"]["post_nms"]:
        raise AssertionError("display threshold did not filter detector presentation")
    if len(result["detector"]["classifier_rois"]) != 8:
        raise AssertionError("display threshold changed classifier ROI selection")

    after = client.get("/api/v1/models")
    _assert_status(after.status_code, 200, after.content)
    runtime = after.json()["runtime"]
    if set(runtime["resident_models"]) != {
        "focalnet-dino-detector",
        "mmbcd-classifier",
    }:
        raise AssertionError(f"models are not dual-resident: {runtime!r}")

    warm_started = monotonic()
    detection = client.post(
        "/api/v1/predictions",
        {"dicom": _upload(dicom), "mode": "detection"},
        HTTP_IDEMPOTENCY_KEY="real-l4-warm-detection",
    )
    warm_elapsed = monotonic() - warm_started
    _assert_status(detection.status_code, 200, detection.content)
    detection_result = detection.json()["result"]
    if detection_result["classification"] is not None:
        raise AssertionError("detection mode unexpectedly ran the classifier")
    if detection_result["detector"]["prediction_sha256"] != EXPECTED_DETECTOR_SHA256:
        raise AssertionError("warm HTTP detector result differs from the L4 golden")

    broker = b"".join(redis.dump(key) or b"" for key in redis.scan_iter("*"))
    if dicom[128:256] in broker or CLINICAL_HISTORY.encode("utf-8") in broker:
        raise AssertionError("private request content leaked into Redis")

    print(
        json.dumps(
            {
                "classifier_sha256": EXPECTED_CLASSIFIER_SHA256,
                "detector_sha256": EXPECTED_DETECTOR_SHA256,
                "display_threshold_preserved_classifier_rois": True,
                "dual_resident_models": sorted(runtime["resident_models"]),
                "readiness": readiness.status_code,
                "redis": redis.info("server")["redis_version"],
                "warm_sync_http_seconds": warm_elapsed,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _upload(dicom: bytes):
    from django.core.files.uploadedfile import SimpleUploadedFile

    return SimpleUploadedFile(
        "public-mammogram.dcm",
        dicom,
        content_type="application/dicom",
    )


def _wait_for_success(client: object, prediction_id: str, *, timeout_seconds: float):
    deadline = monotonic() + timeout_seconds
    while monotonic() < deadline:
        response = client.get(f"/api/v1/predictions/{prediction_id}")
        _assert_status(response.status_code, 200, response.content)
        payload = response.json()
        if payload["state"] == "succeeded":
            payload["result_url"] = f"/api/v1/predictions/{prediction_id}/result"
            return payload
        if payload["state"] in {"failed", "expired"}:
            raise AssertionError(f"prediction did not succeed: {payload!r}")
        sleep(0.1)
    raise AssertionError("prediction did not complete before the acceptance deadline")


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} must point to real acceptance infrastructure")
    return value


def _assert_status(observed: int, expected: int, content: bytes) -> None:
    if observed != expected:
        raise AssertionError(
            f"expected HTTP {expected}, observed {observed}: "
            f"{content.decode('utf-8', errors='replace')}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
