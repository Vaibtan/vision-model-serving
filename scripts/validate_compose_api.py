#!/usr/bin/env python3
"""Validate the packaged API against real Redis, GPU inference, and a public DICOM."""

from __future__ import annotations

import argparse
import hashlib
import json
from io import BytesIO
from pathlib import Path

from pydicom import dcmread
from redis import Redis

from vision_model_serving.dicom import DicomCanonicalizer
from vision_model_serving.pipeline.contracts import PredictionMode
from vision_model_serving.validation.acceptance_contract import (
    CLASSIFIER_ARTIFACT,
    DETECTOR_ARTIFACT,
    PACKAGED_ACCEPTANCE_HISTORY,
    PACKAGED_ACCEPTANCE,
    PUBLIC_CANONICAL_ARRAY_SHA256,
    PUBLIC_DICOM_SHA256,
)
from vision_model_serving.validation.packaged_http import PackagedPredictionClient
from vision_model_serving.validation.reporting import write_json_atomic


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--redis-url", required=True)
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--dicom", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()
    if args.cycles < 2 or args.timeout_seconds <= 0:
        parser.error("--cycles must be at least 2 and timeout must be positive")

    dicom = args.dicom.read_bytes()
    if _sha256(dicom) != PUBLIC_DICOM_SHA256:
        raise AssertionError("DICOM bytes differ from the pinned public fixture")
    canonical_array_sha256 = _sha256(DicomCanonicalizer().decode(BytesIO(dicom)).pixels.tobytes())
    if canonical_array_sha256 != PUBLIC_CANONICAL_ARRAY_SHA256:
        raise AssertionError("canonical DICOM pixels differ from the pinned fixture")
    client = PackagedPredictionClient(
        args.base_url,
        timeout_seconds=args.timeout_seconds,
    )
    readiness = PACKAGED_ACCEPTANCE.validate_readiness(client.readiness())

    initial = PACKAGED_ACCEPTANCE.validate_inventory(client.model_inventory())
    if (
        initial["runtime"]["state"] != "unloaded"
        or initial["runtime"]["inference_warm"] is not False
    ):
        raise AssertionError(f"executor did not start cold: {initial['runtime']!r}")
    cycles = []
    for cycle in range(1, args.cycles + 1):
        detection = client.predict(dicom, mode=PredictionMode.DETECTION)
        PACKAGED_ACCEPTANCE.validate_prediction(detection, mode=PredictionMode.DETECTION)
        after_detection = PACKAGED_ACCEPTANCE.validate_inventory(client.model_inventory())[
            "runtime"
        ]
        PACKAGED_ACCEPTANCE.validate_runtime(
            after_detection,
            active_model=DETECTOR_ARTIFACT.model_id,
        )

        repeated_detection = client.predict(dicom, mode=PredictionMode.DETECTION)
        PACKAGED_ACCEPTANCE.validate_prediction(repeated_detection, mode=PredictionMode.DETECTION)
        after_repeated_detection = PACKAGED_ACCEPTANCE.validate_inventory(client.model_inventory())[
            "runtime"
        ]
        PACKAGED_ACCEPTANCE.validate_runtime(
            after_repeated_detection,
            active_model=DETECTOR_ARTIFACT.model_id,
        )
        if not repeated_detection["timings"]["detector"]["runtime"]["reused"]:
            raise AssertionError("consecutive detection did not reuse the detector")

        full = client.predict(dicom, mode=PredictionMode.FULL)
        PACKAGED_ACCEPTANCE.validate_prediction(full, mode=PredictionMode.FULL)
        after_full = PACKAGED_ACCEPTANCE.validate_inventory(client.model_inventory())["runtime"]
        PACKAGED_ACCEPTANCE.validate_runtime(
            after_full,
            active_model=CLASSIFIER_ARTIFACT.model_id,
        )
        if not full["timings"]["detector"]["runtime"]["reused"]:
            raise AssertionError("full prediction did not reuse the resident detector")
        if full["timings"]["classifier"]["runtime"]["reused"]:
            raise AssertionError("full prediction reused an evicted classifier")
        if detection["timings"]["detector"]["runtime"]["reused"]:
            raise AssertionError("detection reused a detector evicted by prior full")
        cycles.append(
            {
                "cycle": cycle,
                "detection": PACKAGED_ACCEPTANCE.behavior_snapshot(detection, after_detection),
                "repeated_detection": PACKAGED_ACCEPTANCE.behavior_snapshot(
                    repeated_detection,
                    after_repeated_detection,
                ),
                "full": PACKAGED_ACCEPTANCE.behavior_snapshot(full, after_full),
            }
        )

    _assert_private_content_absent(args.redis_url, args.job_root, dicom)
    record = {
        "schema_version": 2,
        "identity": _identity(initial, canonical_array_sha256),
        "behavior": {"cycles": cycles},
        "privacy": {
            "redis_contains_request_content": False,
            "job_volume_contains_request_content": False,
            "request_files_remaining_after_completion": False,
        },
        "readiness_checks": readiness["checks"],
        "restart_baseline_matched": None,
        "validation_boundary": (
            "Packaged deterministic execution on one public Secondary Capture "
            "fixture; not accuracy, calibration, robustness, or clinical validation."
        ),
    }
    output = args.output.resolve()
    baseline = args.baseline.resolve()
    if output != baseline:
        if not baseline.is_file():
            raise AssertionError("restart baseline report is missing")
        prior = json.loads(baseline.read_text(encoding="utf-8"))
        if (prior.get("identity"), prior.get("behavior")) != (
            record["identity"],
            record["behavior"],
        ):
            raise AssertionError("restart changed model identity or API behavior")
        record["restart_baseline_matched"] = True
    write_json_atomic(output, record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


def _assert_private_content_absent(redis_url: str, job_root: Path, dicom: bytes) -> None:
    redis = Redis.from_url(redis_url)
    if not redis.ping():
        raise AssertionError("real Redis did not answer ping")
    dataset = dcmread(BytesIO(dicom), stop_before_pixels=True)
    private_values = [dicom[128:256], PACKAGED_ACCEPTANCE_HISTORY.encode()]
    private_values.extend(
        str(value).encode()
        for value in (
            dataset.get("PatientID"),
            dataset.get("PatientName"),
            dataset.get("StudyInstanceUID"),
            dataset.get("SOPInstanceUID"),
        )
        if value
    )
    broker = b"".join(redis.dump(key) or b"" for key in redis.scan_iter("*"))
    if any(value in broker for value in private_values):
        raise AssertionError("request content persisted in Redis")
    files = [path for path in job_root.rglob("*") if path.is_file()]
    if any(path.name in {"input.dcm", "request.json"} for path in files):
        raise AssertionError("private request files remained in the job volume")
    stored_results = b"".join(path.read_bytes() for path in files)
    if any(value in stored_results for value in private_values):
        raise AssertionError("request content persisted in the job volume")


def _identity(inventory: dict, canonical_array_sha256: str) -> dict:
    root = Path(__file__).resolve().parents[1]
    return {
        "dicom_sha256": PUBLIC_DICOM_SHA256,
        "canonical_array_sha256": canonical_array_sha256,
        "manifest_id": inventory["manifest_id"],
        "config_sha256": {
            name: _sha256((root / "config" / name).read_bytes())
            for name in ("model-artifacts.json", "l4-fp32-environment.json")
        },
        "models": {model["id"]: model["sha256"] for model in inventory["models"]},
    }


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
