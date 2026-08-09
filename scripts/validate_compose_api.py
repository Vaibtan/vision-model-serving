#!/usr/bin/env python3
"""Validate the packaged API against real Redis, GPU inference, and a public DICOM."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from io import BytesIO
from pathlib import Path

from pydicom import dcmread
from redis import Redis

from vision_model_serving.dicom import DicomCanonicalizer
from vision_model_serving.model_ids import CLASSIFIER_MODEL_ID, DETECTOR_MODEL_ID
from vision_model_serving.pipeline.contracts import PredictionMode
from vision_model_serving.validation.packaged_http import (
    PACKAGED_ACCEPTANCE_HISTORY,
    PackagedPredictionClient,
)
from vision_model_serving.validation.reporting import write_json_atomic

DICOM_SHA256 = "9f70081672a460f29231bb471e8a9e26dd3ed26a2ebbd91c064e575e7842a19c"
CANONICAL_ARRAY_SHA256 = (
    "97fa0f80a696ce7f822c1681a8c3f7c072da9262b2bd91239c9f1637eaf68552"
)
DETECTOR_OUTPUT_SHA256 = (
    "4cdd09d986702e8839acff8d7517a63f263ca2a01b0607d78d6b2086c886a9a5"
)
CLASSIFIER_OUTPUT_SHA256 = (
    "f994ccfad2e1894f95b487cf1068b5c0038b4bb12c7d49f5e0dc396afc83f1a3"
)
DETECTOR = {
    "id": DETECTOR_MODEL_ID,
    "sha256": "67a7b0cd787a3aaba199cf1ff82ed2934c33ffe37544473379d7a837ab1637b4",
    "repository_revision": "23901e021dc6ec8f66bad47983f45a25574452cc",
}
CLASSIFIER = {
    "id": CLASSIFIER_MODEL_ID,
    "sha256": "2264351216f9fb4945af35e300459ff4ce2e7f5445519348024f3bf1eec721a4",
    "repository_revision": "14ac5e099c79253b01e0885d2ebefa6f86cfd8f0",
}
TOKENIZER_REVISION = "e2da8e2f811d1448a5b465c236feacd80ffbac7b"
MANIFEST_ID = "vision-model-serving-l4-fp32-20260807"
DISCLAIMER = "Research use only; not a medical diagnosis."
EXPECTED_WARNINGS = {"secondary_capture_storage", "aspect_ratio_distorted"}
CLASSIFIER_WARNINGS = {
    "class_semantics_and_decision_threshold_unverified",
    "attention_is_inspection_not_causal_or_clinical_evidence",
}


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
    if _sha256(dicom) != DICOM_SHA256:
        raise AssertionError("DICOM bytes differ from the pinned public fixture")
    canonical_array_sha256 = _sha256(
        DicomCanonicalizer().decode(BytesIO(dicom)).pixels.tobytes()
    )
    if canonical_array_sha256 != CANONICAL_ARRAY_SHA256:
        raise AssertionError("canonical DICOM pixels differ from the pinned fixture")
    client = PackagedPredictionClient(
        args.base_url,
        timeout_seconds=args.timeout_seconds,
    )
    readiness = client.readiness()
    if (
        readiness.get("status") != "ready"
        or readiness.get("readiness_scope") != "artifact_ready"
        or not all(readiness.get("checks", {}).values())
    ):
        raise AssertionError(f"packaged API is not ready: {readiness!r}")

    initial = _inventory(client.model_inventory())
    if (
        initial["runtime"]["state"] != "unloaded"
        or initial["runtime"]["inference_warm"] is not False
    ):
        raise AssertionError(f"executor did not start cold: {initial['runtime']!r}")
    cycles = []
    for cycle in range(1, args.cycles + 1):
        detection = client.predict(dicom, mode=PredictionMode.DETECTION)
        _validate_result(detection, mode="detection")
        after_detection = _inventory(client.model_inventory())["runtime"]
        _validate_runtime(
            after_detection,
            active=DETECTOR["id"],
            residents={DETECTOR["id"]},
        )

        repeated_detection = client.predict(dicom, mode=PredictionMode.DETECTION)
        _validate_result(repeated_detection, mode="detection")
        after_repeated_detection = _inventory(client.model_inventory())["runtime"]
        _validate_runtime(
            after_repeated_detection,
            active=DETECTOR["id"],
            residents={DETECTOR["id"]},
        )
        if not repeated_detection["timings"]["detector"]["runtime"]["reused"]:
            raise AssertionError("consecutive detection did not reuse the detector")

        full = client.predict(dicom, mode=PredictionMode.FULL)
        _validate_result(full, mode="full")
        after_full = _inventory(client.model_inventory())["runtime"]
        _validate_runtime(
            after_full,
            active=CLASSIFIER["id"],
            residents={CLASSIFIER["id"]},
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
                "detection": _behavior(detection, after_detection),
                "repeated_detection": _behavior(
                    repeated_detection,
                    after_repeated_detection,
                ),
                "full": _behavior(full, after_full),
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


def _inventory(inventory: dict) -> dict:
    if inventory.get("manifest_id") != MANIFEST_ID:
        raise AssertionError("API exposed an unexpected artifact manifest")
    expected = {
        DETECTOR["id"]: ("detector", DETECTOR["sha256"]),
        CLASSIFIER["id"]: ("classifier", CLASSIFIER["sha256"]),
    }
    observed = {
        model["id"]: (model["role"], model["sha256"])
        for model in inventory.get("models", [])
        if model.get("strict_load_verified") is True
    }
    if observed != expected:
        raise AssertionError(f"model inventory differs from the manifest: {observed!r}")
    residents = set(inventory["runtime"]["resident_models"])
    if len(residents) > 1 or not residents <= set(expected):
        raise AssertionError("executor exposed a model outside the manifest")
    return inventory


def _validate_result(result: dict, *, mode: str) -> None:
    _assert_finite(result)
    if (
        result.get("mode") != mode
        or result["input"]["source_sha256"] != DICOM_SHA256
        or result.get("disclaimer") != DISCLAIMER
        or result["detector"]["prediction_sha256"] != DETECTOR_OUTPUT_SHA256
    ):
        raise AssertionError("prediction identity differs from the golden contract")
    detector = result["detector"]
    if len(detector["top_candidates"]) > 300 or len(detector["post_nms"]) > 300:
        raise AssertionError("detector presentation exceeded its bounded contract")
    if len(detector["classifier_rois"]) != 8:
        raise AssertionError("detector did not produce exactly eight classifier ROIs")
    _validate_detections(result)
    provenance = result["provenance"]
    _assert_artifact(provenance["detector"], DETECTOR)
    warning_codes = {warning["code"] for warning in result["warnings"]}
    if not EXPECTED_WARNINGS <= warning_codes:
        raise AssertionError("required public-DICOM warnings are absent")

    if mode == "detection":
        if result["classification"] is not None or any(
            provenance[field] is not None
            for field in (
                "classifier",
                "tokenizer",
                "precision",
                "offline_assets_only",
                "strict_checkpoint_load",
            )
        ):
            raise AssertionError("detection mode unexpectedly ran the classifier")
        return
    classification = result["classification"]
    if classification["prediction_sha256"] != CLASSIFIER_OUTPUT_SHA256:
        raise AssertionError("classifier output differs from the golden contract")
    if (
        len(classification["logits"]) != 2
        or len(classification["probabilities"]) != 2
        or len(classification["attention"]["roi_weights"]) != 8
        or not math.isclose(sum(classification["probabilities"]), 1.0, abs_tol=1e-6)
    ):
        raise AssertionError("classifier output shape or probabilities are invalid")
    _assert_artifact(provenance["classifier"], CLASSIFIER)
    if (
        provenance["tokenizer"]["revision"] != TOKENIZER_REVISION
        or provenance["precision"] != "float32"
        or provenance["offline_assets_only"] is not True
        or provenance["strict_checkpoint_load"] is not True
        or not CLASSIFIER_WARNINGS <= warning_codes
    ):
        raise AssertionError("classifier provenance or warnings are incomplete")


def _validate_detections(result: dict) -> None:
    geometry = result["geometry"]
    bounds = (
        ("normalized_xyxy", 1.0, 1.0),
        ("canonical_xyxy", geometry["canonical_width"], geometry["canonical_height"]),
        ("original_xyxy", geometry["original_width"], geometry["original_height"]),
    )
    detector = result["detector"]
    for detection in (
        detector["top_candidates"] + detector["post_nms"] + detector["classifier_rois"]
    ):
        if not 0.0 <= detection["score"] <= 1.0:
            raise AssertionError("detector score is outside [0, 1]")
        for field, width, height in bounds:
            x1, y1, x2, y2 = detection[field]
            if not (0 <= x1 <= x2 <= width and 0 <= y1 <= y2 <= height):
                raise AssertionError(f"detector {field} is outside image bounds")


def _validate_runtime(runtime: dict, *, active: str, residents: set[str]) -> None:
    if (
        runtime["state"] != "ready"
        or runtime["initialized"] is not True
        or runtime["artifact_ready"] is not True
        or runtime["inference_warm"] is not True
        or runtime["warm_model"] != active
        or runtime["active_model"] != active
        or set(runtime["resident_models"]) != residents
        or runtime["last_error"] is not None
    ):
        raise AssertionError(f"unexpected executor lifecycle state: {runtime!r}")


def _assert_private_content_absent(
    redis_url: str, job_root: Path, dicom: bytes
) -> None:
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
        "dicom_sha256": DICOM_SHA256,
        "canonical_array_sha256": canonical_array_sha256,
        "manifest_id": inventory["manifest_id"],
        "config_sha256": {
            name: _sha256((root / "config" / name).read_bytes())
            for name in ("model-artifacts.json", "l4-fp32-environment.json")
        },
        "models": {model["id"]: model["sha256"] for model in inventory["models"]},
    }


def _behavior(result: dict, runtime: dict) -> dict:
    classifier = result["classification"]
    return {
        "mode": result["mode"],
        "detector_sha256": result["detector"]["prediction_sha256"],
        "classifier_sha256": (
            None if classifier is None else classifier["prediction_sha256"]
        ),
        "warning_codes": sorted(warning["code"] for warning in result["warnings"]),
        "detector_reused": result["timings"]["detector"]["runtime"]["reused"],
        "classifier_reused": (
            None
            if result["timings"]["classifier"] is None
            else result["timings"]["classifier"]["runtime"]["reused"]
        ),
        "active_model": runtime["active_model"],
        "resident_models": sorted(runtime["resident_models"]),
    }


def _assert_artifact(observed: dict, expected: dict[str, str]) -> None:
    if any(observed.get(field) != value for field, value in expected.items()):
        raise AssertionError(f"artifact provenance differs: {observed!r}")


def _assert_finite(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise AssertionError("prediction contains a non-finite number")
    if isinstance(value, dict):
        for item in value.values():
            _assert_finite(item)
    elif isinstance(value, list):
        for item in value:
            _assert_finite(item)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
