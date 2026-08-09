#!/usr/bin/env python3
"""Run one exact packaged full prediction for the Compose GPU smoke gate."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vision_model_serving.model_ids import CLASSIFIER_MODEL_ID  # noqa: E402
from vision_model_serving.pipeline.contracts import PredictionMode  # noqa: E402
from vision_model_serving.validation.packaged_http import (  # noqa: E402
    PackagedPredictionClient,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--dicom", type=Path, required=True)
    parser.add_argument("--expected-detector-sha256", required=True)
    parser.add_argument("--expected-classifier-sha256", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    args = parser.parse_args()
    client = PackagedPredictionClient(
        args.base_url,
        timeout_seconds=args.timeout_seconds,
    )
    readiness = client.readiness()
    if (
        readiness.get("status") != "ready"
        or readiness.get("readiness_scope") != "artifact_ready"
    ):
        raise RuntimeError("packaged service is not artifact-ready")
    result = client.predict(args.dicom.read_bytes(), mode=PredictionMode.FULL)
    if result["detector"]["prediction_sha256"] != args.expected_detector_sha256:
        raise RuntimeError("detector output differs from the packaged golden")
    classification = result.get("classification")
    if (
        not isinstance(classification, dict)
        or classification.get("prediction_sha256")
        != args.expected_classifier_sha256
    ):
        raise RuntimeError("classifier output differs from the packaged golden")
    inventory = client.model_inventory()["runtime"]
    if (
        inventory.get("resident_models") != [CLASSIFIER_MODEL_ID]
        or inventory.get("active_model") != CLASSIFIER_MODEL_ID
    ):
        raise RuntimeError("GPU smoke ended outside the one-resident contract")
    print("PACKAGED SINGLE-RESIDENCY GPU SMOKE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
