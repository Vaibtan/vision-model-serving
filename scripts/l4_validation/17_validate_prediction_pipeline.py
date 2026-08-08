#!/usr/bin/env python3
"""Validate the public prediction pipeline against the archived L4 goldens."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from _common import (
    DETECTOR_PREDICTION_SHA256,
    DICOM_SHA256,
    MMBCD_INPUT_TENSOR_SHA256,
    MMBCD_PREDICTION_SHA256,
    default_paths,
    write_json_atomic,
)


def main() -> None:
    defaults = default_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-repo", type=Path, default=defaults["project_repo"])
    parser.add_argument("--artifact-dir", type=Path, default=defaults["artifact_dir"])
    parser.add_argument("--focalnet-repo", type=Path, default=defaults["focalnet_repo"])
    parser.add_argument("--mmbcd-repo", type=Path, default=defaults["mmbcd_repo"])
    parser.add_argument("--dino-repo", type=Path, default=defaults["dino_repo"])
    parser.add_argument("--tokenizer-dir", type=Path, default=defaults["tokenizer_dir"])
    parser.add_argument("--fixture-dir", type=Path, default=defaults["fixture_dir"])
    parser.add_argument(
        "--output",
        type=Path,
        default=defaults["fixture_dir"] / "pipeline" / "prediction-manifest.json",
    )
    parser.add_argument("--cycles", type=int, default=2)
    args = parser.parse_args()

    if args.cycles < 2:
        raise ValueError("at least two full-pipeline cycles are required")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError(
            "launch with CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python starts"
        )

    project_root = args.project_repo.expanduser().resolve()
    sys.path.insert(0, str(project_root / "src"))

    from vision_model_serving.pipeline import (
        CaseInput,
        LocalCudaPipelineConfig,
        PredictionMode,
        build_local_cuda_pipeline,
        prediction_to_dict,
    )

    fixture_root = args.fixture_dir.expanduser().resolve()
    dicom_files = sorted(fixture_root.glob("*.dcm"))
    if len(dicom_files) != 1:
        raise RuntimeError("fixture directory must contain exactly one DICOM")

    pipeline = build_local_cuda_pipeline(
        LocalCudaPipelineConfig(
            project_root=project_root,
            artifact_root=args.artifact_dir,
            tokenizer_root=args.tokenizer_dir,
            focalnet_root=args.focalnet_repo,
            mmbcd_root=args.mmbcd_repo,
            dino_root=args.dino_repo,
            require_history_for_full=False,
        )
    )

    with dicom_files[0].open("rb") as stream:
        detection = pipeline.infer(CaseInput(stream), PredictionMode.DETECTION)
    _verify_detection(detection)
    if detection.classification is not None:
        raise RuntimeError("detection mode unexpectedly returned classification")

    records: list[dict[str, object]] = []
    expected_logits: tuple[float, float] | None = None
    for cycle in range(1, args.cycles + 1):
        with dicom_files[0].open("rb") as stream:
            result = pipeline.infer(
                CaseInput(stream, clinical_history=""),
                PredictionMode.FULL,
            )
        _verify_detection(result)
        classification = result.classification
        if classification is None:
            raise RuntimeError("full mode omitted classification")
        if classification.input.crop_tensor_sha256 != MMBCD_INPUT_TENSOR_SHA256:
            raise RuntimeError("pipeline MMBCD input differs from the L4 reference")
        if classification.prediction_sha256 != MMBCD_PREDICTION_SHA256:
            raise RuntimeError("pipeline MMBCD output differs from the L4 reference")
        if expected_logits is None:
            expected_logits = classification.logits
        elif classification.logits != expected_logits:
            raise RuntimeError("pipeline classifier logits are not deterministic")
        if result.provenance.classifier is None or result.provenance.tokenizer is None:
            raise RuntimeError("full mode omitted classifier provenance")
        records.append(
            {
                "cycle": cycle,
                "detector_prediction_sha256": result.detector.prediction_sha256,
                "classifier_prediction_sha256": classification.prediction_sha256,
                "classifier_logits": list(classification.logits),
                "timings": prediction_to_dict(result)["timings"],
            }
        )

    manifest = {
        "pipeline": "prediction-pipeline-real-dicom-l4-fp32-v1",
        "mode": "full",
        "cycles": records,
        "input_sha256": detection.input.source_sha256,
        "detector_artifact": {
            "id": detection.provenance.detector.id,
            "sha256": detection.provenance.detector.sha256,
            "repository_revision": (
                detection.provenance.detector.repository_revision
            ),
        },
        "warnings": [warning.code for warning in result.warnings],
        "disclaimer": result.disclaimer,
    }
    write_json_atomic(args.output.expanduser().resolve(), manifest)
    print("PREDICTION PIPELINE L4 PASSED")


def _verify_detection(result: object) -> None:
    if result.input.source_sha256 != DICOM_SHA256:
        raise RuntimeError("pipeline DICOM input differs from the L4 reference")
    if result.detector.prediction_sha256 != DETECTOR_PREDICTION_SHA256:
        raise RuntimeError("pipeline detector output differs from the L4 reference")
    if result.detector.raw_logits.shape != (1, 900, 1):
        raise RuntimeError("pipeline detector logits shape differs")
    if result.detector.raw_boxes_cxcywh.shape != (1, 900, 4):
        raise RuntimeError("pipeline detector boxes shape differs")
    if len(result.detector.classifier_rois) != 8:
        raise RuntimeError("pipeline did not retain exactly eight classifier ROIs")


if __name__ == "__main__":
    main()
