#!/usr/bin/env python3
"""Prove repeated real-model switches through the single-residency runtime."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import os
from pathlib import Path
import sys

from _common import (
    DETECTOR_PREDICTION_SHA256,
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
        default=defaults["fixture_dir"] / "residency" / "runtime-manifest.json",
    )
    parser.add_argument("--cycles", type=int, default=2)
    args = parser.parse_args()

    if args.cycles < 2:
        raise ValueError("at least two detector-classifier cycles are required")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError(
            "launch with CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python starts"
        )

    project_root = args.project_repo.expanduser().resolve()
    sys.path.insert(0, str(project_root / "src"))

    from vision_model_serving.artifacts import ArtifactRegistry
    from vision_model_serving.classifier import MmbcdClassifierAdapter
    from vision_model_serving.detector import FocalNetDinoAdapter
    from vision_model_serving.dicom import DicomCanonicalizer
    from vision_model_serving.model_ids import CLASSIFIER_MODEL_ID, DETECTOR_MODEL_ID
    from vision_model_serving.residency import (
        ModelBinding,
        RuntimeState,
        SingleResidencyRuntime,
        TorchCudaLifecycle,
    )

    artifact_root = args.artifact_dir.expanduser().resolve()
    tokenizer_root = args.tokenizer_dir.expanduser().resolve()
    focalnet_root = args.focalnet_repo.expanduser().resolve()
    mmbcd_root = args.mmbcd_repo.expanduser().resolve()
    dino_root = args.dino_repo.expanduser().resolve()
    fixture_root = args.fixture_dir.expanduser().resolve()
    dicom_files = sorted(fixture_root.glob("*.dcm"))
    if len(dicom_files) != 1:
        raise RuntimeError("fixture directory must contain exactly one DICOM")
    with dicom_files[0].open("rb") as stream:
        canonical = DicomCanonicalizer().decode(stream)

    registry = ArtifactRegistry(
        project_root / "config" / "model-artifacts.json",
        artifact_root=artifact_root,
        tokenizer_root=tokenizer_root,
        repository_root=project_root,
    )
    detector_artifact = registry.resolve(DETECTOR_MODEL_ID)
    classifier_artifact = registry.resolve(CLASSIFIER_MODEL_ID)
    classifier_inputs: list[tuple[object, object, str]] = []

    class DetectorResident:
        artifact = detector_artifact

        def __init__(self) -> None:
            self._adapter = FocalNetDinoAdapter.from_local_source(
                detector_artifact,
                repository_root=focalnet_root,
                project_root=project_root,
            )

        def warmup(self) -> None:
            self._adapter.predict(canonical)

        def execute(self, inputs: object) -> object:
            return self._adapter.predict(inputs)

    class ClassifierResident:
        artifact = classifier_artifact

        def __init__(self) -> None:
            self._adapter = MmbcdClassifierAdapter.from_local_assets(
                classifier_artifact,
                tokenizer_root=tokenizer_root,
                dino_root=dino_root,
                mmbcd_root=mmbcd_root,
                project_root=project_root,
            )

        def warmup(self) -> None:
            if not classifier_inputs:
                raise RuntimeError("detector proposals are unavailable for warmup")
            self.execute(classifier_inputs[-1])

        def execute(self, inputs: object) -> object:
            mammogram, rois, history = inputs
            return self._adapter.predict(mammogram, rois, history)

    runtime = SingleResidencyRuntime(
        bindings=(
            ModelBinding(
                model_id=DETECTOR_MODEL_ID,
                load=DetectorResident,
                failure_token=lambda: (
                    f"{detector_artifact.sha256}:"
                    f"{detector_artifact.repository_revision}"
                ),
            ),
            ModelBinding(
                model_id=CLASSIFIER_MODEL_ID,
                load=ClassifierResident,
                failure_token=lambda: (
                    f"{classifier_artifact.sha256}:"
                    f"{classifier_artifact.repository_revision}"
                ),
            ),
        ),
        accelerator=TorchCudaLifecycle(device="cuda:0"),
    )

    records: list[dict[str, object]] = []
    for cycle in range(1, args.cycles + 1):
        detector_output = runtime.execute(DETECTOR_MODEL_ID, canonical)
        detector_result = detector_output.value
        detector_hash = detector_result.proposals.prediction_sha256
        if detector_hash != DETECTOR_PREDICTION_SHA256:
            raise RuntimeError("detector prediction hash differs from L4 reference")
        after_detector = runtime.status()
        if (
            after_detector.active_model != DETECTOR_MODEL_ID
            or after_detector.resident_models != (DETECTOR_MODEL_ID,)
        ):
            raise RuntimeError("detector stage violated single residency")

        classifier_input = (
            canonical,
            detector_result.proposals.classifier_rois,
            "",
        )
        classifier_inputs.append(classifier_input)
        classifier_output = runtime.execute(CLASSIFIER_MODEL_ID, classifier_input)
        classifier_result = classifier_output.value
        if classifier_result.prediction_sha256 != MMBCD_PREDICTION_SHA256:
            raise RuntimeError("classifier prediction hash differs from L4 reference")
        after_classifier = runtime.status()
        if (
            after_classifier.active_model != CLASSIFIER_MODEL_ID
            or after_classifier.resident_models != (CLASSIFIER_MODEL_ID,)
        ):
            raise RuntimeError("classifier stage violated single residency")
        records.append(
            {
                "cycle": cycle,
                "detector_prediction_sha256": detector_hash,
                "classifier_prediction_sha256": classifier_result.prediction_sha256,
                "detector_timings": asdict(detector_output.timings),
                "detector_memory": asdict(detector_output.memory),
                "classifier_timings": asdict(classifier_output.timings),
                "classifier_memory": asdict(classifier_output.memory),
                "resident_after_detector": list(after_detector.resident_models),
                "resident_after_classifier": list(
                    after_classifier.resident_models
                ),
            }
        )
        classifier_inputs.clear()

    status = runtime.status()
    expected_loads = args.cycles * 2
    if status.state is not RuntimeState.READY:
        raise RuntimeError("single-residency runtime is not ready after live cycles")
    if status.active_model != CLASSIFIER_MODEL_ID:
        raise RuntimeError("unexpected final resident model")
    if status.resident_models != (CLASSIFIER_MODEL_ID,):
        raise RuntimeError("final runtime violated single residency")
    if status.active_inferences != 0 or status.last_error is not None:
        raise RuntimeError("single-residency runtime has unfinished or failed work")
    if status.metrics.load_count != expected_loads:
        raise RuntimeError("live runtime load count differs")
    if status.metrics.switch_count != expected_loads - 1:
        raise RuntimeError("live runtime switch count differs")
    if status.metrics.unload_count != expected_loads - 1:
        raise RuntimeError("live runtime unload count differs")
    if status.metrics.failure_count != 0:
        raise RuntimeError("live runtime recorded a lifecycle failure")

    output_path = args.output.expanduser().resolve()
    manifest = {
        "pipeline": "single-residency-real-dicom-l4-fp32-v1",
        "cycles": records,
        "final_status": {
            "state": status.state.value,
            "active_model": status.active_model,
            "resident_models": list(status.resident_models),
            "active_inferences": status.active_inferences,
            "artifact": asdict(status.artifact),
            "memory": asdict(status.memory),
            "metrics": asdict(status.metrics),
            "timings": asdict(status.timings),
        },
        "validation_boundary": (
            "Serving lifecycle and exact-hash smoke test on one public fixture; "
            "not medical accuracy, calibration, or clinical validation."
        ),
    }
    write_json_atomic(output_path, manifest)
    print("Cycles:", args.cycles)
    print("Loads:", status.metrics.load_count)
    print("Switches:", status.metrics.switch_count)
    print("Unloads:", status.metrics.unload_count)
    print("Allocated bytes:", status.memory.allocated_bytes)
    print("Reserved bytes:", status.memory.reserved_bytes)
    print("Peak allocated bytes:", status.memory.peak_allocated_bytes)
    print("Peak reserved bytes:", status.memory.peak_reserved_bytes)
    print("Manifest:", output_path)
    print("REAL DICOM DETECTOR INFERENCE PASSED")
    print("REAL DICOM MMBCD INFERENCE PASSED")
    print("SINGLE RESIDENCY L4 PASSED")


if __name__ == "__main__":
    main()
