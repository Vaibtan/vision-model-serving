#!/usr/bin/env python3
"""Measure the complete PyTorch candidate matrix on the pinned L4 fixture."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping

import numpy as np

from _common import (
    DETECTOR_PREDICTION_SHA256,
    MMBCD_PREDICTION_SHA256,
    decode_dicom_file,
    default_paths,
    load_json,
    sha256_array,
    sha256_file,
    write_json_atomic,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from _optimization_measurement import (  # noqa: E402
    SwitchTarget,
    observe_candidate,
    observe_repeated_switches,
)

from vision_model_serving.artifacts import ArtifactRegistry  # noqa: E402
from vision_model_serving.classifier.runtime import (  # noqa: E402
    LocalMmbcdModelFactory,
    _load_verified_mmbcd_model,
)
from vision_model_serving.detector.postprocessing import (  # noqa: E402
    DetectorProposal,
    DetectorPostprocessor,
    DetectorPreprocessor,
    ProposalSelection,
)
from vision_model_serving.detector.runtime import (  # noqa: E402
    _LocalFocalNetDinoFactory,
    _load_verified_detector_model,
    probe_focalnet_native_operator,
)
from vision_model_serving.dicom import DicomCanonicalizer  # noqa: E402
from vision_model_serving.model_ids import (  # noqa: E402
    CLASSIFIER_MODEL_ID,
    DETECTOR_MODEL_ID,
)
from vision_model_serving.validation.optimization import (  # noqa: E402
    CandidateMeasurement,
    DetectorParityComparison,
    DetectorParityRecord,
    OptimizationExperiment,
    OptimizationPolicy,
    OPTIMIZATION_POLICIES,
    compare_detector_records,
    validate_optimization_report,
)
from vision_model_serving.validation.revision import require_clean_revision  # noqa: E402


def main() -> int:
    defaults = default_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=defaults["artifact_dir"])
    parser.add_argument("--tokenizer-root", type=Path, default=defaults["tokenizer_dir"])
    parser.add_argument("--focalnet-root", type=Path, default=defaults["focalnet_repo"])
    parser.add_argument("--mmbcd-root", type=Path, default=defaults["mmbcd_repo"])
    parser.add_argument("--dino-root", type=Path, default=defaults["dino_repo"])
    parser.add_argument("--dicom", type=Path, required=True)
    parser.add_argument(
        "--mmbcd-input-bundle",
        type=Path,
        default=defaults["mmbcd_input_dir"] / "mmbcd-inputs.npz",
    )
    parser.add_argument("--single-residency-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--measured-runs", type=int, default=10)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    if args.warmup_runs < 1 or args.measured_runs < 5:
        parser.error("warmup/measured runs must be at least 1/5")
    if len(args.revision) != 40 or any(c not in "0123456789abcdef" for c in args.revision):
        parser.error("--revision must be a full lowercase Git commit")
    require_clean_revision(args.project_root, args.revision)
    residency_identity = _verify_single_residency_evidence(
        args.single_residency_evidence,
        args.revision,
    )
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    project_root = args.project_root.expanduser().resolve()
    focalnet_root = args.focalnet_root.expanduser().resolve()
    registry = ArtifactRegistry(
        project_root / "config" / "model-artifacts.json",
        artifact_root=args.artifact_root,
        tokenizer_root=args.tokenizer_root,
        repository_root=project_root,
        native_operator_probe=lambda: probe_focalnet_native_operator(
            focalnet_root,
            device="cuda:0",
        ),
    )
    artifact_report = registry.verify_all()
    if not artifact_report.ready:
        raise artifact_report.errors[0]
    artifacts = {item.id: item for item in artifact_report.verified_artifacts}

    canonical = decode_dicom_file(args.dicom, DicomCanonicalizer())
    detector_input = DetectorPreprocessor().prepare(canonical.pixels)
    detector_host = np.array(detector_input.tensor, dtype=np.float32, copy=True)
    detector_postprocessor = DetectorPostprocessor()

    def load_detector() -> tuple[object, tuple[object, ...], float]:
        loaded = _load_verified_detector_model(
            artifacts[DETECTOR_MODEL_ID],
            model_factory=_LocalFocalNetDinoFactory(
                repository_root=focalnet_root,
                project_root=project_root,
                expected_revision=artifacts[DETECTOR_MODEL_ID].repository_revision,
                device="cuda:0",
            ).build,
            device="cuda:0",
        )
        tensor = loaded.torch.from_numpy(detector_host).to("cuda:0")
        return loaded.model, (tensor,), loaded.load_ms

    def call_detector(model: object, inputs: tuple[object, ...]) -> object:
        return model([inputs[0]])

    def project_detector(output: object) -> dict[str, np.ndarray]:
        if not isinstance(output, Mapping):
            raise RuntimeError("detector candidate returned an invalid output")
        return {
            "pred_logits": _numpy(output["pred_logits"]),
            "pred_boxes": _numpy(output["pred_boxes"]),
        }

    def verify_detector_reference(values: Mapping[str, np.ndarray]) -> None:
        proposals = detector_postprocessor.process(
            values["pred_logits"],
            values["pred_boxes"],
            canonical.geometry,
        )
        if proposals.prediction_sha256 != DETECTOR_PREDICTION_SHA256:
            raise RuntimeError("detector FP32 baseline differs from the L4 golden")

    def compare_detector(
        baseline: Mapping[str, object],
        observed: Mapping[str, object],
        score_tolerance: float,
    ) -> Mapping[str, object]:
        baseline_arrays = _arrays(baseline)
        observed_arrays = _arrays(observed)
        if (
            baseline_arrays.keys() != observed_arrays.keys()
            or baseline_arrays.keys() != {"pred_logits", "pred_boxes"}
            or any(
                observed_arrays[name].shape != reference.shape
                for name, reference in baseline_arrays.items()
            )
        ):
            raise RuntimeError("detector candidate raw output contract differs")
        baseline_selection = detector_postprocessor.process(
            baseline_arrays["pred_logits"],
            baseline_arrays["pred_boxes"],
            canonical.geometry,
        )
        observed_selection = detector_postprocessor.process(
            observed_arrays["pred_logits"],
            observed_arrays["pred_boxes"],
            canonical.geometry,
        )
        minimum_iou = 1.0 if score_tolerance == 0.0 else 0.99
        baseline_stages = {
            name: _detector_stage_records(baseline_selection, name)
            for name in ("raw", "post_nms", "selected_rois")
        }
        observed_stages = {
            name: _detector_stage_records(observed_selection, name) for name in baseline_stages
        }
        comparisons = {
            name: compare_detector_records(
                baseline_stages[name],
                observed_stages[name],
                score_tolerance=score_tolerance,
                logit_tolerance=score_tolerance,
                box_tolerance=score_tolerance,
                minimum_iou=minimum_iou,
            )
            for name in baseline_stages
        }
        return {
            "passed": all(comparison.passed for comparison in comparisons.values()),
            "method": "raw_post_nms_selected_roi_score_class_iou",
            "score_tolerance": score_tolerance,
            "logit_tolerance": score_tolerance,
            "box_tolerance": score_tolerance,
            "minimum_iou": minimum_iou,
            "raw_output_shapes": {
                name: list(value.shape) for name, value in observed_arrays.items()
            },
            "stages": {
                name: _detector_comparison_evidence(
                    comparison,
                    baseline_count=len(baseline_stages[name]),
                    candidate_count=len(observed_stages[name]),
                )
                for name, comparison in comparisons.items()
            },
            "output_sha256": {name: sha256_array(value) for name, value in observed_arrays.items()},
        }

    classifier_host = _classifier_inputs(args.mmbcd_input_bundle)

    def load_classifier() -> tuple[object, tuple[object, ...], float]:
        loaded = _load_verified_mmbcd_model(
            artifacts[CLASSIFIER_MODEL_ID],
            model_factory=LocalMmbcdModelFactory(
                dino_root=args.dino_root,
                mmbcd_root=args.mmbcd_root,
                project_root=project_root,
                expected_mmbcd_revision=(artifacts[CLASSIFIER_MODEL_ID].repository_revision),
            ).build,
            device="cuda:0",
        )
        inputs = tuple(loaded.torch.from_numpy(value).to("cuda:0") for value in classifier_host)
        return loaded.model, inputs, loaded.load_ms

    def call_classifier(model: object, inputs: tuple[object, ...]) -> object:
        return model(*inputs)

    def project_classifier(output: object) -> dict[str, np.ndarray]:
        if not isinstance(output, tuple) or len(output) != 3:
            raise RuntimeError("classifier candidate returned an invalid output")
        return {
            "logits": _numpy(output[0]),
            "fused_embeddings": _numpy(output[1]),
            "roi_attention": _numpy(output[2]),
        }

    def verify_classifier_reference(values: Mapping[str, np.ndarray]) -> None:
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(values["logits"]).tobytes())
        digest.update(np.ascontiguousarray(values["fused_embeddings"]).tobytes())
        if digest.hexdigest() != MMBCD_PREDICTION_SHA256:
            raise RuntimeError("classifier FP32 baseline differs from the L4 golden")

    def compare_classifier(
        baseline: Mapping[str, object],
        observed: Mapping[str, object],
        tolerance: float,
    ) -> Mapping[str, object]:
        baseline_arrays = _arrays(baseline)
        observed_arrays = _arrays(observed)
        if baseline_arrays.keys() != observed_arrays.keys() or any(
            observed_arrays[name].shape != reference.shape
            for name, reference in baseline_arrays.items()
        ):
            raise RuntimeError("classifier candidate output contract differs")
        differences = {
            name: float(np.max(np.abs(observed_arrays[name] - reference)))
            for name, reference in baseline_arrays.items()
        }
        return {
            "passed": all(value <= tolerance for value in differences.values()),
            "method": "elementwise_absolute",
            "absolute_tolerance": tolerance,
            "max_absolute_difference": differences,
            "output_sha256": {name: sha256_array(value) for name, value in observed_arrays.items()},
        }

    measured_at = datetime.now(UTC).isoformat()
    tolerances = {
        "fp32": 0.0,
        "tf32": 1e-4,
        "fp16": 5e-3,
        "bf16": 1e-2,
        "compile": 1e-5,
    }
    evidence_roots = (
        project_root,
        focalnet_root,
        args.mmbcd_root,
        args.dino_root,
    )
    switch_targets = (
        SwitchTarget(DETECTOR_MODEL_ID, load_detector, call_detector),
        SwitchTarget(CLASSIFIER_MODEL_ID, load_classifier, call_classifier),
    )
    switch_reliability = {
        policy.name: observe_repeated_switches(
            torch=torch,
            policy=policy,
            targets=switch_targets,
            cycles=2,
            memory_growth_threshold_percent=20.0,
            evidence_roots=evidence_roots,
        )
        for policy in OPTIMIZATION_POLICIES
    }

    def measure_detector(policy: OptimizationPolicy) -> CandidateMeasurement:
        return observe_candidate(
            torch=torch,
            policy=policy,
            load_model=load_detector,
            call_model=call_detector,
            project_output=project_detector,
            warmup_runs=args.warmup_runs,
            measured_runs=args.measured_runs,
            evidence_roots=evidence_roots,
        )

    def measure_classifier(policy: OptimizationPolicy) -> CandidateMeasurement:
        return observe_candidate(
            torch=torch,
            policy=policy,
            load_model=load_classifier,
            call_model=call_classifier,
            project_output=project_classifier,
            warmup_runs=args.warmup_runs,
            measured_runs=args.measured_runs,
            evidence_roots=evidence_roots,
        )

    detector_matrix = OptimizationExperiment(
        model_id=DETECTOR_MODEL_ID,
        tolerances=tolerances,
        verify_reference=verify_detector_reference,
        compare=compare_detector,
    ).run(measure_detector, switch_reliability=switch_reliability)
    classifier_matrix = OptimizationExperiment(
        model_id=CLASSIFIER_MODEL_ID,
        tolerances=tolerances,
        verify_reference=verify_classifier_reference,
        compare=compare_classifier,
    ).run(measure_classifier, switch_reliability=switch_reliability)
    device = torch.cuda.get_device_properties("cuda:0")
    report = {
        "schema_version": 3,
        "revision": args.revision,
        "environment": {
            "measured_at": measured_at,
            "device": device.name,
            "compute_capability": f"{device.major}.{device.minor}",
            "driver": _nvidia_value("driver_version"),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": str(torch.backends.cudnn.version()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "measurement_module_sha256": sha256_file(
                Path(__file__).with_name("_optimization_measurement.py").resolve()
            ),
            "single_residency_evidence_sha256": residency_identity,
        },
        "policy": {
            "warmup_runs": args.warmup_runs,
            "measured_runs": args.measured_runs,
            "retention_threshold_percent": 15,
            "memory_retention_threshold_percent": 20,
            "candidate_order": [policy.name for policy in OPTIMIZATION_POLICIES],
            "one_change_at_a_time": True,
        },
        "switch_reliability": switch_reliability,
        "models": {
            DETECTOR_MODEL_ID: detector_matrix,
            CLASSIFIER_MODEL_ID: classifier_matrix,
        },
        "final_runtime": {
            "backend": "pytorch-eager",
            "precision": "float32",
            "tf32": False,
        },
        "validation_boundary": (
            "Optimization screening on one checksum-pinned public DICOM and one "
            "NVIDIA L4. Every promotion gate includes two candidate-specific "
            "detector-to-classifier switch cycles, post-release CUDA memory, and "
            "standalone release evidence; production selection still requires the "
            "packaged single-residency benchmark."
        ),
    }
    validate_optimization_report(report)
    write_json_atomic(args.output, report)
    markdown = args.output.with_suffix(".md")
    markdown.write_text(_render_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("PYTORCH OPTIMIZATION MATRIX PASSED")
    return 0


def _classifier_inputs(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path.expanduser().resolve(), allow_pickle=False) as bundle:
        crops = np.ascontiguousarray(bundle["crops"][None, ...], dtype=np.float32)
        input_ids = np.ascontiguousarray(bundle["input_ids"], dtype=np.int64)
        attention_mask = np.ascontiguousarray(bundle["attention_mask"], dtype=np.int64)
    if (
        input_ids.ndim != 2
        or input_ids.shape[0] != 1
        or not 1 <= input_ids.shape[1] <= 90
        or attention_mask.shape != input_ids.shape
    ):
        raise RuntimeError("MMBCD token tensors differ from the pinned request")
    return crops, input_ids, attention_mask


def _numpy(value: object) -> np.ndarray:
    return np.ascontiguousarray(value.detach().float().cpu().numpy())


def _arrays(values: Mapping[str, object]) -> dict[str, np.ndarray]:
    if not values or any(not isinstance(value, np.ndarray) for value in values.values()):
        raise RuntimeError("optimization outputs must be NumPy arrays")
    return {name: value for name, value in values.items() if isinstance(value, np.ndarray)}


def _raw_detector_records(
    selection: ProposalSelection,
) -> tuple[DetectorParityRecord, ...]:
    logits = selection.raw_logits[0]
    scores = selection.raw_scores[0]
    boxes = selection.raw_boxes_cxcywh[0]
    records: list[DetectorParityRecord] = []
    for query_index in range(logits.shape[0]):
        cxcywh = tuple(float(value) for value in boxes[query_index])
        xyxy = _clipped_xyxy(cxcywh)
        for class_index in range(logits.shape[1]):
            records.append(
                DetectorParityRecord(
                    class_index=class_index,
                    raw_logit=float(logits[query_index, class_index]),
                    score=float(scores[query_index, class_index]),
                    normalized_cxcywh=cxcywh,
                    normalized_xyxy=xyxy,
                )
            )
    return tuple(records)


def _selected_detector_records(
    proposals: tuple[DetectorProposal, ...],
) -> tuple[DetectorParityRecord, ...]:
    return tuple(
        DetectorParityRecord(
            class_index=proposal.class_index,
            raw_logit=proposal.raw_logit,
            score=proposal.score,
            normalized_cxcywh=proposal.normalized_cxcywh,
            normalized_xyxy=proposal.normalized_xyxy,
        )
        for proposal in proposals
    )


def _detector_stage_records(
    selection: ProposalSelection,
    stage: str,
) -> tuple[DetectorParityRecord, ...]:
    if stage == "raw":
        return _raw_detector_records(selection)
    if stage == "post_nms":
        return _selected_detector_records(selection.post_nms)
    if stage == "selected_rois":
        return _selected_detector_records(selection.classifier_rois)
    raise ValueError(f"unknown detector parity stage: {stage}")


def _detector_comparison_evidence(
    comparison: DetectorParityComparison,
    *,
    baseline_count: int,
    candidate_count: int,
) -> dict[str, object]:
    return {
        "baseline_count": baseline_count,
        "candidate_count": candidate_count,
        "matched_pairs": [list(pair) for pair in comparison.matched_pairs],
        "unmatched_baseline": list(comparison.unmatched_baseline),
        "unmatched_candidate": list(comparison.unmatched_candidate),
        "max_score_difference": comparison.max_score_difference,
        "max_logit_difference": comparison.max_logit_difference,
        "max_box_coordinate_difference": (comparison.max_box_coordinate_difference),
        "minimum_observed_iou": comparison.minimum_observed_iou,
    }


def _clipped_xyxy(
    cxcywh: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    center_x, center_y, width, height = cxcywh
    return (
        min(max(center_x - width / 2.0, 0.0), 1.0),
        min(max(center_y - height / 2.0, 0.0), 1.0),
        min(max(center_x + width / 2.0, 0.0), 1.0),
        min(max(center_y + height / 2.0, 0.0), 1.0),
    )


def _verify_single_residency_evidence(path: Path, revision: str) -> str:
    payload = load_json(path)
    if (
        payload.get("schema_version") != 2
        or payload.get("pipeline") != "single-residency-real-dicom-l4-fp32-v2"
        or payload.get("revision") != revision
    ):
        raise RuntimeError("single-residency evidence revision differs")
    cycles = payload.get("cycles")
    if not isinstance(cycles, list) or len(cycles) < 2:
        raise RuntimeError("single-residency evidence has too few cycles")
    first_reserved: dict[str, int] = {}
    maximum_growth = 0.0
    for expected_cycle, cycle in enumerate(cycles, start=1):
        if (
            cycle.get("cycle") != expected_cycle
            or cycle.get("resident_after_detector") != [DETECTOR_MODEL_ID]
            or cycle.get("resident_after_classifier") != [CLASSIFIER_MODEL_ID]
            or cycle.get("detector_prediction_sha256") != DETECTOR_PREDICTION_SHA256
            or cycle.get("classifier_prediction_sha256") != MMBCD_PREDICTION_SHA256
        ):
            raise RuntimeError("single-residency evidence contains an invalid snapshot")
        for model_id, field in (
            (DETECTOR_MODEL_ID, "detector_memory"),
            (CLASSIFIER_MODEL_ID, "classifier_memory"),
        ):
            memory = _validated_memory_snapshot(cycle.get(field), field)
            initial = first_reserved.setdefault(model_id, memory["reserved_bytes"])
            maximum_growth = max(
                maximum_growth,
                max(
                    0.0,
                    (memory["reserved_bytes"] - initial) * 100.0 / initial,
                ),
            )
    final_status = payload.get("final_status", {})
    metrics = final_status.get("metrics", {})
    expected_loads = len(cycles) * 2
    if (
        final_status.get("state") != "ready"
        or final_status.get("active_model") != CLASSIFIER_MODEL_ID
        or final_status.get("resident_models") != [CLASSIFIER_MODEL_ID]
        or final_status.get("active_inferences") != 0
        or metrics.get("load_count") != expected_loads
        or metrics.get("switch_count") != expected_loads - 1
        or metrics.get("unload_count") != expected_loads - 1
        or metrics.get("failure_count") != 0
    ):
        raise RuntimeError("single-residency evidence has an invalid final resident")
    _validated_memory_snapshot(final_status.get("memory"), "final_status.memory")
    if maximum_growth > 20.0:
        raise RuntimeError("single-residency evidence exceeds the VRAM growth gate")
    return sha256_file(path.expanduser().resolve())


def _validated_memory_snapshot(value: object, path: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{path} is missing")
    required = {
        "allocated_bytes",
        "reserved_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    }
    if set(value) != required or any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value.values()
    ):
        raise RuntimeError(f"{path} is invalid")
    result = {name: int(item) for name, item in value.items()}
    if (
        result["allocated_bytes"] > result["reserved_bytes"]
        or result["allocated_bytes"] > result["peak_allocated_bytes"]
        or result["reserved_bytes"] > result["peak_reserved_bytes"]
        or result["peak_allocated_bytes"] > result["peak_reserved_bytes"]
    ):
        raise RuntimeError(f"{path} is impossible")
    return result


def _nvidia_value(field: str) -> str:
    return (
        subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        )
        .splitlines()[0]
        .strip()
    )


def _render_markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# PyTorch optimization matrix",
        "",
        f"Revision: `{report['revision']}`",
        "",
        (
            "| Model | Candidate | Status | Parity | Candidate release | "
            "Repeated A→B switch | "
            "Warm p50 ms | Peak reserved bytes | Accepted |"
        ),
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for model_id, matrix in report["models"].items():
        for candidate in matrix["candidates"]:
            performance = candidate["performance"]
            latency = performance["latency_ms"]
            lines.append(
                "| "
                + " | ".join(
                    (
                        model_id,
                        candidate["name"],
                        candidate["status"],
                        str(candidate["parity"]["passed"]).lower(),
                        candidate["reliability"]["release"]["status"],
                        report["switch_reliability"][candidate["name"]]["status"],
                        f"{latency['p50']:.3f}" if latency else "n/a",
                        str(performance["peak_reserved_bytes"] or "n/a"),
                        str(candidate["promotion"]["accepted"]).lower(),
                    )
                )
                + " |"
            )
    lines.extend(
        (
            "",
            (
                "The production runtime remains eager FP32 until a candidate also "
                "passes the packaged benchmark and restart gates."
            ),
            "",
            str(report["validation_boundary"]),
            "",
        )
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
