#!/usr/bin/env python3
"""Measure the complete PyTorch candidate matrix on the pinned L4 fixture."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
from time import perf_counter
from typing import Any, Callable, Mapping

import numpy as np

from _common import (
    DETECTOR_PREDICTION_SHA256,
    MMBCD_PREDICTION_SHA256,
    default_paths,
    load_json,
    sha256_array,
    sha256_file,
    write_json_atomic,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vision_model_serving.artifacts import ArtifactRegistry  # noqa: E402
from vision_model_serving.classifier.runtime import (  # noqa: E402
    LocalMmbcdModelFactory,
    _load_verified_mmbcd_model,
)
from vision_model_serving.detector.postprocessing import (  # noqa: E402
    DetectorPostprocessor,
    DetectorPreprocessor,
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
    candidate_promotion,
    validate_optimization_report,
)


@dataclass(frozen=True, slots=True)
class CandidatePolicy:
    name: str
    autocast_dtype: str | None = None
    tf32: bool = False
    compile: bool = False


POLICIES = (
    CandidatePolicy("fp32"),
    CandidatePolicy("tf32", tf32=True),
    CandidatePolicy("fp16", autocast_dtype="float16"),
    CandidatePolicy("bf16", autocast_dtype="bfloat16"),
    CandidatePolicy("compile", compile=True),
)


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
    _verify_clean_revision(args.project_root, args.revision)
    residency_identity = _verify_single_residency_evidence(
        args.single_residency_evidence
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

    canonical = DicomCanonicalizer().decode(args.dicom.read_bytes())
    detector_input = DetectorPreprocessor().prepare(canonical.pixels)
    detector_host = np.array(detector_input.tensor, dtype=np.float32, copy=True)
    detector_postprocessor = DetectorPostprocessor()

    def load_detector() -> tuple[object, object, float]:
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

    classifier_host = _classifier_inputs(args.mmbcd_input_bundle)

    def load_classifier() -> tuple[object, object, float]:
        loaded = _load_verified_mmbcd_model(
            artifacts[CLASSIFIER_MODEL_ID],
            model_factory=LocalMmbcdModelFactory(
                dino_root=args.dino_root,
                mmbcd_root=args.mmbcd_root,
                project_root=project_root,
                expected_mmbcd_revision=(
                    artifacts[CLASSIFIER_MODEL_ID].repository_revision
                ),
            ).build,
            device="cuda:0",
        )
        inputs = tuple(
            loaded.torch.from_numpy(value).to("cuda:0") for value in classifier_host
        )
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

    measured_at = datetime.now(UTC).isoformat()
    detector_matrix = _measure_model(
        torch=torch,
        load_model=load_detector,
        call_model=call_detector,
        project_output=project_detector,
        verify_reference=verify_detector_reference,
        tolerances={"fp32": 0.0, "tf32": 1e-4, "fp16": 5e-3, "bf16": 1e-2, "compile": 1e-5},
        warmup_runs=args.warmup_runs,
        measured_runs=args.measured_runs,
    )
    classifier_matrix = _measure_model(
        torch=torch,
        load_model=load_classifier,
        call_model=call_classifier,
        project_output=project_classifier,
        verify_reference=verify_classifier_reference,
        tolerances={"fp32": 0.0, "tf32": 1e-4, "fp16": 5e-3, "bf16": 1e-2, "compile": 1e-5},
        warmup_runs=args.warmup_runs,
        measured_runs=args.measured_runs,
    )
    device = torch.cuda.get_device_properties("cuda:0")
    report = {
        "schema_version": 1,
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
            "single_residency_evidence_sha256": residency_identity,
        },
        "policy": {
            "warmup_runs": args.warmup_runs,
            "measured_runs": args.measured_runs,
            "retention_threshold_percent": 15,
            "memory_retention_threshold_percent": 20,
            "candidate_order": [policy.name for policy in POLICIES],
            "one_change_at_a_time": True,
        },
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
            "NVIDIA L4. A candidate is not selected for serving until the packaged "
            "single-residency benchmark also passes parity and reliability gates."
        ),
    }
    validate_optimization_report(report)
    write_json_atomic(args.output, report)
    markdown = args.output.with_suffix(".md")
    markdown.write_text(_render_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("PYTORCH OPTIMIZATION MATRIX PASSED")
    return 0


def _measure_model(
    *,
    torch: object,
    load_model: Callable[[], tuple[object, tuple[object, ...], float]],
    call_model: Callable[[object, tuple[object, ...]], object],
    project_output: Callable[[object], dict[str, np.ndarray]],
    verify_reference: Callable[[Mapping[str, np.ndarray]], None],
    tolerances: Mapping[str, float],
    warmup_runs: int,
    measured_runs: int,
) -> dict[str, object]:
    baseline_outputs: dict[str, np.ndarray] | None = None
    candidates: list[dict[str, object]] = []
    for policy in POLICIES:
        try:
            candidate = _measure_candidate(
                torch=torch,
                policy=policy,
                load_model=load_model,
                call_model=call_model,
                project_output=project_output,
                warmup_runs=warmup_runs,
                measured_runs=measured_runs,
            )
            outputs = candidate.pop("_outputs")
            if baseline_outputs is None:
                baseline_outputs = outputs
                verify_reference(outputs)
            tolerance = tolerances[policy.name]
            differences = {
                name: float(np.max(np.abs(outputs[name] - reference)))
                for name, reference in baseline_outputs.items()
            }
            parity = all(value <= tolerance for value in differences.values())
            candidate["parity"] = {
                "passed": parity,
                "absolute_tolerance": tolerance,
                "max_absolute_difference": differences,
                "output_sha256": {
                    name: sha256_array(value) for name, value in outputs.items()
                },
            }
            candidate["status"] = "passed" if parity else "rejected"
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            candidate = _failed_candidate(policy, "cuda_out_of_memory", cuda_oom=True)
        except Exception as error:  # one failed candidate must not hide the matrix
            candidate = _failed_candidate(
                policy,
                f"candidate_failed:{type(error).__name__}",
                cuda_oom=False,
            )
        candidates.append(candidate)
        gc.collect()
        torch.cuda.empty_cache()
    baseline = candidates[0]
    if baseline["status"] != "passed":
        raise RuntimeError("FP32 optimization baseline did not pass")
    baseline_performance = baseline["performance"]
    for candidate in candidates:
        if candidate["name"] == "fp32":
            candidate["promotion"] = {
                "accepted": False,
                "reasons": ["baseline_reference"],
            }
            continue
        performance = candidate["performance"]
        if candidate["status"] == "failed":
            candidate["promotion"] = {
                "accepted": False,
                "reasons": [candidate["failure_code"]],
            }
            continue
        decision = candidate_promotion(
            baseline_p50_ms=baseline_performance["latency_ms"]["p50"],
            baseline_throughput=baseline_performance["throughput_per_second"],
            baseline_peak_bytes=baseline_performance["peak_reserved_bytes"],
            candidate_p50_ms=performance["latency_ms"]["p50"],
            candidate_throughput=performance["throughput_per_second"],
            candidate_peak_bytes=performance["peak_reserved_bytes"],
            parity_passed=candidate["parity"]["passed"],
            cuda_oom=candidate["reliability"]["cuda_oom"],
            model_switch_leak=False,
            graph_break_count=candidate["compile"]["graph_break_count"],
            recompilation_count=candidate["compile"]["recompilation_count"],
        )
        candidate["promotion"] = asdict(decision)
    return {"baseline": "fp32", "candidates": candidates, "selected": "fp32"}


def _measure_candidate(
    *,
    torch: object,
    policy: CandidatePolicy,
    load_model: Callable[[], tuple[object, tuple[object, ...], float]],
    call_model: Callable[[object, tuple[object, ...]], object],
    project_output: Callable[[object], dict[str, np.ndarray]],
    warmup_runs: int,
    measured_runs: int,
) -> dict[str, object]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats("cuda:0")
    model, inputs, load_ms = load_model()
    compilation_ms: float | None = None
    compilation_started: float | None = None
    if policy.compile:
        torch._dynamo.reset()
        compilation_started = perf_counter()
        model = torch.compile(model, fullgraph=True, mode="reduce-overhead")
    torch.backends.cuda.matmul.allow_tf32 = policy.tf32
    torch.backends.cudnn.allow_tf32 = policy.tf32
    torch.set_float32_matmul_precision("high" if policy.tf32 else "highest")
    dtype = getattr(torch, policy.autocast_dtype) if policy.autocast_dtype else None
    context = (
        torch.autocast(device_type="cuda", dtype=dtype)
        if dtype is not None
        else nullcontext()
    )
    with torch.inference_mode(), context:
        for warmup_index in range(warmup_runs):
            output = call_model(model, inputs)
            if policy.compile and warmup_index == 0:
                torch.cuda.synchronize("cuda:0")
                compilation_ms = (perf_counter() - compilation_started) * 1_000.0
        torch.cuda.synchronize("cuda:0")
        measured: list[float] = []
        for _ in range(measured_runs):
            started = torch.cuda.Event(enable_timing=True)
            completed = torch.cuda.Event(enable_timing=True)
            started.record()
            output = call_model(model, inputs)
            completed.record()
            torch.cuda.synchronize("cuda:0")
            measured.append(float(started.elapsed_time(completed)))
        projected = project_output(output)
    distribution = _distribution(measured)
    unique_graphs = (
        int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
        if policy.compile
        else 0
    )
    graph_breaks = (
        sum(torch._dynamo.utils.counters["graph_break"].values())
        if policy.compile
        else 0
    )
    return {
        "name": policy.name,
        "status": "passed",
        "load_ms": load_ms,
        "parity": {"passed": False},
        "performance": {
            "latency_ms": distribution,
            "throughput_per_second": 1_000.0 / statistics.mean(measured),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated("cuda:0")),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved("cuda:0")),
        },
        "reliability": {"cuda_oom": False, "model_switch_leak": False},
        "compile": {
            "enabled": policy.compile,
            "compilation_ms": compilation_ms,
            "recompilation_count": max(0, unique_graphs - 1),
            "graph_break_count": graph_breaks,
            "fullgraph_required": policy.compile,
        },
        "precision": policy.autocast_dtype or "float32",
        "tf32": policy.tf32,
        "_outputs": projected,
    }


def _failed_candidate(
    policy: CandidatePolicy,
    failure_code: str,
    *,
    cuda_oom: bool,
) -> dict[str, object]:
    return {
        "name": policy.name,
        "status": "failed",
        "failure_code": failure_code,
        "parity": {"passed": False},
        "performance": {
            "latency_ms": None,
            "throughput_per_second": None,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
        },
        "reliability": {"cuda_oom": cuda_oom, "model_switch_leak": False},
        "compile": {
            "enabled": policy.compile,
            "compilation_ms": None,
            "recompilation_count": None,
            "graph_break_count": None,
            "fullgraph_required": policy.compile,
        },
        "precision": policy.autocast_dtype or "float32",
        "tf32": policy.tf32,
    }


def _classifier_inputs(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path.expanduser().resolve(), allow_pickle=False) as bundle:
        crops = np.ascontiguousarray(bundle["crops"][None, ...], dtype=np.float32)
        input_ids = _pad_tokens(bundle["input_ids"], value=1)
        attention_mask = _pad_tokens(bundle["attention_mask"], value=0)
    return crops, input_ids, attention_mask


def _pad_tokens(values: np.ndarray, *, value: int) -> np.ndarray:
    if values.ndim != 2 or values.shape[0] != 1 or values.shape[1] > 90:
        raise RuntimeError("MMBCD token tensor exceeds the fixed candidate contract")
    result = np.full((1, 90), value, dtype=np.int64)
    result[:, : values.shape[1]] = values
    return result


def _numpy(value: object) -> np.ndarray:
    return np.ascontiguousarray(value.detach().float().cpu().numpy())


def _distribution(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    percentile = lambda p: ordered[max(0, math.ceil(p * len(ordered)) - 1)]
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
        "mean": statistics.mean(ordered),
        "population_stddev": statistics.pstdev(ordered),
    }


def _verify_single_residency_evidence(path: Path) -> str:
    payload = load_json(path)
    cycles = payload.get("cycles")
    if not isinstance(cycles, list) or len(cycles) < 2:
        raise RuntimeError("single-residency evidence has too few cycles")
    for cycle in cycles:
        if (
            cycle.get("resident_after_detector") != [DETECTOR_MODEL_ID]
            or cycle.get("resident_after_classifier") != [CLASSIFIER_MODEL_ID]
        ):
            raise RuntimeError("single-residency evidence contains an invalid snapshot")
    if payload.get("final_status", {}).get("resident_models") != [CLASSIFIER_MODEL_ID]:
        raise RuntimeError("single-residency evidence has an invalid final resident")
    return sha256_file(path.expanduser().resolve())


def _verify_clean_revision(root: Path, revision: str) -> None:
    prefix = ["git", "-c", f"safe.directory={root.resolve().as_posix()}", "-C", str(root)]
    observed = subprocess.check_output([*prefix, "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output([*prefix, "status", "--porcelain"], text=True).strip()
    if observed != revision or dirty:
        raise RuntimeError("optimization evidence requires the exact clean revision")


def _nvidia_value(field: str) -> str:
    return subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits"],
        text=True,
        timeout=10,
    ).splitlines()[0].strip()


def _render_markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# PyTorch optimization matrix",
        "",
        f"Revision: `{report['revision']}`",
        "",
        "| Model | Candidate | Status | Parity | Warm p50 ms | Peak reserved bytes | Accepted |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
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
            "The production runtime remains eager FP32 until a candidate also passes the packaged benchmark and restart gates.",
            "",
            str(report["validation_boundary"]),
            "",
        )
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
