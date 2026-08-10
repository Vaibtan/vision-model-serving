#!/usr/bin/env python3
"""Build and gate strict MMBCD TensorRT and detector coverage candidates."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import UTC, datetime
import gc
import hashlib
from importlib import metadata
from io import StringIO
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from time import perf_counter
from typing import Any, Mapping

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
    validate_optimization_report,
)
from vision_model_serving.validation.revision import require_clean_revision  # noqa: E402


_COMMIT = re.compile(r"[0-9a-f]{40}")
_DEPENDENCIES = {
    "torch-tensorrt": "2.8.0",
    "tensorrt": "10.12.0.36",
    "onnx": "1.16.0",
    "polygraphy": "0.49.24",
    "cuda-python": "12.8.0",
}


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
    parser.add_argument("--optimization-evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--measured-runs", type=int, default=10)
    args = parser.parse_args()
    if _COMMIT.fullmatch(args.revision) is None:
        parser.error("--revision must be a full lowercase Git commit")
    if args.warmup_runs < 1 or args.measured_runs < 5:
        parser.error("warmup/measured runs must be at least 1/5")
    project_root = args.project_root.expanduser().resolve()
    require_clean_revision(project_root, args.revision)
    optimization = load_json(args.optimization_evidence)
    validate_optimization_report(optimization)
    if optimization["revision"] != args.revision:
        raise RuntimeError("optimization evidence revision differs")
    dependencies = _verify_dependencies()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import tensorrt as trt
    import torch
    import torch_tensorrt

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
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
    detector_host = np.array(
        DetectorPreprocessor().prepare(canonical.pixels).tensor,
        dtype=np.float32,
        copy=True,
    )
    detector = _probe_detector(
        torch=torch,
        torch_tensorrt=torch_tensorrt,
        artifact=artifacts[DETECTOR_MODEL_ID],
        project_root=project_root,
        focalnet_root=focalnet_root,
        detector_host=detector_host,
        canonical=canonical,
        output_dir=output_dir,
    )
    gc.collect()
    torch.cuda.empty_cache()
    try:
        classifier = _build_classifier(
            torch=torch,
            torch_tensorrt=torch_tensorrt,
            trt=trt,
            artifact=artifacts[CLASSIFIER_MODEL_ID],
            project_root=project_root,
            dino_root=args.dino_root,
            mmbcd_root=args.mmbcd_root,
            input_bundle=args.mmbcd_input_bundle,
            output_dir=output_dir,
            warmup_runs=args.warmup_runs,
            measured_runs=args.measured_runs,
        )
    except Exception as error:
        candidate_plan = output_dir / "mmbcd-fp32.candidate.plan"
        if candidate_plan.exists():
            candidate_plan.unlink()
        failure_path = output_dir / "mmbcd-tensorrt-failure.txt"
        failure_path.write_text(
            _sanitize_error(
                error,
                project_root,
                args.dino_root.expanduser().resolve(),
                args.mmbcd_root.expanduser().resolve(),
            ),
            encoding="utf-8",
        )
        classifier = {
            "strict_export": False,
            "dryrun_completed": False,
            "require_full_compilation": True,
            "pytorch_partition_count": None,
            "unsupported_operators": [],
            "engine_built": False,
            "tensorrt_only_runtime_passed": False,
            "parity": {"passed": False},
            "performance": {"promotion_threshold_passed": False},
            "decision": "stop",
            "failure_code": f"classifier_tensorrt_failed:{type(error).__name__}",
            "failure_report_sha256": sha256_file(failure_path),
        }
    if classifier["decision"] == "go" and detector["decision"] != "go":
        decision = "partial"
    elif classifier["decision"] == "go" and detector["decision"] == "go":
        decision = "go"
    else:
        decision = "stop"
    report = {
        "schema_version": 1,
        "revision": args.revision,
        "measured_at": datetime.now(UTC).isoformat(),
        "environment": {
            "dependencies": dependencies,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name("cuda:0"),
            "compute_capability": ".".join(
                str(value) for value in torch.cuda.get_device_capability("cuda:0")
            ),
            "driver": _nvidia_value("driver_version"),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "optimization_evidence_sha256": sha256_file(
                args.optimization_evidence.expanduser().resolve()
            ),
        },
        "models": {
            DETECTOR_MODEL_ID: detector,
            CLASSIFIER_MODEL_ID: classifier,
        },
        "decision": decision,
        "production_selection": {
            "backend": "pytorch-eager",
            "reason": (
                "TensorRT remains evidence-gated. A classifier engine GO is a "
                "PARTIAL result until the packaged single-residency endpoint "
                "passes same-revision parity, restart, and end-to-end benchmarks."
            ),
        },
        "validation_boundary": (
            "Fixed-shape FP32/TF32-disabled TensorRT feasibility on one NVIDIA L4 "
            "and one public DICOM. It is not clinical or cross-hardware evidence."
        ),
    }
    write_json_atomic(output_dir / "tensorrt-spike.json", report)
    (output_dir / "tensorrt-spike.md").write_text(
        _render_markdown(report),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"TENSORRT SPIKE CONCLUDED: {decision.upper()}")
    return 0


def _probe_detector(
    *,
    torch: object,
    torch_tensorrt: object,
    artifact: object,
    project_root: Path,
    focalnet_root: Path,
    detector_host: np.ndarray,
    canonical: object,
    output_dir: Path,
) -> dict[str, object]:
    loaded = _load_verified_detector_model(
        artifact,
        model_factory=_LocalFocalNetDinoFactory(
            repository_root=focalnet_root,
            project_root=project_root,
            expected_revision=artifact.repository_revision,
            device="cuda:0",
        ).build,
        device="cuda:0",
    )
    tensor = torch.from_numpy(detector_host).to("cuda:0")
    with torch.inference_mode():
        baseline = loaded.model([tensor])
    proposals = DetectorPostprocessor().process(
        _numpy(baseline["pred_logits"]),
        _numpy(baseline["pred_boxes"]),
        canonical.geometry,
    )
    if proposals.prediction_sha256 != DETECTOR_PREDICTION_SHA256:
        raise RuntimeError("detector TensorRT baseline differs from the L4 golden")
    capture = StringIO()
    strict_export = False
    dryrun_completed = False
    failure_code: str | None = None
    started = perf_counter()
    try:
        exported = torch.export.export(loaded.model, ([tensor],), strict=True)
        strict_export = True
        with redirect_stdout(capture), redirect_stderr(capture):
            torch_tensorrt.dynamo.compile(
                exported,
                arg_inputs=([tensor],),
                enabled_precisions={torch.float32},
                require_full_compilation=True,
                disable_tf32=True,
                dryrun=True,
            )
        dryrun_completed = True
        failure_code = "runtime_parity_and_plugin_gate_not_implemented"
    except Exception as error:
        failure_code = f"strict_coverage_failed:{type(error).__name__}"
        capture.write("\n" + _sanitize_error(error, project_root, focalnet_root))
    report_path = output_dir / "focalnet-dino-tensorrt-coverage.txt"
    report_path.write_text(capture.getvalue(), encoding="utf-8")
    return {
        "strict_export": strict_export,
        "dryrun_completed": dryrun_completed,
        "require_full_compilation": True,
        "coverage_report_sha256": sha256_file(report_path),
        "elapsed_ms": (perf_counter() - started) * 1_000.0,
        "custom_operator": "MultiScaleDeformableAttention",
        "plugin_required": not dryrun_completed,
        "engine_built": False,
        "parity_passed": False,
        "performance_threshold_passed": False,
        "decision": "stop",
        "failure_code": failure_code,
        "next_action": (
            "Implement an exact TensorRT IPluginV3 plus Torch-TensorRT converter "
            "only if the assignment budget justifies custom deformable-attention work."
        ),
    }


def _build_classifier(
    *,
    torch: object,
    torch_tensorrt: object,
    trt: object,
    artifact: object,
    project_root: Path,
    dino_root: Path,
    mmbcd_root: Path,
    input_bundle: Path,
    output_dir: Path,
    warmup_runs: int,
    measured_runs: int,
) -> dict[str, object]:
    loaded = _load_verified_mmbcd_model(
        artifact,
        model_factory=LocalMmbcdModelFactory(
            dino_root=dino_root,
            mmbcd_root=mmbcd_root,
            project_root=project_root,
            expected_mmbcd_revision=artifact.repository_revision,
        ).build,
        device="cuda:0",
    )
    host_inputs = _classifier_inputs(input_bundle)
    inputs = tuple(torch.from_numpy(value).to("cuda:0") for value in host_inputs)
    with torch.inference_mode():
        for _ in range(warmup_runs):
            eager = loaded.model(*inputs)
        torch.cuda.synchronize("cuda:0")
        eager_latencies: list[float] = []
        for _ in range(measured_runs):
            started = perf_counter()
            eager = loaded.model(*inputs)
            torch.cuda.synchronize("cuda:0")
            eager_latencies.append((perf_counter() - started) * 1_000.0)
    eager_outputs = {
        "logits": _numpy(eager[0]),
        "fused_embeddings": _numpy(eager[1]),
        "roi_attention": _numpy(eager[2]),
    }
    digest = hashlib.sha256()
    digest.update(eager_outputs["logits"].tobytes())
    digest.update(eager_outputs["fused_embeddings"].tobytes())
    if digest.hexdigest() != MMBCD_PREDICTION_SHA256:
        raise RuntimeError("MMBCD fixed-shape baseline differs from the L4 golden")

    capture = StringIO()
    export_started = perf_counter()
    exported = torch.export.export(loaded.model, inputs, strict=True)
    export_ms = (perf_counter() - export_started) * 1_000.0
    with torch.inference_mode():
        exported_outputs = exported.module()(*inputs)
    export_differences = {
        name: float(np.max(np.abs(_numpy(value) - eager_outputs[name])))
        for name, value in zip(eager_outputs, exported_outputs, strict=True)
    }
    if any(value > 1e-6 for value in export_differences.values()):
        raise RuntimeError("strict MMBCD export differs from eager FP32")
    with redirect_stdout(capture), redirect_stderr(capture):
        torch_tensorrt.dynamo.compile(
            exported,
            arg_inputs=inputs,
            enabled_precisions={torch.float32},
            require_full_compilation=True,
            disable_tf32=True,
            dryrun=True,
        )
    dryrun_path = output_dir / "mmbcd-tensorrt-dryrun.txt"
    dryrun_path.write_text(capture.getvalue(), encoding="utf-8")
    build_started = perf_counter()
    engine_bytes = torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
        exported,
        arg_inputs=inputs,
        enabled_precisions={torch.float32},
        require_full_compilation=True,
        disable_tf32=True,
        use_python_runtime=True,
        version_compatible=False,
        hardware_compatible=False,
        pass_through_build_failures=False,
    )
    build_ms = (perf_counter() - build_started) * 1_000.0
    candidate_plan = output_dir / "mmbcd-fp32.candidate.plan"
    candidate_plan.write_bytes(engine_bytes)
    runtime_bundle = output_dir / "mmbcd-tensorrt-outputs.npz"
    runtime_report = output_dir / "mmbcd-tensorrt-runtime.json"
    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "tensorrt_runtime_verify.py"),
            "--plan",
            str(candidate_plan),
            "--input-bundle",
            str(input_bundle),
            "--output-bundle",
            str(runtime_bundle),
            "--report",
            str(runtime_report),
            "--warmup-runs",
            str(warmup_runs),
            "--measured-runs",
            str(measured_runs),
        ],
        check=True,
        timeout=900,
    )
    with np.load(runtime_bundle, allow_pickle=False) as bundle:
        trt_outputs = {name: np.ascontiguousarray(bundle[name]) for name in eager_outputs}
    differences = {
        name: float(np.max(np.abs(value - eager_outputs[name])))
        for name, value in trt_outputs.items()
    }
    predicted_class_equal = int(np.argmax(trt_outputs["logits"])) == int(
        np.argmax(eager_outputs["logits"])
    )
    parity_passed = predicted_class_equal and all(
        value <= 1e-4 for value in differences.values()
    )
    parity = {
        "passed": parity_passed,
        "absolute_tolerance": 1e-4,
        "max_absolute_difference": differences,
        "predicted_class_equal": predicted_class_equal,
        "eager_output_sha256": {
            name: sha256_array(value) for name, value in eager_outputs.items()
        },
        "tensorrt_output_sha256": {
            name: sha256_array(value) for name, value in trt_outputs.items()
        },
    }
    parity_path = output_dir / "mmbcd-tensorrt-parity.json"
    write_json_atomic(parity_path, parity)
    runtime = load_json(runtime_report)
    eager_p50 = float(np.median(eager_latencies))
    trt_p50 = float(runtime["performance"]["p50_ms"])
    performance_passed = trt_p50 <= eager_p50 * 0.85
    performance = {
        "eager_latencies_ms": eager_latencies,
        "eager_p50_ms": eager_p50,
        "tensorrt_p50_ms": trt_p50,
        "warm_p50_improvement_percent": (1.0 - trt_p50 / eager_p50) * 100.0,
        "promotion_threshold_passed": performance_passed,
        "scope": "fixed-shape classifier forward only",
    }
    performance_path = output_dir / "mmbcd-tensorrt-performance.json"
    write_json_atomic(performance_path, performance)
    decision = "go" if parity_passed and performance_passed else "stop"
    plan_path = output_dir / "mmbcd-fp32.plan"
    if decision == "go":
        candidate_plan.replace(plan_path)
        manifest = {
            "schema_version": 1,
            "model": {
                "id": CLASSIFIER_MODEL_ID,
                "checkpoint_sha256": artifact.sha256,
                "repository_revision": artifact.repository_revision,
                "wrapper_contract_version": 1,
            },
            "engine": {
                "filename": plan_path.name,
                "sha256": sha256_file(plan_path),
                "precision": "float32",
                "tf32": False,
            },
            "builder": {
                "torch": torch.__version__,
                "torch_tensorrt": metadata.version("torch-tensorrt"),
                "tensorrt": trt.__version__,
                "cuda": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name("cuda:0"),
                "compute_capability": ".".join(
                    str(value)
                    for value in torch.cuda.get_device_capability("cuda:0")
                ),
            },
            "inputs": [
                {"name": "roi_crops", "dtype": "float32", "shape": [1, 8, 3, 224, 224]},
                {"name": "input_ids", "dtype": "int64", "shape": [1, 90]},
                {"name": "attention_mask", "dtype": "int64", "shape": [1, 90]},
            ],
            "outputs": [
                {"name": "logits", "dtype": "float32", "shape": [1, 2]},
                {"name": "fused_embeddings", "dtype": "float32", "shape": [1, 768]},
                {"name": "roi_attention", "dtype": "float32", "shape": [1, 1, 8]},
            ],
            "coverage": {
                "strict_export": True,
                "require_full_compilation": True,
                "pytorch_partition_count": 0,
                "unsupported_operators": [],
                "dry_run_report_sha256": sha256_file(dryrun_path),
            },
            "plugin": None,
            "parity": {
                "passed": True,
                "report_sha256": sha256_file(parity_path),
            },
            "performance": {
                "promotion_threshold_passed": True,
                "report_sha256": sha256_file(performance_path),
            },
            "decision": "go",
        }
        write_json_atomic(output_dir / "mmbcd-tensorrt-manifest.json", manifest)
    else:
        candidate_plan.unlink()
    return {
        "strict_export": True,
        "export_ms": export_ms,
        "export_max_absolute_difference": export_differences,
        "dryrun_completed": True,
        "require_full_compilation": True,
        "pytorch_partition_count": 0,
        "unsupported_operators": [],
        "dryrun_report_sha256": sha256_file(dryrun_path),
        "engine_built": True,
        "engine_build_ms": build_ms,
        "tensorrt_only_runtime_passed": all(runtime["gates"].values()),
        "runtime_report_sha256": sha256_file(runtime_report),
        "parity": parity,
        "performance": performance,
        "decision": decision,
        "promotion_boundary": (
            "Engine GO permits the immutable plan to be considered by a later "
            "packaged endpoint experiment; it does not select production TensorRT."
        ),
    }


def _classifier_inputs(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path.expanduser().resolve(), allow_pickle=False) as bundle:
        crops = np.ascontiguousarray(bundle["crops"][None, ...], dtype=np.float32)
        ids = _pad(bundle["input_ids"], 1)
        mask = _pad(bundle["attention_mask"], 0)
    return crops, ids, mask


def _pad(values: np.ndarray, fill: int) -> np.ndarray:
    if values.ndim != 2 or values.shape[0] != 1 or values.shape[1] > 90:
        raise RuntimeError("MMBCD token input differs from the fixed TensorRT contract")
    result = np.full((1, 90), fill, dtype=np.int64)
    result[:, : values.shape[1]] = values
    return result


def _verify_dependencies() -> dict[str, str]:
    observed = {name: metadata.version(name) for name in _DEPENDENCIES}
    if observed != _DEPENDENCIES:
        raise RuntimeError(f"TensorRT dependency lane differs: {observed!r}")
    return observed


def _sanitize_error(error: Exception, *roots: Path) -> str:
    value = str(error)
    for root in roots:
        value = value.replace(str(root), f"<{root.name}>")
    return value[:20_000]


def _numpy(value: object) -> np.ndarray:
    return np.ascontiguousarray(value.detach().float().cpu().numpy())


def _nvidia_value(field: str) -> str:
    return subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits"],
        text=True,
        timeout=10,
    ).splitlines()[0].strip()


def _render_markdown(report: Mapping[str, object]) -> str:
    detector = report["models"][DETECTOR_MODEL_ID]
    classifier = report["models"][CLASSIFIER_MODEL_ID]
    return "\n".join(
        (
            "# TensorRT spike",
            "",
            f"Revision: `{report['revision']}`",
            f"Decision: **{str(report['decision']).upper()}**",
            "",
            "| Model | Strict export | Full compilation | Engine | Parity | Performance gate | Decision |",
            "| --- | --- | --- | --- | --- | --- | --- |",
            f"| {DETECTOR_MODEL_ID} | {detector['strict_export']} | {detector['dryrun_completed']} | {detector['engine_built']} | {detector['parity_passed']} | {detector['performance_threshold_passed']} | {detector['decision']} |",
            f"| {CLASSIFIER_MODEL_ID} | {classifier['strict_export']} | {classifier['dryrun_completed']} | {classifier['engine_built']} | {classifier['parity']['passed']} | {classifier['performance']['promotion_threshold_passed']} | {classifier['decision']} |",
            "",
            str(report["production_selection"]["reason"]),
            "",
            str(report["validation_boundary"]),
            "",
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
