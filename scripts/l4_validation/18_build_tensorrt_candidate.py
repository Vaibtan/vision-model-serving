#!/usr/bin/env python3
"""Build and publish strict MMBCD TensorRT and detector coverage evidence."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

from _common import (
    decode_dicom_file,
    default_paths,
    load_json,
    sha256_file,
    write_json_atomic,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vision_model_serving.artifacts import ArtifactRegistry  # noqa: E402
from vision_model_serving.detector.postprocessing import (  # noqa: E402
    DetectorPreprocessor,
)
from vision_model_serving.detector.runtime import (  # noqa: E402
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
from vision_model_serving.validation.tensorrt_experiment import (  # noqa: E402
    TensorRtReportContext,
    TorchTensorRtMeasurements,
    run_tensorrt_experiment,
)


_COMMIT = re.compile(r"[0-9a-f]{40}")
_DEPENDENCIES = {
    "torch-tensorrt": "2.8.0",
    "tensorrt": "10.12.0.36",
    "onnx": "1.16.0",
    "polygraphy": "0.49.24",
    "cuda-python": "12.8.0",
}


def main() -> int:
    args = _parse_args()
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
    __import__("tensorrt")
    import torch
    import torch_tensorrt

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_root = args.artifact_root.expanduser().resolve()
    tokenizer_root = args.tokenizer_root.expanduser().resolve()
    focalnet_root = args.focalnet_root.expanduser().resolve()
    dino_root = args.dino_root.expanduser().resolve()
    mmbcd_root = args.mmbcd_root.expanduser().resolve()
    dicom_path = args.dicom.expanduser().resolve()
    input_bundle = args.mmbcd_input_bundle.expanduser().resolve()
    optimization_evidence = args.optimization_evidence.expanduser().resolve()
    registry = ArtifactRegistry(
        project_root / "config" / "model-artifacts.json",
        artifact_root=artifact_root,
        tokenizer_root=tokenizer_root,
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
    canonical = decode_dicom_file(dicom_path, DicomCanonicalizer())
    detector_host = np.array(
        DetectorPreprocessor().prepare(canonical.pixels).tensor,
        dtype=np.float32,
        copy=True,
    )

    measurements = TorchTensorRtMeasurements(
        torch=torch,
        torch_tensorrt=torch_tensorrt,
        detector_artifact=artifacts[DETECTOR_MODEL_ID],
        classifier_artifact=artifacts[CLASSIFIER_MODEL_ID],
        project_root=project_root,
        focalnet_root=focalnet_root,
        dino_root=dino_root,
        mmbcd_root=mmbcd_root,
        input_bundle=input_bundle,
        output_dir=output_dir,
        runtime_verify_script=project_root / "scripts" / "tensorrt_runtime_verify.py",
        python_executable=sys.executable,
        detector_host=detector_host,
        canonical=canonical,
        warmup_runs=args.warmup_runs,
        measured_runs=args.measured_runs,
    )

    def report_environment() -> dict[str, object]:
        return {
            "dependencies": dependencies,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name("cuda:0"),
            "compute_capability": ".".join(
                str(value) for value in torch.cuda.get_device_capability("cuda:0")
            ),
            "driver": _nvidia_value("driver_version"),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "optimization_evidence_sha256": sha256_file(optimization_evidence),
        }

    context = TensorRtReportContext(
        revision=args.revision,
        environment=report_environment,
    )
    result = run_tensorrt_experiment(
        context,
        measurements,
        output_dir=output_dir,
        failure_roots=(
            project_root,
            artifact_root,
            tokenizer_root,
            focalnet_root,
            dino_root,
            mmbcd_root,
            dicom_path.parent,
            input_bundle.parent,
            optimization_evidence.parent,
        ),
    )
    write_json_atomic(output_dir / "tensorrt-spike.json", result.report)
    (output_dir / "tensorrt-spike.md").write_text(
        result.markdown,
        encoding="utf-8",
    )
    print(json.dumps(result.report, indent=2, sort_keys=True))
    print(f"TENSORRT SPIKE CONCLUDED: {result.report['decision'].upper()}")
    return 0


def _parse_args() -> argparse.Namespace:
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
    return args


def _verify_dependencies() -> dict[str, str]:
    observed = {name: metadata.version(name) for name in _DEPENDENCIES}
    if observed != _DEPENDENCIES:
        raise RuntimeError(f"TensorRT dependency lane differs: {observed!r}")
    return observed


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


if __name__ == "__main__":
    raise SystemExit(main())
