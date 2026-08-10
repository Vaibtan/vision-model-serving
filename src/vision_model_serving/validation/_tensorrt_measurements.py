"""Torch/TensorRT measurement adapter for the public experiment interface."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
import gc
import hashlib
from io import StringIO
import json
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any

import numpy as np

from vision_model_serving.acceleration.tensorrt import detector_plugin_requirement
from vision_model_serving.classifier.runtime import (
    LocalMmbcdModelFactory,
    _load_verified_mmbcd_model,
)
from vision_model_serving.detector.postprocessing import (
    DetectorPostprocessor,
)
from vision_model_serving.detector.runtime import (
    _LocalFocalNetDinoFactory,
    _load_verified_detector_model,
)
from vision_model_serving.validation.evidence import sanitize_error_detail
from vision_model_serving.validation.golden import (
    REFERENCE_DETECTOR_PREDICTION_SHA256,
    REFERENCE_MMBCD_PREDICTION_SHA256,
)
from vision_model_serving.validation.reporting import write_json_atomic


@dataclass(slots=True)
class TorchTensorRtMeasurements:
    """Production GPU adapter for one detector/classifier TensorRT experiment."""

    torch: Any
    torch_tensorrt: Any
    detector_artifact: Any
    classifier_artifact: Any
    project_root: Path
    focalnet_root: Path
    dino_root: Path
    mmbcd_root: Path
    input_bundle: Path
    output_dir: Path
    runtime_verify_script: Path
    python_executable: str
    detector_host: np.ndarray
    canonical: Any
    warmup_runs: int
    measured_runs: int

    def measure_detector(self) -> dict[str, object]:
        return _probe_detector(
            torch=self.torch,
            torch_tensorrt=self.torch_tensorrt,
            artifact=self.detector_artifact,
            project_root=self.project_root,
            focalnet_root=self.focalnet_root,
            detector_host=self.detector_host,
            canonical=self.canonical,
            output_dir=self.output_dir,
        )

    def release_detector(self) -> None:
        gc.collect()
        self.torch.cuda.empty_cache()

    def measure_classifier(self) -> dict[str, object]:
        return _build_classifier(
            torch=self.torch,
            torch_tensorrt=self.torch_tensorrt,
            artifact=self.classifier_artifact,
            project_root=self.project_root,
            dino_root=self.dino_root,
            mmbcd_root=self.mmbcd_root,
            input_bundle=self.input_bundle,
            output_dir=self.output_dir,
            runtime_verify_script=self.runtime_verify_script,
            python_executable=self.python_executable,
            warmup_runs=self.warmup_runs,
            measured_runs=self.measured_runs,
        )


def load_classifier_inputs(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and validate the exact classifier bundle used by TensorRT."""

    with np.load(path.expanduser().resolve(), allow_pickle=False) as bundle:
        crops = np.ascontiguousarray(bundle["crops"][None, ...], dtype=np.float32)
        ids = _token_input(bundle["input_ids"])
        mask = _token_input(bundle["attention_mask"])
    if ids.shape != mask.shape:
        raise RuntimeError("MMBCD token inputs have different shapes")
    return crops, ids, mask


def _probe_detector(
    *,
    torch: Any,
    torch_tensorrt: Any,
    artifact: Any,
    project_root: Path,
    focalnet_root: Path,
    detector_host: np.ndarray,
    canonical: Any,
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
    if proposals.prediction_sha256 != REFERENCE_DETECTOR_PREDICTION_SHA256:
        raise RuntimeError("detector TensorRT baseline differs from the L4 golden")
    capture = StringIO()
    strict_export = False
    dryrun_completed = False
    failure_code: str | None = None
    unsupported_operators: tuple[str, ...] = ()
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
        stage = "coverage_analysis_failed" if strict_export else "strict_export_failed"
        failure_code = f"{stage}:{type(error).__name__}"
        detail = sanitize_error_detail(error, (project_root, focalnet_root))
        capture.write("\n" + detail)
        if strict_export and "MultiScaleDeformableAttention" in detail:
            unsupported_operators = ("MultiScaleDeformableAttention",)
    report_path = output_dir / "focalnet-dino-tensorrt-coverage.txt"
    report_path.write_text(capture.getvalue(), encoding="utf-8")
    plugin = detector_plugin_requirement(
        strict_export=strict_export,
        dryrun_completed=dryrun_completed,
        unsupported_operators=unsupported_operators,
    )
    next_action = (
        "Correct strict export and rerun coverage before deciding whether a "
        "TensorRT plugin is required."
        if plugin.status == "not_reached"
        else (
            "Implement and validate an exact TensorRT IPluginV3 plus Torch-TensorRT converter."
            if plugin.required is True
            else "Complete coverage analysis before implementing conversion code."
        )
    )
    return {
        "strict_export": strict_export,
        "dryrun_completed": dryrun_completed,
        "require_full_compilation": True,
        "coverage_report_sha256": _sha256_file(report_path),
        "elapsed_ms": (perf_counter() - started) * 1_000.0,
        "custom_operator": "MultiScaleDeformableAttention",
        "unsupported_operators": list(unsupported_operators),
        "plugin_requirement_status": plugin.status,
        "plugin_required": plugin.required,
        "engine_built": False,
        "parity_passed": False,
        "performance_threshold_passed": False,
        "decision": "stop",
        "failure_code": failure_code,
        "next_action": next_action,
    }


def _build_classifier(
    *,
    torch: Any,
    torch_tensorrt: Any,
    artifact: Any,
    project_root: Path,
    dino_root: Path,
    mmbcd_root: Path,
    input_bundle: Path,
    output_dir: Path,
    runtime_verify_script: Path,
    python_executable: str,
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
    host_inputs = load_classifier_inputs(input_bundle)
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
    if digest.hexdigest() != REFERENCE_MMBCD_PREDICTION_SHA256:
        raise RuntimeError("MMBCD TensorRT baseline differs from the L4 golden")

    capture = StringIO()
    dynamic_report_path = output_dir / "mmbcd-dynamic-export.txt"
    try:
        token_width = torch.export.Dim("token_width", min=2, max=90)
        torch.export.export(
            loaded.model,
            inputs,
            dynamic_shapes=({}, {1: token_width}, {1: token_width}),
            strict=True,
        )
        dynamic_export = {
            "passed": True,
            "failure_code": None,
        }
        dynamic_report_path.write_text(
            "strict dynamic export passed\n",
            encoding="utf-8",
        )
    except Exception as error:
        dynamic_export = {
            "passed": False,
            "failure_code": f"dynamic_export_failed:{type(error).__name__}",
        }
        dynamic_report_path.write_text(
            sanitize_error_detail(error, (project_root, dino_root, mmbcd_root)),
            encoding="utf-8",
        )
    dynamic_export["report_sha256"] = _sha256_file(dynamic_report_path)
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
    trt_inputs = (
        torch_tensorrt.Input(
            shape=(1, 8, 3, 224, 224),
            dtype=torch.float32,
            name="roi_crops",
        ),
        torch_tensorrt.Input(
            shape=tuple(inputs[1].shape),
            dtype=torch.int64,
            name="input_ids",
        ),
        torch_tensorrt.Input(
            shape=tuple(inputs[2].shape),
            dtype=torch.int64,
            name="attention_mask",
        ),
    )
    with redirect_stdout(capture), redirect_stderr(capture):
        torch_tensorrt.dynamo.compile(
            exported,
            arg_inputs=trt_inputs,
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
        arg_inputs=trt_inputs,
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
            python_executable,
            str(runtime_verify_script),
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
    parity_passed = predicted_class_equal and all(value <= 1e-4 for value in differences.values())
    parity = {
        "passed": parity_passed,
        "absolute_tolerance": 1e-4,
        "max_absolute_difference": differences,
        "predicted_class_equal": predicted_class_equal,
        "eager_output_sha256": {
            name: _sha256_array(value) for name, value in eager_outputs.items()
        },
        "tensorrt_output_sha256": {
            name: _sha256_array(value) for name, value in trt_outputs.items()
        },
    }
    parity_path = output_dir / "mmbcd-tensorrt-parity.json"
    write_json_atomic(parity_path, parity)
    runtime = _load_json(runtime_report)
    eager_p50 = float(np.median(eager_latencies))
    trt_p50 = float(runtime["performance"]["p50_ms"])
    performance_passed = trt_p50 <= eager_p50 * 0.85
    performance = {
        "eager_latencies_ms": eager_latencies,
        "eager_p50_ms": eager_p50,
        "tensorrt_p50_ms": trt_p50,
        "warm_p50_improvement_percent": (1.0 - trt_p50 / eager_p50) * 100.0,
        "promotion_threshold_passed": performance_passed,
        "scope": (
            "classifier forward only for the exact static token-width-5 public "
            "fixture; it is not production shape coverage"
        ),
    }
    performance_path = output_dir / "mmbcd-tensorrt-performance.json"
    write_json_atomic(performance_path, performance)
    fixture_engine_sha256 = _sha256_file(candidate_plan)
    shape_coverage = {
        "passed": False,
        "required_profile": {
            "min_token_width": 2,
            "opt_token_width": 5,
            "max_token_width": 90,
        },
        "static_fixture_token_width": 5,
        "dynamic_strict_export": dynamic_export,
        "failure_code": (
            "dynamic_profile_engine_not_built"
            if dynamic_export["passed"]
            else dynamic_export["failure_code"]
        ),
    }
    return {
        "strict_export": True,
        "export_ms": export_ms,
        "export_max_absolute_difference": export_differences,
        "dryrun_completed": True,
        "require_full_compilation": True,
        "pytorch_partition_count": 0,
        "unsupported_operators": [],
        "dryrun_report_sha256": _sha256_file(dryrun_path),
        "engine_built": True,
        "engine_build_ms": build_ms,
        "static_fixture_engine_sha256": fixture_engine_sha256,
        "tensorrt_only_runtime_passed": all(runtime["gates"].values()),
        "runtime_report_sha256": _sha256_file(runtime_report),
        "parity": parity,
        "performance": performance,
        "shape_coverage": shape_coverage,
        "decision": "stop",
        "promotion_boundary": (
            "The static diagnostic plan is always deleted. Production requires "
            "one strict 2..90 token profile with the same parity/performance gates."
        ),
    }


def _token_input(values: np.ndarray) -> np.ndarray:
    if values.ndim != 2 or values.shape[0] != 1 or values.shape[1] < 2 or values.shape[1] > 90:
        raise RuntimeError("MMBCD token input is outside the TensorRT profile")
    return np.ascontiguousarray(values, dtype=np.int64)


def _numpy(value: Any) -> np.ndarray:
    return np.ascontiguousarray(value.detach().float().cpu().numpy())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(array: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("TensorRT runtime report must be a JSON object")
    return value
