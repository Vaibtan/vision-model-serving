"""Validate immutable TensorRT plans without importing PyTorch or model code."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any

from vision_model_serving.model_ids import CLASSIFIER_MODEL_ID, MODEL_IDS


_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_CLASSIFIER_INPUTS = (
    ("roi_crops", "float32", (1, 8, 3, 224, 224)),
    ("input_ids", "int64", (1, 90)),
    ("attention_mask", "int64", (1, 90)),
)
_CLASSIFIER_OUTPUTS = (
    ("logits", "float32", (1, 2)),
    ("fused_embeddings", "float32", (1, 768)),
    ("roi_attention", "float32", (1, 1, 8)),
)


class TensorRtManifestError(ValueError):
    """Raised when a plan cannot be selected without fallback."""


@dataclass(frozen=True, slots=True)
class EngineRuntimeCompatibility:
    tensorrt: str
    cuda: str
    gpu_name: str
    compute_capability: str

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value
            for value in (
                self.tensorrt,
                self.cuda,
                self.gpu_name,
                self.compute_capability,
            )
        ):
            raise ValueError("TensorRT runtime compatibility values are required")


@dataclass(frozen=True, slots=True)
class TensorRtEngineManifest:
    model_id: str
    checkpoint_sha256: str
    repository_revision: str
    plan_path: Path
    plan_sha256: str
    precision: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    decision: str
    plugin_library: Path | None = None
    plugin_sha256: str | None = None


def load_engine_manifest(
    path: str | Path,
    *,
    engine_root: str | Path,
    expected_model_id: str,
    expected_checkpoint_sha256: str,
    compatibility: EngineRuntimeCompatibility,
) -> TensorRtEngineManifest:
    """Return one verified full engine or raise; no eager fallback is represented."""

    manifest_path = Path(path).expanduser().resolve()
    root = Path(engine_root).expanduser().resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise TensorRtManifestError("TensorRT manifest is unavailable") from None
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version",
        "model",
        "engine",
        "builder",
        "inputs",
        "outputs",
        "coverage",
        "plugin",
        "parity",
        "performance",
        "decision",
    }:
        raise TensorRtManifestError("TensorRT manifest schema is invalid")
    if payload.get("schema_version") != 1:
        raise TensorRtManifestError("TensorRT manifest schema version is invalid")
    if expected_model_id not in MODEL_IDS:
        raise TensorRtManifestError("expected TensorRT model id is invalid")
    if _SHA256.fullmatch(expected_checkpoint_sha256) is None:
        raise TensorRtManifestError("expected checkpoint identity is invalid")

    model = _mapping(payload.get("model"), "model")
    if set(model) != {
        "id",
        "checkpoint_sha256",
        "repository_revision",
        "wrapper_contract_version",
    }:
        raise TensorRtManifestError("TensorRT model identity is incomplete")
    if (
        model.get("id") != expected_model_id
        or model.get("checkpoint_sha256") != expected_checkpoint_sha256
        or _COMMIT.fullmatch(str(model.get("repository_revision", ""))) is None
        or model.get("wrapper_contract_version") != 1
    ):
        raise TensorRtManifestError("TensorRT model identity differs")

    engine = _mapping(payload.get("engine"), "engine")
    if set(engine) != {"filename", "sha256", "precision", "tf32"}:
        raise TensorRtManifestError("TensorRT engine identity is incomplete")
    filename = _safe_basename(engine.get("filename"), "engine.filename")
    plan_sha256 = str(engine.get("sha256", ""))
    if (
        _SHA256.fullmatch(plan_sha256) is None
        or engine.get("precision") != "float32"
        or engine.get("tf32") is not False
    ):
        raise TensorRtManifestError("TensorRT engine precision contract differs")
    plan_path = _resolve_child(root, filename)
    if _sha256_file(plan_path) != plan_sha256:
        raise TensorRtManifestError("TensorRT plan identity differs")

    builder = _mapping(payload.get("builder"), "builder")
    if set(builder) != {
        "torch",
        "torch_tensorrt",
        "tensorrt",
        "cuda",
        "gpu_name",
        "compute_capability",
    }:
        raise TensorRtManifestError("TensorRT builder identity is incomplete")
    observed_compatibility = {
        "tensorrt": compatibility.tensorrt,
        "cuda": compatibility.cuda,
        "gpu_name": compatibility.gpu_name,
        "compute_capability": compatibility.compute_capability,
    }
    if any(builder.get(name) != value for name, value in observed_compatibility.items()):
        raise TensorRtManifestError("TensorRT runtime is incompatible with the plan")
    if builder.get("torch") != "2.8.0+cu128" or builder.get(
        "torch_tensorrt"
    ) != "2.8.0":
        raise TensorRtManifestError("TensorRT builder dependency lane differs")

    inputs = _tensor_contract(payload.get("inputs"), "inputs")
    outputs = _tensor_contract(payload.get("outputs"), "outputs")
    if expected_model_id == CLASSIFIER_MODEL_ID and (
        inputs != _CLASSIFIER_INPUTS or outputs != _CLASSIFIER_OUTPUTS
    ):
        raise TensorRtManifestError("MMBCD TensorRT tensor contract differs")

    coverage = _mapping(payload.get("coverage"), "coverage")
    if set(coverage) != {
        "strict_export",
        "require_full_compilation",
        "pytorch_partition_count",
        "unsupported_operators",
        "dry_run_report_sha256",
    }:
        raise TensorRtManifestError("TensorRT coverage evidence is incomplete")
    if (
        coverage.get("strict_export") is not True
        or coverage.get("require_full_compilation") is not True
        or coverage.get("pytorch_partition_count") != 0
        or coverage.get("unsupported_operators") != []
        or _SHA256.fullmatch(
            str(coverage.get("dry_run_report_sha256", ""))
        )
        is None
    ):
        raise TensorRtManifestError("TensorRT engine contains fallback coverage")

    parity = _mapping(payload.get("parity"), "parity")
    performance = _mapping(payload.get("performance"), "performance")
    if (
        set(parity) != {"passed", "report_sha256"}
        or parity.get("passed") is not True
        or _SHA256.fullmatch(str(parity.get("report_sha256", ""))) is None
        or set(performance) != {"promotion_threshold_passed", "report_sha256"}
        or performance.get("promotion_threshold_passed") is not True
        or _SHA256.fullmatch(str(performance.get("report_sha256", ""))) is None
        or payload.get("decision") != "go"
    ):
        raise TensorRtManifestError("TensorRT promotion evidence did not pass")

    plugin_path, plugin_sha256 = _plugin_contract(payload.get("plugin"), root)
    return TensorRtEngineManifest(
        model_id=expected_model_id,
        checkpoint_sha256=expected_checkpoint_sha256,
        repository_revision=str(model["repository_revision"]),
        plan_path=plan_path,
        plan_sha256=plan_sha256,
        precision="float32",
        input_names=tuple(name for name, _, _ in inputs),
        output_names=tuple(name for name, _, _ in outputs),
        decision="go",
        plugin_library=plugin_path,
        plugin_sha256=plugin_sha256,
    )


def _plugin_contract(
    value: object, root: Path
) -> tuple[Path | None, str | None]:
    if value is None:
        return None, None
    plugin = _mapping(value, "plugin")
    if set(plugin) != {"filename", "sha256", "name", "version", "namespace"}:
        raise TensorRtManifestError("TensorRT plugin identity is incomplete")
    filename = _safe_basename(plugin.get("filename"), "plugin.filename")
    digest = str(plugin.get("sha256", ""))
    if _SHA256.fullmatch(digest) is None or any(
        not isinstance(plugin.get(name), str) or not plugin.get(name)
        for name in ("name", "version", "namespace")
    ):
        raise TensorRtManifestError("TensorRT plugin identity is invalid")
    plugin_path = _resolve_child(root, filename)
    if _sha256_file(plugin_path) != digest:
        raise TensorRtManifestError("TensorRT plugin identity differs")
    return plugin_path, digest


def _tensor_contract(
    value: object, path: str
) -> tuple[tuple[str, str, tuple[int, ...]], ...]:
    if not isinstance(value, list) or not value:
        raise TensorRtManifestError(f"TensorRT {path} contract is invalid")
    result: list[tuple[str, str, tuple[int, ...]]] = []
    for item in value:
        tensor = _mapping(item, path)
        if set(tensor) != {"name", "dtype", "shape"}:
            raise TensorRtManifestError(f"TensorRT {path} contract is incomplete")
        name = tensor.get("name")
        dtype = tensor.get("dtype")
        shape = tensor.get("shape")
        if (
            not isinstance(name, str)
            or not name
            or dtype not in {"float32", "int64"}
            or not isinstance(shape, list)
            or not shape
            or any(
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension <= 0
                for dimension in shape
            )
        ):
            raise TensorRtManifestError(f"TensorRT {path} tensor is invalid")
        result.append((name, dtype, tuple(shape)))
    if len({name for name, _, _ in result}) != len(result):
        raise TensorRtManifestError(f"TensorRT {path} names must be unique")
    return tuple(result)


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TensorRtManifestError(f"TensorRT {path} must be an object")
    return value


def _safe_basename(value: object, path: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or PurePosixPath(value).name != value
        or "\\" in value
    ):
        raise TensorRtManifestError(f"TensorRT {path} is unsafe")
    return value


def _resolve_child(root: Path, filename: str) -> Path:
    candidate = (root / filename).resolve()
    if candidate.parent != root:
        raise TensorRtManifestError("TensorRT artifact escaped the engine root")
    return candidate


def _sha256_file(path: Path) -> str:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        raise TensorRtManifestError("TensorRT artifact is unavailable") from None
