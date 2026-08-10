"""Bind benchmark evidence to the repository's exact L4 compatibility lane."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any

from vision_model_serving.compatibility import load_environment_spec
from vision_model_serving.validation._benchmark_primitives import BenchmarkContractError


_IMAGE_ID = re.compile(r"(?:sha256:)?[0-9a-f]{64}")
BENCHMARK_LANE_ID = "lightning-l4-fp32-cu128"
BENCHMARK_LANE_CONFIG_SHA256 = "438d3e344c0862d4106795aaa8b1b7c1215d87381bf3b221b330aec05690e2e9"
BENCHMARK_LANE_PACKAGES = MappingProxyType(
    {
        "huggingface-hub": "1.26.0",
        "ninja": "1.13.0",
        "numpy": "1.26.4",
        "opencv-python-headless": "4.11.0.86",
        "pillow": "12.3.0",
        "pydicom": "3.0.2",
        "scipy": "1.11.4",
        "timm": "1.0.28",
        "tokenizers": "0.22.2",
        "torch": "2.8.0+cu128",
        "torchvision": "0.23.0+cu128",
        "transformers": "5.14.1",
    }
)
BENCHMARK_LANE_HARDWARE = MappingProxyType(
    {
        "gpu_name": "NVIDIA L4",
        "compute_capability": "8.9",
        "driver": "580.173.02",
    }
)
BENCHMARK_LANE_SOFTWARE = MappingProxyType(
    {
        "python": "3.12.11",
        "torch": "2.8.0+cu128",
        "torchvision": "0.23.0+cu128",
        "cuda_runtime": "12.8",
        "cuda_toolkit": "12.8",
    }
)
BENCHMARK_LANE_NATIVE_OPERATOR = MappingProxyType(
    {
        "module": "MultiScaleDeformableAttention",
        "reference_sha256": ("f8ec513bbab7d3ae134b2b4c7ae934cc45995e96d58dd4d53810268784b86147"),
        "verification": "functional_forward_parity",
        "torch_arch_list": "8.9",
    }
)
BENCHMARK_EXECUTOR_DOCKERFILE_SHA256 = (
    "19b6edcb9b20ae288d4813a83ac4d16088fed8d3a779ac6fecb3cd280bedf787"
)
BENCHMARK_EXECUTOR_BASE_IMAGES = (
    "ghcr.io/astral-sh/uv:0.8.4@sha256:"
    "40775a79214294fb51d097c9117592f193bcfdfc634f4daa0e169ee965b10ef0",
    "nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04@sha256:"
    "24c8e3581ea6330038b0d374920721983312627f8adbfcf390bdb4b399d280ed",
    "nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04@sha256:"
    "ac55d124da4882b497f732d8dfd9a702d5447a5f29d08d56da6f64f0a1eb34bc",
)


def benchmark_environment_identity(
    project_root: Path,
    evidence_path: Path,
    *,
    executor_image_id: str,
) -> dict[str, object]:
    """Validate current-lane evidence and return its self-contained identity."""

    project_root = project_root.resolve()
    if _IMAGE_ID.fullmatch(executor_image_id) is None:
        raise BenchmarkContractError("executor image id must be a SHA-256 digest")
    normalized_image_id = (
        executor_image_id
        if executor_image_id.startswith("sha256:")
        else f"sha256:{executor_image_id}"
    )
    lane_path = project_root / "config" / "l4-fp32-environment.json"
    try:
        lane = load_environment_spec(lane_path)
    except ValueError as error:
        raise BenchmarkContractError("current L4 lane configuration is invalid") from error
    if (
        hashlib.sha256(lane_path.read_bytes()).hexdigest() != BENCHMARK_LANE_CONFIG_SHA256
        or lane.lane_id != BENCHMARK_LANE_ID
        or lane.packages != dict(BENCHMARK_LANE_PACKAGES)
    ):
        raise BenchmarkContractError("current L4 lane differs from benchmark constants")

    evidence = _json_object(evidence_path)
    snapshot = evidence.get("snapshot")
    if (
        evidence.get("status") != "passed"
        or evidence.get("gpu_gate") != "passed"
        or evidence.get("marker") != "LIGHTNING L4 ENVIRONMENT PASSED"
        or evidence.get("issues") != []
        or evidence.get("lane_id") != lane.lane_id
        or not isinstance(snapshot, dict)
        or snapshot.get("collection_errors") != []
    ):
        raise BenchmarkContractError("L4 environment evidence did not pass")

    packages = snapshot.get("packages")
    exact_lane_facts = {
        "python": BENCHMARK_LANE_SOFTWARE["python"],
        "torch_cuda": BENCHMARK_LANE_SOFTWARE["cuda_runtime"],
        "device_name": BENCHMARK_LANE_HARDWARE["gpu_name"],
        "compute_capability": BENCHMARK_LANE_HARDWARE["compute_capability"],
        "nvcc_release": BENCHMARK_LANE_SOFTWARE["cuda_toolkit"],
        "driver": BENCHMARK_LANE_HARDWARE["driver"],
    }
    if packages != dict(BENCHMARK_LANE_PACKAGES) or any(
        snapshot.get(name) != expected for name, expected in exact_lane_facts.items()
    ):
        raise BenchmarkContractError("environment evidence differs from the current L4 lane")
    if (
        snapshot.get("cuda_available") is not True
        or snapshot.get("cuda_home_present") is not True
        or snapshot.get("cuda_sanity_passed") is not True
        or not _nonempty_string(snapshot.get("platform"))
        or not _nonempty_string(snapshot.get("compiler"))
        or not _nonempty_string(snapshot.get("cudnn_version"))
        or not _positive_integer(snapshot.get("total_device_memory_bytes"))
    ):
        raise BenchmarkContractError("current L4 runtime/compiler identity is incomplete")

    dockerfile = project_root / "docker" / "executor.Dockerfile"
    dockerfile_sha256 = hashlib.sha256(dockerfile.read_bytes()).hexdigest()
    base_images = _docker_base_images(dockerfile)
    if (
        dockerfile_sha256 != BENCHMARK_EXECUTOR_DOCKERFILE_SHA256
        or tuple(base_images) != BENCHMARK_EXECUTOR_BASE_IMAGES
    ):
        raise BenchmarkContractError("executor Dockerfile differs from benchmark constants")
    return {
        "lane": {
            "id": BENCHMARK_LANE_ID,
            "config_sha256": BENCHMARK_LANE_CONFIG_SHA256,
            "packages": dict(BENCHMARK_LANE_PACKAGES),
        },
        "hardware": {
            "gpu_name": snapshot["device_name"],
            "total_memory_bytes": snapshot["total_device_memory_bytes"],
            "compute_capability": snapshot["compute_capability"],
            "driver": snapshot["driver"],
        },
        "software": {
            "python": snapshot["python"],
            "torch": BENCHMARK_LANE_SOFTWARE["torch"],
            "torchvision": BENCHMARK_LANE_SOFTWARE["torchvision"],
            "cuda_runtime": snapshot["torch_cuda"],
            "cuda_toolkit": snapshot["nvcc_release"],
            "cudnn": snapshot["cudnn_version"],
            "compiler": snapshot["compiler"],
        },
        "native_operator": {
            **BENCHMARK_LANE_NATIVE_OPERATOR,
        },
        "container": {
            "executor_image_id": normalized_image_id,
            "executor_dockerfile_sha256": BENCHMARK_EXECUTOR_DOCKERFILE_SHA256,
            "pinned_base_images": list(BENCHMARK_EXECUTOR_BASE_IMAGES),
        },
    }


def _docker_base_images(path: Path) -> list[str]:
    images: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("ARG ") and "IMAGE=" in line and "@sha256:" in line:
            images.append(line.split("=", 1)[1])
    if len(images) != 3:
        raise BenchmarkContractError("executor base-image identity is incomplete")
    return images


def _json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise BenchmarkContractError(f"evidence file is unavailable: {path.name}") from None
    if not isinstance(payload, dict):
        raise BenchmarkContractError(f"evidence file is not an object: {path.name}")
    return payload


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value)


def _positive_integer(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0
