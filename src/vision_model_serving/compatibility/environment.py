"""Evaluate a host against the pinned Lightning L4 FP32 compatibility lane."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
from typing import Any, Mapping


_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_NVCC_RELEASE = re.compile(r"release\s+(\d+\.\d+)")


class EnvironmentSpecError(ValueError):
    """Raised when the checked-in compatibility specification is malformed."""


@dataclass(frozen=True, slots=True)
class L4EnvironmentSpec:
    lane_id: str
    python: str
    packages: dict[str, str]
    cuda_runtime: str
    cuda_toolkit_release: str
    torch_arch_list: str
    device_name: str
    compute_capability: str
    reference_driver: str
    upstream_commits: dict[str, str]
    patches: tuple[dict[str, str], ...]
    native_operator: dict[str, str]
    artifact_manifest: str
    requirements: str
    success_markers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EnvironmentSnapshot:
    python: str
    platform: str
    packages: dict[str, str | None]
    cuda_available: bool
    torch_cuda: str | None
    device_name: str | None
    compute_capability: str | None
    cuda_home_present: bool
    cuda_sanity_passed: bool | None
    nvcc_release: str | None
    compiler: str | None
    driver: str | None
    collection_errors: tuple[str, ...] = ()

    @classmethod
    def collect(cls, spec: L4EnvironmentSpec) -> EnvironmentSnapshot:
        """Collect non-sensitive compatibility facts from the current host."""

        packages: dict[str, str | None] = {}
        errors: list[str] = []
        for distribution in spec.packages:
            try:
                packages[distribution] = metadata.version(distribution)
            except metadata.PackageNotFoundError:
                packages[distribution] = None

        cuda_available = False
        torch_cuda: str | None = None
        device_name: str | None = None
        compute_capability: str | None = None
        cuda_home_present = False
        cuda_sanity_passed: bool | None = None
        if packages.get("torch") is not None:
            try:
                import torch
                from torch.utils.cpp_extension import CUDA_HOME

                torch_cuda = torch.version.cuda
                cuda_home_present = bool(CUDA_HOME and Path(CUDA_HOME).is_dir())
                cuda_available = torch.cuda.is_available()
                if cuda_available:
                    device_name = torch.cuda.get_device_name(0)
                    compute_capability = ".".join(
                        str(part) for part in torch.cuda.get_device_capability(0)
                    )
                    value = torch.randn(128, 128, device="cuda")
                    result = value @ value.T
                    cuda_sanity_passed = bool(torch.isfinite(result).all().item())
                    if not cuda_sanity_passed:
                        errors.append("CUDA matrix sanity check produced non-finite values")
                    torch.cuda.synchronize()
            except Exception as error:  # pragma: no cover - host-specific import
                errors.append(f"torch probe failed: {type(error).__name__}: {error}")

        nvcc_output = _command_output(["nvcc", "--version"])
        nvcc_release = None
        if nvcc_output:
            match = _NVCC_RELEASE.search(nvcc_output)
            nvcc_release = match.group(1) if match else None

        compiler_output = _command_output(["c++", "--version"])
        compiler = compiler_output.splitlines()[0] if compiler_output else None
        driver = _command_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ]
        )
        return cls(
            python=platform.python_version(),
            platform=f"{platform.system()} {platform.release()}",
            packages=packages,
            cuda_available=cuda_available,
            torch_cuda=torch_cuda,
            device_name=device_name,
            compute_capability=compute_capability,
            cuda_home_present=cuda_home_present,
            cuda_sanity_passed=cuda_sanity_passed,
            nvcc_release=nvcc_release,
            compiler=compiler,
            driver=driver.splitlines()[0].strip() if driver else None,
            collection_errors=tuple(errors),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "python": self.python,
            "platform": self.platform,
            "packages": dict(sorted(self.packages.items())),
            "cuda_available": self.cuda_available,
            "torch_cuda": self.torch_cuda,
            "device_name": self.device_name,
            "compute_capability": self.compute_capability,
            "cuda_home_present": self.cuda_home_present,
            "cuda_sanity_passed": self.cuda_sanity_passed,
            "nvcc_release": self.nvcc_release,
            "compiler": self.compiler,
            "driver": self.driver,
            "collection_errors": list(self.collection_errors),
        }


@dataclass(frozen=True, slots=True)
class EnvironmentGateResult:
    status: str
    marker: str
    issues: tuple[str, ...]
    gpu_gate: str
    snapshot: EnvironmentSnapshot

    @property
    def succeeded(self) -> bool:
        return self.status in {"passed", "gpu_skipped"}

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "marker": self.marker,
            "gpu_gate": self.gpu_gate,
            "issues": list(self.issues),
            "snapshot": self.snapshot.as_dict(),
        }


def load_environment_spec(path: str | Path) -> L4EnvironmentSpec:
    spec_path = Path(path)
    try:
        payload = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EnvironmentSpecError(
            f"Cannot load environment spec {spec_path}: {error}"
        ) from error
    if not isinstance(payload, Mapping):
        raise EnvironmentSpecError("Environment spec root must be an object")
    if payload.get("schema_version") != 1:
        raise EnvironmentSpecError("schema_version must be 1")

    packages = _string_mapping(payload.get("packages"), "packages")
    cuda = _mapping(payload.get("cuda"), "cuda")
    device = _mapping(payload.get("device"), "device")
    commits = _string_mapping(payload.get("upstream_commits"), "upstream_commits")
    if set(commits) != {"focalnet_dino", "mmbcd", "dino"}:
        raise EnvironmentSpecError(
            "upstream_commits must contain focalnet_dino, mmbcd, and dino"
        )
    for name, commit in commits.items():
        if _COMMIT.fullmatch(commit) is None:
            raise EnvironmentSpecError(f"upstream_commits.{name} is not a full commit")

    patch_values = payload.get("patches")
    if not isinstance(patch_values, list) or not patch_values:
        raise EnvironmentSpecError("patches must be a non-empty array")
    patches: list[dict[str, str]] = []
    for index, value in enumerate(patch_values):
        patch = _string_mapping(value, f"patches[{index}]")
        if set(patch) != {"path", "sha256"}:
            raise EnvironmentSpecError(f"patches[{index}] must contain path and sha256")
        if not _safe_relative_path(patch["path"]):
            raise EnvironmentSpecError(
                f"patches[{index}].path must be repository-relative"
            )
        if _SHA256.fullmatch(patch["sha256"]) is None:
            raise EnvironmentSpecError(f"patches[{index}].sha256 is invalid")
        patches.append(patch)

    native = _string_mapping(payload.get("native_operator"), "native_operator")
    if _SHA256.fullmatch(native.get("reference_sha256", "")) is None:
        raise EnvironmentSpecError("native_operator.reference_sha256 is invalid")
    markers = payload.get("success_markers")
    if not isinstance(markers, list) or not markers or not all(
        isinstance(marker, str) and marker for marker in markers
    ):
        raise EnvironmentSpecError("success_markers must be non-empty strings")

    return L4EnvironmentSpec(
        lane_id=_string(payload.get("lane_id"), "lane_id"),
        python=_string(payload.get("python"), "python"),
        packages=packages,
        cuda_runtime=_string(cuda.get("runtime"), "cuda.runtime"),
        cuda_toolkit_release=_string(
            cuda.get("toolkit_release"), "cuda.toolkit_release"
        ),
        torch_arch_list=_string(
            cuda.get("torch_arch_list"), "cuda.torch_arch_list"
        ),
        device_name=_string(device.get("name"), "device.name"),
        compute_capability=_string(
            device.get("compute_capability"), "device.compute_capability"
        ),
        reference_driver=_string(
            device.get("reference_driver"), "device.reference_driver"
        ),
        upstream_commits=commits,
        patches=tuple(patches),
        native_operator=native,
        artifact_manifest=_relative_path(
            payload.get("artifact_manifest"), "artifact_manifest"
        ),
        requirements=_relative_path(payload.get("requirements"), "requirements"),
        success_markers=tuple(markers),
    )


def evaluate_environment(
    spec: L4EnvironmentSpec,
    snapshot: EnvironmentSnapshot,
    *,
    allow_cpu: bool = False,
) -> EnvironmentGateResult:
    issues = list(snapshot.collection_errors)
    if snapshot.python != spec.python:
        issues.append(f"python: expected {spec.python}, observed {snapshot.python}")
    for distribution, expected in spec.packages.items():
        observed = snapshot.packages.get(distribution)
        if observed != expected:
            issues.append(
                f"package {distribution}: expected {expected}, observed {observed or 'missing'}"
            )

    if not snapshot.cuda_available:
        gpu_gate = "skipped" if allow_cpu else "failed"
        if not allow_cpu:
            issues.append(
                "CUDA is unavailable; rerun with --allow-cpu only for metadata checks"
            )
    else:
        gpu_gate = "passed"
        expected_gpu_facts = {
            "torch CUDA runtime": (snapshot.torch_cuda, spec.cuda_runtime),
            "GPU": (snapshot.device_name, spec.device_name),
            "compute capability": (
                snapshot.compute_capability,
                spec.compute_capability,
            ),
            "nvcc release": (snapshot.nvcc_release, spec.cuda_toolkit_release),
        }
        for label, (observed, expected) in expected_gpu_facts.items():
            if observed != expected:
                issues.append(
                    f"{label}: expected {expected}, observed {observed or 'missing'}"
                )
        if not snapshot.cuda_home_present:
            issues.append("CUDA_HOME does not resolve to a directory")
        if snapshot.cuda_sanity_passed is not True:
            issues.append("CUDA matrix sanity check did not pass")
        if snapshot.compiler is None:
            issues.append("C++ compiler is unavailable")

    if issues:
        return EnvironmentGateResult(
            status="failed",
            marker="L4 ENVIRONMENT FAILED",
            issues=tuple(issues),
            gpu_gate=gpu_gate,
            snapshot=snapshot,
        )
    if gpu_gate == "skipped":
        return EnvironmentGateResult(
            status="gpu_skipped",
            marker="L4 GPU GATES SKIPPED",
            issues=(),
            gpu_gate=gpu_gate,
            snapshot=snapshot,
        )
    return EnvironmentGateResult(
        status="passed",
        marker="LIGHTNING L4 ENVIRONMENT PASSED",
        issues=(),
        gpu_gate=gpu_gate,
        snapshot=snapshot,
    )


def _command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
            env=os.environ.copy(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = result.stdout or result.stderr
    return output.strip() if result.returncode == 0 and output.strip() else None


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EnvironmentSpecError(f"{path} must be an object")
    return value


def _string_mapping(value: Any, path: str) -> dict[str, str]:
    mapping = _mapping(value, path)
    if not all(
        isinstance(key, str) and isinstance(item, str) and item
        for key, item in mapping.items()
    ):
        raise EnvironmentSpecError(f"{path} must contain non-empty string values")
    return dict(mapping)


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise EnvironmentSpecError(f"{path} must be a non-empty string")
    return value


def _safe_relative_path(value: str) -> bool:
    candidate = Path(value)
    return (
        not candidate.is_absolute()
        and ".." not in candidate.parts
        and "\\" not in value
    )


def _relative_path(value: Any, path: str) -> str:
    result = _string(value, path)
    if not _safe_relative_path(result):
        raise EnvironmentSpecError(f"{path} must be repository-relative")
    return result
