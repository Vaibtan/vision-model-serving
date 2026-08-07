"""Idempotent preparation of the pinned FocalNet-DINO native-operator source."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import subprocess
import sys

from .environment import L4EnvironmentSpec


class PatchCheckError(RuntimeError):
    """Raised when source identity or patch applicability is not trustworthy."""


@dataclass(frozen=True, slots=True)
class PatchResult:
    path: str
    sha256: str
    state: str


def prepare_focalnet_patches(
    repository: str | Path,
    project_root: str | Path,
    spec: L4EnvironmentSpec,
    *,
    apply: bool = False,
) -> tuple[PatchResult, ...]:
    """Verify or idempotently apply both patches to the pinned checkout."""

    repository_path = Path(repository).expanduser().resolve()
    project_path = Path(project_root).expanduser().resolve()
    if not (repository_path / ".git").exists():
        raise PatchCheckError(f"Not a Git checkout: {repository_path}")
    head = _git(repository_path, "rev-parse", "HEAD").stdout.strip()
    expected_head = spec.upstream_commits["focalnet_dino"]
    if head != expected_head:
        raise PatchCheckError(
            f"FocalNet-DINO commit mismatch: expected {expected_head}, observed {head}"
        )

    results: list[PatchResult] = []
    for record in spec.patches:
        patch = (project_path / record["path"]).resolve()
        if not patch.is_relative_to(project_path) or not patch.is_file():
            raise PatchCheckError(f"Patch is missing or outside the project: {patch}")
        observed_hash = _sha256(patch)
        if observed_hash != record["sha256"]:
            raise PatchCheckError(
                f"Patch hash mismatch for {record['path']}: "
                f"expected {record['sha256']}, observed {observed_hash}"
            )

        applicable = _git(
            repository_path, "apply", "--check", str(patch), check=False
        )
        if applicable.returncode == 0:
            if apply:
                _git(repository_path, "apply", str(patch))
                state = "applied"
            else:
                state = "applicable"
        else:
            reversed_check = _git(
                repository_path,
                "apply",
                "--reverse",
                "--check",
                str(patch),
                check=False,
            )
            if reversed_check.returncode != 0:
                diagnostic = (applicable.stderr or applicable.stdout).strip()
                raise PatchCheckError(
                    f"Patch is neither applicable nor already applied: {record['path']}. "
                    f"git apply diagnostic: {diagnostic or 'none'}"
                )
            state = "already_applied"
        results.append(PatchResult(record["path"], observed_hash, state))

    whitespace = _git(repository_path, "diff", "--check", check=False)
    if whitespace.returncode != 0:
        raise PatchCheckError(
            "Patched checkout fails git diff --check: "
            + (whitespace.stdout or whitespace.stderr).strip()
        )
    return tuple(results)


def build_focalnet_extension(
    repository: str | Path,
    spec: L4EnvironmentSpec,
    *,
    max_jobs: int = 4,
) -> Path:
    """Build the native operator for the spec's CUDA architecture."""

    if max_jobs <= 0:
        raise PatchCheckError("max_jobs must be positive")

    repository_path = Path(repository).expanduser().resolve()
    ops_dir = repository_path / "models" / "dino" / "ops"
    setup = ops_dir / "setup.py"
    if not setup.is_file():
        raise PatchCheckError(f"FocalNet-DINO operator setup is missing: {setup}")

    environment = os.environ.copy()
    cuda_home_value = environment.get("CUDA_HOME") or environment.get("CONDA_PREFIX")
    if not cuda_home_value or not Path(cuda_home_value).is_dir():
        raise PatchCheckError(
            "CUDA_HOME or CONDA_PREFIX must identify the CUDA 12.8 toolkit root"
        )
    cuda_home = Path(cuda_home_value).resolve()
    nvcc = cuda_home / "bin" / "nvcc"
    if not nvcc.is_file():
        raise PatchCheckError(f"nvcc is missing: {nvcc}")
    nvcc_report = subprocess.run(
        [str(nvcc), "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if f"release {spec.cuda_toolkit_release}" not in nvcc_report:
        raise PatchCheckError(
            "nvcc release mismatch: expected "
            f"{spec.cuda_toolkit_release}; output was {nvcc_report.strip()}"
        )

    environment.update(
        {
            "CUDA_HOME": str(cuda_home),
            "CUDACXX": str(nvcc),
            "TORCH_CUDA_ARCH_LIST": spec.torch_arch_list,
            "MAX_JOBS": str(max_jobs),
            "PATH": str(cuda_home / "bin") + os.pathsep + environment.get("PATH", ""),
        }
    )
    subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=ops_dir,
        env=environment,
        check=True,
    )
    candidates = sorted(ops_dir.glob("MultiScaleDeformableAttention*.so"))
    if not candidates:
        raise PatchCheckError("Build completed without an in-place extension module")
    return candidates[0]


def _git(
    repository: Path, *arguments: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=check,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        diagnostic = (error.stderr or error.stdout).strip()
        raise PatchCheckError(
            f"git {' '.join(arguments)} failed: {diagnostic or error.returncode}"
        ) from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
