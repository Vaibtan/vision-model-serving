#!/usr/bin/env python3
"""Collect small validation evidence without copying model weights or raw DICOM."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

from _common import default_paths, sha256_file


def command_output(command: list[str]) -> str:
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)


def copy_required(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    defaults = default_paths()
    validation_root = defaults["studio_root"] / "validation-evidence"
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, default=defaults["fixture_dir"])
    parser.add_argument("--artifact-dir", type=Path, default=defaults["artifact_dir"])
    parser.add_argument("--project-repo", type=Path, default=defaults["project_repo"])
    parser.add_argument("--focalnet-repo", type=Path, default=defaults["focalnet_repo"])
    parser.add_argument("--mmbcd-repo", type=Path, default=defaults["mmbcd_repo"])
    parser.add_argument("--dino-repo", type=Path, default=defaults["dino_repo"])
    parser.add_argument(
        "--evidence-root", type=Path, default=validation_root / "l4-fp32-reproduction"
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=validation_root / "vision-model-serving-l4-fp32-reproduction.tar.gz",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    case_dir = args.case_dir.expanduser().resolve()
    artifact_dir = args.artifact_dir.expanduser().resolve()
    project_repo = args.project_repo.expanduser().resolve()
    evidence_root = args.evidence_root.expanduser().resolve()
    archive = args.archive.expanduser().resolve()
    sidecar = archive.with_suffix(archive.suffix + ".sha256")
    validation_root = validation_root.resolve()

    if (
        evidence_root == validation_root
        or not evidence_root.is_relative_to(validation_root)
        or not archive.is_relative_to(validation_root)
        or not sidecar.is_relative_to(validation_root)
    ):
        raise RuntimeError(
            "Evidence root, archive, and sidecar must remain below "
            f"{validation_root}; refusing unsafe output paths"
        )

    for target in (evidence_root, archive, sidecar):
        if target.exists() and not args.overwrite:
            raise FileExistsError(
                f"Refusing to replace {target}; pass --overwrite intentionally"
            )
    if args.overwrite and evidence_root.exists():
        shutil.rmtree(evidence_root)
    if args.overwrite:
        archive.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)

    copy_map = {
        case_dir / "manifest.json": evidence_root / "manifests/dicom-manifest.json",
        case_dir / "preprocessed/preprocess-manifest.json": (
            evidence_root / "manifests/preprocess-manifest.json"
        ),
        case_dir / "detector/inference-manifest.json": (
            evidence_root / "manifests/detector-inference-manifest.json"
        ),
        case_dir / "mmbcd/input/mmbcd-input-manifest.json": (
            evidence_root / "manifests/mmbcd-input-manifest.json"
        ),
        case_dir / "mmbcd/output/inference-manifest.json": (
            evidence_root / "manifests/mmbcd-inference-manifest.json"
        ),
        case_dir / "detector/detections-top8.txt": (
            evidence_root / "bundles/detections-top8.txt"
        ),
        case_dir / "mmbcd/input/mmbcd-inputs.npz": (
            evidence_root / "bundles/mmbcd-inputs.npz"
        ),
        case_dir / "mmbcd/output/mmbcd-outputs.npz": (
            evidence_root / "bundles/mmbcd-outputs.npz"
        ),
        case_dir / "preprocessed/upstream-1024.png": (
            evidence_root / "visuals/upstream-1024.png"
        ),
        case_dir / "detector/overlay-top8-1024.png": (
            evidence_root / "visuals/overlay-top8-1024.png"
        ),
        case_dir / "detector/overlay-top8-original.png": (
            evidence_root / "visuals/overlay-top8-original.png"
        ),
        case_dir / "mmbcd/input/roi-montage.png": (
            evidence_root / "visuals/roi-montage.png"
        ),
        project_repo / "patches/focalnet-pytorch-2.8-compat.patch": (
            evidence_root / "patches/focalnet-pytorch-2.8-compat.patch"
        ),
    }
    for source, destination in copy_map.items():
        copy_required(source, destination)

    environment_dir = evidence_root / "environment"
    environment_dir.mkdir(parents=True, exist_ok=True)
    package_json = command_output(
        ["python", "-m", "pip", "list", "--format=json"]
    )
    json.loads(package_json)
    (environment_dir / "python-packages.json").write_text(
        package_json, encoding="utf-8"
    )
    conda_json = command_output(["conda", "list", "--json"])
    json.loads(conda_json)
    (environment_dir / "conda-packages.json").write_text(
        conda_json, encoding="utf-8"
    )
    (environment_dir / "python.txt").write_text(
        command_output(["python", "--version"]), encoding="utf-8"
    )
    (environment_dir / "nvcc.txt").write_text(
        command_output(["nvcc", "--version"]), encoding="utf-8"
    )
    (environment_dir / "gpu.txt").write_text(
        command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,compute_cap,pstate,power.limit",
                "--format=csv,noheader",
            ]
        ),
        encoding="utf-8",
    )
    repositories = {
        "focalnet": args.focalnet_repo,
        "mmbcd": args.mmbcd_repo,
        "dino": args.dino_repo,
    }
    for name, repository in repositories.items():
        commit = command_output(
            ["git", "-C", str(repository.expanduser().resolve()), "rev-parse", "HEAD"]
        )
        (environment_dir / f"{name}-commit.txt").write_text(
            commit, encoding="utf-8"
        )

    extension_files = sorted(
        args.focalnet_repo.expanduser().resolve().rglob(
            "MultiScaleDeformableAttention*.so"
        )
    )
    if not extension_files:
        raise RuntimeError("No compiled MultiScaleDeformableAttention extension found")
    (environment_dir / "cuda-extension.sha256").write_text(
        "".join(f"{sha256_file(path)}  {path}\n" for path in extension_files),
        encoding="utf-8",
    )
    model_files = sorted(
        path
        for path in artifact_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".pt", ".pth"}
    )
    (environment_dir / "model-artifacts.sha256").write_text(
        "".join(f"{sha256_file(path)}  {path}\n" for path in model_files),
        encoding="utf-8",
    )

    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, mode="w:gz") as tar:
        for path in sorted(evidence_root.rglob("*")):
            tar.add(path, arcname=path.relative_to(evidence_root), recursive=False)
    archive_hash = sha256_file(archive)
    sidecar.write_text(f"{archive_hash}  {archive}\n", encoding="utf-8")
    print("Evidence root:", evidence_root)
    print("Archive:", archive)
    print("Archive SHA256:", archive_hash)
    print("Sidecar:", sidecar)
    print("LIGHTNING L4 EVIDENCE COLLECTED")


if __name__ == "__main__":
    main()
