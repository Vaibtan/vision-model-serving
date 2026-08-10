from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import tomllib
import unittest
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from vision_model_serving.compatibility import (  # noqa: E402
    EnvironmentSnapshot,
    evaluate_environment,
    load_environment_spec,
    prepare_focalnet_patches,
)
from vision_model_serving.compatibility.focalnet import (  # noqa: E402
    build_focalnet_extension,
)


SPEC_PATH = REPOSITORY_ROOT / "config" / "l4-fp32-environment.json"


def _snapshot(spec, *, cuda_available: bool) -> EnvironmentSnapshot:
    return EnvironmentSnapshot(
        python=spec.python,
        platform="Linux test",
        packages=dict(spec.packages),
        cuda_available=cuda_available,
        torch_cuda=spec.cuda_runtime if cuda_available else None,
        device_name=spec.device_name if cuda_available else None,
        compute_capability=spec.compute_capability if cuda_available else None,
        cuda_home_present=cuda_available,
        cuda_sanity_passed=True if cuda_available else None,
        nvcc_release=spec.cuda_toolkit_release if cuda_available else None,
        compiler="g++ test" if cuda_available else None,
        driver=spec.reference_driver if cuda_available else None,
    )


class L4EnvironmentTests(unittest.TestCase):
    def test_spec_matches_artifact_manifest_and_requirements(self) -> None:
        spec = load_environment_spec(SPEC_PATH)
        manifest = json.loads(
            (REPOSITORY_ROOT / spec.artifact_manifest).read_text(encoding="utf-8")
        )

        self.assertEqual(spec.lane_id, "lightning-l4-fp32-cu128")
        self.assertEqual(
            spec.upstream_commits,
            {
                "focalnet_dino": manifest["revisions"]["focalnet_dino"],
                "mmbcd": manifest["revisions"]["mmbcd"],
                "dino": manifest["revisions"]["dino"],
            },
        )

        requirement_pins: dict[str, str] = {}
        for line in (REPOSITORY_ROOT / spec.requirements).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith(("#", "--")):
                continue
            name, version = line.split("==", 1)
            normalized = re.sub(r"[-_.]+", "-", name).lower()
            requirement_pins[normalized] = version
        self.assertEqual(requirement_pins, spec.packages)

        project = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        project_requirements = [
            *project["project"]["dependencies"],
            *project["project"]["optional-dependencies"]["gpu"],
        ]
        project_pins = {}
        for requirement in project_requirements:
            name, version = requirement.split("==", 1)
            project_pins[re.sub(r"[-_.]+", "-", name).lower()] = version
        self.assertLessEqual(set(spec.packages), set(project_pins))
        self.assertEqual(
            {name: project_pins[name] for name in spec.packages},
            spec.packages,
        )

        for patch_record in spec.patches:
            content = (REPOSITORY_ROOT / patch_record["path"]).read_bytes()
            self.assertEqual(
                hashlib.sha256(content).hexdigest(),
                patch_record["sha256"],
            )

    def test_exact_l4_snapshot_passes(self) -> None:
        spec = load_environment_spec(SPEC_PATH)
        result = evaluate_environment(spec, _snapshot(spec, cuda_available=True))

        self.assertTrue(result.succeeded)
        self.assertEqual(result.status, "passed")
        self.assertEqual(result.marker, "LIGHTNING L4 ENVIRONMENT PASSED")
        self.assertEqual(result.gpu_gate, "passed")

    def test_exact_cpu_snapshot_skips_gpu_without_claiming_l4_success(self) -> None:
        spec = load_environment_spec(SPEC_PATH)
        result = evaluate_environment(
            spec,
            _snapshot(spec, cuda_available=False),
            allow_cpu=True,
        )

        self.assertTrue(result.succeeded)
        self.assertEqual(result.status, "gpu_skipped")
        self.assertEqual(result.marker, "L4 GPU GATES SKIPPED")
        self.assertNotEqual(result.marker, "LIGHTNING L4 ENVIRONMENT PASSED")

    def test_cpu_skip_does_not_hide_dependency_mismatches(self) -> None:
        spec = load_environment_spec(SPEC_PATH)
        snapshot = _snapshot(spec, cuda_available=False)
        packages = dict(snapshot.packages)
        packages["torch"] = "2.8.0+cpu"
        snapshot = replace(snapshot, packages=packages)

        result = evaluate_environment(spec, snapshot, allow_cpu=True)

        self.assertFalse(result.succeeded)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.gpu_gate, "skipped")
        self.assertIn("package torch", "\n".join(result.issues))

    def test_patch_preparation_is_checked_and_idempotent(self) -> None:
        base_spec = load_environment_spec(SPEC_PATH)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "upstream"
            project = root / "project"
            patch_dir = project / "patches"
            repository.mkdir()
            patch_dir.mkdir(parents=True)

            self._git(repository, "init")
            source = repository / "source.txt"
            source.write_text("before\n", encoding="utf-8")
            self._git(repository, "add", "source.txt")
            self._git(
                repository,
                "-c",
                "user.name=Codex Test",
                "-c",
                "user.email=codex@example.invalid",
                "commit",
                "-m",
                "fixture",
            )
            commit = self._git(repository, "rev-parse", "HEAD").stdout.strip()

            source.write_text("after\n", encoding="utf-8")
            patch_content = self._git(repository, "diff", "--", "source.txt").stdout
            patch_path = patch_dir / "change.patch"
            patch_path.write_text(patch_content, encoding="utf-8")
            self._git(repository, "checkout", "--", "source.txt")
            patch_hash = hashlib.sha256(patch_path.read_bytes()).hexdigest()

            spec = replace(
                base_spec,
                upstream_commits={
                    **base_spec.upstream_commits,
                    "focalnet_dino": commit,
                },
                patches=({"path": "patches/change.patch", "sha256": patch_hash},),
            )

            checked = prepare_focalnet_patches(repository, project, spec)
            self.assertEqual(checked[0].state, "applicable")
            applied = prepare_focalnet_patches(repository, project, spec, apply=True)
            self.assertEqual(applied[0].state, "applied")
            self.assertEqual(source.read_text(encoding="utf-8"), "after\n")
            checked_again = prepare_focalnet_patches(repository, project, spec)
            self.assertEqual(checked_again[0].state, "already_applied")

    def test_environment_and_strict_load_scripts_expose_help_without_ml_packages(self) -> None:
        scripts = (
            "00_probe_environment.py",
            "prepare_focalnet.py",
            "01_validate_cuda_extension.py",
            "02_strict_load_detector.py",
            "09_strict_load_mmbcd.py",
        )
        for name in scripts:
            with self.subTest(script=name):
                result = subprocess.run(
                    [
                        sys.executable,
                        str(REPOSITORY_ROOT / "scripts" / "l4_validation" / name),
                        "--help",
                    ],
                    cwd=REPOSITORY_ROOT,
                    check=False,
                    capture_output=True,
                    text=True,
                    env=os.environ.copy(),
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)

    def test_extension_build_supports_conda_target_cuda_layout(self) -> None:
        spec = load_environment_spec(SPEC_PATH)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "upstream"
            ops = repository / "models" / "dino" / "ops"
            ops.mkdir(parents=True)
            (ops / "setup.py").write_text("# fixture\n", encoding="utf-8")
            toolkit = root / "cuda-12.8"
            (toolkit / "bin").mkdir(parents=True)
            (toolkit / "bin" / "nvcc").write_text("fixture\n", encoding="utf-8")
            target = toolkit / "targets" / "x86_64-linux"
            (target / "include").mkdir(parents=True)
            (target / "include" / "cuda_runtime_api.h").write_text("fixture\n", encoding="utf-8")
            (target / "lib").mkdir()
            captured: dict[str, str] = {}

            def run(arguments, **kwargs):
                if arguments[0] == str(toolkit / "bin" / "nvcc"):
                    return subprocess.CompletedProcess(
                        arguments, 0, stdout="Cuda compilation tools, release 12.8"
                    )
                captured.update(kwargs["env"])
                (ops / "MultiScaleDeformableAttention.fixture.so").write_bytes(b"so")
                return subprocess.CompletedProcess(arguments, 0)

            with patch.dict(os.environ, {"CUDA_HOME": str(toolkit)}, clear=False):
                with patch(
                    "vision_model_serving.compatibility.focalnet.subprocess.run",
                    side_effect=run,
                ):
                    build_focalnet_extension(repository, spec)

            self.assertEqual(captured["CPATH"].split(os.pathsep)[0], str(target / "include"))
            self.assertEqual(captured["LIBRARY_PATH"].split(os.pathsep)[0], str(target / "lib"))
            self.assertEqual(captured["LD_LIBRARY_PATH"].split(os.pathsep)[0], str(target / "lib"))
            self.assertEqual(
                captured["PATH"].split(os.pathsep)[0], str(Path(sys.executable).parent)
            )

    @staticmethod
    def _git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )


if __name__ == "__main__":
    unittest.main()
