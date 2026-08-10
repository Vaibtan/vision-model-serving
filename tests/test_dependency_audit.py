from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "dependency_audit_verifier",
    REPOSITORY_ROOT / ".github" / "scripts" / "verify_dependency_audit.py",
)
assert spec is not None and spec.loader is not None
dependency_audit_verifier = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = dependency_audit_verifier
spec.loader.exec_module(dependency_audit_verifier)


def dependency(
    name: str,
    version: str = "1.0",
    *,
    vulnerabilities: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "name": name,
        "version": version,
        "vulns": vulnerabilities or [],
    }


class DependencyAuditVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.report = {
            "dependencies": [
                dependency("django"),
                dependency("pillow"),
                dependency("playwright"),
                dependency(
                    "torch",
                    "2.8.0",
                    vulnerabilities=[{"id": "ADV-1", "fix_versions": ["2.9.0"]}],
                ),
                dependency("torchvision"),
                dependency("cuda-python"),
                dependency("onnx"),
                dependency("polygraphy"),
                dependency("tensorrt"),
                dependency("torch-tensorrt"),
                dependency("indirect-library"),
            ]
        }
        self.baseline = {
            "schema_version": 1,
            "required_packages": {
                "cpu": ["django", "pillow"],
                "browser": ["playwright"],
                "gpu": ["torch", "torchvision"],
                "tensorrt": [
                    "cuda-python",
                    "onnx",
                    "polygraphy",
                    "tensorrt",
                    "torch-tensorrt",
                ],
            },
            "exception_packages": ["torch"],
            "reviewed_exceptions": [
                {
                    "package": "torch",
                    "locked_version": "2.8.0+cu128",
                    "audited_version": "2.8.0",
                    "id": "ADV-1",
                    "fix_versions": ["2.9.0"],
                }
            ],
        }

    def verify(
        self,
        report: dict[str, object] | None = None,
        baseline: dict[str, object] | None = None,
        source_version_overrides: dict[str, str] | None = None,
    ) -> None:
        with TemporaryDirectory() as directory:
            report_path = Path(directory) / "report.json"
            baseline_path = Path(directory) / "baseline.json"
            requirements_path = Path(directory) / "requirements.txt"
            source_requirements_path = Path(directory) / "source-requirements.txt"
            report_path.write_text(json.dumps(report or self.report), encoding="utf-8")
            baseline_path.write_text(json.dumps(baseline or self.baseline), encoding="utf-8")
            requirements_path.write_text(
                "\n".join(
                    f"{item['name']}=={item['version']}" for item in self.report["dependencies"]
                ),
                encoding="utf-8",
            )
            source_versions = {"torch": "2.8.0+cu128"}
            source_versions.update(source_version_overrides or {})
            source_requirements_path.write_text(
                "\n".join(
                    f"{item['name']}=={source_versions.get(item['name'], item['version'])}"
                    for item in self.report["dependencies"]
                ),
                encoding="utf-8",
            )
            dependency_audit_verifier.verify(
                report_path,
                baseline_path,
                requirements_path,
                source_requirements_path,
            )

    def test_matching_report_passes(self) -> None:
        self.verify()

    def test_unreviewed_vulnerability_fails(self) -> None:
        report = deepcopy(self.report)
        torch = report["dependencies"][3]
        torch["vulns"].append({"id": "ADV-2", "fix_versions": []})

        with self.assertRaisesRegex(ValueError, "unreviewed vulnerabilities"):
            self.verify(report=report)

    def test_stale_baseline_entry_fails(self) -> None:
        baseline = deepcopy(self.baseline)
        baseline["reviewed_exceptions"].append(
            {
                "package": "torch",
                "locked_version": "2.8.0+cu128",
                "audited_version": "2.8.0",
                "id": "ADV-REMOVED",
                "fix_versions": [],
            }
        )

        with self.assertRaisesRegex(ValueError, "stale baseline entries"):
            self.verify(baseline=baseline)

    def test_changed_fix_versions_fail(self) -> None:
        report = deepcopy(self.report)
        report["dependencies"][3]["vulns"][0]["fix_versions"] = ["2.10.0"]

        with self.assertRaisesRegex(ValueError, "changed fix-version evidence"):
            self.verify(report=report)

    def test_changed_cuda_local_pin_fails(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "locked_version=2.8.0\\+cu128.*source version=2.8.0\\+cu129",
        ):
            self.verify(source_version_overrides={"torch": "2.8.0+cu129"})

    def test_skipped_dependency_fails(self) -> None:
        report = deepcopy(self.report)
        report["dependencies"][0]["skip_reason"] = "not found on advisory service"

        with self.assertRaisesRegex(ValueError, "pip-audit skipped django"):
            self.verify(report=report)

    def test_missing_required_package_fails(self) -> None:
        report = deepcopy(self.report)
        report["dependencies"] = [
            item for item in report["dependencies"] if item["name"] != "playwright"
        ]

        with self.assertRaisesRegex(
            ValueError, "audit omitted required browser package.*playwright"
        ):
            self.verify(report=report)

    def test_omitted_non_sentinel_package_fails_inventory_check(self) -> None:
        report = deepcopy(self.report)
        report["dependencies"] = [
            item for item in report["dependencies"] if item["name"] != "indirect-library"
        ]

        with self.assertRaisesRegex(
            ValueError, "audit report inventory does not match input.*indirect-library"
        ):
            self.verify(report=report)


if __name__ == "__main__":
    unittest.main()
