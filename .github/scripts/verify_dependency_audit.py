from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _canonical_package_name(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return str(canonicalize_name(value.strip()))


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _string_tuple(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{field} must be a list of non-empty strings")
    return tuple(sorted(item.strip() for item in value))


def _format_key(key: tuple[str, str, str]) -> str:
    package, version, vulnerability_id = key
    return f"{package}=={version} {vulnerability_id}"


def _requirements_inventory(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"cannot read {path}: {error}") from error
    inventory: dict[str, str] = {}
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith(("#", "--")):
            continue
        try:
            requirement = Requirement(line)
        except InvalidRequirement as error:
            raise ValueError(f"invalid requirement at {path}:{line_number}: {error}") from error
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        specifiers = list(requirement.specifier)
        if (
            requirement.url is not None
            or requirement.extras
            or len(specifiers) != 1
            or specifiers[0].operator != "=="
            or "*" in specifiers[0].version
        ):
            raise ValueError(f"audit input must be an exact package pin at {path}:{line_number}")
        package = str(canonicalize_name(requirement.name))
        version = specifiers[0].version
        previous = inventory.get(package)
        if previous is not None and previous != version:
            raise ValueError(
                f"audit input selects conflicting versions for {package}: {previous} and {version}"
            )
        inventory[package] = version
    if not inventory:
        raise ValueError("audit requirements have no applicable exact package pins")
    return inventory


def verify(
    report_path: Path,
    baseline_path: Path,
    requirements_path: Path,
    source_requirements_path: Path,
) -> None:
    report = _load_json_object(report_path)
    baseline = _load_json_object(baseline_path)
    if baseline.get("schema_version") != 1:
        raise ValueError("dependency audit baseline schema_version must be 1")

    dependencies = report.get("dependencies")
    if not isinstance(dependencies, list) or not dependencies:
        raise ValueError("pip-audit report has no dependency inventory")

    package_names: set[str] = set()
    reported_inventory: dict[str, str] = {}
    actual: dict[tuple[str, str, str], tuple[str, ...]] = {}
    for index, dependency in enumerate(dependencies):
        if not isinstance(dependency, dict):
            raise ValueError(f"report dependency {index} must be an object")
        package = _canonical_package_name(
            dependency.get("name"), field=f"report dependency {index} name"
        )
        version = _string(dependency.get("version"), field=f"report dependency {index} version")
        previous_version = reported_inventory.get(package)
        if previous_version is not None:
            raise ValueError(
                f"pip-audit reported duplicate package {package}: {previous_version} and {version}"
            )
        reported_inventory[package] = version
        package_names.add(package)
        skip_reason = dependency.get("skip_reason")
        if skip_reason:
            raise ValueError(f"pip-audit skipped {package}=={version}: {skip_reason}")
        vulnerabilities = dependency.get("vulns")
        if not isinstance(vulnerabilities, list):
            raise ValueError(f"report dependency {package} has no vulnerability list")
        for vulnerability_index, vulnerability in enumerate(vulnerabilities):
            if not isinstance(vulnerability, dict):
                raise ValueError(
                    f"report vulnerability {package}[{vulnerability_index}] must be an object"
                )
            vulnerability_id = _string(
                vulnerability.get("id"),
                field=f"report vulnerability {package}[{vulnerability_index}] id",
            )
            fix_versions = _string_tuple(
                vulnerability.get("fix_versions", []),
                field=(f"report vulnerability {package}[{vulnerability_index}] fix_versions"),
            )
            key = (package, version, vulnerability_id)
            if key in actual:
                raise ValueError(f"pip-audit reported a duplicate: {_format_key(key)}")
            actual[key] = fix_versions

    required_planes = baseline.get("required_packages")
    if not isinstance(required_planes, dict) or not required_planes:
        raise ValueError("baseline required_packages must be a non-empty object")
    for plane, packages in required_planes.items():
        required = {
            _canonical_package_name(package, field=f"required_packages.{plane}")
            for package in _string_tuple(packages, field=f"required_packages.{plane}")
        }
        missing = sorted(required - package_names)
        if missing:
            raise ValueError(f"audit omitted required {plane} package(s): {', '.join(missing)}")

    expected_inventory = _requirements_inventory(requirements_path)
    source_inventory = _requirements_inventory(source_requirements_path)
    missing_inventory = sorted(expected_inventory.keys() - reported_inventory.keys())
    unexpected_inventory = sorted(reported_inventory.keys() - expected_inventory.keys())
    changed_versions = sorted(
        package
        for package in expected_inventory.keys() & reported_inventory.keys()
        if expected_inventory[package] != reported_inventory[package]
    )
    inventory_problems: list[str] = []
    if missing_inventory:
        inventory_problems.append(
            "omitted package pins: "
            + ", ".join(
                f"{package}=={expected_inventory[package]}" for package in missing_inventory
            )
        )
    if unexpected_inventory:
        inventory_problems.append("unexpected audited packages: " + ", ".join(unexpected_inventory))
    if changed_versions:
        inventory_problems.append(
            "audited version mismatches: "
            + ", ".join(
                (
                    f"{package} expected={expected_inventory[package]} "
                    f"reported={reported_inventory[package]}"
                )
                for package in changed_versions
            )
        )
    if inventory_problems:
        raise ValueError(
            "audit report inventory does not match input: " + "; ".join(inventory_problems)
        )

    exception_packages = {
        _canonical_package_name(package, field="exception_packages")
        for package in _string_tuple(baseline.get("exception_packages"), field="exception_packages")
    }
    exceptions = baseline.get("reviewed_exceptions")
    if not isinstance(exceptions, list):
        raise ValueError("baseline reviewed_exceptions must be a list")

    expected: dict[tuple[str, str, str], tuple[str, ...]] = {}
    for index, exception in enumerate(exceptions):
        if not isinstance(exception, dict):
            raise ValueError(f"baseline exception {index} must be an object")
        package = _canonical_package_name(
            exception.get("package"), field=f"baseline exception {index} package"
        )
        if package not in exception_packages:
            raise ValueError(f"baseline exception package {package} is outside exception_packages")
        audited_version = _string(
            exception.get("audited_version"),
            field=f"baseline exception {index} audited_version",
        )
        locked_version = _string(
            exception.get("locked_version"),
            field=f"baseline exception {index} locked_version",
        )
        selected_version = source_inventory.get(package)
        if selected_version != locked_version:
            raise ValueError(
                f"baseline exception {package} locked_version={locked_version} "
                f"does not match selected source version={selected_version}"
            )
        vulnerability_id = _string(exception.get("id"), field=f"baseline exception {index} id")
        fix_versions = _string_tuple(
            exception.get("fix_versions"),
            field=f"baseline exception {index} fix_versions",
        )
        key = (package, audited_version, vulnerability_id)
        if key in expected:
            raise ValueError(f"baseline contains a duplicate: {_format_key(key)}")
        expected[key] = fix_versions

    unreviewed = sorted(actual.keys() - expected.keys())
    stale = sorted(expected.keys() - actual.keys())
    changed_fixes = sorted(
        key for key in actual.keys() & expected.keys() if actual[key] != expected[key]
    )
    problems: list[str] = []
    if unreviewed:
        problems.append(
            "unreviewed vulnerabilities: " + ", ".join(_format_key(key) for key in unreviewed)
        )
    if stale:
        problems.append("stale baseline entries: " + ", ".join(_format_key(key) for key in stale))
    if changed_fixes:
        details = ", ".join(
            (f"{_format_key(key)} expected={list(expected[key])} reported={list(actual[key])}")
            for key in changed_fixes
        )
        problems.append(f"changed fix-version evidence: {details}")
    if problems:
        raise ValueError("; ".join(problems))

    print(
        f"Audited {len(package_names)} package identities; "
        f"all {len(actual)} vulnerability records match the reviewed GPU baseline."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail on unreviewed, changed, skipped, or stale dependency findings."
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--source-requirements", type=Path, required=True)
    args = parser.parse_args()
    try:
        verify(
            args.report,
            args.baseline,
            args.requirements,
            args.source_requirements,
        )
    except ValueError as error:
        print(f"dependency audit verification failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
