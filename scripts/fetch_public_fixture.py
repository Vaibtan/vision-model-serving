#!/usr/bin/env python3
"""Fetch and fail-closed verify one attributed public validation fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import BinaryIO

_CHUNK_BYTES = 1024 * 1024


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")

    fixture = _load_fixture(args.manifest, args.fixture)
    output_root = args.output_root.expanduser().resolve()
    relative = Path(_text(fixture, "relative_directory"))
    destination = (output_root / relative).resolve()
    if not destination.is_relative_to(output_root):
        raise RuntimeError("fixture destination escapes the output root")
    existing = _verify_destination(destination, fixture)
    if existing is not None:
        print(json.dumps(existing, indent=2, sort_keys=True))
        return 0
    if args.offline:
        raise RuntimeError("offline verification requires an existing fixture")

    output_root.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".fixture-fetch-",
        dir=destination.parent,
    ) as temporary:
        work = Path(temporary)
        staging = work / "payload"
        staging.mkdir()
        archive_path = work / "download.zip"
        _download(
            _download_url(fixture),
            archive_path,
            max_bytes=_positive_int(fixture["download"], "max_download_bytes"),
            timeout_seconds=args.timeout_seconds,
        )
        _extract_verified(archive_path, staging, fixture)
        report = _report(destination, fixture, reused=False)
        (staging / "ATTRIBUTION.md").write_text(
            _attribution_text(fixture),
            encoding="utf-8",
        )
        (staging / "fetch-report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if destination.exists():
            raise RuntimeError("fixture destination appeared during download")
        staging.replace(destination)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _load_fixture(path: Path, fixture_id: str) -> dict[str, object]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise RuntimeError("public fixture manifest schema is unsupported")
    fixtures = payload.get("fixtures")
    if not isinstance(fixtures, list):
        raise TypeError("public fixture manifest has no fixture list")
    matches = [
        item
        for item in fixtures
        if isinstance(item, dict) and item.get("id") == fixture_id
    ]
    if len(matches) != 1:
        raise RuntimeError("public fixture identity is missing or ambiguous")
    fixture = matches[0]
    for section in ("download", "dicom", "license", "attribution"):
        if not isinstance(fixture.get(section), dict):
            raise TypeError(f"public fixture {section} section is invalid")
    return fixture


def _download_url(fixture: dict[str, object]) -> str:
    url = _text(fixture["download"], "url")
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != "nbia.cancerimagingarchive.net":
        raise RuntimeError("fixture download URL is outside the pinned TCIA host")
    return url


def _download(
    url: str,
    destination: Path,
    *,
    max_bytes: int,
    timeout_seconds: float,
) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "vision-model-serving-fixture-fetch/1"},
    )
    total = 0
    with (
        urllib.request.urlopen(request, timeout=timeout_seconds) as response,
        destination.open("xb") as output,
    ):
        while chunk := response.read(_CHUNK_BYTES):
            total += len(chunk)
            if total > max_bytes:
                raise RuntimeError("fixture download exceeds the manifest limit")
            output.write(chunk)
    if total == 0:
        raise RuntimeError("fixture download was empty")


def _extract_verified(
    archive_path: Path,
    staging: Path,
    fixture: dict[str, object],
) -> None:
    download = fixture["download"]
    with zipfile.ZipFile(archive_path) as archive:
        members = [member for member in archive.infolist() if not member.is_dir()]
        if not 1 <= len(members) <= _positive_int(download, "max_members"):
            raise RuntimeError("fixture archive member count is outside the limit")
        if sum(member.file_size for member in members) > _positive_int(
            download,
            "max_uncompressed_bytes",
        ):
            raise RuntimeError("fixture archive expands beyond the manifest limit")
        for member in members:
            path = PurePosixPath(member.filename)
            if path.is_absolute() or ".." in path.parts or member.flag_bits & 0x1:
                raise RuntimeError("fixture archive contains an unsafe member")
        for section_name in ("dicom", "license"):
            section = fixture[section_name]
            basename = _text(section, "member_basename")
            matches = [
                member
                for member in members
                if PurePosixPath(member.filename).name == basename
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"fixture archive has an ambiguous {section_name} member"
                )
            member = matches[0]
            if member.file_size != _positive_int(section, "size_bytes"):
                raise RuntimeError(
                    f"fixture {section_name} size differs from the manifest"
                )
            destination = staging / basename
            with archive.open(member) as source, destination.open("xb") as output:
                digest = _copy_and_hash(source, output)
            if digest != _sha256_text(section):
                raise RuntimeError(
                    f"fixture {section_name} hash differs from the manifest"
                )


def _verify_destination(
    destination: Path,
    fixture: dict[str, object],
) -> dict[str, object] | None:
    if not destination.exists():
        return None
    if not destination.is_dir():
        raise RuntimeError("fixture destination exists and is not a directory")
    for section_name in ("dicom", "license"):
        section = fixture[section_name]
        path = destination / _text(section, "member_basename")
        if not path.is_file() or path.stat().st_size != _positive_int(
            section, "size_bytes"
        ):
            raise RuntimeError(f"existing fixture {section_name} size is invalid")
        if _sha256_file(path) != _sha256_text(section):
            raise RuntimeError(f"existing fixture {section_name} hash is invalid")
    return _report(destination, fixture, reused=True)


def _report(
    directory: Path,
    fixture: dict[str, object],
    *,
    reused: bool,
) -> dict[str, object]:
    return {
        "fixture_id": fixture["id"],
        "series_instance_uid": fixture["series_instance_uid"],
        "dicom_sha256": _sha256_text(fixture["dicom"]),
        "license_sha256": _sha256_text(fixture["license"]),
        "license": fixture["license"]["spdx_expression"],
        "destination": str(directory.resolve()),
        "reused": reused,
        "validation_boundary": fixture["validation_boundary"],
    }


def _attribution_text(fixture: dict[str, object]) -> str:
    attribution = fixture["attribution"]
    license_record = fixture["license"]
    return (
        "# CBIS-DDSM attribution\n\n"
        f"{_text(attribution, 'data_citation')} "
        f"[{_text(attribution, 'doi')}]({_text(attribution, 'doi')})\n\n"
        f"Collection: {_text(attribution, 'collection_url')}\n\n"
        f"License: {_text(license_record, 'spdx_expression')} "
        f"({_text(license_record, 'url')})\n\n"
        f"TCIA usage policy: {_text(attribution, 'usage_policy_url')}\n"
    )


def _copy_and_hash(source: BinaryIO, output: BinaryIO) -> str:
    digest = hashlib.sha256()
    while chunk := source.read(_CHUNK_BYTES):
        output.write(chunk)
        digest.update(chunk)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(section: object) -> str:
    value = _text(section, "sha256")
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise RuntimeError("fixture SHA-256 is invalid")
    return value


def _positive_int(section: object, name: str) -> int:
    if not isinstance(section, dict):
        raise TypeError("fixture manifest section is invalid")
    value = section.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"fixture {name} is invalid")
    return value


def _text(section: object, name: str) -> str:
    if not isinstance(section, dict):
        raise TypeError("fixture manifest section is invalid")
    value = section.get(name)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"fixture {name} is invalid")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
