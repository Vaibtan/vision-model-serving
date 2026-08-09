"""Live TCIA fixture-fetch acceptance through the public CLI."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SERIES_UID = "1.3.6.1.4.1.9590.100.1.2.100131208110604806117271735422083351547"
DICOM_SHA256 = "9f70081672a460f29231bb471e8a9e26dd3ed26a2ebbd91c064e575e7842a19c"
LICENSE_SHA256 = "77585fc0bd3e537c198f8f761607302ad3ac90261b2bb199aa13ba2073460d67"


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="vms-cbis-ddsm-") as temporary:
        output_root = Path(temporary)
        completed = subprocess.run(
            [
                sys.executable,
                str(REPOSITORY_ROOT / "scripts" / "fetch_public_fixture.py"),
                "--manifest",
                str(REPOSITORY_ROOT / "config" / "public-fixtures.json"),
                "--fixture",
                "cbis-ddsm-l4-reference",
                "--output-root",
                str(output_root),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
        fixture = output_root / "cbis-ddsm" / SERIES_UID
        dicom = fixture / "1-1.dcm"
        license_file = fixture / "LICENSE"
        if _sha256(dicom) != DICOM_SHA256:
            raise AssertionError("downloaded DICOM differs from the pinned fixture")
        if _sha256(license_file) != LICENSE_SHA256:
            raise AssertionError("downloaded TCIA license differs from the pinned copy")
        report = json.loads(completed.stdout)
        if report["dicom_sha256"] != DICOM_SHA256:
            raise AssertionError("fetch report does not identify the pinned DICOM")
        if report["license_sha256"] != LICENSE_SHA256:
            raise AssertionError("fetch report does not identify the pinned license")
        offline = subprocess.run(
            [
                sys.executable,
                str(REPOSITORY_ROOT / "scripts" / "fetch_public_fixture.py"),
                "--manifest",
                str(REPOSITORY_ROOT / "config" / "public-fixtures.json"),
                "--fixture",
                "cbis-ddsm-l4-reference",
                "--output-root",
                str(output_root),
                "--offline",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        offline_report = json.loads(offline.stdout)
        if offline_report != {**report, "reused": True}:
            raise AssertionError("offline verification changed the fixture identity")
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
