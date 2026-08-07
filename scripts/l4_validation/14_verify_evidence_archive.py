#!/usr/bin/env python3
"""Safely verify the downloaded L4 evidence archive without extraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vision_model_serving.validation import (  # noqa: E402
    EvidenceVerificationError,
    verify_evidence_archive,
)
from vision_model_serving.validation.golden import (  # noqa: E402
    REFERENCE_ARCHIVE_SHA256,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("sidecar", type=Path)
    parser.add_argument("--expected-sha256", default=REFERENCE_ARCHIVE_SHA256)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        summary = verify_evidence_archive(
            args.archive,
            args.sidecar,
            expected_sha256=args.expected_sha256,
        )
    except EvidenceVerificationError as error:
        print(f"L4 EVIDENCE VERIFICATION FAILED: {error}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(summary.as_dict(), indent=2, sort_keys=True))
    else:
        print("Archive SHA256:", summary.archive_sha256)
        print("Members:", summary.member_count)
        print("Detector prediction SHA256:", summary.detector_prediction_sha256)
        print("MMBCD prediction SHA256:", summary.mmbcd_prediction_sha256)
    print("LIGHTNING L4 EVIDENCE ARCHIVE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
