#!/usr/bin/env python3
"""Measure the packaged strict-residency L4 path and publish schema-v4 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vision_model_serving.validation.benchmark import (  # noqa: E402
    BenchmarkContractError,
    render_benchmark_markdown,
)
from vision_model_serving.validation.benchmark_campaign import (  # noqa: E402
    BenchmarkCampaignPlan,
    NvidiaSampler,
    run_benchmark_campaign,
)
from vision_model_serving.validation.benchmark_environment import (  # noqa: E402
    benchmark_environment_identity,
)
from vision_model_serving.validation.acceptance_contract import (  # noqa: E402
    PUBLIC_DICOM_SHA256,
)
from vision_model_serving.validation.packaged_http import (  # noqa: E402
    PackagedPredictionClient,
)
from vision_model_serving.validation.reporting import write_json_atomic  # noqa: E402
from vision_model_serving.validation.revision import (  # noqa: E402
    RevisionEvidenceError,
    require_clean_revision,
)


_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"(?:sha256:)?[0-9a-f]{64}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--dicom", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=float, default=180)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--environment-evidence", type=Path, required=True)
    parser.add_argument("--executor-image-id", required=True)
    parser.add_argument("--expected-detector-sha256", required=True)
    parser.add_argument("--expected-classifier-sha256", required=True)
    parser.add_argument("--resource-sample-interval-ms", type=int, default=200)
    args = parser.parse_args()
    if args.runs < 5:
        parser.error("--runs must be at least five")
    if args.resource_sample_interval_ms < 50:
        parser.error("--resource-sample-interval-ms must be at least 50")
    if _COMMIT.fullmatch(args.revision) is None:
        parser.error("--revision must be a full lowercase Git commit")
    for name in ("expected_detector_sha256", "expected_classifier_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", getattr(args, name)) is None:
            parser.error(f"--{name.replace('_', '-')} must be a SHA-256 digest")
    if _SHA256.fullmatch(args.executor_image_id) is None:
        parser.error("--executor-image-id must be a SHA-256 image id")

    _verify_clean_revision(PROJECT_ROOT, args.revision)
    dicom = args.dicom.read_bytes()
    if hashlib.sha256(dicom).hexdigest() != PUBLIC_DICOM_SHA256:
        raise BenchmarkContractError("benchmark DICOM differs from the packaged public fixture")
    plan = BenchmarkCampaignPlan(
        dicom=dicom,
        runs=args.runs,
        revision=args.revision,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        environment=benchmark_environment_identity(
            PROJECT_ROOT,
            args.environment_evidence,
            executor_image_id=args.executor_image_id,
        ),
        identity=_benchmark_identity(PROJECT_ROOT, dicom),
        expected_detector_sha256=args.expected_detector_sha256,
        expected_classifier_sha256=args.expected_classifier_sha256,
    )
    record = run_benchmark_campaign(
        PackagedPredictionClient(
            args.base_url,
            timeout_seconds=args.timeout_seconds,
        ),
        plan,
        NvidiaSampler(args.resource_sample_interval_ms),
    )
    write_json_atomic(args.output, record)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(
        render_benchmark_markdown(record),
        encoding="utf-8",
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["outcome"] == "passed" else 1


def _verify_clean_revision(project_root: Path, revision: str) -> None:
    try:
        require_clean_revision(project_root, revision)
    except RevisionEvidenceError as error:
        raise BenchmarkContractError(
            f"benchmark must run from the exact clean committed revision: {error}"
        ) from None


def _benchmark_identity(project_root: Path, dicom: bytes) -> dict[str, object]:
    path = project_root / "config" / "model-artifacts.json"
    manifest = _json_object(path)
    return {
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "models": manifest["artifacts"],
        "tokenizer": manifest["tokenizer"],
        "repository_assets": manifest["repository_assets"],
        "dicom_sha256": hashlib.sha256(dicom).hexdigest(),
    }


def _json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise BenchmarkContractError(f"configuration file is unavailable: {path.name}") from None
    if not isinstance(payload, dict):
        raise BenchmarkContractError(f"configuration file is not an object: {path.name}")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
