#!/usr/bin/env python3
"""Run a real HTTP prediction benchmark and write a bounded JSON record."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--dicom", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=120)
    parser.add_argument("--expected-detector-sha256", required=True)
    parser.add_argument("--expected-classifier-sha256", required=True)
    args = parser.parse_args()
    if args.runs <= 0:
        parser.error("--runs must be positive")

    dicom = args.dicom.read_bytes()
    samples: list[float] = []
    for index in range(args.runs + 1):
        started = time.monotonic()
        result = _predict(
            args.base_url.rstrip("/"),
            dicom,
            timeout_seconds=args.timeout_seconds,
        )
        elapsed = time.monotonic() - started
        detector_hash = result["detector"]["prediction_sha256"]
        classifier_hash = result["classification"]["prediction_sha256"]
        if detector_hash != args.expected_detector_sha256:
            raise RuntimeError("detector output differs from the pinned golden")
        if classifier_hash != args.expected_classifier_sha256:
            raise RuntimeError("classifier output differs from the pinned golden")
        if index:
            samples.append(elapsed)

    ordered = sorted(samples)
    record = {
        "schema_version": 1,
        "runs": args.runs,
        "warmup_runs": 1,
        "dicom_sha256": hashlib.sha256(dicom).hexdigest(),
        "detector_sha256": args.expected_detector_sha256,
        "classifier_sha256": args.expected_classifier_sha256,
        "latency_seconds": {
            "min": min(samples),
            "median": statistics.median(samples),
            "p95": ordered[max(0, int(len(ordered) * 0.95 + 0.999) - 1)],
            "max": max(samples),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


def _predict(base_url: str, dicom: bytes, *, timeout_seconds: float) -> dict:
    boundary = "vms-real-infrastructure-boundary"
    body = _multipart(
        boundary,
        dicom,
        {"mode": "full", "clinical_history": "real public mammogram acceptance."},
    )
    request = urllib.request.Request(
        f"{base_url}/api/v1/predictions",
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Prefer": "respond-async",
        },
        method="POST",
    )
    submitted = _json(request, timeout=min(30.0, timeout_seconds))
    prediction_id = submitted["prediction_id"]
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        status = _json(
            urllib.request.Request(f"{base_url}/api/v1/predictions/{prediction_id}"),
            timeout=min(10.0, timeout_seconds),
        )
        if status["state"] == "succeeded":
            return _json(
                urllib.request.Request(
                    f"{base_url}/api/v1/predictions/{prediction_id}/result"
                ),
                timeout=min(10.0, timeout_seconds),
            )["result"]
        if status["state"] in {"failed", "expired"}:
            raise RuntimeError(f"prediction ended in state {status['state']}")
        time.sleep(0.1)
    raise TimeoutError("prediction did not finish before the benchmark deadline")


def _json(request: urllib.request.Request, *, timeout: float) -> dict:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        message = error.read(4096).decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {message}") from error


def _multipart(boundary: str, dicom: bytes, fields: dict[str, str]) -> bytes:
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            (
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode(),
                b"\r\n",
            )
        )
    chunks.extend(
        (
            f"--{boundary}\r\n".encode(),
            b'Content-Disposition: form-data; name="dicom"; filename="input.dcm"\r\n',
            b"Content-Type: application/dicom\r\n\r\n",
            dicom,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        )
    )
    return b"".join(chunks)


if __name__ == "__main__":
    raise SystemExit(main())
