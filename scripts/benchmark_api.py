#!/usr/bin/env python3
"""Benchmark the packaged HTTP path and emit bounded JSON/Markdown evidence."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

from vision_model_serving.model_ids import MODEL_IDS
from vision_model_serving.pipeline.contracts import PredictionMode
from vision_model_serving.validation.packaged_http import PackagedPredictionClient
from vision_model_serving.validation.reporting import write_json_atomic


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--dicom", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=120)
    parser.add_argument("--revision", default="unrecorded")
    parser.add_argument("--expected-detector-sha256", required=True)
    parser.add_argument("--expected-classifier-sha256", required=True)
    args = parser.parse_args()
    if args.runs <= 0:
        parser.error("--runs must be positive")

    dicom = args.dicom.read_bytes()
    client = PackagedPredictionClient(
        args.base_url,
        timeout_seconds=args.timeout_seconds,
    )
    readiness = client.readiness()
    if readiness.get("status") != "ready" or not all(
        readiness.get("checks", {}).values()
    ):
        raise RuntimeError("service readiness did not pass every packaged check")
    before = _inventory(client.model_inventory())
    if before["runtime"]["state"] != "unloaded" or before["runtime"][
        "resident_models"
    ]:
        raise RuntimeError(
            "cold benchmark requires a fresh executor with no resident models"
        )

    cold_full = _run_sample(
        client,
        dicom,
        mode=PredictionMode.FULL,
        expected_detector_sha256=args.expected_detector_sha256,
        expected_classifier_sha256=args.expected_classifier_sha256,
    )
    if cold_full["lifecycle"] != {
        "detector_reused": False,
        "classifier_reused": False,
    }:
        raise RuntimeError("cold full request did not perform both first loads")

    warm_detection = [
        _run_sample(
            client,
            dicom,
            mode=PredictionMode.DETECTION,
            expected_detector_sha256=args.expected_detector_sha256,
            expected_classifier_sha256=args.expected_classifier_sha256,
        )
        for _ in range(args.runs)
    ]
    warm_full = [
        _run_sample(
            client,
            dicom,
            mode=PredictionMode.FULL,
            expected_detector_sha256=args.expected_detector_sha256,
            expected_classifier_sha256=args.expected_classifier_sha256,
        )
        for _ in range(args.runs)
    ]
    if not all(
        sample["lifecycle"]["detector_reused"] for sample in warm_detection
    ) or not all(
        sample["lifecycle"]
        == {"detector_reused": True, "classifier_reused": True}
        for sample in warm_full
    ):
        raise RuntimeError("warm benchmark unexpectedly reloaded a model")

    after = _inventory(client.model_inventory())
    expected_residents = set(MODEL_IDS)
    if set(after["runtime"]["resident_models"]) != expected_residents:
        raise RuntimeError("final executor inventory is not dual resident")

    record = {
        "schema_version": 2,
        "measured_at": datetime.now(UTC).isoformat(),
        "revision": args.revision,
        "benchmark_script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "policy": {
            "queue_concurrency": 1,
            "runs_per_warm_mode": args.runs,
            "cold_start": "fresh executor, then one full request",
            "warm_sequence": "detection requests, then full requests",
            "precision": "float32",
        },
        "identity": {
            "dicom_sha256": hashlib.sha256(dicom).hexdigest(),
            "detector_prediction_sha256": args.expected_detector_sha256,
            "classifier_prediction_sha256": args.expected_classifier_sha256,
            "manifest_id": before["manifest_id"],
            "models": before["models"],
        },
        "runtime": {
            "device": after["runtime"]["device"],
            "initial_state": before["runtime"]["state"],
            "final_state": after["runtime"]["state"],
            "final_active_model": after["runtime"]["active_model"],
            "final_resident_models": sorted(after["runtime"]["resident_models"]),
            "readiness_checks": readiness["checks"],
        },
        "measurements": {
            "cold_full": cold_full,
            "warm_detection": _aggregate(warm_detection),
            "warm_full": _aggregate(warm_full),
        },
        "gates": {
            "exact_output_hashes": True,
            "cold_loaded_each_model_once": True,
            "warm_requests_reused_residents": True,
            "final_dual_residency": True,
            "all_readiness_checks_passed": True,
        },
        "validation_boundary": (
            "One serialized NVIDIA L4, one public Secondary Capture DICOM, and "
            "the pinned FP32 artifacts; not concurrency scaling, accuracy, "
            "calibration, robustness, or clinical performance."
        ),
    }
    write_json_atomic(args.output, record)
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(
            _markdown_report(record),
            encoding="utf-8",
        )
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


def _run_sample(
    client: PackagedPredictionClient,
    dicom: bytes,
    *,
    mode: PredictionMode,
    expected_detector_sha256: str,
    expected_classifier_sha256: str,
) -> dict[str, Any]:
    started = time.monotonic()
    result = client.predict(dicom, mode=mode)
    wall_seconds = time.monotonic() - started
    detector = result["detector"]
    classification = result["classification"]
    if detector["prediction_sha256"] != expected_detector_sha256:
        raise RuntimeError("detector output differs from the pinned golden")
    if mode is PredictionMode.FULL:
        if (
            classification is None
            or classification["prediction_sha256"]
            != expected_classifier_sha256
        ):
            raise RuntimeError("classifier output differs from the pinned golden")
    elif classification is not None:
        raise RuntimeError("detection benchmark unexpectedly ran the classifier")

    timings = result["timings"]
    classifier_timings = timings["classifier"]
    peak_reserved = timings["detector"]["memory"]["peak_reserved_bytes"]
    if classifier_timings is not None:
        peak_reserved = max(
            peak_reserved,
            classifier_timings["memory"]["peak_reserved_bytes"],
        )
    return {
        "mode": mode.value,
        "wall_seconds": wall_seconds,
        "pipeline_total_seconds": timings["total_ms"] / 1000.0,
        "http_queue_overhead_seconds": max(
            0.0,
            wall_seconds - (timings["total_ms"] / 1000.0),
        ),
        "stage_seconds": {
            "dicom_decode": timings["decode_ms"] / 1000.0,
            "detector": timings["detector"]["runtime"]["inference_ms"] / 1000.0,
            "classifier": (
                None
                if classifier_timings is None
                else classifier_timings["runtime"]["inference_ms"] / 1000.0
            ),
        },
        "lifecycle": {
            "detector_reused": timings["detector"]["runtime"]["reused"],
            "classifier_reused": (
                None
                if classifier_timings is None
                else classifier_timings["runtime"]["reused"]
            ),
        },
        "peak_reserved_bytes": peak_reserved,
    }


def _aggregate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    walls = [sample["wall_seconds"] for sample in samples]
    return {
        "samples": samples,
        "latency_seconds": _distribution(walls),
        "serialized_throughput_per_second": len(walls) / sum(walls),
        "median_stage_seconds": {
            "dicom_decode": statistics.median(
                sample["stage_seconds"]["dicom_decode"] for sample in samples
            ),
            "detector": statistics.median(
                sample["stage_seconds"]["detector"] for sample in samples
            ),
            "classifier": _optional_median(
                sample["stage_seconds"]["classifier"] for sample in samples
            ),
        },
        "maximum_peak_reserved_bytes": max(
            sample["peak_reserved_bytes"] for sample in samples
        ),
    }


def _distribution(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "min": min(samples),
        "median": statistics.median(samples),
        "p95": ordered[max(0, int(len(ordered) * 0.95 + 0.999) - 1)],
        "max": max(samples),
    }


def _optional_median(values: object) -> float | None:
    present = [float(value) for value in values if value is not None]
    return statistics.median(present) if present else None


def _inventory(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "manifest_id": payload["manifest_id"],
        "models": [
            {
                "id": model["id"],
                "role": model["role"],
                "sha256": model["sha256"],
                "strict_load_verified": model["strict_load_verified"],
                "semantics_status": model["semantics_status"],
            }
            for model in payload["models"]
        ],
        "runtime": payload["runtime"],
    }


def _markdown_report(record: dict[str, Any]) -> str:
    measurements = record["measurements"]
    cold = measurements["cold_full"]
    rows = []
    for name in ("warm_detection", "warm_full"):
        value = measurements[name]
        latency = value["latency_seconds"]
        rows.append(
            "| {name} | {runs} | {minimum:.3f} | {median:.3f} | {p95:.3f} | "
            "{maximum:.3f} | {throughput:.3f} |".format(
                name=name.replace("_", " "),
                runs=len(value["samples"]),
                minimum=latency["min"],
                median=latency["median"],
                p95=latency["p95"],
                maximum=latency["max"],
                throughput=value["serialized_throughput_per_second"],
            )
        )
    model_lines = "\n".join(
        f"| {model['role']} | `{model['sha256']}` |"
        for model in record["identity"]["models"]
    )
    gate_lines = "\n".join(
        f"| {name.replace('_', ' ')} | {'PASS' if passed else 'FAIL'} |"
        for name, passed in record["gates"].items()
    )
    return f"""# Packaged L4 FP32 benchmark

Measured at `{record['measured_at']}` from revision `{record['revision']}`.

## Identity

| Input or artifact | SHA-256 |
| --- | --- |
| public DICOM | `{record['identity']['dicom_sha256']}` |
| detector output | `{record['identity']['detector_prediction_sha256']}` |
| classifier output | `{record['identity']['classifier_prediction_sha256']}` |
{model_lines}

Device: `{record['runtime']['device']}`. Final residents:
`{', '.join(record['runtime']['final_resident_models'])}`.

## End-to-end results

The cold full request took **{cold['wall_seconds']:.3f} s** wall time and
**{cold['pipeline_total_seconds']:.3f} s** inside the typed pipeline. Warm
throughput is serialized request throughput at queue concurrency one.

| path | runs | min s | median s | p95 s | max s | requests/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

## Promotion gates

| Gate | Result |
| --- | --- |
{gate_lines}

## Boundary

{record['validation_boundary']}
"""


if __name__ == "__main__":
    raise SystemExit(main())
