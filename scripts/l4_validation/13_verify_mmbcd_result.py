#!/usr/bin/env python3
"""Verify persisted MMBCD postconditions without loading a model or requiring CUDA."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from _common import MMBCD_PREDICTION_SHA256, default_paths, load_json, sha256_file


def main() -> None:
    defaults = default_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=defaults["mmbcd_output_dir"])
    parser.add_argument("--expected-prediction-sha256", default=MMBCD_PREDICTION_SHA256)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    manifest = load_json(output_dir / "inference-manifest.json")
    output_bundle = output_dir / manifest["outputs"]["bundle"]
    assert manifest["pipeline"] == "real-dicom-mmbcd-fp32-v1"
    assert manifest["model"]["strict_load"] is True
    assert manifest["model"]["checkpoint_aliases_equal"] is True
    assert manifest["determinism"]["bitwise_equal_logits"] is True
    assert manifest["determinism"]["bitwise_equal_embeddings"] is True
    assert manifest["determinism"]["logits_max_abs_diff"] == 0.0
    assert manifest["determinism"]["embeddings_max_abs_diff"] == 0.0
    assert manifest["inputs"]["label_information_used"] is False
    assert manifest["outputs"]["fused_embeddings_shape"] == [1, 768]
    assert sha256_file(output_bundle) == manifest["outputs"]["bundle_sha256"]
    assert manifest["outputs"]["prediction_sha256"] == args.expected_prediction_sha256
    probabilities = manifest["outputs"]["probabilities"][0]
    assert len(probabilities) == 2
    assert all(math.isfinite(value) for value in probabilities)
    assert abs(sum(probabilities) - 1.0) < 1e-5
    print("Prediction SHA256:", manifest["outputs"]["prediction_sha256"])
    print("Median latency ms:", manifest["performance"]["median_latency_ms"])
    print("REAL DICOM MMBCD INFERENCE PASSED")


if __name__ == "__main__":
    main()
