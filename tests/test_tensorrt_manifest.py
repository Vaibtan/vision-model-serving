from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from vision_model_serving.acceleration.tensorrt import (  # noqa: E402
    EngineRuntimeCompatibility,
    TensorRtManifestError,
    load_engine_manifest,
)
from vision_model_serving.model_ids import CLASSIFIER_MODEL_ID  # noqa: E402


CHECKPOINT_SHA256 = "a" * 64


def manifest(plan: bytes) -> dict[str, object]:
    return {
        "schema_version": 1,
        "model": {
            "id": CLASSIFIER_MODEL_ID,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "repository_revision": "b" * 40,
            "wrapper_contract_version": 1,
        },
        "engine": {
            "filename": "mmbcd-fp32.plan",
            "sha256": hashlib.sha256(plan).hexdigest(),
            "precision": "float32",
            "tf32": False,
        },
        "builder": {
            "torch": "2.8.0+cu128",
            "torch_tensorrt": "2.8.0",
            "tensorrt": "10.12.0.36",
            "cuda": "12.8",
            "gpu_name": "NVIDIA L4",
            "compute_capability": "8.9",
        },
        "inputs": [
            {"name": "roi_crops", "dtype": "float32", "shape": [1, 8, 3, 224, 224]},
            {"name": "input_ids", "dtype": "int64", "shape": [1, 90]},
            {"name": "attention_mask", "dtype": "int64", "shape": [1, 90]},
        ],
        "outputs": [
            {"name": "logits", "dtype": "float32", "shape": [1, 2]},
            {
                "name": "fused_embeddings",
                "dtype": "float32",
                "shape": [1, 768],
            },
            {"name": "roi_attention", "dtype": "float32", "shape": [1, 1, 8]},
        ],
        "coverage": {
            "strict_export": True,
            "require_full_compilation": True,
            "pytorch_partition_count": 0,
            "unsupported_operators": [],
            "dry_run_report_sha256": "c" * 64,
        },
        "plugin": None,
        "parity": {"passed": True, "report_sha256": "d" * 64},
        "performance": {
            "promotion_threshold_passed": True,
            "report_sha256": "e" * 64,
        },
        "decision": "go",
    }


def compatibility() -> EngineRuntimeCompatibility:
    return EngineRuntimeCompatibility(
        tensorrt="10.12.0.36",
        cuda="12.8",
        gpu_name="NVIDIA L4",
        compute_capability="8.9",
    )


class TensorRtManifestTests(unittest.TestCase):
    def test_valid_full_engine_manifest_loads_exact_plan(self) -> None:
        plan = b"serialized-tensorrt-plan"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "mmbcd-fp32.plan").write_bytes(plan)
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest(plan)), encoding="utf-8")

            observed = load_engine_manifest(
                path,
                engine_root=root,
                expected_model_id=CLASSIFIER_MODEL_ID,
                expected_checkpoint_sha256=CHECKPOINT_SHA256,
                compatibility=compatibility(),
            )

        self.assertEqual(observed.plan_sha256, hashlib.sha256(plan).hexdigest())
        self.assertEqual(observed.decision, "go")
        self.assertEqual(observed.input_names, (
            "roi_crops",
            "input_ids",
            "attention_mask",
        ))

    def test_corrupt_plan_or_wrong_runtime_fails_without_fallback(self) -> None:
        plan = b"serialized-tensorrt-plan"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "mmbcd-fp32.plan").write_bytes(b"corrupt")
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest(plan)), encoding="utf-8")

            with self.assertRaises(TensorRtManifestError):
                load_engine_manifest(
                    path,
                    engine_root=root,
                    expected_model_id=CLASSIFIER_MODEL_ID,
                    expected_checkpoint_sha256=CHECKPOINT_SHA256,
                    compatibility=compatibility(),
                )

            (root / "mmbcd-fp32.plan").write_bytes(plan)
            with self.assertRaises(TensorRtManifestError):
                load_engine_manifest(
                    path,
                    engine_root=root,
                    expected_model_id=CLASSIFIER_MODEL_ID,
                    expected_checkpoint_sha256=CHECKPOINT_SHA256,
                    compatibility=EngineRuntimeCompatibility(
                        tensorrt="10.13.0",
                        cuda="12.8",
                        gpu_name="NVIDIA L4",
                        compute_capability="8.9",
                    ),
                )

    def test_hybrid_or_unsupported_engine_is_never_accepted(self) -> None:
        plan = b"serialized-tensorrt-plan"
        for change in (
            {"pytorch_partition_count": 1},
            {"unsupported_operators": ["aten.opaque"]},
            {"require_full_compilation": False},
        ):
            with self.subTest(change=change), TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "mmbcd-fp32.plan").write_bytes(plan)
                payload = manifest(plan)
                payload["coverage"].update(change)
                path = root / "manifest.json"
                path.write_text(json.dumps(payload), encoding="utf-8")

                with self.assertRaises(TensorRtManifestError):
                    load_engine_manifest(
                        path,
                        engine_root=root,
                        expected_model_id=CLASSIFIER_MODEL_ID,
                        expected_checkpoint_sha256=CHECKPOINT_SHA256,
                        compatibility=compatibility(),
                    )

    def test_plan_filename_cannot_escape_the_engine_root(self) -> None:
        plan = b"serialized-tensorrt-plan"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            payload = manifest(plan)
            payload["engine"]["filename"] = "../outside.plan"
            path = root / "manifest.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaises(TensorRtManifestError):
                load_engine_manifest(
                    path,
                    engine_root=root,
                    expected_model_id=CLASSIFIER_MODEL_ID,
                    expected_checkpoint_sha256=CHECKPOINT_SHA256,
                    compatibility=compatibility(),
                )


if __name__ == "__main__":
    unittest.main()
