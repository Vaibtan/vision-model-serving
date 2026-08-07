from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from vision_model_serving.validation import (  # noqa: E402
    EvidenceVerificationError,
    compare_box_records,
    verify_evidence_archive,
)
from vision_model_serving.validation.golden import (  # noqa: E402
    REFERENCE_COMMITS,
    REFERENCE_DETECTOR_PREDICTION_SHA256,
    REFERENCE_DICOM_SHA256,
    REFERENCE_EXTENSION_SHA256,
    REFERENCE_MMBCD_INPUT_SHA256,
    REFERENCE_MMBCD_LOGITS,
    REFERENCE_MMBCD_PREDICTION_SHA256,
    REFERENCE_MMBCD_PROBABILITIES,
    REFERENCE_MODEL_HASHES,
    REFERENCE_PREPROCESSED_ARRAY_SHA256,
    REQUIRED_FILES,
)


class GoldenComparisonTests(unittest.TestCase):
    def test_box_comparison_is_permutation_aware(self) -> None:
        expected = (
            (0.1, 0.2, 0.3, 0.4, 0.9),
            (0.5, 0.6, 0.2, 0.1, 0.8),
        )
        observed = (
            (0.50001, 0.6, 0.2, 0.1, 0.8),
            (0.1, 0.2, 0.3, 0.4, 0.90001),
        )

        result = compare_box_records(
            expected,
            observed,
            absolute_tolerance=0.0001,
        )

        self.assertTrue(result.equivalent)
        self.assertEqual(result.matched_pairs, ((0, 1), (1, 0)))
        self.assertAlmostEqual(result.max_abs_difference, 0.00001)

    def test_box_comparison_reports_unmatched_records(self) -> None:
        result = compare_box_records(
            ((0.1, 0.2, 0.3, 0.4, 0.9),),
            ((0.8, 0.8, 0.1, 0.1, 0.2),),
            absolute_tolerance=0.01,
        )

        self.assertFalse(result.equivalent)
        self.assertEqual(result.unmatched_expected, (0,))
        self.assertEqual(result.unmatched_observed, (0,))

    def test_box_comparison_rejects_non_finite_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-finite"):
            compare_box_records(
                ((float("nan"), 0.2, 0.3, 0.4, 0.9),),
                ((0.1, 0.2, 0.3, 0.4, 0.9),),
                absolute_tolerance=0.01,
            )


class EvidenceArchiveTests(unittest.TestCase):
    def test_valid_archive_passes_without_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive, sidecar, digest = self._write_fixture(Path(directory))

            summary = verify_evidence_archive(
                archive,
                sidecar,
                expected_sha256=digest,
            )

            self.assertEqual(summary.archive_sha256, digest)
            self.assertEqual(summary.member_count, len(REQUIRED_FILES))
            self.assertEqual(
                summary.detector_prediction_sha256,
                REFERENCE_DETECTOR_PREDICTION_SHA256,
            )

    def test_sidecar_mismatch_fails_with_typed_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive, sidecar, digest = self._write_fixture(Path(directory))
            sidecar.write_text("0" * 64 + "  evidence.tar.gz\n", encoding="utf-8")

            with self.assertRaisesRegex(
                EvidenceVerificationError,
                "Sidecar hash mismatch",
            ):
                verify_evidence_archive(
                    archive,
                    sidecar,
                    expected_sha256=digest,
                )

    def test_unsafe_member_fails_before_manifest_use(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive, sidecar, digest = self._write_fixture(
                Path(directory),
                extra_member="../escape.txt",
            )

            with self.assertRaisesRegex(EvidenceVerificationError, "Unsafe"):
                verify_evidence_archive(
                    archive,
                    sidecar,
                    expected_sha256=digest,
                )

    def test_missing_required_member_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive, sidecar, digest = self._write_fixture(
                Path(directory),
                omit="bundles/mmbcd-outputs.npz",
            )

            with self.assertRaisesRegex(
                EvidenceVerificationError,
                "mmbcd-outputs.npz",
            ):
                verify_evidence_archive(
                    archive,
                    sidecar,
                    expected_sha256=digest,
                )

    def test_prediction_hash_tampering_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive, sidecar, digest = self._write_fixture(
                Path(directory),
                detector_prediction="f" * 64,
            )

            with self.assertRaisesRegex(
                EvidenceVerificationError,
                "detector prediction SHA-256 mismatch",
            ):
                verify_evidence_archive(
                    archive,
                    sidecar,
                    expected_sha256=digest,
                )

    def test_wrong_upstream_commit_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive, sidecar, digest = self._write_fixture(
                Path(directory),
                focalnet_commit="0" * 40,
            )

            with self.assertRaisesRegex(
                EvidenceVerificationError,
                "Commit mismatch",
            ):
                verify_evidence_archive(
                    archive,
                    sidecar,
                    expected_sha256=digest,
                )

    def test_every_harness_program_compiles_and_exposes_help(self) -> None:
        scripts = sorted(
            (REPOSITORY_ROOT / "scripts" / "l4_validation").glob("[0-9][0-9]_*.py")
        )
        scripts.append(
            REPOSITORY_ROOT / "scripts" / "l4_validation" / "prepare_focalnet.py"
        )
        for script in scripts:
            with self.subTest(script=script.name):
                compile_result = subprocess.run(
                    [sys.executable, "-m", "py_compile", str(script)],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=os.environ.copy(),
                )
                self.assertEqual(compile_result.returncode, 0, compile_result.stderr)
                help_result = subprocess.run(
                    [sys.executable, str(script), "--help"],
                    cwd=REPOSITORY_ROOT,
                    check=False,
                    capture_output=True,
                    text=True,
                    env=os.environ.copy(),
                )
                self.assertEqual(help_result.returncode, 0, help_result.stderr)
                self.assertIn("usage:", help_result.stdout)

    @staticmethod
    def _write_fixture(
        directory: Path,
        *,
        omit: str | None = None,
        extra_member: str | None = None,
        detector_prediction: str = REFERENCE_DETECTOR_PREDICTION_SHA256,
        focalnet_commit: str = REFERENCE_COMMITS[
            "environment/focalnet-commit.txt"
        ],
    ) -> tuple[Path, Path, str]:
        upstream = b"upstream image"
        montage = b"roi montage"
        detections = b"top eight detections"
        mmbcd_inputs = b"mmbcd inputs"
        mmbcd_outputs = b"mmbcd outputs"
        patch = (
            REPOSITORY_ROOT
            / "patches"
            / "focalnet-pytorch-2.8-compat.patch"
        ).read_bytes()

        manifests = {
            "manifests/dicom-manifest.json": {
                "sha256": REFERENCE_DICOM_SHA256,
            },
            "manifests/preprocess-manifest.json": {
                "artifacts": {
                    "resized": {
                        "array_sha256": REFERENCE_PREPROCESSED_ARRAY_SHA256,
                        "file_sha256": hashlib.sha256(upstream).hexdigest(),
                    }
                }
            },
            "manifests/detector-inference-manifest.json": {
                "model": {"strict_load": True},
                "outputs": {
                    "prediction_sha256": detector_prediction,
                    "determinism_max_abs_diff": 0.0,
                    "mmbcd_topk_count": 8,
                    "pred_logits_shape": [1, 900, 1],
                    "pred_boxes_shape": [1, 900, 4],
                    "pre_nms_count": 300,
                    "post_nms_count": 8,
                },
                "semantic_validation": False,
            },
            "manifests/mmbcd-input-manifest.json": {
                "image_transform": {
                    "tensor_sha256": REFERENCE_MMBCD_INPUT_SHA256,
                },
                "text": {
                    "label_information_used": False,
                    "prompt": "Indication:",
                },
                "artifacts": {
                    "montage_sha256": hashlib.sha256(montage).hexdigest(),
                    "bundle_sha256": hashlib.sha256(mmbcd_inputs).hexdigest(),
                },
                "source": {
                    "detections_sha256": hashlib.sha256(detections).hexdigest(),
                },
                "proposals": {"records": [{} for _ in range(8)]},
            },
            "manifests/mmbcd-inference-manifest.json": {
                "model": {"strict_load": True},
                "determinism": {
                    "bitwise_equal_logits": True,
                    "bitwise_equal_embeddings": True,
                },
                "outputs": {
                    "prediction_sha256": REFERENCE_MMBCD_PREDICTION_SHA256,
                    "bundle_sha256": hashlib.sha256(mmbcd_outputs).hexdigest(),
                    "logits": REFERENCE_MMBCD_LOGITS,
                    "probabilities": REFERENCE_MMBCD_PROBABILITIES,
                },
            },
        }
        files: dict[str, bytes] = {
            "visuals/upstream-1024.png": upstream,
            "visuals/overlay-top8-1024.png": b"overlay 1024",
            "visuals/roi-montage.png": montage,
            "visuals/overlay-top8-original.png": b"overlay original",
            "environment/conda-packages.json": b"[]",
            "environment/nvcc.txt": b"Cuda compilation tools, release 12.8",
            "environment/python.txt": b"Python 3.12.11",
            "environment/gpu.txt": b"NVIDIA L4",
            "environment/python-packages.json": b"[]",
            "environment/cuda-extension.sha256": (
                REFERENCE_EXTENSION_SHA256 + "  extension.so\n"
            ).encode(),
            "environment/model-artifacts.sha256": (
                "\n".join(
                    f"{digest}  artifact-{index}.pth"
                    for index, digest in enumerate(sorted(REFERENCE_MODEL_HASHES))
                )
                + "\n"
            ).encode(),
            "patches/focalnet-pytorch-2.8-compat.patch": patch,
            "bundles/detections-top8.txt": detections,
            "bundles/mmbcd-outputs.npz": mmbcd_outputs,
            "bundles/mmbcd-inputs.npz": mmbcd_inputs,
        }
        for path, commit in REFERENCE_COMMITS.items():
            if path == "environment/focalnet-commit.txt":
                commit = focalnet_commit
            files[path] = (commit + "\n").encode()
        for path, manifest in manifests.items():
            files[path] = json.dumps(manifest, sort_keys=True).encode()
        if omit is not None:
            files.pop(omit)
        if extra_member is not None:
            files[extra_member] = b"unsafe"

        archive = directory / "evidence.tar.gz"
        with tarfile.open(archive, mode="w:gz") as tar:
            for path, content in sorted(files.items()):
                info = tarfile.TarInfo(path)
                info.size = len(content)
                info.mtime = 0
                tar.addfile(info, io.BytesIO(content))
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        sidecar = directory / "evidence.tar.gz.sha256"
        sidecar.write_text(f"{digest}  evidence.tar.gz\n", encoding="utf-8")
        return archive, sidecar, digest


if __name__ == "__main__":
    unittest.main()
