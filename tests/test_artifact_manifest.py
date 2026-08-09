from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from vision_model_serving.artifacts import (  # noqa: E402
    ManifestValidationError,
    load_manifest,
)
from vision_model_serving.model_ids import MODEL_IDS  # noqa: E402


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
COMMIT_A = "1" * 40
COMMIT_B = "2" * 40
COMMIT_C = "3" * 40
COMMIT_D = "4" * 40


def _artifact(artifact_id: str, role: str, sha256: str) -> dict[str, object]:
    detector = role == "detector"
    repository_revision = COMMIT_A if detector else COMMIT_B
    checkpoint: dict[str, object] = {
        "container": "wrapped_state_dict" if detector else "raw_state_dict",
        "state_dict_key": "model" if detector else None,
        "key_normalization": "none" if detector else "strip_module_prefix",
        "state_tensor_count": 6 if detector else 2,
        "tensor_element_count": 1,
        "tensor_bytes": 4,
        "dtypes": ["float32"] if detector else ["float32", "int64"],
        "strict_load_verified": True,
        "strict_load_evidence": "l4_archive",
    }
    if detector:
        checkpoint.update(
            {
                "serving_backbone_embedded": True,
                "required_key_groups": {
                    "backbone": 1,
                    "transformer": 1,
                    "bbox_embed": 1,
                    "input_proj": 1,
                    "class_embed": 1,
                    "label_enc": 1,
                },
                "required_key_prefixes": {
                    "backbone": "backbone.",
                    "transformer": "transformer.",
                    "bbox_embed": "bbox_embed.",
                    "input_proj": "input_proj.",
                    "class_embed": "class_embed.",
                    "label_enc": "label_enc.",
                },
                "required_tensor_shapes": {
                    "class_embed.0.weight": [1, 256],
                },
                "output_contract": {
                    "pred_logits_shape": [1, 900, 1],
                    "pred_boxes_shape": [1, 900, 4],
                    "box_format": "normalized_cxcywh",
                    "num_select": 300,
                    "internal_nms_iou_threshold": -1,
                },
                "derived_inference_checkpoint": {
                    "filename": "detector-inference.pth",
                    "sha256": SHA_C,
                    "state_tensor_count": 6,
                    "retention": "not_retained_in_repository",
                },
            }
        )
    else:
        checkpoint.update(
            {
                "module_prefixed_key_count": 2,
                "dtype_counts": {"float32": 1, "int64": 1},
                "checkpoint_aliases_equal": True,
                "required_key_groups": {
                    "image_encoder": 1,
                    "image_projection": 1,
                    "text_encoder": 1,
                    "text_projection": 1,
                    "cross_attention": 1,
                    "classifier": 1,
                },
                "required_key_prefixes": {
                    "image_encoder": "image_encoder.",
                    "image_projection": "img_fc_layer.",
                    "text_encoder": "text_encoder.",
                    "text_projection": "txt_fc_layer.",
                    "cross_attention": "attention.",
                    "classifier": "model_fc2.",
                },
                "required_tensor_shapes": {
                    "model_fc2.weight": [2, 768],
                },
                "output_contract": {
                    "logits_shape": [1, 2],
                    "fused_embeddings_shape": [1, 768],
                    "roi_count": 8,
                    "text_max_length": 90,
                },
            }
        )
    return {
        "id": artifact_id,
        "role": role,
        "filename": f"{artifact_id}.pth",
        "size_bytes": 128,
        "sha256": sha256,
        "provenance": {
            "source": "evaluator_supplied",
            "custody": "external_read_only_mount",
            "repository": "https://example.invalid/model-source",
            "repository_revision": repository_revision,
        },
        "trust": {
            "checksum_pinned": True,
            "load_authorized": True,
            "user_upload_allowed": False,
        },
        "license": {
            "weights_status": "unknown",
            "redistribution_status": "not_authorized",
        },
        "checkpoint": checkpoint,
        "semantics": {
            "status": "unverified",
            "class_names": None,
            "decision_threshold": None,
            "medical_validation": False,
        },
    }


def _valid_manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "manifest_id": "tiny-ci-fixture",
        "evidence": {
            "archive": {"filename": "evidence.tar.gz", "sha256": SHA_C},
            "reference_result": {
                "path": "docs/validation/reference.json",
                "sha256": SHA_A,
            },
        },
        "pipeline": {
            "kind": "detector_then_classifier",
            "stages": ["detector", "classifier"],
            "proposal_contract": {
                "format": "normalized_cxcywh_confidence",
                "detector_output_count": 300,
                "nms_iou_threshold": 0.1,
                "nms_comparison": "strictly_greater",
                "classifier_roi_count": 8,
                "insufficient_proposal_policy": "duplicate_existing",
                "empty_proposal_policy": "fail_closed",
            },
        },
        "revisions": {
            "focalnet_dino": COMMIT_A,
            "mmbcd": COMMIT_B,
            "dino": COMMIT_C,
            "roberta_tokenizer": COMMIT_D,
        },
        "runtime_lane": {
            "device": "NVIDIA L4",
            "python": "3.12.11",
            "torch": "2.8.0+cu128",
            "torchvision": "0.23.0+cu128",
            "transformers": "5.14.1",
            "numpy": "1.26.4",
            "cuda_runtime": "12.8",
        },
        "artifacts": [
            _artifact("detector", "detector", SHA_A),
            _artifact("classifier", "classifier", SHA_B),
        ],
        "tokenizer": {
            "id": "roberta-base",
            "repository": "FacebookAI/roberta-base",
            "revision": COMMIT_D,
            "local_files_only": True,
            "files": [
                {"filename": name, "size_bytes": index + 1, "sha256": SHA_C}
                for index, name in enumerate(
                    (
                        "config.json",
                        "merges.txt",
                        "tokenizer.json",
                        "tokenizer_config.json",
                        "vocab.json",
                    )
                )
            ],
        },
        "repository_assets": [
            {
                "id": "detector-config",
                "kind": "config",
                "path": "config_cfg.py",
                "size_bytes": 10,
                "sha256": SHA_A,
            },
            {
                "id": "compat-patch",
                "kind": "patch",
                "path": "patches/compat.patch",
                "size_bytes": 10,
                "sha256": SHA_B,
                "applies_to_revision": COMMIT_A,
            },
        ],
    }


class ArtifactManifestTests(unittest.TestCase):
    def _load_fixture(self, payload: dict[str, object]):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            return load_manifest(path)

    def test_tiny_manifest_loads_without_touching_named_artifact_files(self) -> None:
        manifest = self._load_fixture(_valid_manifest())

        self.assertEqual(manifest.manifest_id, "tiny-ci-fixture")
        self.assertEqual(manifest.pipeline_stages, ("detector", "classifier"))
        self.assertEqual(
            tuple(artifact.id for artifact in manifest.artifacts),
            ("detector", "classifier"),
        )

    def test_repository_manifest_matches_the_verified_l4_contract(self) -> None:
        manifest_path = REPOSITORY_ROOT / "config" / "model-artifacts.json"
        manifest = load_manifest(manifest_path)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        reference = json.loads(
            (
                REPOSITORY_ROOT
                / "docs"
                / "validation"
                / "reference-l4-fp32-20260807.json"
            ).read_text(encoding="utf-8")
        )

        self.assertEqual(
            manifest.pipeline_stages,
            MODEL_IDS,
        )
        self.assertEqual(
            manifest.revisions,
            {
                "focalnet_dino": "23901e021dc6ec8f66bad47983f45a25574452cc",
                "mmbcd": "14ac5e099c79253b01e0885d2ebefa6f86cfd8f0",
                "dino": "7c446df5b9f45747937fb0d72314eb9f7b66930a",
                "roberta_tokenizer": "e2da8e2f811d1448a5b465c236feacd80ffbac7b",
            },
        )
        self.assertEqual(
            [artifact.sha256 for artifact in manifest.artifacts],
            [
                reference["artifacts"]["focalnet_dino_finetuned_sha256"],
                reference["artifacts"]["mmbcd_sha256"],
            ],
        )
        self.assertEqual(
            manifest.evidence_archive_sha256,
            reference["archive"]["sha256"],
        )
        self.assertEqual(
            payload["artifacts"][0]["checkpoint"]["derived_inference_checkpoint"][
                "sha256"
            ],
            reference["artifacts"]["focalnet_dino_inference_sha256"],
        )
        for artifact in manifest.artifacts:
            self.assertTrue(artifact.strict_load_verified)
            self.assertFalse(artifact.user_upload_allowed)
            self.assertEqual(artifact.semantics_status, "unverified")
            self.assertIsNone(artifact.class_names)
            self.assertIsNone(artifact.decision_threshold)

    def test_repository_asset_records_match_checked_in_bytes(self) -> None:
        payload = json.loads(
            (
                REPOSITORY_ROOT / "config" / "model-artifacts.json"
            ).read_text(encoding="utf-8")
        )

        for asset in payload["repository_assets"]:
            path = REPOSITORY_ROOT / asset["path"]
            content = path.read_bytes()
            self.assertEqual(len(content), asset["size_bytes"], path)
            self.assertEqual(hashlib.sha256(content).hexdigest(), asset["sha256"], path)
            if "l4_validated_sha256" in asset:
                self.assertTrue(content.endswith(b"\n"), path)
                self.assertEqual(
                    hashlib.sha256(content[:-1]).hexdigest(),
                    asset["l4_validated_sha256"],
                    path,
                )

        reference = payload["evidence"]["reference_result"]
        reference_content = (REPOSITORY_ROOT / reference["path"]).read_bytes()
        self.assertEqual(
            hashlib.sha256(reference_content).hexdigest(),
            reference["sha256"],
        )

    def test_rejects_unpinned_or_user_supplied_pickle_artifacts(self) -> None:
        payload = _valid_manifest()
        artifact = payload["artifacts"][0]  # type: ignore[index]
        artifact["sha256"] = "not-a-sha256"  # type: ignore[index]
        artifact["trust"]["user_upload_allowed"] = True  # type: ignore[index]

        with self.assertRaises(ManifestValidationError) as raised:
            self._load_fixture(payload)

        message = str(raised.exception)
        self.assertIn("artifacts[0].sha256", message)
        self.assertIn("artifacts[0].trust.user_upload_allowed", message)

    def test_unverified_semantics_cannot_gain_labels_or_a_threshold(self) -> None:
        payload = _valid_manifest()
        semantics = payload["artifacts"][1]["semantics"]  # type: ignore[index]
        semantics["class_names"] = ["benign", "malignant"]  # type: ignore[index]
        semantics["decision_threshold"] = 0.5  # type: ignore[index]
        semantics["medical_validation"] = True  # type: ignore[index]

        with self.assertRaises(ManifestValidationError) as raised:
            self._load_fixture(payload)

        message = str(raised.exception)
        self.assertIn("class_names must be null", message)
        self.assertIn("decision_threshold must be null", message)
        self.assertIn("medical_validation must be false", message)

    def test_requires_detector_then_classifier_stage_order(self) -> None:
        payload = _valid_manifest()
        payload["pipeline"]["stages"] = ["classifier", "detector"]  # type: ignore[index]

        with self.assertRaises(ManifestValidationError) as raised:
            self._load_fixture(payload)

        self.assertIn("pipeline.stages", str(raised.exception))

    def test_requires_explicit_weight_and_redistribution_status(self) -> None:
        payload = _valid_manifest()
        del payload["artifacts"][0]["license"]["weights_status"]  # type: ignore[index]

        with self.assertRaises(ManifestValidationError) as raised:
            self._load_fixture(payload)

        self.assertIn("license.weights_status", str(raised.exception))

    def test_requires_the_role_specific_checkpoint_audit(self) -> None:
        payload = _valid_manifest()
        checkpoint = payload["artifacts"][1]["checkpoint"]  # type: ignore[index]
        del checkpoint["required_key_groups"]["text_encoder"]  # type: ignore[index]

        with self.assertRaises(ManifestValidationError) as raised:
            self._load_fixture(payload)

        self.assertIn(
            "checkpoint.required_key_groups",
            str(raised.exception),
        )

    def test_rejects_artifact_revision_that_differs_from_pinned_source(self) -> None:
        payload = _valid_manifest()
        provenance = payload["artifacts"][0]["provenance"]  # type: ignore[index]
        provenance["repository_revision"] = "a" * 40  # type: ignore[index]

        with self.assertRaises(ManifestValidationError) as raised:
            self._load_fixture(payload)

        self.assertIn("must match revisions.focalnet_dino", str(raised.exception))

    def test_rejects_patch_revision_that_differs_from_pinned_source(self) -> None:
        payload = _valid_manifest()
        patch = payload["repository_assets"][1]  # type: ignore[index]
        patch["applies_to_revision"] = "a" * 40  # type: ignore[index]

        with self.assertRaises(ManifestValidationError) as raised:
            self._load_fixture(payload)

        self.assertIn("must match revisions.focalnet_dino", str(raised.exception))

    def test_cli_emits_a_machine_readable_inventory(self) -> None:
        environment = copy.copy(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            filter(
                None,
                (str(SOURCE_ROOT), environment.get("PYTHONPATH", "")),
            )
        )
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "vision_model_serving.artifacts",
                str(REPOSITORY_ROOT / "config" / "model-artifacts.json"),
                "--json",
            ],
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        inventory = json.loads(result.stdout)
        self.assertEqual(inventory["status"], "valid")
        self.assertEqual(inventory["manifest_id"], "vision-model-serving-l4-fp32-20260807")
        self.assertEqual(
            [artifact["role"] for artifact in inventory["artifacts"]],
            ["detector", "classifier"],
        )


if __name__ == "__main__":
    unittest.main()
