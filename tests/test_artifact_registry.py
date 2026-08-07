from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from vision_model_serving.artifacts import (  # noqa: E402
    ArtifactChangedError,
    ArtifactIntegrityError,
    ArtifactNotFoundError,
    ArtifactRegistry,
    ArtifactStructureError,
    CheckpointSummary,
    RegistryConfigurationError,
)


class DescriptorInspector:
    """Safe test inspector for tiny JSON checkpoint descriptors."""

    def inspect(self, stream, artifact_spec) -> CheckpointSummary:
        descriptor = json.loads(stream.read())
        return CheckpointSummary(
            container=descriptor["container"],
            state_dict_key=descriptor["state_dict_key"],
            key_normalization=descriptor["key_normalization"],
            state_tensor_count=descriptor["state_tensor_count"],
            tensor_element_count=descriptor["tensor_element_count"],
            tensor_bytes=descriptor["tensor_bytes"],
            dtype_counts=descriptor["dtype_counts"],
            group_counts=descriptor["group_counts"],
            tensor_shapes={
                name: tuple(shape)
                for name, shape in descriptor["tensor_shapes"].items()
            },
            aliases_equal=descriptor.get("aliases_equal"),
        )


class RegistryFixture:
    def __init__(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.artifact_root = self.root / "models"
        self.tokenizer_root = self.root / "tokenizer"
        self.repository_root = self.root / "repository"
        self.artifact_root.mkdir()
        self.tokenizer_root.mkdir()
        self.repository_root.mkdir()
        self.manifest_path = self.root / "model-artifacts.json"
        self.payload = copy.deepcopy(
            json.loads(
                (
                    REPOSITORY_ROOT / "config" / "model-artifacts.json"
                ).read_text(encoding="utf-8")
            )
        )
        self.descriptors: dict[str, dict[str, object]] = {}
        self.artifact_paths: dict[str, Path] = {}
        self._create_model_artifacts()
        self._create_tokenizer()
        self._create_repository_assets()
        self.write_manifest()

    def close(self) -> None:
        self.temporary.cleanup()

    def registry(
        self,
        *,
        runtime_versions: dict[str, str] | None = None,
        native_operator: bool = True,
    ) -> ArtifactRegistry:
        return ArtifactRegistry(
            self.manifest_path,
            artifact_root=self.artifact_root,
            tokenizer_root=self.tokenizer_root,
            repository_root=self.repository_root,
            checkpoint_inspector=DescriptorInspector(),
            runtime_versions=runtime_versions or self.expected_runtime(),
            native_operator_probe=lambda: native_operator,
        )

    def expected_runtime(self) -> dict[str, str]:
        lane = self.payload["runtime_lane"]
        return {
            "python": lane["python"],
            "torch": lane["torch"],
            "torchvision": lane["torchvision"],
            "transformers": lane["transformers"],
            "numpy": lane["numpy"],
            "cuda_runtime": lane["cuda_runtime"],
        }

    def rewrite_descriptor(
        self,
        model_id: str,
        descriptor: dict[str, object],
        *,
        update_manifest_identity: bool,
    ) -> None:
        content = json.dumps(descriptor, sort_keys=True).encode()
        path = self.artifact_paths[model_id]
        path.write_bytes(content)
        self.descriptors[model_id] = descriptor
        if update_manifest_identity:
            artifact = self._artifact_spec(model_id)
            artifact["size_bytes"] = len(content)
            artifact["sha256"] = hashlib.sha256(content).hexdigest()
            self.write_manifest()

    def write_manifest(self) -> None:
        self.manifest_path.write_text(
            json.dumps(self.payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _create_model_artifacts(self) -> None:
        for artifact in self.payload["artifacts"]:
            detector = artifact["role"] == "detector"
            checkpoint = artifact["checkpoint"]
            if detector:
                checkpoint.update(
                    {
                        "state_tensor_count": 6,
                        "tensor_element_count": 6,
                        "tensor_bytes": 24,
                        "dtypes": ["float32"],
                        "required_key_groups": {
                            "backbone": 1,
                            "transformer": 1,
                            "bbox_embed": 1,
                            "input_proj": 1,
                            "class_embed": 1,
                            "label_enc": 1,
                        },
                        "required_tensor_shapes": {
                            "class_embed.0.weight": [1, 2],
                        },
                    }
                )
                checkpoint["derived_inference_checkpoint"][
                    "state_tensor_count"
                ] = 6
                dtype_counts = {"float32": 6}
                aliases_equal = None
            else:
                checkpoint.update(
                    {
                        "module_prefixed_key_count": 6,
                        "state_tensor_count": 6,
                        "tensor_element_count": 6,
                        "tensor_bytes": 28,
                        "dtypes": ["float32", "int64"],
                        "dtype_counts": {"float32": 5, "int64": 1},
                        "required_key_groups": {
                            "image_encoder": 1,
                            "image_projection": 1,
                            "text_encoder": 1,
                            "text_projection": 1,
                            "cross_attention": 1,
                            "classifier": 1,
                        },
                        "required_tensor_shapes": {
                            "model_fc2.weight": [2, 3],
                        },
                    }
                )
                dtype_counts = {"float32": 5, "int64": 1}
                aliases_equal = True
            descriptor: dict[str, object] = {
                "container": checkpoint["container"],
                "state_dict_key": checkpoint["state_dict_key"],
                "key_normalization": checkpoint["key_normalization"],
                "state_tensor_count": checkpoint["state_tensor_count"],
                "tensor_element_count": checkpoint["tensor_element_count"],
                "tensor_bytes": checkpoint["tensor_bytes"],
                "dtype_counts": dtype_counts,
                "group_counts": checkpoint["required_key_groups"],
                "tensor_shapes": checkpoint["required_tensor_shapes"],
                "aliases_equal": aliases_equal,
            }
            content = json.dumps(descriptor, sort_keys=True).encode()
            path = self.artifact_root / artifact["filename"]
            path.write_bytes(content)
            artifact["size_bytes"] = len(content)
            artifact["sha256"] = hashlib.sha256(content).hexdigest()
            self.descriptors[artifact["id"]] = descriptor
            self.artifact_paths[artifact["id"]] = path

    def _create_tokenizer(self) -> None:
        for record in self.payload["tokenizer"]["files"]:
            content = f"tokenizer fixture: {record['filename']}\n".encode()
            (self.tokenizer_root / record["filename"]).write_bytes(content)
            record["size_bytes"] = len(content)
            record["sha256"] = hashlib.sha256(content).hexdigest()

    def _create_repository_assets(self) -> None:
        for record in self.payload["repository_assets"]:
            path = self.repository_root.joinpath(*Path(record["path"]).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            content = f"repository fixture: {record['id']}\n".encode()
            path.write_bytes(content)
            record["size_bytes"] = len(content)
            record["sha256"] = hashlib.sha256(content).hexdigest()
            record.pop("l4_validated_sha256", None)
            record.pop("equivalence", None)

    def _artifact_spec(self, model_id: str) -> dict[str, object]:
        return next(
            artifact
            for artifact in self.payload["artifacts"]
            if artifact["id"] == model_id
        )


class ArtifactRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = RegistryFixture()

    def tearDown(self) -> None:
        self.fixture.close()

    def test_verify_all_returns_ready_without_public_paths_or_hashes(self) -> None:
        report = self.fixture.registry().verify_all()

        self.assertTrue(report.ready)
        self.assertEqual(len(report.verified_artifacts), 2)
        public = json.dumps(report.as_public_dict(), sort_keys=True)
        self.assertNotIn(str(self.fixture.root), public)
        for artifact in report.verified_artifacts:
            self.assertNotIn(artifact.sha256, public)

    def test_read_only_model_and_tokenizer_mounts_pass_readiness(self) -> None:
        files = [
            *self.fixture.artifact_root.iterdir(),
            *self.fixture.tokenizer_root.iterdir(),
            *(
                path
                for path in self.fixture.repository_root.rglob("*")
                if path.is_file()
            ),
        ]
        try:
            for path in files:
                path.chmod(stat.S_IREAD)

            report = self.fixture.registry().verify_all()

            self.assertTrue(report.ready)
        finally:
            for path in files:
                path.chmod(stat.S_IREAD | stat.S_IWRITE)

    def test_resolve_returns_only_manifest_owned_model_ids(self) -> None:
        registry = self.fixture.registry()

        detector = registry.resolve("focalnet-dino-detector")

        self.assertEqual(detector.role, "detector")
        with self.assertRaises(ArtifactNotFoundError):
            registry.resolve("user-uploaded-checkpoint")

    def test_missing_and_corrupt_artifacts_have_redacted_typed_errors(self) -> None:
        model_id = "focalnet-dino-detector"
        path = self.fixture.artifact_paths[model_id]
        expected_hash = self.fixture._artifact_spec(model_id)["sha256"]
        path.unlink()

        with self.assertRaises(ArtifactNotFoundError) as missing:
            self.fixture.registry().resolve(model_id)
        self.assertNotIn(str(self.fixture.root), str(missing.exception))
        self.assertNotIn(str(expected_hash), str(missing.exception))

        self.fixture.rewrite_descriptor(
            model_id,
            self.fixture.descriptors[model_id],
            update_manifest_identity=False,
        )
        path.write_bytes(path.read_bytes() + b"corrupt")
        with self.assertRaises(ArtifactIntegrityError) as corrupt:
            self.fixture.registry().resolve(model_id)
        self.assertNotIn(str(self.fixture.root), str(corrupt.exception))
        self.assertNotIn(str(expected_hash), str(corrupt.exception))

    def test_wrong_checkpoint_shape_fails_after_identity_verification(self) -> None:
        model_id = "mmbcd-classifier"
        descriptor = copy.deepcopy(self.fixture.descriptors[model_id])
        descriptor["tensor_shapes"] = {"model_fc2.weight": [3, 2]}
        self.fixture.rewrite_descriptor(
            model_id,
            descriptor,
            update_manifest_identity=True,
        )

        with self.assertRaisesRegex(ArtifactStructureError, "tensor shapes"):
            self.fixture.registry().resolve(model_id)

    def test_same_size_wrong_hash_fails_before_checkpoint_inspection(self) -> None:
        model_id = "focalnet-dino-detector"
        path = self.fixture.artifact_paths[model_id]
        content = bytearray(path.read_bytes())
        content[-2] = ord("0") if content[-2] != ord("0") else ord("1")
        path.write_bytes(content)

        with self.assertRaisesRegex(ArtifactIntegrityError, "checksum"):
            self.fixture.registry().resolve(model_id)

    def test_wrong_required_key_count_fails_after_identity_verification(self) -> None:
        model_id = "focalnet-dino-detector"
        descriptor = copy.deepcopy(self.fixture.descriptors[model_id])
        descriptor["group_counts"] = {
            **descriptor["group_counts"],
            "backbone": 2,
        }
        self.fixture.rewrite_descriptor(
            model_id,
            descriptor,
            update_manifest_identity=True,
        )

        with self.assertRaisesRegex(ArtifactStructureError, "state-key groups"):
            self.fixture.registry().resolve(model_id)

    def test_wrong_role_is_rejected_as_a_manifest_configuration_error(self) -> None:
        self.fixture.payload["artifacts"][0]["role"] = "classifier"
        self.fixture.write_manifest()

        with self.assertRaises(RegistryConfigurationError):
            self.fixture.registry()

    def test_readiness_fails_for_tokenizer_runtime_and_operator_gates(self) -> None:
        tokenizer_file = self.fixture.tokenizer_root / "vocab.json"
        tokenizer_file.unlink()
        runtime = self.fixture.expected_runtime()
        runtime["torch"] = "wrong-version"

        report = self.fixture.registry(
            runtime_versions=runtime,
            native_operator=False,
        ).verify_all()

        self.assertFalse(report.ready)
        self.assertEqual(report.tokenizer.code, "artifact_missing")
        self.assertEqual(report.runtime.code, "runtime_incompatible")
        self.assertEqual(report.native_operator.code, "native_operator_unavailable")

    def test_cached_verification_detects_a_changed_file(self) -> None:
        model_id = "focalnet-dino-detector"
        registry = self.fixture.registry()
        registry.resolve(model_id)
        path = self.fixture.artifact_paths[model_id]
        original_stat = path.stat()
        content = bytearray(path.read_bytes())
        content[-2] = ord("0") if content[-2] != ord("0") else ord("1")
        path.write_bytes(content)
        os.utime(
            path,
            ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns + 2_000_000_000),
        )

        with self.assertRaises(ArtifactChangedError):
            registry.resolve(model_id)

    def test_verified_stream_detects_replacement_before_model_load(self) -> None:
        model_id = "focalnet-dino-detector"
        artifact = self.fixture.registry().resolve(model_id)
        path = self.fixture.artifact_paths[model_id]
        content = bytearray(path.read_bytes())
        content[-2] = ord("0") if content[-2] != ord("0") else ord("1")
        path.write_bytes(content)

        with self.assertRaises(ArtifactChangedError):
            with artifact.open_checkpoint():
                self.fail("a changed artifact must not be yielded")


if __name__ == "__main__":
    unittest.main()
