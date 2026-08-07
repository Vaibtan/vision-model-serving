"""Fail-closed local artifact resolution for startup and readiness."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
from importlib import metadata
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
from typing import Any, BinaryIO, Iterator, Protocol

from .manifest import ManifestValidationError, load_manifest


class ArtifactRegistryError(RuntimeError):
    """Base class for typed, path-redacted registry failures."""

    code = "artifact_registry_error"

    def __init__(self, subject: str, detail: str):
        self.subject = subject
        self.detail = detail
        super().__init__(f"{self.code}: {subject}: {detail}")


class RegistryConfigurationError(ArtifactRegistryError):
    code = "registry_configuration_invalid"


class ArtifactNotFoundError(ArtifactRegistryError):
    code = "artifact_missing"


class ArtifactIntegrityError(ArtifactRegistryError):
    code = "artifact_integrity_failed"


class ArtifactChangedError(ArtifactIntegrityError):
    code = "artifact_changed"


class ArtifactStructureError(ArtifactRegistryError):
    code = "artifact_structure_invalid"


class RuntimeCompatibilityError(ArtifactRegistryError):
    code = "runtime_incompatible"


class NativeOperatorError(ArtifactRegistryError):
    code = "native_operator_unavailable"


@dataclass(frozen=True, slots=True)
class CheckpointSummary:
    container: str
    state_dict_key: str | None
    key_normalization: str
    state_tensor_count: int
    tensor_element_count: int
    tensor_bytes: int
    dtype_counts: dict[str, int]
    group_counts: dict[str, int]
    tensor_shapes: dict[str, tuple[int, ...]]
    aliases_equal: bool | None = None

    @property
    def dtypes(self) -> frozenset[str]:
        return frozenset(self.dtype_counts)


class CheckpointInspector(Protocol):
    """Internal seam for restricted checkpoint structure inspection."""

    def inspect(
        self,
        stream: BinaryIO,
        artifact_spec: Mapping[str, Any],
    ) -> CheckpointSummary: ...


class TorchCheckpointInspector:
    """Inspect checksum-pinned PyTorch checkpoints on CPU with restricted loading."""

    def inspect(
        self,
        stream: BinaryIO,
        artifact_spec: Mapping[str, Any],
    ) -> CheckpointSummary:
        try:
            import torch
        except ImportError as error:
            raise ArtifactStructureError(
                str(artifact_spec.get("id", "unknown")),
                "PyTorch is not installed in the pinned runtime",
            ) from error

        checkpoint_spec = artifact_spec["checkpoint"]
        stream.seek(0)
        try:
            with torch.serialization.safe_globals([argparse.Namespace]):
                checkpoint = torch.load(
                    stream,
                    map_location="cpu",
                    weights_only=True,
                )
        except Exception as error:
            raise ArtifactStructureError(
                str(artifact_spec["id"]),
                f"restricted CPU checkpoint loading failed ({type(error).__name__})",
            ) from error

        expected_container = checkpoint_spec["container"]
        state_dict_key = checkpoint_spec["state_dict_key"]
        if expected_container == "wrapped_state_dict":
            if not isinstance(checkpoint, Mapping) or state_dict_key not in checkpoint:
                raise ArtifactStructureError(
                    str(artifact_spec["id"]),
                    "expected checkpoint state-dict root is absent",
                )
            state_dict = checkpoint[state_dict_key]
        else:
            state_dict = checkpoint
        if not _is_tensor_mapping(state_dict, torch):
            raise ArtifactStructureError(
                str(artifact_spec["id"]),
                "checkpoint root is not a non-empty tensor state dictionary",
            )

        canonical: dict[str, Any] = {}
        normalization = checkpoint_spec["key_normalization"]
        for key, tensor in state_dict.items():
            canonical_key = (
                key[7:]
                if normalization == "strip_module_prefix" and key.startswith("module.")
                else key
            )
            if canonical_key in canonical:
                raise ArtifactStructureError(
                    str(artifact_spec["id"]),
                    "state key collision after approved prefix normalization",
                )
            canonical[canonical_key] = tensor

        prefixes = checkpoint_spec["required_key_prefixes"]
        group_counts = {
            name: sum(key.startswith(prefix) for key in canonical)
            for name, prefix in prefixes.items()
        }
        required_shapes = checkpoint_spec["required_tensor_shapes"]
        tensor_shapes = {
            name: tuple(canonical[name].shape)
            for name in required_shapes
            if name in canonical
        }
        dtype_counts = Counter(
            str(tensor.dtype).removeprefix("torch.") for tensor in canonical.values()
        )
        aliases_equal = None
        if checkpoint_spec.get("checkpoint_aliases_equal") is True:
            alias_pairs = (
                ("img_fc1.weight", "img_fc_layer.1.weight"),
                ("img_fc1.bias", "img_fc_layer.1.bias"),
                ("txt_fc1.weight", "txt_fc_layer.1.weight"),
                ("txt_fc1.bias", "txt_fc_layer.1.bias"),
            )
            aliases_equal = all(
                left in canonical
                and right in canonical
                and torch.equal(canonical[left], canonical[right])
                for left, right in alias_pairs
            )
        return CheckpointSummary(
            container=expected_container,
            state_dict_key=state_dict_key,
            key_normalization=normalization,
            state_tensor_count=len(canonical),
            tensor_element_count=sum(tensor.numel() for tensor in canonical.values()),
            tensor_bytes=sum(
                tensor.numel() * tensor.element_size() for tensor in canonical.values()
            ),
            dtype_counts=dict(dtype_counts),
            group_counts=group_counts,
            tensor_shapes=tensor_shapes,
            aliases_equal=aliases_equal,
        )


@dataclass(frozen=True, slots=True)
class _FileFingerprint:
    size: int
    modified_ns: int
    changed_ns: int
    inode: int
    device: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> _FileFingerprint:
        return cls(
            size=value.st_size,
            modified_ns=value.st_mtime_ns,
            changed_ns=value.st_ctime_ns,
            inode=value.st_ino,
            device=value.st_dev,
        )


@dataclass(frozen=True, slots=True)
class VerifiedArtifact:
    id: str
    role: str
    filename: str
    size_bytes: int
    sha256: str
    checkpoint: CheckpointSummary
    provenance_source: str
    provenance_custody: str
    repository_revision: str
    strict_load_verified: bool
    weights_license_status: str
    redistribution_status: str
    _path: Path = field(repr=False, compare=False)
    _fingerprint: _FileFingerprint = field(repr=False, compare=False)

    @contextmanager
    def open_checkpoint(self) -> Iterator[BinaryIO]:
        """Yield a reverified read-only stream without exposing its local path."""

        descriptor = _open_readonly(self._path, self.id)
        with os.fdopen(descriptor, "rb") as stream:
            before = _FileFingerprint.from_stat(os.fstat(stream.fileno()))
            if before != self._fingerprint:
                raise ArtifactChangedError(
                    self.id,
                    "artifact identity changed after verification",
                )
            observed_hash = _sha256_stream(stream)
            if observed_hash != self.sha256:
                raise ArtifactChangedError(
                    self.id,
                    "artifact checksum changed after verification",
                )
            stream.seek(0)
            try:
                yield stream
            finally:
                try:
                    after = _FileFingerprint.from_stat(os.fstat(stream.fileno()))
                except OSError as error:
                    raise ArtifactChangedError(
                        self.id,
                        "artifact stream was closed during loading",
                    ) from error
                if after != before:
                    raise ArtifactChangedError(
                        self.id,
                        "artifact changed during loading",
                    )


@dataclass(frozen=True, slots=True)
class VerificationStatus:
    subject: str
    ok: bool
    code: str | None = None

    def as_public_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"subject": self.subject, "ok": self.ok}
        if self.code is not None:
            result["code"] = self.code
        return result


@dataclass(frozen=True, slots=True)
class ArtifactReport:
    ready: bool
    artifacts: tuple[VerificationStatus, ...]
    tokenizer: VerificationStatus
    repository_assets: VerificationStatus
    runtime: VerificationStatus
    native_operator: VerificationStatus
    verified_artifacts: tuple[VerifiedArtifact, ...] = field(repr=False)
    errors: tuple[ArtifactRegistryError, ...] = field(repr=False)

    def as_public_dict(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "artifacts": [status.as_public_dict() for status in self.artifacts],
            "tokenizer": self.tokenizer.as_public_dict(),
            "repository_assets": self.repository_assets.as_public_dict(),
            "runtime": self.runtime.as_public_dict(),
            "native_operator": self.native_operator.as_public_dict(),
        }


class ArtifactRegistry:
    """Resolve only manifest-owned artifacts and aggregate startup readiness."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        artifact_root: str | Path,
        tokenizer_root: str | Path,
        repository_root: str | Path,
        checkpoint_inspector: CheckpointInspector | None = None,
        runtime_versions: Mapping[str, str] | None = None,
        native_operator_probe: Callable[[], bool] | None = None,
    ):
        self._manifest_path = Path(manifest_path).expanduser().resolve()
        self._artifact_root = Path(artifact_root).expanduser().resolve()
        self._tokenizer_root = Path(tokenizer_root).expanduser().resolve()
        self._repository_root = Path(repository_root).expanduser().resolve()
        self._inspector = checkpoint_inspector or TorchCheckpointInspector()
        self._runtime_versions = (
            dict(runtime_versions) if runtime_versions is not None else None
        )
        self._native_operator_probe = native_operator_probe or _probe_native_operator
        self._cache: dict[str, VerifiedArtifact] = {}

        try:
            load_manifest(self._manifest_path)
            payload = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except (ManifestValidationError, OSError, json.JSONDecodeError) as error:
            raise RegistryConfigurationError(
                "manifest",
                f"manifest validation failed ({type(error).__name__})",
            ) from error
        self._payload: dict[str, Any] = payload
        self._artifact_specs = {
            artifact["id"]: artifact for artifact in payload["artifacts"]
        }

    def resolve(self, model_id: str) -> VerifiedArtifact:
        """Return verified metadata after same-handle hashing and inspection."""

        spec = self._artifact_specs.get(model_id)
        if spec is None:
            raise ArtifactNotFoundError(model_id, "model id is not manifest-owned")
        path = self._resolve_inside(
            self._artifact_root,
            spec["filename"],
            model_id,
        )
        if path.is_symlink():
            raise ArtifactIntegrityError(model_id, "symbolic-link artifacts are forbidden")

        descriptor = _open_readonly(path, model_id)

        with os.fdopen(descriptor, "rb") as stream:
            before = _FileFingerprint.from_stat(os.fstat(stream.fileno()))
            cached = self._cache.get(model_id)
            if cached is not None and cached._fingerprint == before:
                return cached
            if before.size != spec["size_bytes"]:
                error_type = ArtifactChangedError if cached else ArtifactIntegrityError
                raise error_type(model_id, "artifact size does not match the manifest")
            observed_hash = _sha256_stream(stream)
            if observed_hash != spec["sha256"]:
                error_type = ArtifactChangedError if cached else ArtifactIntegrityError
                raise error_type(model_id, "artifact checksum does not match the manifest")
            stream.seek(0)
            summary = self._inspector.inspect(stream, spec)
            after = _FileFingerprint.from_stat(os.fstat(stream.fileno()))
            if after != before:
                raise ArtifactChangedError(model_id, "artifact changed during verification")

        self._validate_checkpoint_summary(spec, summary)
        verified = VerifiedArtifact(
            id=model_id,
            role=spec["role"],
            filename=spec["filename"],
            size_bytes=spec["size_bytes"],
            sha256=spec["sha256"],
            checkpoint=summary,
            provenance_source=spec["provenance"]["source"],
            provenance_custody=spec["provenance"]["custody"],
            repository_revision=spec["provenance"]["repository_revision"],
            strict_load_verified=spec["checkpoint"]["strict_load_verified"],
            weights_license_status=spec["license"]["weights_status"],
            redistribution_status=spec["license"]["redistribution_status"],
            _path=path,
            _fingerprint=after,
        )
        self._cache[model_id] = verified
        return verified

    def verify_all(self) -> ArtifactReport:
        """Return aggregate readiness without leaking paths or full hashes."""

        errors: list[ArtifactRegistryError] = []
        verified: list[VerifiedArtifact] = []
        artifact_statuses: list[VerificationStatus] = []
        for model_id in self._artifact_specs:
            try:
                artifact = self.resolve(model_id)
                verified.append(artifact)
                artifact_statuses.append(VerificationStatus(model_id, True))
            except ArtifactRegistryError as error:
                errors.append(error)
                artifact_statuses.append(
                    VerificationStatus(model_id, False, error.code)
                )

        tokenizer_status = self._capture_status(
            "tokenizer",
            self._verify_tokenizer,
            errors,
        )
        repository_status = self._capture_status(
            "repository_assets",
            self._verify_repository_assets,
            errors,
        )
        runtime_status = self._capture_status(
            "runtime",
            self._verify_runtime,
            errors,
        )
        operator_status = self._capture_status(
            "native_operator",
            self._verify_native_operator,
            errors,
        )
        return ArtifactReport(
            ready=not errors,
            artifacts=tuple(artifact_statuses),
            tokenizer=tokenizer_status,
            repository_assets=repository_status,
            runtime=runtime_status,
            native_operator=operator_status,
            verified_artifacts=tuple(verified),
            errors=tuple(errors),
        )

    def _validate_checkpoint_summary(
        self,
        artifact_spec: Mapping[str, Any],
        summary: CheckpointSummary,
    ) -> None:
        model_id = str(artifact_spec["id"])
        expected = artifact_spec["checkpoint"]
        scalar_checks = {
            "container": (summary.container, expected["container"]),
            "state-dict root": (summary.state_dict_key, expected["state_dict_key"]),
            "key normalization": (
                summary.key_normalization,
                expected["key_normalization"],
            ),
            "state tensor count": (
                summary.state_tensor_count,
                expected["state_tensor_count"],
            ),
            "tensor element count": (
                summary.tensor_element_count,
                expected["tensor_element_count"],
            ),
            "tensor bytes": (summary.tensor_bytes, expected["tensor_bytes"]),
        }
        for label, (observed, wanted) in scalar_checks.items():
            if observed != wanted:
                raise ArtifactStructureError(model_id, f"{label} does not match")
        if summary.dtypes != frozenset(expected["dtypes"]):
            raise ArtifactStructureError(model_id, "checkpoint dtypes do not match")
        if "dtype_counts" in expected and summary.dtype_counts != expected["dtype_counts"]:
            raise ArtifactStructureError(model_id, "checkpoint dtype counts do not match")
        if summary.group_counts != expected["required_key_groups"]:
            raise ArtifactStructureError(model_id, "required state-key groups do not match")
        expected_shapes = {
            name: tuple(shape)
            for name, shape in expected["required_tensor_shapes"].items()
        }
        if summary.tensor_shapes != expected_shapes:
            raise ArtifactStructureError(model_id, "required tensor shapes do not match")
        if (
            expected.get("checkpoint_aliases_equal") is True
            and summary.aliases_equal is not True
        ):
            raise ArtifactStructureError(model_id, "checkpoint aliases differ")

    def _verify_tokenizer(self) -> None:
        tokenizer = self._payload["tokenizer"]
        for record in tokenizer["files"]:
            path = self._resolve_inside(
                self._tokenizer_root,
                record["filename"],
                "tokenizer",
            )
            self._verify_plain_file(
                path,
                int(record["size_bytes"]),
                str(record["sha256"]),
                "tokenizer",
            )

    def _verify_repository_assets(self) -> None:
        for record in self._payload["repository_assets"]:
            path = self._resolve_inside(
                self._repository_root,
                record["path"],
                "repository_assets",
            )
            self._verify_plain_file(
                path,
                int(record["size_bytes"]),
                str(record["sha256"]),
                "repository_assets",
            )

    def _verify_runtime(self) -> None:
        observed = self._runtime_versions or _collect_runtime_versions()
        lane = self._payload["runtime_lane"]
        expected = {
            "python": lane["python"],
            "torch": lane["torch"],
            "torchvision": lane["torchvision"],
            "transformers": lane["transformers"],
            "numpy": lane["numpy"],
            "cuda_runtime": lane["cuda_runtime"],
        }
        mismatches = [
            name
            for name, value in expected.items()
            if observed.get(name) != value
        ]
        if mismatches:
            raise RuntimeCompatibilityError(
                "runtime",
                "pinned runtime fields differ: " + ", ".join(sorted(mismatches)),
            )

    def _verify_native_operator(self) -> None:
        try:
            valid = self._native_operator_probe()
        except Exception as error:
            raise NativeOperatorError(
                "native_operator",
                f"operator probe failed ({type(error).__name__})",
            ) from error
        if valid is not True:
            raise NativeOperatorError(
                "native_operator",
                "operator import or CUDA availability check failed",
            )

    def _verify_plain_file(
        self,
        path: Path,
        expected_size: int,
        expected_hash: str,
        subject: str,
    ) -> None:
        if path.is_symlink():
            raise ArtifactIntegrityError(subject, "symbolic-link assets are forbidden")
        descriptor = _open_readonly(path, subject)
        try:
            with os.fdopen(descriptor, "rb") as stream:
                before = _FileFingerprint.from_stat(os.fstat(stream.fileno()))
                if before.size != expected_size:
                    raise ArtifactIntegrityError(subject, "asset size does not match")
                observed_hash = _sha256_stream(stream)
                after = _FileFingerprint.from_stat(os.fstat(stream.fileno()))
        except OSError as error:
            raise ArtifactIntegrityError(
                subject,
                f"asset cannot be opened ({type(error).__name__})",
            ) from error
        if observed_hash != expected_hash:
            raise ArtifactIntegrityError(subject, "asset checksum does not match")
        if before != after:
            raise ArtifactChangedError(subject, "asset changed during verification")

    def _resolve_inside(self, root: Path, relative: str, subject: str) -> Path:
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or "\\" in relative:
            raise RegistryConfigurationError(subject, "configured path escapes its root")
        candidate = root.joinpath(*pure.parts)
        if not candidate.is_relative_to(root):
            raise RegistryConfigurationError(subject, "configured path escapes its root")
        current = root
        for part in pure.parts:
            current = current / part
            if current.is_symlink():
                raise ArtifactIntegrityError(
                    subject,
                    "symbolic-link assets are forbidden",
                )
        return candidate

    @staticmethod
    def _capture_status(
        subject: str,
        check: Callable[[], None],
        errors: list[ArtifactRegistryError],
    ) -> VerificationStatus:
        try:
            check()
        except ArtifactRegistryError as error:
            errors.append(error)
            return VerificationStatus(subject, False, error.code)
        return VerificationStatus(subject, True)


def _is_tensor_mapping(value: Any, torch: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and bool(value)
        and all(isinstance(key, str) for key in value)
        and all(torch.is_tensor(tensor) for tensor in value.values())
    )


def _sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _open_readonly(path: Path, subject: str) -> int:
    try:
        return os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
    except FileNotFoundError as error:
        raise ArtifactNotFoundError(subject, "required local asset is missing") from error
    except OSError as error:
        raise ArtifactIntegrityError(
            subject,
            f"asset cannot be opened ({type(error).__name__})",
        ) from error


def _collect_runtime_versions() -> dict[str, str]:
    result = {"python": platform.python_version()}
    distributions = {
        "torch": "torch",
        "torchvision": "torchvision",
        "transformers": "transformers",
        "numpy": "numpy",
    }
    for field_name, distribution in distributions.items():
        try:
            result[field_name] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            result[field_name] = "missing"
    try:
        torch = importlib.import_module("torch")
        result["cuda_runtime"] = str(torch.version.cuda or "missing")
    except Exception:
        result["cuda_runtime"] = "missing"
    return result


def _probe_native_operator() -> bool:
    torch = importlib.import_module("torch")
    if not torch.cuda.is_available():
        return False
    importlib.import_module("MultiScaleDeformableAttention")
    return True
