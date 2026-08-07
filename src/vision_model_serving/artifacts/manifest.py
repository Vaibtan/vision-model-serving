"""Load and validate the repository-owned model-artifact evidence manifest.

This module validates metadata only. It deliberately does not resolve artifact
paths, hash checkpoint files, or deserialize PyTorch checkpoints; those runtime
responsibilities belong to the ArtifactRegistry implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence


_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
_ROLES = ("detector", "classifier")
_TOKENIZER_FILES = {
    "config.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
}


class ManifestValidationError(ValueError):
    """Raised when a manifest violates the versioned artifact contract."""

    def __init__(self, issues: Sequence[str]):
        self.issues = tuple(issues)
        detail = "\n".join(f"- {issue}" for issue in self.issues)
        super().__init__(f"Artifact manifest is invalid:\n{detail}")


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    """The serving-relevant, validated view of one model artifact."""

    id: str
    role: str
    filename: str
    size_bytes: int
    sha256: str
    strict_load_verified: bool
    user_upload_allowed: bool
    semantics_status: str
    class_names: tuple[str, ...] | None
    decision_threshold: float | None
    weights_license_status: str
    redistribution_status: str


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    """Validated metadata needed to identify the two-model serving contract."""

    manifest_id: str
    pipeline_stages: tuple[str, ...]
    revisions: dict[str, str]
    artifacts: tuple[ArtifactRecord, ...]
    evidence_archive_sha256: str

    def inventory(self) -> dict[str, object]:
        """Return a stable, JSON-serializable inventory summary."""

        return {
            "status": "valid",
            "manifest_id": self.manifest_id,
            "pipeline_stages": list(self.pipeline_stages),
            "revisions": dict(self.revisions),
            "evidence_archive_sha256": self.evidence_archive_sha256,
            "artifacts": [
                {
                    "id": artifact.id,
                    "role": artifact.role,
                    "filename": artifact.filename,
                    "size_bytes": artifact.size_bytes,
                    "sha256": artifact.sha256,
                    "strict_load_verified": artifact.strict_load_verified,
                    "user_upload_allowed": artifact.user_upload_allowed,
                    "semantics_status": artifact.semantics_status,
                    "class_names": (
                        list(artifact.class_names)
                        if artifact.class_names is not None
                        else None
                    ),
                    "decision_threshold": artifact.decision_threshold,
                    "weights_license_status": artifact.weights_license_status,
                    "redistribution_status": artifact.redistribution_status,
                }
                for artifact in self.artifacts
            ],
        }


def load_manifest(path: str | Path) -> ArtifactManifest:
    """Load and validate manifest metadata without touching model files."""

    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ManifestValidationError(
            [f"manifest: cannot read {manifest_path}: {error.strerror or error}"]
        ) from error
    except json.JSONDecodeError as error:
        raise ManifestValidationError(
            [
                "manifest: invalid JSON at "
                f"line {error.lineno}, column {error.colno}: {error.msg}"
            ]
        ) from error

    issues: list[str] = []
    root = _mapping(payload, "manifest", issues)
    _validate_root(root, issues)
    if issues:
        raise ManifestValidationError(issues)

    artifacts = tuple(_artifact_record(item) for item in root["artifacts"])
    return ArtifactManifest(
        manifest_id=root["manifest_id"],
        pipeline_stages=tuple(root["pipeline"]["stages"]),
        revisions=dict(root["revisions"]),
        artifacts=artifacts,
        evidence_archive_sha256=root["evidence"]["archive"]["sha256"],
    )


def _validate_root(root: Mapping[str, Any], issues: list[str]) -> None:
    if root.get("schema_version") != 1:
        issues.append("schema_version: expected integer 1")
    _required_string(root, "manifest_id", "manifest_id", issues)

    evidence = _child_mapping(root, "evidence", "evidence", issues)
    archive = _child_mapping(evidence, "archive", "evidence.archive", issues)
    _filename(archive.get("filename"), "evidence.archive.filename", issues)
    _sha256(archive.get("sha256"), "evidence.archive.sha256", issues)
    reference = _child_mapping(
        evidence, "reference_result", "evidence.reference_result", issues
    )
    _relative_path(reference.get("path"), "evidence.reference_result.path", issues)
    _sha256(reference.get("sha256"), "evidence.reference_result.sha256", issues)

    revisions = _child_mapping(root, "revisions", "revisions", issues)
    required_revisions = {
        "focalnet_dino",
        "mmbcd",
        "dino",
        "roberta_tokenizer",
    }
    if set(revisions) != required_revisions:
        issues.append(
            "revisions: expected exactly focalnet_dino, mmbcd, dino, "
            "and roberta_tokenizer"
        )
    for name in required_revisions:
        _git_commit(revisions.get(name), f"revisions.{name}", issues)

    runtime = _child_mapping(root, "runtime_lane", "runtime_lane", issues)
    for name in ("device", "python", "torch", "cuda_runtime"):
        _required_string(runtime, name, f"runtime_lane.{name}", issues)

    artifacts_value = root.get("artifacts")
    artifacts = _list(artifacts_value, "artifacts", issues)
    if len(artifacts) != 2:
        issues.append("artifacts: expected exactly one detector and one classifier")

    artifact_maps: list[Mapping[str, Any]] = []
    for index, value in enumerate(artifacts):
        artifact = _mapping(value, f"artifacts[{index}]", issues)
        artifact_maps.append(artifact)
        _validate_artifact(artifact, index, issues)

    ids = [artifact.get("id") for artifact in artifact_maps]
    if len(ids) != len(set(_hashable(value) for value in ids)):
        issues.append("artifacts: artifact ids must be unique")
    roles = [artifact.get("role") for artifact in artifact_maps]
    if sorted(value for value in roles if isinstance(value, str)) != sorted(_ROLES):
        issues.append("artifacts: roles must be exactly detector and classifier")

    pipeline = _child_mapping(root, "pipeline", "pipeline", issues)
    if pipeline.get("kind") != "detector_then_classifier":
        issues.append("pipeline.kind: expected detector_then_classifier")
    stages = _list(pipeline.get("stages"), "pipeline.stages", issues)
    expected_stages = [
        next(
            (
                artifact.get("id")
                for artifact in artifact_maps
                if artifact.get("role") == role
            ),
            None,
        )
        for role in _ROLES
    ]
    if stages != expected_stages:
        issues.append(
            "pipeline.stages: must name the detector then classifier artifact ids"
        )
    _validate_proposal_contract(pipeline, issues)

    _validate_tokenizer(root, revisions, issues)
    _validate_repository_assets(root, issues)


def _validate_artifact(
    artifact: Mapping[str, Any], index: int, issues: list[str]
) -> None:
    base = f"artifacts[{index}]"
    _required_string(artifact, "id", f"{base}.id", issues)
    role = artifact.get("role")
    if role not in _ROLES:
        issues.append(f"{base}.role: expected detector or classifier")
    _filename(artifact.get("filename"), f"{base}.filename", issues)
    _positive_int(artifact.get("size_bytes"), f"{base}.size_bytes", issues)
    _sha256(artifact.get("sha256"), f"{base}.sha256", issues)

    provenance = _child_mapping(
        artifact, "provenance", f"{base}.provenance", issues
    )
    _required_string(provenance, "source", f"{base}.provenance.source", issues)
    _required_string(provenance, "custody", f"{base}.provenance.custody", issues)

    trust = _child_mapping(artifact, "trust", f"{base}.trust", issues)
    if trust.get("checksum_pinned") is not True:
        issues.append(f"{base}.trust.checksum_pinned: must be true")
    if trust.get("load_authorized") is not True:
        issues.append(f"{base}.trust.load_authorized: must be true")
    if trust.get("user_upload_allowed") is not False:
        issues.append(f"{base}.trust.user_upload_allowed: must be false")

    license_info = _child_mapping(artifact, "license", f"{base}.license", issues)
    weights_status = license_info.get("weights_status")
    if weights_status not in {"verified", "unknown"}:
        issues.append(f"{base}.license.weights_status: expected verified or unknown")
    redistribution = license_info.get("redistribution_status")
    if redistribution != "not_authorized":
        issues.append(
            f"{base}.license.redistribution_status: must be not_authorized"
        )

    checkpoint = _child_mapping(
        artifact, "checkpoint", f"{base}.checkpoint", issues
    )
    if checkpoint.get("container") not in {"wrapped_state_dict", "raw_state_dict"}:
        issues.append(
            f"{base}.checkpoint.container: expected wrapped_state_dict or raw_state_dict"
        )
    state_dict_key = checkpoint.get("state_dict_key")
    if checkpoint.get("container") == "wrapped_state_dict":
        if not isinstance(state_dict_key, str) or not state_dict_key:
            issues.append(
                f"{base}.checkpoint.state_dict_key: required for wrapped_state_dict"
            )
    elif state_dict_key is not None:
        issues.append(
            f"{base}.checkpoint.state_dict_key: must be null for raw_state_dict"
        )
    if checkpoint.get("key_normalization") not in {"none", "strip_module_prefix"}:
        issues.append(
            f"{base}.checkpoint.key_normalization: expected none or strip_module_prefix"
        )
    for name in ("state_tensor_count", "tensor_element_count", "tensor_bytes"):
        _positive_int(checkpoint.get(name), f"{base}.checkpoint.{name}", issues)
    dtypes = _list(checkpoint.get("dtypes"), f"{base}.checkpoint.dtypes", issues)
    if not dtypes or any(not isinstance(dtype, str) or not dtype for dtype in dtypes):
        issues.append(f"{base}.checkpoint.dtypes: expected non-empty dtype names")
    if checkpoint.get("strict_load_verified") is not True:
        issues.append(f"{base}.checkpoint.strict_load_verified: must be true")
    _required_string(
        checkpoint,
        "strict_load_evidence",
        f"{base}.checkpoint.strict_load_evidence",
        issues,
    )
    if role == "detector":
        _validate_detector_checkpoint(checkpoint, base, issues)
    elif role == "classifier":
        _validate_classifier_checkpoint(checkpoint, base, issues)

    semantics = _child_mapping(
        artifact, "semantics", f"{base}.semantics", issues
    )
    if semantics.get("status") != "unverified":
        issues.append(f"{base}.semantics.status: must be unverified")
    if semantics.get("class_names") is not None:
        issues.append(
            f"{base}.semantics.class_names must be null while semantics are unverified"
        )
    if semantics.get("decision_threshold") is not None:
        issues.append(
            f"{base}.semantics.decision_threshold must be null while semantics are unverified"
        )
    if semantics.get("medical_validation") is not False:
        issues.append(
            f"{base}.semantics.medical_validation must be false while semantics are unverified"
        )


def _validate_detector_checkpoint(
    checkpoint: Mapping[str, Any], base: str, issues: list[str]
) -> None:
    checkpoint_path = f"{base}.checkpoint"
    if checkpoint.get("serving_backbone_embedded") is not True:
        issues.append(f"{checkpoint_path}.serving_backbone_embedded: must be true")
    if set(checkpoint.get("dtypes", [])) != {"float32"}:
        issues.append(f"{checkpoint_path}.dtypes: detector must be float32")

    group_counts = _required_groups(
        checkpoint,
        checkpoint_path,
        {
            "backbone",
            "transformer",
            "bbox_embed",
            "input_proj",
            "class_embed",
            "label_enc",
        },
        issues,
    )
    state_tensor_count = checkpoint.get("state_tensor_count")
    if group_counts and sum(group_counts.values()) != state_tensor_count:
        issues.append(
            f"{checkpoint_path}.required_key_groups: counts must sum to "
            "state_tensor_count"
        )

    output = _child_mapping(
        checkpoint,
        "output_contract",
        f"{checkpoint_path}.output_contract",
        issues,
    )
    expected_output: dict[str, object] = {
        "pred_logits_shape": [1, 900, 1],
        "pred_boxes_shape": [1, 900, 4],
        "box_format": "normalized_cxcywh",
        "num_select": 300,
        "internal_nms_iou_threshold": -1,
    }
    _exact_values(output, f"{checkpoint_path}.output_contract", expected_output, issues)

    derived = _child_mapping(
        checkpoint,
        "derived_inference_checkpoint",
        f"{checkpoint_path}.derived_inference_checkpoint",
        issues,
    )
    _filename(
        derived.get("filename"),
        f"{checkpoint_path}.derived_inference_checkpoint.filename",
        issues,
    )
    _sha256(
        derived.get("sha256"),
        f"{checkpoint_path}.derived_inference_checkpoint.sha256",
        issues,
    )
    _positive_int(
        derived.get("state_tensor_count"),
        f"{checkpoint_path}.derived_inference_checkpoint.state_tensor_count",
        issues,
    )
    if derived.get("state_tensor_count") != state_tensor_count:
        issues.append(
            f"{checkpoint_path}.derived_inference_checkpoint.state_tensor_count: "
            "must match the supplied checkpoint"
        )
    if derived.get("retention") != "not_retained_in_repository":
        issues.append(
            f"{checkpoint_path}.derived_inference_checkpoint.retention: "
            "must be not_retained_in_repository"
        )


def _validate_classifier_checkpoint(
    checkpoint: Mapping[str, Any], base: str, issues: list[str]
) -> None:
    checkpoint_path = f"{base}.checkpoint"
    if set(checkpoint.get("dtypes", [])) != {"float32", "int64"}:
        issues.append(
            f"{checkpoint_path}.dtypes: classifier must record float32 and int64"
        )
    state_tensor_count = checkpoint.get("state_tensor_count")
    module_count = checkpoint.get("module_prefixed_key_count")
    _positive_int(
        module_count, f"{checkpoint_path}.module_prefixed_key_count", issues
    )
    if module_count != state_tensor_count:
        issues.append(
            f"{checkpoint_path}.module_prefixed_key_count: must match "
            "state_tensor_count"
        )

    dtype_counts = _child_mapping(
        checkpoint, "dtype_counts", f"{checkpoint_path}.dtype_counts", issues
    )
    if set(dtype_counts) != {"float32", "int64"}:
        issues.append(
            f"{checkpoint_path}.dtype_counts: expected exactly float32 and int64"
        )
    for name in ("float32", "int64"):
        _positive_int(
            dtype_counts.get(name), f"{checkpoint_path}.dtype_counts.{name}", issues
        )
    if all(isinstance(dtype_counts.get(name), int) for name in ("float32", "int64")):
        if sum(dtype_counts[name] for name in ("float32", "int64")) != state_tensor_count:
            issues.append(
                f"{checkpoint_path}.dtype_counts: counts must sum to "
                "state_tensor_count"
            )

    if checkpoint.get("checkpoint_aliases_equal") is not True:
        issues.append(f"{checkpoint_path}.checkpoint_aliases_equal: must be true")
    _required_groups(
        checkpoint,
        checkpoint_path,
        {
            "image_encoder",
            "image_projection",
            "text_encoder",
            "text_projection",
            "cross_attention",
            "classifier",
        },
        issues,
    )

    output = _child_mapping(
        checkpoint,
        "output_contract",
        f"{checkpoint_path}.output_contract",
        issues,
    )
    expected_output: dict[str, object] = {
        "logits_shape": [1, 2],
        "fused_embeddings_shape": [1, 768],
        "roi_count": 8,
        "text_max_length": 90,
    }
    _exact_values(output, f"{checkpoint_path}.output_contract", expected_output, issues)


def _required_groups(
    checkpoint: Mapping[str, Any],
    checkpoint_path: str,
    expected_names: set[str],
    issues: list[str],
) -> dict[str, int]:
    groups = _child_mapping(
        checkpoint,
        "required_key_groups",
        f"{checkpoint_path}.required_key_groups",
        issues,
    )
    if set(groups) != expected_names:
        issues.append(
            f"{checkpoint_path}.required_key_groups: expected exactly "
            + ", ".join(sorted(expected_names))
        )
    valid_counts: dict[str, int] = {}
    for name in expected_names:
        value = groups.get(name)
        _positive_int(
            value, f"{checkpoint_path}.required_key_groups.{name}", issues
        )
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            valid_counts[name] = value
    return valid_counts


def _exact_values(
    actual: Mapping[str, Any],
    path: str,
    expected: Mapping[str, object],
    issues: list[str],
) -> None:
    for name, expected_value in expected.items():
        if actual.get(name) != expected_value:
            issues.append(f"{path}.{name}: expected {expected_value!r}")


def _validate_proposal_contract(
    pipeline: Mapping[str, Any], issues: list[str]
) -> None:
    contract = _child_mapping(
        pipeline, "proposal_contract", "pipeline.proposal_contract", issues
    )
    expected: dict[str, object] = {
        "format": "normalized_cxcywh_confidence",
        "detector_output_count": 300,
        "nms_iou_threshold": 0.1,
        "nms_comparison": "strictly_greater",
        "classifier_roi_count": 8,
        "insufficient_proposal_policy": "duplicate_existing",
        "empty_proposal_policy": "fail_closed",
    }
    for name, expected_value in expected.items():
        if contract.get(name) != expected_value:
            issues.append(
                f"pipeline.proposal_contract.{name}: expected {expected_value!r}"
            )


def _validate_tokenizer(
    root: Mapping[str, Any], revisions: Mapping[str, Any], issues: list[str]
) -> None:
    tokenizer = _child_mapping(root, "tokenizer", "tokenizer", issues)
    _required_string(tokenizer, "id", "tokenizer.id", issues)
    _required_string(tokenizer, "repository", "tokenizer.repository", issues)
    revision = tokenizer.get("revision")
    _git_commit(revision, "tokenizer.revision", issues)
    if revision != revisions.get("roberta_tokenizer"):
        issues.append("tokenizer.revision: must match revisions.roberta_tokenizer")
    if tokenizer.get("local_files_only") is not True:
        issues.append("tokenizer.local_files_only: must be true")

    files = _list(tokenizer.get("files"), "tokenizer.files", issues)
    names: set[object] = set()
    for index, value in enumerate(files):
        entry = _mapping(value, f"tokenizer.files[{index}]", issues)
        name = entry.get("filename")
        _filename(name, f"tokenizer.files[{index}].filename", issues)
        names.add(_hashable(name))
        _positive_int(
            entry.get("size_bytes"), f"tokenizer.files[{index}].size_bytes", issues
        )
        _sha256(entry.get("sha256"), f"tokenizer.files[{index}].sha256", issues)
    if names != _TOKENIZER_FILES:
        issues.append(
            "tokenizer.files: expected config.json, merges.txt, tokenizer.json, "
            "tokenizer_config.json, and vocab.json"
        )


def _validate_repository_assets(root: Mapping[str, Any], issues: list[str]) -> None:
    assets = _list(root.get("repository_assets"), "repository_assets", issues)
    kinds: set[object] = set()
    ids: set[object] = set()
    for index, value in enumerate(assets):
        asset = _mapping(value, f"repository_assets[{index}]", issues)
        base = f"repository_assets[{index}]"
        asset_id = asset.get("id")
        _required_string(asset, "id", f"{base}.id", issues)
        ids.add(_hashable(asset_id))
        kind = asset.get("kind")
        if kind not in {"config", "patch"}:
            issues.append(f"{base}.kind: expected config or patch")
        kinds.add(_hashable(kind))
        _relative_path(asset.get("path"), f"{base}.path", issues)
        _positive_int(asset.get("size_bytes"), f"{base}.size_bytes", issues)
        _sha256(asset.get("sha256"), f"{base}.sha256", issues)
        if kind == "patch":
            _git_commit(
                asset.get("applies_to_revision"),
                f"{base}.applies_to_revision",
                issues,
            )
        if "l4_validated_sha256" in asset:
            _sha256(
                asset.get("l4_validated_sha256"),
                f"{base}.l4_validated_sha256",
                issues,
            )
            _required_string(asset, "equivalence", f"{base}.equivalence", issues)
    if len(ids) != len(assets):
        issues.append("repository_assets: ids must be unique")
    if not {"config", "patch"}.issubset(kinds):
        issues.append("repository_assets: expected at least one config and one patch")


def _artifact_record(artifact: Mapping[str, Any]) -> ArtifactRecord:
    semantics = artifact["semantics"]
    class_names = semantics["class_names"]
    return ArtifactRecord(
        id=artifact["id"],
        role=artifact["role"],
        filename=artifact["filename"],
        size_bytes=artifact["size_bytes"],
        sha256=artifact["sha256"],
        strict_load_verified=artifact["checkpoint"]["strict_load_verified"],
        user_upload_allowed=artifact["trust"]["user_upload_allowed"],
        semantics_status=semantics["status"],
        class_names=tuple(class_names) if class_names is not None else None,
        decision_threshold=semantics["decision_threshold"],
        weights_license_status=artifact["license"]["weights_status"],
        redistribution_status=artifact["license"]["redistribution_status"],
    )


def _mapping(value: Any, path: str, issues: list[str]) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    issues.append(f"{path}: expected an object")
    return {}


def _child_mapping(
    parent: Mapping[str, Any], key: str, path: str, issues: list[str]
) -> Mapping[str, Any]:
    return _mapping(parent.get(key), path, issues)


def _list(value: Any, path: str, issues: list[str]) -> list[Any]:
    if isinstance(value, list):
        return value
    issues.append(f"{path}: expected an array")
    return []


def _required_string(
    parent: Mapping[str, Any], key: str, path: str, issues: list[str]
) -> None:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        issues.append(f"{path}: expected a non-empty string")


def _sha256(value: Any, path: str, issues: list[str]) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        issues.append(f"{path}: expected a lowercase SHA-256 hex digest")


def _git_commit(value: Any, path: str, issues: list[str]) -> None:
    if not isinstance(value, str) or _GIT_COMMIT.fullmatch(value) is None:
        issues.append(f"{path}: expected a full lowercase Git commit")


def _positive_int(value: Any, path: str, issues: list[str]) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        issues.append(f"{path}: expected a positive integer")


def _filename(value: Any, path: str, issues: list[str]) -> None:
    if (
        not isinstance(value, str)
        or not value
        or PurePosixPath(value).name != value
        or "\\" in value
    ):
        issues.append(f"{path}: expected a basename without directory components")


def _relative_path(value: Any, path: str, issues: list[str]) -> None:
    if not isinstance(value, str) or not value:
        issues.append(f"{path}: expected a repository-relative path")
        return
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or ".." in candidate.parts or "\\" in value:
        issues.append(f"{path}: expected a safe repository-relative POSIX path")


def _hashable(value: Any) -> object:
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value
