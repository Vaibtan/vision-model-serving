"""Verify the immutable L4 evidence and compare repository-adapter outputs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import tarfile
from typing import Any, Sequence


REFERENCE_ARCHIVE_SHA256 = (
    "5a1311b121edd6c03feb279e5aa7170be5a545d859a1e1fc2f70fc32252f097f"
)
REFERENCE_DICOM_SHA256 = (
    "9f70081672a460f29231bb471e8a9e26dd3ed26a2ebbd91c064e575e7842a19c"
)
REFERENCE_PREPROCESSED_ARRAY_SHA256 = (
    "97fa0f80a696ce7f822c1681a8c3f7c072da9262b2bd91239c9f1637eaf68552"
)
REFERENCE_DETECTOR_PREDICTION_SHA256 = (
    "4cdd09d986702e8839acff8d7517a63f263ca2a01b0607d78d6b2086c886a9a5"
)
REFERENCE_MMBCD_INPUT_SHA256 = (
    "89cda9694e3696f63eb70706a6ae4cc2dd16e9be214f27ba6f106479eac90155"
)
REFERENCE_MMBCD_PREDICTION_SHA256 = (
    "43ec1c4593c0549510098ea082ea7092c7fd5631c95d8b912ecf31633185899b"
)
REFERENCE_MMBCD_LOGITS = [[3.582984209060669, -4.886208534240723]]
REFERENCE_MMBCD_PROBABILITIES = [[0.9997902512550354, 0.0002097902470268309]]
REFERENCE_COMMITS = {
    "environment/focalnet-commit.txt": (
        "23901e021dc6ec8f66bad47983f45a25574452cc"
    ),
    "environment/mmbcd-commit.txt": "14ac5e099c79253b01e0885d2ebefa6f86cfd8f0",
    "environment/dino-commit.txt": "7c446df5b9f45747937fb0d72314eb9f7b66930a",
}
REFERENCE_MODEL_HASHES = {
    "67a7b0cd787a3aaba199cf1ff82ed2934c33ffe37544473379d7a837ab1637b4",
    "a7fed981c7309d12c19532624e148b45087462c56568797cb62b60055bb62f04",
    "2264351216f9fb4945af35e300459ff4ce2e7f5445519348024f3bf1eec721a4",
}
REFERENCE_EXTENSION_SHA256 = (
    "f8ec513bbab7d3ae134b2b4c7ae934cc45995e96d58dd4d53810268784b86147"
)
REFERENCE_PATCH_SHA256 = (
    "10718bf12f2ac98c15014f14a6578ccb97f604a3f46a6db8724f4c9aa34f319c"
)
REQUIRED_FILES = {
    "visuals/upstream-1024.png",
    "visuals/overlay-top8-1024.png",
    "visuals/roi-montage.png",
    "visuals/overlay-top8-original.png",
    "environment/conda-packages.json",
    "environment/nvcc.txt",
    "environment/dino-commit.txt",
    "environment/focalnet-commit.txt",
    "environment/python.txt",
    "environment/gpu.txt",
    "environment/mmbcd-commit.txt",
    "environment/python-packages.json",
    "environment/cuda-extension.sha256",
    "environment/model-artifacts.sha256",
    "manifests/detector-inference-manifest.json",
    "manifests/preprocess-manifest.json",
    "manifests/dicom-manifest.json",
    "manifests/mmbcd-inference-manifest.json",
    "manifests/mmbcd-input-manifest.json",
    "patches/focalnet-pytorch-2.8-compat.patch",
    "bundles/detections-top8.txt",
    "bundles/mmbcd-outputs.npz",
    "bundles/mmbcd-inputs.npz",
}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_JSON_BYTES = 1024 * 1024
_MAX_MEMBER_BYTES = 64 * 1024 * 1024
_MAX_MEMBERS = 1000


class EvidenceVerificationError(RuntimeError):
    """Raised when downloaded evidence is unsafe, incomplete, or inconsistent."""


@dataclass(frozen=True, slots=True)
class EvidenceSummary:
    archive_sha256: str
    member_count: int
    detector_prediction_sha256: str
    mmbcd_prediction_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "archive_sha256": self.archive_sha256,
            "member_count": self.member_count,
            "detector_prediction_sha256": self.detector_prediction_sha256,
            "mmbcd_prediction_sha256": self.mmbcd_prediction_sha256,
            "status": "passed",
        }


@dataclass(frozen=True, slots=True)
class BoxComparison:
    matched_pairs: tuple[tuple[int, int], ...]
    unmatched_expected: tuple[int, ...]
    unmatched_observed: tuple[int, ...]
    max_abs_difference: float

    @property
    def equivalent(self) -> bool:
        return not self.unmatched_expected and not self.unmatched_observed


def compare_box_records(
    expected: Sequence[Sequence[float]],
    observed: Sequence[Sequence[float]],
    *,
    absolute_tolerance: float,
) -> BoxComparison:
    """Compare box/score records using deterministic permutation-aware matching."""

    if not math.isfinite(absolute_tolerance) or absolute_tolerance < 0:
        raise ValueError("absolute_tolerance must be finite and non-negative")
    expected_rows = _numeric_rows(expected, "expected")
    observed_rows = _numeric_rows(observed, "observed")
    widths = {len(row) for row in (*expected_rows, *observed_rows)}
    if len(widths) > 1:
        raise ValueError("all expected and observed records must have equal width")

    candidates: list[list[tuple[float, int]]] = []
    for expected_row in expected_rows:
        row_candidates: list[tuple[float, int]] = []
        for observed_index, observed_row in enumerate(observed_rows):
            difference = max(
                (abs(left - right) for left, right in zip(expected_row, observed_row)),
                default=0.0,
            )
            if difference <= absolute_tolerance:
                row_candidates.append((difference, observed_index))
        candidates.append(sorted(row_candidates))

    observed_matches: dict[int, int] = {}

    def assign(expected_index: int, visited: set[int]) -> bool:
        for _, observed_index in candidates[expected_index]:
            if observed_index in visited:
                continue
            visited.add(observed_index)
            previous = observed_matches.get(observed_index)
            if previous is None or assign(previous, visited):
                observed_matches[observed_index] = expected_index
                return True
        return False

    for expected_index in range(len(expected_rows)):
        assign(expected_index, set())

    pairs = tuple(
        sorted(
            (expected_index, observed_index)
            for observed_index, expected_index in observed_matches.items()
        )
    )
    matched_expected = {expected_index for expected_index, _ in pairs}
    matched_observed = {observed_index for _, observed_index in pairs}
    differences = [
        max(
            abs(left - right)
            for left, right in zip(
                expected_rows[expected_index], observed_rows[observed_index]
            )
        )
        for expected_index, observed_index in pairs
    ]
    return BoxComparison(
        matched_pairs=pairs,
        unmatched_expected=tuple(
            index for index in range(len(expected_rows)) if index not in matched_expected
        ),
        unmatched_observed=tuple(
            index for index in range(len(observed_rows)) if index not in matched_observed
        ),
        max_abs_difference=max(differences, default=0.0),
    )


def verify_evidence_archive(
    archive: str | Path,
    sidecar: str | Path,
    *,
    expected_sha256: str = REFERENCE_ARCHIVE_SHA256,
) -> EvidenceSummary:
    archive_path = Path(archive).expanduser().resolve()
    sidecar_path = Path(sidecar).expanduser().resolve()
    if not archive_path.is_file():
        raise EvidenceVerificationError(f"Evidence archive is missing: {archive_path}")
    if not sidecar_path.is_file():
        raise EvidenceVerificationError(f"Evidence sidecar is missing: {sidecar_path}")
    if _SHA256.fullmatch(expected_sha256.lower()) is None:
        raise EvidenceVerificationError("Expected archive SHA-256 is malformed")

    if sidecar_path.stat().st_size > 4096:
        raise EvidenceVerificationError("Evidence sidecar is unexpectedly large")
    try:
        sidecar_parts = sidecar_path.read_text(encoding="utf-8").split()
    except (OSError, UnicodeDecodeError) as error:
        raise EvidenceVerificationError(f"Cannot read evidence sidecar: {error}") from error
    if not sidecar_parts or _SHA256.fullmatch(sidecar_parts[0].lower()) is None:
        raise EvidenceVerificationError("Sidecar does not start with a SHA-256 digest")
    observed_hash = _sha256_file(archive_path)
    sidecar_hash = sidecar_parts[0].lower()
    if sidecar_hash != observed_hash:
        raise EvidenceVerificationError(
            f"Sidecar hash mismatch: expected {sidecar_hash}, observed {observed_hash}"
        )
    if observed_hash != expected_sha256.lower():
        raise EvidenceVerificationError(
            f"Archive hash mismatch: expected {expected_sha256.lower()}, "
            f"observed {observed_hash}"
        )

    try:
        tar = tarfile.open(archive_path, mode="r:gz")
    except (tarfile.TarError, OSError) as error:
        raise EvidenceVerificationError(f"Cannot open evidence archive: {error}") from error
    with tar:
        members = _safe_members(tar)
        missing = sorted(REQUIRED_FILES - members.keys())
        if missing:
            raise EvidenceVerificationError(
                f"Evidence archive is incomplete; missing: {missing}"
            )

        dicom = _read_json(tar, members, "manifests/dicom-manifest.json")
        preprocess = _read_json(tar, members, "manifests/preprocess-manifest.json")
        detector = _read_json(
            tar, members, "manifests/detector-inference-manifest.json"
        )
        mmbcd_input = _read_json(
            tar, members, "manifests/mmbcd-input-manifest.json"
        )
        mmbcd = _read_json(
            tar, members, "manifests/mmbcd-inference-manifest.json"
        )
        _verify_manifests(dicom, preprocess, detector, mmbcd_input, mmbcd)
        _verify_provenance(tar, members)
        _verify_internal_hashes(tar, members, preprocess, mmbcd_input, mmbcd)

    return EvidenceSummary(
        archive_sha256=observed_hash,
        member_count=len(members),
        detector_prediction_sha256=detector["outputs"]["prediction_sha256"],
        mmbcd_prediction_sha256=mmbcd["outputs"]["prediction_sha256"],
    )


def _safe_members(tar: tarfile.TarFile) -> dict[str, tarfile.TarInfo]:
    members: dict[str, tarfile.TarInfo] = {}
    archive_members = tar.getmembers()
    if len(archive_members) > _MAX_MEMBERS:
        raise EvidenceVerificationError(
            f"Archive contains too many members: {len(archive_members)}"
        )
    for member in archive_members:
        name = _normalize_member(member.name)
        pure = PurePosixPath(name)
        if (
            not name
            or pure.is_absolute()
            or ".." in pure.parts
            or "\\" in name
            or (pure.parts and pure.parts[0].endswith(":"))
        ):
            if name:
                raise EvidenceVerificationError(f"Unsafe archive member: {member.name}")
            if not member.isdir():
                raise EvidenceVerificationError(f"Unsafe archive member: {member.name}")
        if member.issym() or member.islnk():
            raise EvidenceVerificationError(f"Archive links are forbidden: {member.name}")
        if not member.isfile() and not member.isdir():
            raise EvidenceVerificationError(
                f"Archive special files are forbidden: {member.name}"
            )
        if member.isfile() and member.size > _MAX_MEMBER_BYTES:
            raise EvidenceVerificationError(
                f"Archive member is too large: {member.name} ({member.size} bytes)"
            )
        if name in members:
            raise EvidenceVerificationError(f"Duplicate archive member: {name}")
        members[name] = member
    return members


def _verify_manifests(
    dicom: dict[str, Any],
    preprocess: dict[str, Any],
    detector: dict[str, Any],
    mmbcd_input: dict[str, Any],
    mmbcd: dict[str, Any],
) -> None:
    checks = {
        "DICOM SHA-256": (dicom.get("sha256"), REFERENCE_DICOM_SHA256),
        "preprocessed array SHA-256": (
            preprocess.get("artifacts", {}).get("resized", {}).get("array_sha256"),
            REFERENCE_PREPROCESSED_ARRAY_SHA256,
        ),
        "detector prediction SHA-256": (
            detector.get("outputs", {}).get("prediction_sha256"),
            REFERENCE_DETECTOR_PREDICTION_SHA256,
        ),
        "MMBCD input tensor SHA-256": (
            mmbcd_input.get("image_transform", {}).get("tensor_sha256"),
            REFERENCE_MMBCD_INPUT_SHA256,
        ),
        "MMBCD prediction SHA-256": (
            mmbcd.get("outputs", {}).get("prediction_sha256"),
            REFERENCE_MMBCD_PREDICTION_SHA256,
        ),
    }
    for label, (observed, expected) in checks.items():
        if observed != expected:
            raise EvidenceVerificationError(
                f"{label} mismatch: expected {expected}, observed {observed!r}"
            )
    required_true = {
        "detector strict load": detector.get("model", {}).get("strict_load"),
        "detector determinism": (
            detector.get("outputs", {}).get("determinism_max_abs_diff") == 0.0
        ),
        "MMBCD strict load": mmbcd.get("model", {}).get("strict_load"),
        "MMBCD logits determinism": (
            mmbcd.get("determinism", {}).get("bitwise_equal_logits")
        ),
        "MMBCD embedding determinism": (
            mmbcd.get("determinism", {}).get("bitwise_equal_embeddings")
        ),
    }
    for label, observed in required_true.items():
        if observed is not True:
            raise EvidenceVerificationError(f"{label} was not proven by the manifest")
    if detector.get("outputs", {}).get("mmbcd_topk_count") != 8:
        raise EvidenceVerificationError("Detector manifest does not retain eight ROIs")
    if detector.get("semantic_validation") is not False:
        raise EvidenceVerificationError("Detector semantic validation must remain false")
    if mmbcd_input.get("text", {}).get("label_information_used") is not False:
        raise EvidenceVerificationError("MMBCD prompt used forbidden label information")
    if mmbcd_input.get("text", {}).get("prompt") != "Indication:":
        raise EvidenceVerificationError("Golden empty-history prompt must be 'Indication:'")
    detector_outputs = detector.get("outputs", {})
    detector_shape = (
        detector_outputs.get("pred_logits_shape"),
        detector_outputs.get("pred_boxes_shape"),
    )
    if detector_shape != ([1, 900, 1], [1, 900, 4]):
        raise EvidenceVerificationError(
            f"Detector raw output shapes are invalid: {detector_shape!r}"
        )
    if (
        detector_outputs.get("pre_nms_count"),
        detector_outputs.get("post_nms_count"),
    ) != (300, 8):
        raise EvidenceVerificationError("Detector proposal counts are invalid")
    mmbcd_outputs = mmbcd.get("outputs", {})
    if mmbcd_outputs.get("logits") != REFERENCE_MMBCD_LOGITS:
        raise EvidenceVerificationError("MMBCD raw logits differ from the reference")
    if mmbcd_outputs.get("probabilities") != REFERENCE_MMBCD_PROBABILITIES:
        raise EvidenceVerificationError(
            "MMBCD raw probabilities differ from the reference"
        )
    proposals = mmbcd_input.get("proposals", {}).get("records")
    if not isinstance(proposals, list) or len(proposals) != 8:
        raise EvidenceVerificationError("MMBCD input does not preserve eight proposals")


def _verify_provenance(
    tar: tarfile.TarFile, members: dict[str, tarfile.TarInfo]
) -> None:
    for path, expected in REFERENCE_COMMITS.items():
        observed = _read_bytes(tar, members[path]).decode("utf-8").strip()
        if observed != expected:
            raise EvidenceVerificationError(
                f"Commit mismatch for {path}: expected {expected}, observed {observed}"
            )
    model_hash_text = _read_bytes(
        tar, members["environment/model-artifacts.sha256"]
    ).decode("utf-8")
    observed_model_hashes = {
        line.split()[0].lower()
        for line in model_hash_text.splitlines()
        if line.split()
    }
    missing_hashes = REFERENCE_MODEL_HASHES - observed_model_hashes
    if missing_hashes:
        raise EvidenceVerificationError(
            f"Model artifact hash inventory is incomplete: {sorted(missing_hashes)}"
        )
    extension_text = _read_bytes(
        tar, members["environment/cuda-extension.sha256"]
    ).decode("utf-8")
    if REFERENCE_EXTENSION_SHA256 not in extension_text:
        raise EvidenceVerificationError("Reference CUDA extension hash is missing")
    patch_hash = _sha256_bytes(
        _read_bytes(tar, members["patches/focalnet-pytorch-2.8-compat.patch"])
    )
    if patch_hash != REFERENCE_PATCH_SHA256:
        raise EvidenceVerificationError(
            f"Archived compatibility patch hash mismatch: {patch_hash}"
        )
    for path in (
        "environment/conda-packages.json",
        "environment/python-packages.json",
    ):
        package_inventory = _read_json_value(tar, members, path)
        if not isinstance(package_inventory, list):
            raise EvidenceVerificationError(
                f"Package inventory root must be an array: {path}"
            )


def _verify_internal_hashes(
    tar: tarfile.TarFile,
    members: dict[str, tarfile.TarInfo],
    preprocess: dict[str, Any],
    mmbcd_input: dict[str, Any],
    mmbcd: dict[str, Any],
) -> None:
    links = {
        "visuals/upstream-1024.png": preprocess["artifacts"]["resized"]["file_sha256"],
        "visuals/roi-montage.png": mmbcd_input["artifacts"]["montage_sha256"],
        "bundles/detections-top8.txt": mmbcd_input["source"]["detections_sha256"],
        "bundles/mmbcd-inputs.npz": mmbcd_input["artifacts"]["bundle_sha256"],
        "bundles/mmbcd-outputs.npz": mmbcd["outputs"]["bundle_sha256"],
    }
    for path, expected in links.items():
        observed = _sha256_bytes(_read_bytes(tar, members[path]))
        if observed != expected:
            raise EvidenceVerificationError(
                f"Internal evidence hash mismatch for {path}: "
                f"expected {expected}, observed {observed}"
            )


def _read_json(
    tar: tarfile.TarFile,
    members: dict[str, tarfile.TarInfo],
    path: str,
) -> dict[str, Any]:
    value = _read_json_value(tar, members, path)
    if not isinstance(value, dict):
        raise EvidenceVerificationError(f"JSON evidence root must be an object: {path}")
    return value


def _read_json_value(
    tar: tarfile.TarFile,
    members: dict[str, tarfile.TarInfo],
    path: str,
) -> Any:
    member = members[path]
    if member.size > _MAX_JSON_BYTES:
        raise EvidenceVerificationError(f"JSON evidence member is too large: {path}")
    try:
        value = json.loads(_read_bytes(tar, member))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise EvidenceVerificationError(f"Invalid JSON evidence in {path}: {error}") from error
    return value


def _read_bytes(tar: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    extracted = tar.extractfile(member)
    if extracted is None:
        raise EvidenceVerificationError(f"Cannot read archive member: {member.name}")
    return extracted.read()


def _numeric_rows(
    records: Sequence[Sequence[float]], label: str
) -> tuple[tuple[float, ...], ...]:
    rows: list[tuple[float, ...]] = []
    for row_index, record in enumerate(records):
        try:
            row = tuple(float(value) for value in record)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label}[{row_index}] contains a non-numeric value") from error
        if not row:
            raise ValueError(f"{label}[{row_index}] must not be empty")
        if not all(math.isfinite(value) for value in row):
            raise ValueError(f"{label}[{row_index}] contains a non-finite value")
        rows.append(row)
    return tuple(rows)


def _normalize_member(name: str) -> str:
    while name.startswith("./"):
        name = name[2:]
    return name.rstrip("/")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
