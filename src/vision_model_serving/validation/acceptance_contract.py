"""Observable contract for the checksum-pinned packaged acceptance lane."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Final

from vision_model_serving.model_ids import CLASSIFIER_MODEL_ID, DETECTOR_MODEL_ID
from vision_model_serving.pipeline.contracts import PredictionMode

from .acceptance_constants import (
    ARCHIVED_CLASSIFIER_OUTPUT_SHA256 as ARCHIVED_CLASSIFIER_OUTPUT_SHA256,
    PACKAGED_ACCEPTANCE_HISTORY as PACKAGED_ACCEPTANCE_HISTORY,
    PACKAGED_DISCLAIMER,
    PACKAGED_MANIFEST_ID,
    PACKAGED_MANIFEST_SHA256,
    PUBLIC_CANONICAL_ARRAY_SHA256 as PUBLIC_CANONICAL_ARRAY_SHA256,
    PUBLIC_DICOM_SHA256,
    SERVED_CLASSIFIER_OUTPUT_SHA256,
    SERVED_DETECTOR_OUTPUT_SHA256,
    TOKENIZER_REVISION,
)

_TOKENIZER_FILES: Final = {
    "config.json": (
        481,
        "ef0185e2aae6e06c5f105a285006952c340e20c7dbf43c86ec82601b13fc45e9",
    ),
    "merges.txt": (
        456318,
        "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
    ),
    "tokenizer.json": (
        1355863,
        "847bbeab6174d66a88898f729d52fa8d355fafe1bea101cf960dd404581df70e",
    ),
    "tokenizer_config.json": (
        25,
        "994f46754c5bf4014f1aa92d34b1374319c3a6b3f702105cd5b742beaecd18ce",
    ),
    "vocab.json": (
        898823,
        "9e7f63c2d15d666b52e21d250d2e513b87c9b713cfa6987a82ed89e5e6e50655",
    ),
}
_REPOSITORY_ASSETS: Final = {
    "focalnet-dino-config": (
        "config",
        "config_cfg.py",
        "6bf3f0bee489209195879b71fbcff6d5f0368a8488e8cf533366e325fbc403db",
    ),
    "focalnet-pytorch-2.8-compat": (
        "patch",
        "patches/focalnet-pytorch-2.8-compat.patch",
        "10718bf12f2ac98c15014f14a6578ccb97f604a3f46a6db8724f4c9aa34f319c",
    ),
    "focalnet-serving-no-backbone-preload": (
        "patch",
        "patches/focalnet-serving-no-backbone-preload.patch",
        "fe88967e13ead7496c68e1a005ba0317b88e0fd3bfc39af7b75038403d7ddf57",
    ),
    "focalnet-docker-force-cuda": (
        "patch",
        "patches/focalnet-docker-force-cuda.patch",
        "8d1639167bc4a27fe1ae2a4f5145ab162ff4dfdcfae352a564400762000ea305",
    ),
}
_IDENTITY_SECTION_SHA256: Final = {
    "models": "8775e19c139880d10b319a142c244308085a1912f94e66abd37151c251d15a7d",
    "tokenizer": "5855aa714e4f4742c91809cc8a4c9bcfba916e72d38d35be99ae12b08fa7b813",
    "repository_assets": ("54f8d7f5c1d39a55c2571b0c79f6f3ed7b7e6bcfd64e3762758a9702f4e33135"),
}


@dataclass(frozen=True, slots=True)
class ArtifactExpectation:
    model_id: str
    role: str
    sha256: str
    repository_revision: str


DETECTOR_ARTIFACT: Final = ArtifactExpectation(
    model_id=DETECTOR_MODEL_ID,
    role="detector",
    sha256="67a7b0cd787a3aaba199cf1ff82ed2934c33ffe37544473379d7a837ab1637b4",
    repository_revision="23901e021dc6ec8f66bad47983f45a25574452cc",
)
CLASSIFIER_ARTIFACT: Final = ArtifactExpectation(
    model_id=CLASSIFIER_MODEL_ID,
    role="classifier",
    sha256="2264351216f9fb4945af35e300459ff4ce2e7f5445519348024f3bf1eec721a4",
    repository_revision="14ac5e099c79253b01e0885d2ebefa6f86cfd8f0",
)


class PackagedAcceptanceContract:
    """Validate the packaged lane through its HTTP-visible observations."""

    _required_public_warnings = frozenset({"secondary_capture_storage", "aspect_ratio_distorted"})
    _required_classifier_warnings = frozenset(
        {
            "class_semantics_and_decision_threshold_unverified",
            "attention_is_inspection_not_causal_or_clinical_evidence",
        }
    )
    _artifacts = (DETECTOR_ARTIFACT, CLASSIFIER_ARTIFACT)
    _required_readiness_checks = frozenset(
        {
            "redis",
            "rq_worker",
            "executor_artifact_ready",
            "verified_artifacts",
            "runtime_initialized",
            "device_available",
            "native_operator_available",
            "manifest_available",
            "telemetry_available",
        }
    )

    def validate_readiness(self, payload: Mapping[str, object]) -> Mapping[str, object]:
        checks = _mapping(payload.get("checks"), "readiness checks")
        if (
            payload.get("schema_version") != 2
            or payload.get("status") != "ready"
            or payload.get("readiness_scope") != "artifact_ready"
            or set(checks) != self._required_readiness_checks
            or not all(value is True for value in checks.values())
            or payload.get("reasons") != []
        ):
            raise AssertionError(f"packaged API is not ready: {payload!r}")
        runtime = _mapping(payload.get("runtime"), "readiness runtime")
        if set(runtime) != {
            "initialized",
            "state",
            "inference_warm",
            "warm_model",
        }:
            raise AssertionError("readiness runtime schema is invalid")
        state = runtime.get("state")
        inference_warm = runtime.get("inference_warm")
        warm_model = runtime.get("warm_model")
        allowed_states = ("unloaded", "loading", "ready", "draining", "unloading")
        model_ids = tuple(artifact.model_id for artifact in self._artifacts)
        if (
            runtime.get("initialized") is not True
            or state not in allowed_states
            or inference_warm is not (state == "ready")
            or (warm_model not in model_ids if state == "ready" else warm_model is not None)
        ):
            raise AssertionError(f"readiness runtime is inconsistent: {runtime!r}")
        return payload

    def validate_inventory(self, inventory: Mapping[str, object]) -> dict[str, Any]:
        if inventory.get("manifest_id") != PACKAGED_MANIFEST_ID:
            raise AssertionError("API exposed an unexpected artifact manifest")
        models = inventory.get("models")
        if not isinstance(models, Sequence) or isinstance(models, (str, bytes)):
            raise AssertionError("model inventory is not a sequence")
        if len(models) != len(self._artifacts):
            raise AssertionError("model inventory does not contain exactly two models")
        observed: dict[str, tuple[object, object]] = {}
        for raw_model in models:
            model = _mapping(raw_model, "model inventory entry")
            if model.get("strict_load_verified") is not True:
                raise AssertionError("model inventory contains an unverified model")
            model_id = model.get("id")
            if not isinstance(model_id, str) or model_id in observed:
                raise AssertionError("model inventory identity is invalid or repeated")
            observed[model_id] = (model.get("role"), model.get("sha256"))
        expected = {
            artifact.model_id: (artifact.role, artifact.sha256) for artifact in self._artifacts
        }
        if observed != expected:
            raise AssertionError(f"model inventory differs from the manifest: {observed!r}")
        runtime = _mapping(inventory.get("runtime"), "runtime inventory")
        residents = runtime.get("resident_models")
        if not isinstance(residents, Sequence) or isinstance(residents, (str, bytes)):
            raise AssertionError("resident model inventory is invalid")
        if any(not isinstance(model_id, str) for model_id in residents):
            raise AssertionError("resident model identity is invalid")
        resident_set = set(residents)
        if len(resident_set) != len(residents):
            raise AssertionError("executor repeated a resident model identity")
        if len(resident_set) > 1 or not resident_set <= set(expected):
            raise AssertionError("executor exposed a model outside the manifest")
        return {
            "manifest_id": inventory["manifest_id"],
            "models": list(models),
            "runtime": dict(runtime),
        }

    def validate_benchmark_identity(
        self,
        identity: Mapping[str, object],
        *,
        dicom: bytes | None = None,
    ) -> None:
        """Bind benchmark metadata to the exact packaged fixture and assets."""

        if set(identity) != {
            "manifest_id",
            "manifest_sha256",
            "models",
            "tokenizer",
            "repository_assets",
            "dicom_sha256",
            "detector_prediction_sha256",
            "classifier_prediction_sha256",
        }:
            raise AssertionError("benchmark identity schema is incomplete")
        if (
            identity.get("manifest_id") != PACKAGED_MANIFEST_ID
            or identity.get("manifest_sha256") != PACKAGED_MANIFEST_SHA256
            or identity.get("dicom_sha256") != PUBLIC_DICOM_SHA256
            or identity.get("detector_prediction_sha256") != SERVED_DETECTOR_OUTPUT_SHA256
            or identity.get("classifier_prediction_sha256") != SERVED_CLASSIFIER_OUTPUT_SHA256
            or (dicom is not None and hashlib.sha256(dicom).hexdigest() != PUBLIC_DICOM_SHA256)
        ):
            raise AssertionError("benchmark identity differs from packaged acceptance")

        models = _sequence(identity.get("models"), "benchmark models")
        observed_models: dict[str, tuple[object, object]] = {}
        for raw_model in models:
            model = _mapping(raw_model, "benchmark model")
            model_id = model.get("id")
            if not isinstance(model_id, str) or model_id in observed_models:
                raise AssertionError("benchmark model identity is invalid or repeated")
            observed_models[model_id] = (model.get("role"), model.get("sha256"))
        expected_models = {
            artifact.model_id: (artifact.role, artifact.sha256) for artifact in self._artifacts
        }
        if observed_models != expected_models:
            raise AssertionError("benchmark models differ from packaged acceptance")

        tokenizer = _mapping(identity.get("tokenizer"), "benchmark tokenizer")
        if (
            tokenizer.get("id") != "roberta-base"
            or tokenizer.get("repository") != "FacebookAI/roberta-base"
            or tokenizer.get("revision") != TOKENIZER_REVISION
            or tokenizer.get("local_files_only") is not True
        ):
            raise AssertionError("benchmark tokenizer identity is invalid")
        files = _sequence(tokenizer.get("files"), "benchmark tokenizer files")
        observed_files: dict[str, tuple[object, object]] = {}
        for raw_file in files:
            file_identity = _mapping(raw_file, "benchmark tokenizer file")
            filename = file_identity.get("filename")
            if not isinstance(filename, str) or filename in observed_files:
                raise AssertionError("benchmark tokenizer file is invalid or repeated")
            observed_files[filename] = (
                file_identity.get("size_bytes"),
                file_identity.get("sha256"),
            )
        if observed_files != _TOKENIZER_FILES:
            raise AssertionError("benchmark tokenizer files differ from the manifest")

        assets = _sequence(identity.get("repository_assets"), "repository assets")
        observed_assets: dict[str, tuple[object, object, object]] = {}
        for raw_asset in assets:
            asset = _mapping(raw_asset, "repository asset")
            asset_id = asset.get("id")
            if not isinstance(asset_id, str) or asset_id in observed_assets:
                raise AssertionError("repository asset identity is invalid or repeated")
            observed_assets[asset_id] = (
                asset.get("kind"),
                asset.get("path"),
                asset.get("sha256"),
            )
        if observed_assets != _REPOSITORY_ASSETS:
            raise AssertionError("repository assets differ from packaged acceptance")
        for field in ("models", "tokenizer", "repository_assets"):
            if _canonical_sha256(identity[field]) != _IDENTITY_SECTION_SHA256[field]:
                raise AssertionError(f"benchmark {field} details differ from the packaged manifest")

    def validate_runtime(
        self,
        runtime: Mapping[str, object],
        *,
        active_model: str,
    ) -> None:
        if active_model not in {artifact.model_id for artifact in self._artifacts}:
            raise AssertionError("expected runtime model is outside the manifest")
        if (
            runtime.get("state") != "ready"
            or runtime.get("initialized") is not True
            or runtime.get("artifact_ready") is not True
            or runtime.get("inference_warm") is not True
            or runtime.get("warm_model") != active_model
            or runtime.get("active_model") != active_model
            or runtime.get("resident_models") != [active_model]
            or runtime.get("last_error") is not None
        ):
            raise AssertionError(f"unexpected executor lifecycle state: {runtime!r}")

    def validate_prediction(
        self,
        result: Mapping[str, object],
        *,
        mode: PredictionMode,
    ) -> None:
        if not isinstance(mode, PredictionMode):
            raise TypeError("mode must be a PredictionMode")
        _assert_finite(result)
        input_payload = _mapping(result.get("input"), "prediction input")
        detector = _mapping(result.get("detector"), "detector result")
        if (
            result.get("mode") != mode.value
            or input_payload.get("source_sha256") != PUBLIC_DICOM_SHA256
            or result.get("disclaimer") != PACKAGED_DISCLAIMER
            or detector.get("prediction_sha256") != SERVED_DETECTOR_OUTPUT_SHA256
        ):
            raise AssertionError("prediction identity differs from the served golden contract")
        for field in ("top_candidates", "post_nms"):
            values = _sequence(detector.get(field), f"detector {field}")
            if len(values) > 300:
                raise AssertionError("detector presentation exceeded its bounded contract")
        classifier_rois = _sequence(detector.get("classifier_rois"), "detector classifier ROIs")
        if len(classifier_rois) != 8:
            raise AssertionError("detector did not produce exactly eight classifier ROIs")
        self._validate_detections(result, detector)

        provenance = _mapping(result.get("provenance"), "prediction provenance")
        _assert_artifact(provenance.get("detector"), DETECTOR_ARTIFACT)
        warnings = _sequence(result.get("warnings"), "prediction warnings")
        warning_codes = {
            _mapping(warning, "prediction warning").get("code") for warning in warnings
        }
        if not self._required_public_warnings <= warning_codes:
            raise AssertionError("required public-DICOM warnings are absent")

        if mode is PredictionMode.DETECTION:
            classifier_fields = (
                "classifier",
                "tokenizer",
                "precision",
                "offline_assets_only",
                "strict_checkpoint_load",
            )
            if result.get("classification") is not None or any(
                provenance.get(field) is not None for field in classifier_fields
            ):
                raise AssertionError("detection mode unexpectedly ran the classifier")
            return

        classification = _mapping(result.get("classification"), "classification result")
        if classification.get("prediction_sha256") != SERVED_CLASSIFIER_OUTPUT_SHA256:
            raise AssertionError("classifier output differs from the served golden")
        _finite_real_vector(
            classification.get("logits"),
            "classifier logits",
            length=2,
        )
        _normalized_vector(
            classification.get("probabilities"),
            "classifier probabilities",
            length=2,
        )
        attention = _mapping(classification.get("attention"), "classifier attention")
        _normalized_vector(
            attention.get("roi_weights"),
            "classifier attention weights",
            length=8,
        )
        _assert_artifact(provenance.get("classifier"), CLASSIFIER_ARTIFACT)
        tokenizer = _mapping(provenance.get("tokenizer"), "tokenizer provenance")
        if (
            tokenizer.get("revision") != TOKENIZER_REVISION
            or provenance.get("precision") != "float32"
            or provenance.get("offline_assets_only") is not True
            or provenance.get("strict_checkpoint_load") is not True
            or not self._required_classifier_warnings <= warning_codes
        ):
            raise AssertionError("classifier provenance or warnings are incomplete")

    def behavior_snapshot(
        self,
        result: Mapping[str, object],
        runtime: Mapping[str, object],
    ) -> dict[str, object]:
        detector = _mapping(result.get("detector"), "detector result")
        classification = result.get("classification")
        classifier = (
            None if classification is None else _mapping(classification, "classification result")
        )
        timings = _mapping(result.get("timings"), "prediction timings")
        detector_timing = _mapping(timings.get("detector"), "detector timing")
        detector_runtime = _mapping(detector_timing.get("runtime"), "detector runtime timing")
        classifier_timing = timings.get("classifier")
        classifier_reused: object = None
        if classifier_timing is not None:
            classifier_runtime = _mapping(
                _mapping(classifier_timing, "classifier timing").get("runtime"),
                "classifier runtime timing",
            )
            classifier_reused = classifier_runtime.get("reused")
        warning_codes = sorted(
            str(_mapping(warning, "prediction warning").get("code"))
            for warning in _sequence(result.get("warnings"), "prediction warnings")
        )
        return {
            "mode": result.get("mode"),
            "detector_sha256": detector.get("prediction_sha256"),
            "classifier_sha256": (
                None if classifier is None else classifier.get("prediction_sha256")
            ),
            "warning_codes": warning_codes,
            "detector_reused": detector_runtime.get("reused"),
            "classifier_reused": classifier_reused,
            "active_model": runtime.get("active_model"),
            "resident_models": sorted(
                str(value) for value in _sequence(runtime.get("resident_models"), "resident models")
            ),
        }

    @staticmethod
    def _validate_detections(result: Mapping[str, object], detector: Mapping[str, object]) -> None:
        geometry = _mapping(result.get("geometry"), "prediction geometry")
        canonical_width = _finite_real(geometry.get("canonical_width"), "canonical width")
        canonical_height = _finite_real(geometry.get("canonical_height"), "canonical height")
        original_width = _finite_real(geometry.get("original_width"), "original width")
        original_height = _finite_real(geometry.get("original_height"), "original height")
        if min(canonical_width, canonical_height, original_width, original_height) <= 0.0:
            raise AssertionError("prediction geometry dimensions must be positive")
        detections: list[object] = []
        for field in ("top_candidates", "post_nms", "classifier_rois"):
            detections.extend(_sequence(detector.get(field), f"detector {field}"))
        for raw_detection in detections:
            detection = _mapping(raw_detection, "detection")
            score = _finite_real(detection.get("score"), "detector score")
            if not 0.0 <= score <= 1.0:
                raise AssertionError("detector score is outside [0, 1]")
            normalized = _finite_real_vector(
                detection.get("normalized_xyxy"),
                "detector normalized_xyxy",
                length=4,
            )
            canonical = _finite_real_vector(
                detection.get("canonical_xyxy"),
                "detector canonical_xyxy",
                length=4,
            )
            original = _finite_real_vector(
                detection.get("original_xyxy"),
                "detector original_xyxy",
                length=4,
            )
            if not _has_positive_extent(normalized) or not _has_positive_extent(canonical):
                raise AssertionError("detector box must have positive extent")

            expected_canonical = (
                normalized[0] * canonical_width,
                normalized[1] * canonical_height,
                normalized[2] * canonical_width,
                normalized[3] * canonical_height,
            )
            if any(
                not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-4)
                for actual, expected in zip(canonical, expected_canonical, strict=True)
            ):
                raise AssertionError("detector canonical_xyxy disagrees with normalized_xyxy")

            # The reference detector deliberately keeps normalized/canonical
            # boxes unclipped for NMS and zero-padded classifier crops.  The
            # original-image projection is the clipped representation used by
            # browser overlays, and a fully overhanging proposal may collapse
            # to an image edge after clipping.
            x1, y1, x2, y2 = original
            if not (0.0 <= x1 <= x2 <= original_width and 0.0 <= y1 <= y2 <= original_height):
                raise AssertionError("detector original_xyxy is outside image bounds")


PACKAGED_ACCEPTANCE: Final = PackagedAcceptanceContract()


def _has_positive_extent(box: Sequence[float]) -> bool:
    return box[0] < box[2] and box[1] < box[3]


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AssertionError(f"{name} must be an object")
    return value


def _sequence(value: object, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise AssertionError(f"{name} must be an array")
    return value


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AssertionError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise AssertionError(f"{name} must be finite")
    return result


def _finite_real_vector(
    value: object,
    name: str,
    *,
    length: int,
) -> tuple[float, ...]:
    observed = _sequence(value, name)
    if len(observed) != length:
        raise AssertionError(f"{name} must contain exactly {length} values")
    return tuple(_finite_real(item, name) for item in observed)


def _normalized_vector(
    value: object,
    name: str,
    *,
    length: int,
) -> tuple[float, ...]:
    observed = _finite_real_vector(value, name, length=length)
    if any(item < 0.0 or item > 1.0 for item in observed) or not math.isclose(
        sum(observed),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise AssertionError(f"{name} must be bounded probabilities summing to one")
    return observed


def _assert_artifact(value: object, expected: ArtifactExpectation) -> None:
    observed = _mapping(value, f"{expected.role} artifact provenance")
    fields = {
        "id": expected.model_id,
        "sha256": expected.sha256,
        "repository_revision": expected.repository_revision,
    }
    if any(observed.get(field) != expected_value for field, expected_value in fields.items()):
        raise AssertionError(f"artifact provenance differs: {observed!r}")


def _assert_finite(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise AssertionError("prediction contains a non-finite number")
    if isinstance(value, Mapping):
        for item in value.values():
            _assert_finite(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            _assert_finite(item)


def _canonical_sha256(value: object) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise AssertionError("benchmark identity is not canonical JSON") from error
    return hashlib.sha256(payload).hexdigest()
