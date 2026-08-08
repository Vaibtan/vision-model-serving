"""MMBCD classifier interface and immutable result contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
import re
from time import perf_counter
from typing import Protocol, Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image


_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ClassifierAdapterError(RuntimeError):
    code = "classifier_adapter_failed"

    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(f"{self.code}: {detail}")


class ClassifierInputError(ClassifierAdapterError):
    code = "classifier_input_invalid"


class ClassifierOutputError(ClassifierAdapterError):
    code = "classifier_output_invalid"


@dataclass(frozen=True, slots=True)
class ClassifierArtifactIdentity:
    id: str
    sha256: str
    repository_revision: str

    def __post_init__(self) -> None:
        if self.id != "mmbcd-classifier":
            raise ValueError("artifact identity is not MMBCD")
        if _SHA256.fullmatch(self.sha256) is None:
            raise ValueError("artifact SHA-256 is invalid")
        if _GIT_COMMIT.fullmatch(self.repository_revision) is None:
            raise ValueError("artifact repository revision is invalid")


@dataclass(frozen=True, slots=True)
class LocalTokenizerIdentity:
    id: str
    revision: str
    file_sha256: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if self.id != "roberta-base" or _GIT_COMMIT.fullmatch(self.revision) is None:
            raise ValueError("tokenizer identity is invalid")
        if len(self.file_sha256) != 5 or any(
            _SHA256.fullmatch(digest) is None for _, digest in self.file_sha256
        ):
            raise ValueError("tokenizer file identity is invalid")


@dataclass(frozen=True, slots=True)
class TokenBatch:
    input_ids: NDArray[np.int64] = field(repr=False)
    attention_mask: NDArray[np.int64] = field(repr=False)

    def __post_init__(self) -> None:
        if (
            self.input_ids.dtype != np.int64
            or self.attention_mask.dtype != np.int64
            or self.input_ids.ndim != 2
            or self.input_ids.shape != self.attention_mask.shape
            or self.input_ids.shape[0] != 1
            or not 1 <= self.input_ids.shape[1] <= 90
        ):
            raise ValueError("token batch violates the MMBCD text contract")
        self.input_ids.setflags(write=False)
        self.attention_mask.setflags(write=False)


@dataclass(frozen=True, slots=True)
class ClassifierRuntimeOutput:
    logits: NDArray[np.float32] = field(repr=False)
    fused_embeddings: NDArray[np.float32] = field(repr=False)
    roi_attention: NDArray[np.float32] = field(repr=False)
    inference_ms: float


class OfflineTokenizer(Protocol):
    def encode(self, prompt: str, *, max_length: int) -> TokenBatch: ...


class ClassifierRuntime(Protocol):
    def execute(
        self,
        crops: NDArray[np.float32],
        tokens: TokenBatch,
    ) -> ClassifierRuntimeOutput: ...


@dataclass(frozen=True, slots=True)
class ClassifierInputSummary:
    prompt: str
    label_information_used: bool
    token_count: int
    crop_tensor_sha256: str
    input_ids_sha256: str
    attention_mask_sha256: str


@dataclass(frozen=True, slots=True)
class AttentionInspection:
    kind: str
    roi_weights: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ClassifierProvenance:
    artifact: ClassifierArtifactIdentity
    tokenizer: LocalTokenizerIdentity
    precision: str
    offline_assets_only: bool
    strict_checkpoint_load: bool


@dataclass(frozen=True, slots=True)
class ClassifierTimings:
    load_ms: float
    crop_preprocess_ms: float
    tokenization_ms: float
    inference_ms: float
    result_ms: float

    def __post_init__(self) -> None:
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (
                self.load_ms,
                self.crop_preprocess_ms,
                self.tokenization_ms,
                self.inference_ms,
                self.result_ms,
            )
        ):
            raise ValueError("classifier timings must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class MmbcdResult:
    input: ClassifierInputSummary
    class_indices: tuple[int, int]
    logits: tuple[float, float]
    probabilities: tuple[float, float]
    predicted_class_index: int
    fused_embeddings: NDArray[np.float32] = field(repr=False)
    attention: AttentionInspection
    prediction_sha256: str
    provenance: ClassifierProvenance
    timings: ClassifierTimings
    warnings: tuple[str, ...]

    @property
    def artifact(self) -> ClassifierArtifactIdentity:
        return self.provenance.artifact

    @property
    def tokenizer(self) -> LocalTokenizerIdentity:
        return self.provenance.tokenizer


class MmbcdClassifierAdapter:
    """Hide ROI preparation, label-free prompting, tokenization, and inference."""

    def __init__(
        self,
        *,
        runtime: ClassifierRuntime,
        tokenizer: OfflineTokenizer,
        artifact: ClassifierArtifactIdentity,
        tokenizer_identity: LocalTokenizerIdentity,
        strict_checkpoint_load: bool = False,
    ):
        self._runtime = runtime
        self._tokenizer = tokenizer
        self._artifact = artifact
        self._tokenizer_identity = tokenizer_identity
        self._strict_checkpoint_load = strict_checkpoint_load

    @classmethod
    def from_artifact(
        cls,
        artifact: object,
        *,
        tokenizer: OfflineTokenizer,
        tokenizer_identity: LocalTokenizerIdentity,
        model_factory: Callable[[], object],
        device: str = "cuda:0",
    ) -> MmbcdClassifierAdapter:
        """Strict-load one verified artifact behind the public adapter seam."""

        from .runtime import TorchMmbcdRuntime

        runtime = TorchMmbcdRuntime(
            artifact,
            model_factory=model_factory,
            device=device,
        )
        return cls(
            runtime=runtime,
            tokenizer=tokenizer,
            artifact=runtime.identity,
            tokenizer_identity=tokenizer_identity,
            strict_checkpoint_load=True,
        )

    @classmethod
    def from_local_assets(
        cls,
        artifact: object,
        *,
        tokenizer_root: str | Path,
        dino_root: str | Path,
        mmbcd_root: str | Path,
        project_root: str | Path,
        device: str = "cuda:0",
    ) -> MmbcdClassifierAdapter:
        """Build from checksum-pinned artifacts and exact local source commits."""

        from .runtime import LocalMmbcdModelFactory
        from .tokenizer import LocalRobertaTokenizer

        root = Path(project_root).expanduser().resolve()
        tokenizer = LocalRobertaTokenizer.from_manifest(
            tokenizer_root,
            root / "config" / "model-artifacts.json",
        )
        factory = LocalMmbcdModelFactory(
            dino_root=Path(dino_root).expanduser().resolve(),
            mmbcd_root=Path(mmbcd_root).expanduser().resolve(),
            project_root=root,
            expected_mmbcd_revision=str(
                getattr(artifact, "repository_revision", "")
            ),
        )
        return cls.from_artifact(
            artifact,
            tokenizer=tokenizer,
            tokenizer_identity=tokenizer.identity,
            model_factory=factory.build,
            device=device,
        )

    def predict(
        self,
        mammogram: object,
        classifier_rois: Sequence[object],
        clinical_history: str,
    ) -> MmbcdResult:
        if len(classifier_rois) != 8:
            raise ClassifierInputError("exactly eight classifier ROIs are required")
        crop_started = perf_counter()
        crops = _prepare_crops(mammogram, classifier_rois)
        crop_preprocess_ms = (perf_counter() - crop_started) * 1000.0
        prompt = _format_prompt(clinical_history)
        tokenization_started = perf_counter()
        try:
            tokens = self._tokenizer.encode(prompt, max_length=90)
        except Exception as error:
            raise ClassifierInputError(
                f"offline tokenization failed ({type(error).__name__})"
            ) from None
        tokenization_ms = (perf_counter() - tokenization_started) * 1000.0
        try:
            output = self._runtime.execute(crops, tokens)
        except ClassifierAdapterError:
            raise
        except Exception as error:
            raise ClassifierOutputError(
                f"classifier runtime failed ({type(error).__name__})"
            ) from None
        result_started = perf_counter()
        if not math.isfinite(output.inference_ms) or output.inference_ms < 0.0:
            raise ClassifierOutputError("classifier inference timing is invalid")
        logits = np.asarray(output.logits, dtype=np.float32)
        if logits.shape != (1, 2) or not np.isfinite(logits).all():
            raise ClassifierOutputError(
                "classifier logits violate the verified contract"
            )
        embeddings = np.ascontiguousarray(output.fused_embeddings, dtype=np.float32)
        attention = np.asarray(output.roi_attention, dtype=np.float32)
        if embeddings.shape != (1, 768) or not np.isfinite(embeddings).all():
            raise ClassifierOutputError(
                "classifier fused embeddings violate the verified contract"
            )
        if attention.shape != (1, 1, 8) or not np.isfinite(attention).all():
            raise ClassifierOutputError(
                "classifier ROI attention violates the verified contract"
            )
        weights = attention[0, 0]
        if np.any(weights < 0) or not np.isclose(np.sum(weights), 1.0, atol=1e-5):
            raise ClassifierOutputError("classifier ROI attention is not normalized")
        embeddings.setflags(write=False)
        shifted = logits[0] - np.max(logits[0])
        exponentials = np.exp(shifted)
        probabilities = exponentials / np.sum(exponentials)
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(logits, dtype="<f4").tobytes())
        digest.update(np.ascontiguousarray(embeddings, dtype="<f4").tobytes())
        crop_digest = _array_sha256(crops, "<f4")
        input_ids_digest = _array_sha256(tokens.input_ids, "<i8")
        mask_digest = _array_sha256(tokens.attention_mask, "<i8")
        result_ms = (perf_counter() - result_started) * 1000.0
        return MmbcdResult(
            input=ClassifierInputSummary(
                prompt=prompt,
                label_information_used=False,
                token_count=int(tokens.attention_mask.sum()),
                crop_tensor_sha256=crop_digest,
                input_ids_sha256=input_ids_digest,
                attention_mask_sha256=mask_digest,
            ),
            class_indices=(0, 1),
            logits=(float(logits[0, 0]), float(logits[0, 1])),
            probabilities=(float(probabilities[0]), float(probabilities[1])),
            predicted_class_index=int(np.argmax(logits[0])),
            fused_embeddings=embeddings,
            attention=AttentionInspection(
                kind="model_inspection_not_causal_or_clinical_evidence",
                roi_weights=tuple(float(value) for value in weights),
            ),
            prediction_sha256=digest.hexdigest(),
            provenance=ClassifierProvenance(
                artifact=self._artifact,
                tokenizer=self._tokenizer_identity,
                precision="float32",
                offline_assets_only=True,
                strict_checkpoint_load=self._strict_checkpoint_load,
            ),
            timings=ClassifierTimings(
                load_ms=float(getattr(self._runtime, "load_ms", 0.0)),
                crop_preprocess_ms=crop_preprocess_ms,
                tokenization_ms=tokenization_ms,
                inference_ms=float(output.inference_ms),
                result_ms=result_ms,
            ),
            warnings=(
                "class_semantics_and_decision_threshold_unverified",
                "attention_is_inspection_not_causal_or_clinical_evidence",
            ),
        )


def _format_prompt(clinical_history: str) -> str:
    if not isinstance(clinical_history, str):
        raise ClassifierInputError("clinical history must be text")
    normalized = " ".join(clinical_history.split())
    return f"Indication: {normalized}" if normalized else "Indication:"


def _prepare_crops(
    mammogram: object,
    classifier_rois: Sequence[object],
) -> NDArray[np.float32]:
    pixels = getattr(mammogram, "pixels", None)
    if (
        not isinstance(pixels, np.ndarray)
        or pixels.ndim != 2
        or pixels.dtype != np.uint8
    ):
        raise ClassifierInputError("canonical mammogram must be a uint8 image")
    height, width = pixels.shape
    if width <= 0 or height <= 0:
        raise ClassifierInputError("canonical mammogram must not be empty")
    image = Image.fromarray(pixels, mode="L").convert("RGB")
    tensors: list[NDArray[np.float32]] = []
    for proposal in classifier_rois:
        box = getattr(proposal, "canonical_xyxy", None)
        try:
            values = tuple(float(value) for value in box)
        except (TypeError, ValueError):
            raise ClassifierInputError(
                "classifier ROI coordinates are invalid"
            ) from None
        if len(values) != 4 or not np.isfinite(values).all():
            raise ClassifierInputError("classifier ROI coordinates are invalid")
        float_x0, float_y0, float_x1, float_y1 = values
        if not (
            0.0 <= float_x0 < float_x1 <= float(width)
            and 0.0 <= float_y0 < float_y1 <= float(height)
        ):
            raise ClassifierInputError("classifier ROI is outside the canonical image")
        x0, y0, x1, y1 = (int(value) for value in values)
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ClassifierInputError("classifier ROI is outside the canonical image")
        crop = image.crop((x0, y0, x1, y1)).resize(
            (224, 224),
            resample=Image.Resampling.BILINEAR,
        )
        rgb = np.asarray(crop, dtype=np.float32) / np.float32(255.0)
        normalized = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
        tensors.append(np.ascontiguousarray(normalized.transpose(2, 0, 1)))
    crops = np.ascontiguousarray(np.stack(tensors)[np.newaxis, ...])
    crops.setflags(write=False)
    return crops


def _array_sha256(values: NDArray[object], dtype: str) -> str:
    canonical = np.ascontiguousarray(values, dtype=dtype)
    return hashlib.sha256(canonical.tobytes()).hexdigest()
