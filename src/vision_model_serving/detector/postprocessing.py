"""Deterministic detector preprocessing and proposal selection."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import math

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from vision_model_serving.dicom import GeometryLedger
from vision_model_serving.failures import CaseInputFailure


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DetectorAdapterError(RuntimeError):
    """Base class for stable detector-adapter failures."""

    code = "detector_adapter_failed"

    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(f"{self.code}: {detail}")


class DetectorOutputError(DetectorAdapterError):
    code = "detector_output_invalid"


class NoValidProposalsError(CaseInputFailure, DetectorAdapterError):
    code = "detector_no_valid_proposals"


@dataclass(frozen=True, slots=True)
class DetectorInput:
    tensor: NDArray[np.float32] = field(repr=False)
    source_size: tuple[int, int]
    resized_size: tuple[int, int]
    resize_short_edge: int
    resize_max_edge: int
    normalization_mean: tuple[float, float, float]
    normalization_std: tuple[float, float, float]


class DetectorPreprocessor:
    """Apply the pinned FocalNet-DINO evaluation image transform."""

    def __init__(self, *, short_edge: int = 800, max_edge: int = 1333):
        for name, value in (("short edge", short_edge), ("max edge", max_edge)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self._short_edge = short_edge
        self._max_edge = max_edge

    def prepare(self, pixels: NDArray[np.uint8]) -> DetectorInput:
        """Return an immutable CHW float32 tensor without importing PyTorch."""

        if not isinstance(pixels, np.ndarray):
            raise ValueError("canonical pixels must be a NumPy array")
        if pixels.ndim != 2 or pixels.dtype != np.uint8:
            raise ValueError("canonical pixels must be a two-dimensional uint8 array")
        source_height, source_width = pixels.shape
        if source_width <= 0 or source_height <= 0:
            raise ValueError("canonical pixels must not be empty")
        resized_width, resized_height = self._resized_size(
            source_width,
            source_height,
        )
        image = Image.fromarray(pixels, mode="L").convert("RGB")
        image = image.resize(
            (resized_width, resized_height),
            resample=Image.Resampling.BILINEAR,
        )
        rgb = np.asarray(image, dtype=np.float32) / np.float32(255.0)
        normalized = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
        tensor = np.ascontiguousarray(normalized.transpose(2, 0, 1))
        tensor.setflags(write=False)
        return DetectorInput(
            tensor=tensor,
            source_size=(source_width, source_height),
            resized_size=(resized_width, resized_height),
            resize_short_edge=self._short_edge,
            resize_max_edge=self._max_edge,
            normalization_mean=tuple(float(value) for value in _IMAGENET_MEAN),
            normalization_std=tuple(float(value) for value in _IMAGENET_STD),
        )

    def _resized_size(self, width: int, height: int) -> tuple[int, int]:
        size = self._short_edge
        minimum = float(min(width, height))
        maximum = float(max(width, height))
        if maximum / minimum * size > self._max_edge:
            size = int(round(self._max_edge * minimum / maximum))
        if width < height:
            resized_width = size
            resized_height = int(size * height / width)
        elif height < width:
            resized_height = size
            resized_width = int(size * width / height)
        else:
            resized_width = resized_height = size
        return resized_width, resized_height


@dataclass(frozen=True, slots=True)
class DetectorWarning:
    code: str
    detail: str


@dataclass(frozen=True, slots=True)
class DetectorProposal:
    rank: int
    query_index: int
    class_index: int
    raw_logit: float
    score: float
    normalized_cxcywh: tuple[float, float, float, float]
    normalized_xyxy: tuple[float, float, float, float]
    canonical_xyxy: tuple[float, float, float, float]
    original_xyxy: tuple[float, float, float, float]
    padded: bool = False
    duplicate_of_rank: int | None = None


@dataclass(frozen=True, slots=True)
class ProposalSelection:
    raw_logits: NDArray[np.float32] = field(repr=False)
    raw_scores: NDArray[np.float32] = field(repr=False)
    raw_boxes_cxcywh: NDArray[np.float32] = field(repr=False)
    prediction_sha256: str
    top_candidates: tuple[DetectorProposal, ...]
    post_nms: tuple[DetectorProposal, ...]
    classifier_rois: tuple[DetectorProposal, ...]
    warnings: tuple[DetectorWarning, ...]

    def display_proposals(self, *, min_score: float) -> tuple[DetectorProposal, ...]:
        """Filter presentation proposals without touching classifier ROI selection."""

        if not math.isfinite(float(min_score)):
            raise ValueError("display score must be finite")
        return tuple(proposal for proposal in self.post_nms if proposal.score >= min_score)


class DetectorPostprocessor:
    """Convert raw DINO outputs into the fixed MMBCD proposal contract."""

    def __init__(
        self,
        *,
        num_select: int = 300,
        nms_iou_threshold: float = 0.1,
        roi_count: int = 8,
    ):
        for name, value in (("num_select", num_select), ("roi_count", roi_count)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(float(nms_iou_threshold)) or not (0.0 <= nms_iou_threshold <= 1.0):
            raise ValueError("NMS IoU threshold must lie between zero and one")
        self._num_select = num_select
        self._nms_iou_threshold = float(nms_iou_threshold)
        self._roi_count = roi_count

    def process(
        self,
        raw_logits: NDArray[np.generic],
        raw_boxes_cxcywh: NDArray[np.generic],
        geometry: GeometryLedger,
    ) -> ProposalSelection:
        logits, boxes = _validated_outputs(raw_logits, raw_boxes_cxcywh)
        query_logits = logits[0]
        query_boxes = boxes[0]
        class_count = query_logits.shape[1]
        scores = _sigmoid(query_logits.reshape(-1))
        raw_scores = np.ascontiguousarray(scores.reshape(logits.shape))
        raw_scores.setflags(write=False)
        order = np.argsort(-scores, kind="stable")[: self._num_select]

        candidates: list[DetectorProposal] = []
        rejected = 0
        for rank, flat_index_value in enumerate(order):
            flat_index = int(flat_index_value)
            query_index = flat_index // class_count
            class_index = flat_index % class_count
            cxcywh = tuple(float(value) for value in query_boxes[query_index])
            # Reference MMBCD behavior keeps boxes unclipped: NMS runs over
            # raw normalized coordinates and border-overhanging crops are
            # zero-padded downstream, so clamping here would change both the
            # surviving proposal set and the classifier crop content.
            xyxy = _xyxy(cxcywh)
            if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
                rejected += 1
                continue
            canonical = (
                xyxy[0] * geometry.canonical_width,
                xyxy[1] * geometry.canonical_height,
                xyxy[2] * geometry.canonical_width,
                xyxy[3] * geometry.canonical_height,
            )
            candidates.append(
                DetectorProposal(
                    rank=rank,
                    query_index=query_index,
                    class_index=class_index,
                    raw_logit=float(query_logits[query_index, class_index]),
                    score=float(scores[flat_index]),
                    normalized_cxcywh=cxcywh,
                    normalized_xyxy=xyxy,
                    canonical_xyxy=canonical,
                    original_xyxy=geometry.to_original_box(canonical),
                )
            )

        warnings: list[DetectorWarning] = []
        if rejected:
            warnings.append(
                DetectorWarning(
                    "degenerate_proposals_rejected",
                    "one or more top-ranked proposals had no positive extent",
                )
            )
        if not candidates:
            raise NoValidProposalsError("no proposal with positive extent remained")

        candidate_boxes = np.asarray(
            [proposal.normalized_xyxy for proposal in candidates],
            dtype=np.float64,
        )
        retained_indices = strict_nms(candidate_boxes, self._nms_iou_threshold)
        post_nms = tuple(candidates[index] for index in retained_indices)
        classifier_rois = list(post_nms[: self._roi_count])
        if len(classifier_rois) < self._roi_count:
            original_count = len(classifier_rois)
            for index in range(self._roi_count - original_count):
                source = classifier_rois[index % original_count]
                classifier_rois.append(
                    replace(
                        source,
                        padded=True,
                        duplicate_of_rank=source.rank,
                    )
                )
            warnings.append(
                DetectorWarning(
                    "classifier_rois_padded",
                    "fewer than the required proposals remained and were duplicated round-robin",
                )
            )

        return ProposalSelection(
            raw_logits=logits,
            raw_scores=raw_scores,
            raw_boxes_cxcywh=boxes,
            prediction_sha256=prediction_sha256(logits, boxes),
            top_candidates=tuple(candidates),
            post_nms=post_nms,
            classifier_rois=tuple(classifier_rois),
            warnings=tuple(warnings),
        )


def strict_nms(
    boxes_xyxy: NDArray[np.generic],
    threshold: float,
) -> tuple[int, ...]:
    """Return stable indexes, suppressing only when IoU is strictly greater."""

    boxes = np.asarray(boxes_xyxy, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1:] != (4,):
        raise ValueError("NMS boxes must have shape [N, 4]")
    if not np.isfinite(boxes).all():
        raise ValueError("NMS boxes must be finite")
    if np.any(boxes[:, 2] < boxes[:, 0]) or np.any(boxes[:, 3] < boxes[:, 1]):
        raise ValueError("NMS boxes must use ordered XYXY coordinates")
    if not math.isfinite(float(threshold)) or not 0.0 <= threshold <= 1.0:
        raise ValueError("NMS IoU threshold must lie between zero and one")

    retained: list[int] = []
    for candidate in range(len(boxes)):
        if any(_iou_xyxy(boxes[selected], boxes[candidate]) > threshold for selected in retained):
            continue
        retained.append(candidate)
    return tuple(retained)


def prediction_sha256(
    raw_logits: NDArray[np.generic],
    raw_boxes_cxcywh: NDArray[np.generic],
) -> str:
    """Hash raw predictions using the validated little-endian float32 contract."""

    digest = hashlib.sha256()
    for values in (raw_logits, raw_boxes_cxcywh):
        array = np.ascontiguousarray(values, dtype="<f4")
        digest.update(array.tobytes())
    return digest.hexdigest()


def _validated_outputs(
    raw_logits: NDArray[np.generic],
    raw_boxes: NDArray[np.generic],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    logits = np.asarray(raw_logits)
    boxes = np.asarray(raw_boxes)
    if logits.ndim == 2:
        logits = logits[np.newaxis, ...]
    if boxes.ndim == 2:
        boxes = boxes[np.newaxis, ...]
    if logits.ndim != 3 or boxes.ndim != 3:
        raise DetectorOutputError("raw outputs must include one batch dimension")
    if logits.shape[0] != 1 or boxes.shape[0] != 1:
        raise DetectorOutputError("only detector batch size one is supported")
    if logits.shape[1] <= 0 or logits.shape[2] <= 0:
        raise DetectorOutputError("detector logits must include queries and classes")
    if boxes.shape != (1, logits.shape[1], 4):
        raise DetectorOutputError("detector boxes do not match the query contract")
    if not np.isfinite(logits).all() or not np.isfinite(boxes).all():
        raise DetectorOutputError("detector outputs contain non-finite values")
    canonical_logits = np.ascontiguousarray(logits, dtype=np.float32)
    canonical_boxes = np.ascontiguousarray(boxes, dtype=np.float32)
    canonical_logits.setflags(write=False)
    canonical_boxes.setflags(write=False)
    return canonical_logits, canonical_boxes


def _sigmoid(values: NDArray[np.float32]) -> NDArray[np.float32]:
    result = np.empty(values.shape, dtype=np.float32)
    positive = values >= 0
    result[positive] = np.float32(1.0) / (np.float32(1.0) + np.exp(-values[positive]))
    negative_exponential = np.exp(values[~positive])
    result[~positive] = negative_exponential / (np.float32(1.0) + negative_exponential)
    return result


def _xyxy(
    cxcywh: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    center_x, center_y, width, height = cxcywh
    return (
        center_x - width / 2.0,
        center_y - height / 2.0,
        center_x + width / 2.0,
        center_y + height / 2.0,
    )


def _iou_xyxy(first: NDArray[np.float64], second: NDArray[np.float64]) -> float:
    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(
        0.0,
        min(first[3], second[3]) - max(first[1], second[1]),
    )
    intersection = intersection_width * intersection_height
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(
        0.0,
        second[3] - second[1],
    )
    union = first_area + second_area - intersection
    return 0.0 if union <= 0.0 else float(intersection / union)
