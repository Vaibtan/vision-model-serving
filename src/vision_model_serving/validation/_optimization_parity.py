"""Private detector parity implementation for optimization evidence."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class DetectorParityRecord:
    """One raw or selected detector proposal at the semantic parity seam."""

    class_index: int
    raw_logit: float
    score: float
    normalized_cxcywh: tuple[float, float, float, float]
    normalized_xyxy: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if (
            isinstance(self.class_index, bool)
            or not isinstance(self.class_index, int)
            or self.class_index < 0
        ):
            raise ValueError("detector class index must be non-negative")
        if (
            isinstance(self.raw_logit, bool)
            or not isinstance(self.raw_logit, (int, float))
            or not math.isfinite(self.raw_logit)
        ):
            raise ValueError("detector raw logit must be finite")
        if (
            isinstance(self.score, bool)
            or not isinstance(self.score, (int, float))
            or not math.isfinite(self.score)
            or not 0.0 <= self.score <= 1.0
        ):
            raise ValueError("detector score must lie between zero and one")
        center_x, center_y, width, height = self.normalized_cxcywh
        if (
            any(not math.isfinite(value) for value in self.normalized_cxcywh)
            or not (0.0 <= center_x <= 1.0 and 0.0 <= center_y <= 1.0)
            or not (0.0 <= width <= 1.0 and 0.0 <= height <= 1.0)
        ):
            raise ValueError("detector box must be a valid normalized cxcywh box")
        left, top, right, bottom = self.normalized_xyxy
        if (
            any(not math.isfinite(value) for value in self.normalized_xyxy)
            or not (0.0 <= left <= right <= 1.0)
            or not (0.0 <= top <= bottom <= 1.0)
        ):
            raise ValueError("detector box must be a valid normalized xyxy box")


@dataclass(frozen=True, slots=True)
class DetectorParityComparison:
    matched_pairs: tuple[tuple[int, int], ...]
    unmatched_baseline: tuple[int, ...]
    unmatched_candidate: tuple[int, ...]
    max_score_difference: float
    max_logit_difference: float
    max_box_coordinate_difference: float
    minimum_observed_iou: float

    @property
    def passed(self) -> bool:
        return not self.unmatched_baseline and not self.unmatched_candidate


def compare_detector_records(
    baseline: Sequence[DetectorParityRecord],
    candidate: Sequence[DetectorParityRecord],
    *,
    score_tolerance: float,
    logit_tolerance: float,
    box_tolerance: float,
    minimum_iou: float,
) -> DetectorParityComparison:
    """Match raw/selected proposals semantically instead of by array index."""

    for name, value in (
        ("score", score_tolerance),
        ("logit", logit_tolerance),
        ("box", box_tolerance),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} tolerance must be finite and non-negative")
    if not math.isfinite(minimum_iou) or not 0.0 <= minimum_iou <= 1.0:
        raise ValueError("minimum IoU must lie between zero and one")

    possible: list[list[tuple[float, float, float, float, int]]] = []
    pair_metrics: dict[tuple[int, int], tuple[float, float, float, float]] = {}
    for baseline_index, expected in enumerate(baseline):
        matches: list[tuple[float, float, float, float, int]] = []
        for candidate_index, observed in enumerate(candidate):
            score_difference = abs(expected.score - observed.score)
            logit_difference = abs(expected.raw_logit - observed.raw_logit)
            box_difference = max(
                abs(left - right)
                for left, right in zip(
                    expected.normalized_cxcywh,
                    observed.normalized_cxcywh,
                    strict=True,
                )
            )
            iou = _box_iou(expected.normalized_xyxy, observed.normalized_xyxy)
            if (
                expected.class_index == observed.class_index
                and score_difference <= score_tolerance
                and logit_difference <= logit_tolerance
                and box_difference <= box_tolerance
                and iou >= minimum_iou
            ):
                matches.append(
                    (
                        score_difference,
                        logit_difference,
                        box_difference,
                        -iou,
                        candidate_index,
                    )
                )
                pair_metrics[(baseline_index, candidate_index)] = (
                    score_difference,
                    logit_difference,
                    box_difference,
                    iou,
                )
        possible.append(sorted(matches))

    candidate_matches: dict[int, int] = {}

    def assign(baseline_index: int, visited: set[int]) -> bool:
        for *_, candidate_index in possible[baseline_index]:
            if candidate_index in visited:
                continue
            visited.add(candidate_index)
            previous = candidate_matches.get(candidate_index)
            if previous is None or assign(previous, visited):
                candidate_matches[candidate_index] = baseline_index
                return True
        return False

    for baseline_index in range(len(baseline)):
        assign(baseline_index, set())
    pairs = tuple(
        sorted(
            (baseline_index, candidate_index)
            for candidate_index, baseline_index in candidate_matches.items()
        )
    )
    matched_baseline = {left for left, _ in pairs}
    matched_candidate = {right for _, right in pairs}
    metrics = [pair_metrics[pair] for pair in pairs]
    return DetectorParityComparison(
        matched_pairs=pairs,
        unmatched_baseline=tuple(
            index for index in range(len(baseline)) if index not in matched_baseline
        ),
        unmatched_candidate=tuple(
            index for index in range(len(candidate)) if index not in matched_candidate
        ),
        max_score_difference=max((value[0] for value in metrics), default=0.0),
        max_logit_difference=max((value[1] for value in metrics), default=0.0),
        max_box_coordinate_difference=max(
            (value[2] for value in metrics),
            default=0.0,
        ),
        minimum_observed_iou=min((value[3] for value in metrics), default=1.0),
    )


def _box_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    if union > 0.0:
        return intersection / union
    return 1.0 if first == second else 0.0
