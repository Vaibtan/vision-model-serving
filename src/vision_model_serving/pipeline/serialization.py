"""Serialization at the prediction module boundary."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
from typing import cast

from .contracts import PredictionResult


def prediction_to_dict(result: PredictionResult) -> dict[str, object]:
    """Return plain JSON-compatible values without retaining source objects."""

    if not isinstance(result, PredictionResult):
        raise TypeError("result must be a PredictionResult")
    return cast(dict[str, object], _plain_value(result))


def _plain_value(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _plain_value(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, (tuple, list)):
        return [_plain_value(item) for item in value]
    raise TypeError(f"prediction result contains unsupported {type(value).__name__}")
