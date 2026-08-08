"""Serialization at the prediction module boundary."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
from functools import lru_cache
from types import UnionType
from typing import Any, Union, cast, get_args, get_origin, get_type_hints

from .contracts import PredictionResult


def prediction_to_dict(result: PredictionResult) -> dict[str, object]:
    """Return plain JSON-compatible values without retaining source objects."""

    if not isinstance(result, PredictionResult):
        raise TypeError("result must be a PredictionResult")
    return cast(dict[str, object], _plain_value(result))


def prediction_from_dict(payload: object) -> PredictionResult:
    """Rebuild a validated typed result from trusted JSON-compatible values."""

    return cast(PredictionResult, _typed_value(payload, PredictionResult))


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


def _typed_value(value: object, annotation: object) -> object:
    origin = get_origin(annotation)
    if origin in {Union, UnionType}:
        alternatives = get_args(annotation)
        if value is None and type(None) in alternatives:
            return None
        for alternative in alternatives:
            if alternative is type(None):
                continue
            try:
                return _typed_value(value, alternative)
            except (TypeError, ValueError):
                pass
        raise TypeError("prediction value does not match its declared alternatives")
    if origin is tuple:
        if not isinstance(value, list):
            raise TypeError("prediction tuple must be encoded as a list")
        arguments = get_args(annotation)
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return tuple(_typed_value(item, arguments[0]) for item in value)
        if len(value) != len(arguments):
            raise ValueError("prediction tuple has an unexpected length")
        return tuple(
            _typed_value(item, item_type)
            for item, item_type in zip(value, arguments, strict=True)
        )
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        try:
            return annotation(value)
        except (TypeError, ValueError) as error:
            raise ValueError("prediction enum value is invalid") from error
    if isinstance(annotation, type) and is_dataclass(annotation):
        if not isinstance(value, dict) or not all(
            isinstance(key, str) for key in value
        ):
            raise TypeError("prediction object must be encoded as a string-keyed map")
        declared_fields = fields(annotation)
        declared_names = {item.name for item in declared_fields}
        if set(value) != declared_names:
            raise ValueError("prediction object fields do not match the contract")
        hints = _field_hints(annotation)
        return annotation(
            **{
                item.name: _typed_value(value[item.name], hints[item.name])
                for item in declared_fields
            }
        )
    if annotation is str:
        if not isinstance(value, str):
            raise TypeError("prediction text value is invalid")
        return value
    if annotation is bool:
        if not isinstance(value, bool):
            raise TypeError("prediction boolean value is invalid")
        return value
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("prediction integer value is invalid")
        return value
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("prediction numeric value is invalid")
        return float(value)
    if annotation is Any:
        return value
    raise TypeError("prediction contract contains an unsupported annotation")


@lru_cache(maxsize=None)
def _field_hints(contract: type[object]) -> dict[str, object]:
    return get_type_hints(contract)
