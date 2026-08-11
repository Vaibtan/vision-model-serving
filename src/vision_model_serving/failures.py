"""Typed failure classification shared across pipeline and process seams."""

from __future__ import annotations

from enum import Enum


class FailureKind(str, Enum):
    CASE_INPUT = "case_input"
    RUNTIME = "runtime"


class CaseInputFailure:
    """Marker mixin for data-dependent failures that preserve runtime health."""

    failure_kind = FailureKind.CASE_INPUT


def is_case_input_failure(error: BaseException) -> bool:
    return isinstance(error, CaseInputFailure)
