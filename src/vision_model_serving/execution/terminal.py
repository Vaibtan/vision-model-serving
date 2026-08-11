"""Strict terminal-failure marker shared by RQ workers and the gateway."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math


def terminal_key(key_prefix: str, prediction_id: str) -> str:
    if not key_prefix or not prediction_id:
        raise ValueError("terminal marker identifiers must not be empty")
    return f"{key_prefix}:terminal:{prediction_id}"


class TerminalFailureKind(str, Enum):
    CASE_FAILED = "prediction_case_failed"
    EXECUTION_FAILED = "prediction_execution_failed"
    RESERVATION_EXPIRED = "prediction_reservation_expired"
    RUNTIME_UNAVAILABLE = "prediction_runtime_unavailable"
    TIMEOUT = "prediction_timeout"


@dataclass(frozen=True, slots=True)
class TerminalFailureMarker:
    kind: TerminalFailureKind
    detail: str
    retryable: bool
    submitted_at: float
    completed_at: float

    def __post_init__(self) -> None:
        if not isinstance(self.kind, TerminalFailureKind):
            raise TypeError("terminal failure kind is invalid")
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("terminal failure detail must not be empty")
        if not isinstance(self.retryable, bool):
            raise TypeError("terminal failure retryability must be boolean")
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in (self.submitted_at, self.completed_at)
        ):
            raise ValueError("terminal failure timestamps must be finite")
        if self.completed_at < self.submitted_at:
            raise ValueError("terminal failure completion precedes submission")

    def to_json(self) -> str:
        return json.dumps(
            {
                "schema_version": 1,
                "kind": self.kind.value,
                "detail": self.detail,
                "retryable": self.retryable,
                "submitted_at": self.submitted_at,
                "completed_at": self.completed_at,
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, payload: str | bytes) -> TerminalFailureMarker:
        value = json.loads(payload)
        if not isinstance(value, dict) or set(value) != {
            "schema_version",
            "kind",
            "detail",
            "retryable",
            "submitted_at",
            "completed_at",
        }:
            raise ValueError("terminal failure marker shape is invalid")
        if (
            value["schema_version"] != 1
            or not isinstance(value["kind"], str)
            or not isinstance(value["detail"], str)
            or not isinstance(value["retryable"], bool)
            or any(
                isinstance(value[name], bool) or not isinstance(value[name], (int, float))
                for name in ("submitted_at", "completed_at")
            )
        ):
            raise ValueError("terminal failure marker value is invalid")
        return cls(
            kind=TerminalFailureKind(value["kind"]),
            detail=value["detail"],
            retryable=value["retryable"],
            submitted_at=float(value["submitted_at"]),
            completed_at=float(value["completed_at"]),
        )


def store_terminal_marker(
    redis: object,
    *,
    key_prefix: str,
    prediction_id: str,
    marker: TerminalFailureMarker,
    ttl_seconds: int,
) -> bool:
    if not isinstance(marker, TerminalFailureMarker):
        raise TypeError("marker must be a TerminalFailureMarker")
    if ttl_seconds < 1:
        raise ValueError("terminal marker TTL must be positive")
    return bool(
        redis.set(
            terminal_key(key_prefix, prediction_id),
            marker.to_json(),
            nx=True,
            ex=ttl_seconds,
        )
    )
