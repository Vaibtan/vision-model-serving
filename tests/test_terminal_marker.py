from __future__ import annotations

from pathlib import Path
import sys
import unittest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from vision_model_serving.execution.terminal import (  # noqa: E402
    TerminalFailureKind,
    TerminalFailureMarker,
)


class TerminalFailureMarkerTests(unittest.TestCase):
    def test_round_trip_preserves_typed_values(self) -> None:
        marker = TerminalFailureMarker(
            TerminalFailureKind.RUNTIME_UNAVAILABLE,
            "prediction runtime was unavailable",
            True,
            10.0,
            11.0,
        )

        self.assertEqual(TerminalFailureMarker.from_json(marker.to_json()), marker)

    def test_string_retryable_value_is_rejected(self) -> None:
        payload = (
            '{"schema_version":1,"kind":"prediction_runtime_unavailable",'
            '"detail":"unavailable","retryable":"false",'
            '"submitted_at":10,"completed_at":11}'
        )

        with self.assertRaises(ValueError):
            TerminalFailureMarker.from_json(payload)

    def test_unknown_or_extra_fields_are_rejected(self) -> None:
        payload = (
            '{"schema_version":1,"kind":"unknown","detail":"failed",'
            '"retryable":false,"submitted_at":10,"completed_at":11,"extra":1}'
        )

        with self.assertRaises(ValueError):
            TerminalFailureMarker.from_json(payload)


if __name__ == "__main__":
    unittest.main()
