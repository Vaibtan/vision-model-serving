from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from vision_model_serving.pipeline import PredictionMode  # noqa: E402
from vision_model_serving.validation.packaged_http import (  # noqa: E402
    PACKAGED_ACCEPTANCE_HISTORY,
    PackagedPredictionClient,
)


class _JsonResponse(BytesIO):
    def __init__(self, value: dict[str, object]) -> None:
        super().__init__(json.dumps(value).encode("utf-8"))

    def __enter__(self) -> _JsonResponse:
        return self

    def __exit__(self, *_details: object) -> None:
        self.close()


class PackagedPredictionClientTests(unittest.TestCase):
    def test_full_request_uses_the_pinned_history_and_waits_for_result(self) -> None:
        responses = [
            _JsonResponse({"prediction_id": "accepted"}),
            _JsonResponse({"state": "queued"}),
            _JsonResponse({"state": "succeeded"}),
            _JsonResponse({"result": {"mode": "full"}}),
        ]
        client = PackagedPredictionClient("http://service/", timeout_seconds=10)

        with patch(
            "vision_model_serving.validation.packaged_http.urllib.request.urlopen",
            side_effect=responses,
        ) as urlopen, patch(
            "vision_model_serving.validation.packaged_http.time.sleep"
        ) as sleep:
            result = client.predict(b"DICOM", mode=PredictionMode.FULL)

        submission = urlopen.call_args_list[0].args[0]
        self.assertEqual(submission.full_url, "http://service/api/v1/predictions")
        self.assertIn(b'name="mode"\r\n\r\nfull', submission.data)
        self.assertIn(PACKAGED_ACCEPTANCE_HISTORY.encode(), submission.data)
        self.assertEqual(result, {"mode": "full"})
        self.assertEqual(
            [call.args[0].full_url for call in urlopen.call_args_list[1:]],
            [
                "http://service/api/v1/predictions/accepted",
                "http://service/api/v1/predictions/accepted",
                "http://service/api/v1/predictions/accepted/result",
            ],
        )
        sleep.assert_called_once_with(0.1)

    def test_detection_request_omits_clinical_history(self) -> None:
        responses = [
            _JsonResponse({"prediction_id": "accepted"}),
            _JsonResponse({"state": "succeeded"}),
            _JsonResponse({"result": {"mode": "detection"}}),
        ]
        client = PackagedPredictionClient("http://service", timeout_seconds=10)

        with patch(
            "vision_model_serving.validation.packaged_http.urllib.request.urlopen",
            side_effect=responses,
        ) as urlopen:
            result = client.predict(b"DICOM", mode=PredictionMode.DETECTION)

        submission = urlopen.call_args_list[0].args[0]
        self.assertIn(b'name="mode"\r\n\r\ndetection', submission.data)
        self.assertNotIn(PACKAGED_ACCEPTANCE_HISTORY.encode(), submission.data)
        self.assertEqual(result, {"mode": "detection"})

    def test_terminal_prediction_states_do_not_retry_or_fallback(self) -> None:
        for state in ("failed", "expired"):
            with self.subTest(state=state):
                client = PackagedPredictionClient("http://service", timeout_seconds=10)
                responses = [
                    _JsonResponse({"prediction_id": "accepted"}),
                    _JsonResponse({"state": state}),
                ]

                with patch(
                    "vision_model_serving.validation.packaged_http.urllib.request.urlopen",
                    side_effect=responses,
                ) as urlopen, self.assertRaisesRegex(
                    RuntimeError,
                    f"state {state}",
                ):
                    client.predict(b"DICOM", mode=PredictionMode.DETECTION)

                self.assertEqual(urlopen.call_count, 2)


if __name__ == "__main__":
    unittest.main()
