"""One fixed HTTP client for packaged prediction acceptance workflows."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Final

from vision_model_serving.pipeline.contracts import PredictionMode


PACKAGED_ACCEPTANCE_HISTORY: Final = "real public mammogram acceptance."
_POLL_INTERVAL_SECONDS: Final = 0.1


class PackagedPredictionClient:
    """Submit the pinned acceptance request and wait for its bounded result."""

    def __init__(self, base_url: str, *, timeout_seconds: float) -> None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base URL must be a non-empty string")
        if timeout_seconds <= 0:
            raise ValueError("prediction timeout must be positive")
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = float(timeout_seconds)

    def readiness(self) -> dict[str, Any]:
        return _request_json(
            urllib.request.Request(f"{self._base_url}/readyz"),
            timeout=self._request_timeout(10.0),
        )

    def model_inventory(self) -> dict[str, Any]:
        return _request_json(
            urllib.request.Request(f"{self._base_url}/api/v1/models"),
            timeout=self._request_timeout(10.0),
        )

    def predict(
        self,
        dicom: bytes,
        *,
        mode: PredictionMode,
    ) -> dict[str, Any]:
        if not isinstance(dicom, bytes):
            raise TypeError("DICOM input must be bytes")
        if not isinstance(mode, PredictionMode):
            raise TypeError("mode must be a PredictionMode")
        boundary = "vms-packaged-acceptance"
        fields = {"mode": mode.value}
        if mode is PredictionMode.FULL:
            fields["clinical_history"] = PACKAGED_ACCEPTANCE_HISTORY
        submitted = _request_json(
            urllib.request.Request(
                f"{self._base_url}/api/v1/predictions",
                data=_multipart_body(boundary, dicom, fields),
                headers={
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                    "Prefer": "respond-async",
                },
                method="POST",
            ),
            timeout=self._request_timeout(30.0),
        )
        prediction_id = submitted["prediction_id"]
        deadline = time.monotonic() + self._timeout_seconds
        while time.monotonic() < deadline:
            status = _request_json(
                urllib.request.Request(
                    f"{self._base_url}/api/v1/predictions/{prediction_id}"
                ),
                timeout=self._request_timeout(10.0),
            )
            if status["state"] == "succeeded":
                return _request_json(
                    urllib.request.Request(
                        f"{self._base_url}/api/v1/predictions/{prediction_id}/result"
                    ),
                    timeout=self._request_timeout(10.0),
                )["result"]
            if status["state"] in {"failed", "expired"}:
                raise RuntimeError(f"prediction ended in state {status['state']}")
            time.sleep(_POLL_INTERVAL_SECONDS)
        raise TimeoutError("prediction did not finish before the acceptance deadline")

    def _request_timeout(self, maximum: float) -> float:
        return min(maximum, self._timeout_seconds)


def _request_json(request: urllib.request.Request, *, timeout: float) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        message = error.read(4096).decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {message}") from error
    if not isinstance(payload, dict):
        raise RuntimeError("HTTP response must be a JSON object")
    return payload


def _multipart_body(boundary: str, dicom: bytes, fields: dict[str, str]) -> bytes:
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            (
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode(),
                b"\r\n",
            )
        )
    chunks.extend(
        (
            f"--{boundary}\r\n".encode(),
            b'Content-Disposition: form-data; name="dicom"; filename="input.dcm"\r\n',
            b"Content-Type: application/dicom\r\n\r\n",
            dicom,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        )
    )
    return b"".join(chunks)
