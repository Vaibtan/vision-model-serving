"""One fixed HTTP client for packaged prediction acceptance workflows."""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
import urllib.error
import urllib.request
from typing import Any, Final

from vision_model_serving.pipeline.contracts import PredictionMode
from vision_model_serving.validation.acceptance_contract import (
    PACKAGED_ACCEPTANCE_HISTORY,
)


_POLL_INTERVAL_SECONDS: Final = 0.25


class PackagedHttpError(RuntimeError):
    def __init__(
        self,
        code: str,
        detail: str,
        *,
        status_code: int | None = None,
        retry_after_seconds: float | None = None,
    ):
        self.code = code
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds
        super().__init__(detail)


@dataclass(frozen=True, slots=True)
class PredictionObservation:
    result: dict[str, Any]
    wall_seconds: float
    queue_wait_seconds: float
    states: tuple[str, ...]


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

    def operations(self) -> dict[str, Any]:
        return _request_json(
            urllib.request.Request(f"{self._base_url}/api/v1/operations"),
            timeout=self._request_timeout(10.0),
        )

    def predict(
        self,
        dicom: bytes,
        *,
        mode: PredictionMode,
    ) -> dict[str, Any]:
        return self.predict_observed(dicom, mode=mode).result

    def predict_observed(
        self,
        dicom: bytes,
        *,
        mode: PredictionMode,
    ) -> PredictionObservation:
        if not isinstance(dicom, bytes):
            raise TypeError("DICOM input must be bytes")
        if not isinstance(mode, PredictionMode):
            raise TypeError("mode must be a PredictionMode")
        boundary = "vms-packaged-acceptance"
        fields = {"mode": mode.value}
        if mode is PredictionMode.FULL:
            fields["clinical_history"] = PACKAGED_ACCEPTANCE_HISTORY
        started = time.monotonic()
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
        states = [str(submitted.get("state", "submitted"))]
        deadline = time.monotonic() + self._timeout_seconds
        while time.monotonic() < deadline:
            try:
                status = _request_json(
                    urllib.request.Request(f"{self._base_url}/api/v1/predictions/{prediction_id}"),
                    timeout=self._request_timeout(10.0),
                )
            except PackagedHttpError as error:
                if error.status_code != 429 or error.code != "throttled":
                    raise
                retry_after = error.retry_after_seconds or _POLL_INTERVAL_SECONDS
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                time.sleep(min(max(_POLL_INTERVAL_SECONDS, retry_after), remaining))
                continue
            states.append(str(status["state"]))
            if status["state"] == "succeeded":
                result = _request_json(
                    urllib.request.Request(
                        f"{self._base_url}/api/v1/predictions/{prediction_id}/result"
                    ),
                    timeout=self._request_timeout(10.0),
                )["result"]
                return PredictionObservation(
                    result=result,
                    wall_seconds=time.monotonic() - started,
                    queue_wait_seconds=max(
                        0.0, float(status.get("queue_wait_ms") or 0.0) / 1_000.0
                    ),
                    states=tuple(states),
                )
            if status["state"] in {"failed", "expired"}:
                failure = status.get("failure") or {}
                code = str(failure.get("code") or f"prediction_{status['state']}")
                raise PackagedHttpError(
                    code,
                    f"prediction ended in state {status['state']}",
                )
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
        code = f"http_{error.code}"
        try:
            body = json.loads(message)
            code = str(body.get("error", {}).get("code") or code)
        except (AttributeError, json.JSONDecodeError):
            pass
        retry_after_seconds = _retry_after_seconds(error.headers.get("Retry-After"))
        raise PackagedHttpError(
            code,
            f"HTTP {error.code}: {message}",
            status_code=error.code,
            retry_after_seconds=retry_after_seconds,
        ) from error
    if not isinstance(payload, dict):
        raise RuntimeError("HTTP response must be a JSON object")
    return payload


def _retry_after_seconds(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return seconds if seconds > 0.0 else None


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
