"""Ephemeral filesystem storage for private prediction payloads and results."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
import os
from pathlib import Path
import re
import secrets
import shutil
from time import time
from typing import Callable

from vision_model_serving.pipeline.contracts import (
    CaseInput,
    PredictionMode,
    PredictionResult,
)
from vision_model_serving.pipeline.serialization import (
    prediction_from_dict,
    prediction_to_dict,
)

from .contracts import PredictionGatewayError, PredictionId, PredictionRequest
from .fingerprinting import request_fingerprint


_LOCATOR_PATTERN = re.compile(r"^[A-Za-z0-9_-]{32}$")


class JobPayloadNotFound(PredictionGatewayError):
    code = "prediction_payload_not_found"


class JobResultNotFound(PredictionGatewayError):
    code = "prediction_stored_result_not_found"


class JobResultConflict(PredictionGatewayError):
    code = "prediction_result_write_conflict"


@dataclass(frozen=True, slots=True)
class StoredJobPayload:
    locator: str
    request_fingerprint: str
    expires_at: float


class EphemeralJobStore:
    """Keep private request bytes off the broker under opaque locators."""

    def __init__(self, root: Path, *, clock: Callable[[], float] = time):
        self._root = root.resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._clock = clock

    def store_request(
        self,
        prediction_id: PredictionId,
        request: PredictionRequest,
        *,
        ttl_seconds: float,
    ) -> StoredJobPayload:
        if ttl_seconds <= 0:
            raise ValueError("request TTL must be positive")
        payload = request.case.dicom_stream.read()
        if not isinstance(payload, bytes):
            raise TypeError("DICOM stream must return bytes")
        fingerprint = request_fingerprint(request, payload)
        expires_at = self._clock() + ttl_seconds
        locator = secrets.token_urlsafe(24)
        destination = self._directory(locator)
        temporary = self._root / f".tmp-{secrets.token_hex(16)}"
        try:
            temporary.mkdir(mode=0o700)
            (temporary / "input.dcm").write_bytes(payload)
            (temporary / "request.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "prediction_id": str(prediction_id),
                        "mode": request.mode.value,
                        "clinical_history": request.case.clinical_history,
                        "request_fingerprint": fingerprint,
                        "expires_at": expires_at,
                    },
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            for item in temporary.iterdir():
                item.chmod(0o600)
            temporary.replace(destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return StoredJobPayload(locator, fingerprint, expires_at)

    def load_request(
        self,
        prediction_id: PredictionId,
        locator: str,
    ) -> PredictionRequest:
        directory = self._directory(locator)
        try:
            metadata = json.loads(
                (directory / "request.json").read_text(encoding="utf-8")
            )
            payload = (directory / "input.dcm").read_bytes()
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            raise JobPayloadNotFound(
                "prediction request payload is unavailable"
            ) from None
        if (
            not isinstance(metadata, dict)
            or metadata.get("schema_version") != 1
            or metadata.get("prediction_id") != str(prediction_id)
        ):
            raise JobPayloadNotFound("prediction request payload is unavailable")
        try:
            mode = PredictionMode(metadata["mode"])
            history = metadata["clinical_history"]
            if history is not None and not isinstance(history, str):
                raise TypeError
        except (KeyError, TypeError, ValueError):
            raise JobPayloadNotFound(
                "prediction request payload is unavailable"
            ) from None
        return PredictionRequest(
            case=CaseInput(BytesIO(payload), history),
            mode=mode,
        )

    def purge_request(self, locator: str) -> None:
        directory = self._directory(locator)
        for name in ("input.dcm", "request.json"):
            try:
                (directory / name).unlink()
            except FileNotFoundError:
                pass

    def store_result(
        self,
        locator: str,
        result: PredictionResult,
        *,
        ttl_seconds: float,
    ) -> None:
        if not isinstance(result, PredictionResult):
            raise TypeError("result must be a PredictionResult")
        if ttl_seconds <= 0:
            raise ValueError("result TTL must be positive")
        directory = self._directory(locator)
        if not directory.is_dir():
            raise JobPayloadNotFound("prediction payload locator is unavailable")
        envelope = {
            "schema_version": 1,
            "expires_at": self._clock() + ttl_seconds,
            "result": prediction_to_dict(result),
        }
        encoded = json.dumps(
            envelope,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        temporary = directory / f".result-{secrets.token_hex(16)}.tmp"
        destination = directory / "result.json"
        temporary.write_text(encoded, encoding="utf-8")
        temporary.chmod(0o600)
        try:
            try:
                os.link(temporary, destination)
            except FileExistsError:
                try:
                    existing = json.loads(destination.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    raise JobResultConflict(
                        "stored prediction result is inconsistent"
                    ) from None
                if not isinstance(existing, dict) or existing.get("result") != envelope[
                    "result"
                ]:
                    raise JobResultConflict(
                        "stored prediction result is inconsistent"
                    )
        finally:
            temporary.unlink(missing_ok=True)
        self.purge_request(locator)

    def load_result(self, locator: str) -> PredictionResult:
        try:
            envelope = json.loads(
                (self._directory(locator) / "result.json").read_text(
                    encoding="utf-8"
                )
            )
            if (
                not isinstance(envelope, dict)
                or envelope.get("schema_version") != 1
                or self._clock() >= float(envelope["expires_at"])
            ):
                raise ValueError
            return prediction_from_dict(envelope["result"])
        except (FileNotFoundError, OSError, KeyError, TypeError, ValueError):
            raise JobResultNotFound(
                "prediction result payload is unavailable"
            ) from None

    def cleanup_expired(self) -> int:
        removed = 0
        now = self._clock()
        for directory in self._root.iterdir():
            if (
                directory.is_symlink()
                or not directory.is_dir()
                or not _LOCATOR_PATTERN.fullmatch(directory.name)
            ):
                continue
            metadata_path = (
                directory / "result.json"
                if (directory / "result.json").is_file()
                else directory / "request.json"
            )
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                expired = now >= float(metadata["expires_at"])
            except (
                FileNotFoundError,
                OSError,
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError,
            ):
                expired = True
            if expired:
                shutil.rmtree(directory)
                removed += 1
        return removed

    def discard_job(self, locator: str) -> None:
        directory = self._directory(locator)
        if directory.is_symlink():
            raise JobPayloadNotFound("prediction payload locator is invalid")
        try:
            shutil.rmtree(directory)
        except FileNotFoundError:
            pass

    def _directory(self, locator: str) -> Path:
        if not isinstance(locator, str) or not _LOCATOR_PATTERN.fullmatch(locator):
            raise JobPayloadNotFound("prediction payload locator is invalid")
        directory = self._root / locator
        if directory.is_symlink():
            raise JobPayloadNotFound("prediction payload locator is invalid")
        return directory
