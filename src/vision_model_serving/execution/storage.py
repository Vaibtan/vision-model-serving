"""Ephemeral filesystem storage for private prediction payloads and results."""

from __future__ import annotations

import hmac
import json
import os
import re
import secrets
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import Lock
from time import time

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
_STAGING_PREFIXES = (".tmp-", ".result-", ".gc-")
_LEASE_NAME = ".active.json"
# Staging entries older than this are orphans from a killed process, never
# live writes; store_request/store_result complete in well under a minute.
_STAGING_ORPHAN_SECONDS = 900.0


class JobPayloadNotFound(PredictionGatewayError):
    code = "prediction_payload_not_found"


class JobPayloadExpired(JobPayloadNotFound):
    code = "prediction_payload_expired"


class JobResultNotFound(PredictionGatewayError):
    code = "prediction_stored_result_not_found"


class JobResultConflict(PredictionGatewayError):
    code = "prediction_result_write_conflict"


class JobLeaseConflict(PredictionGatewayError):
    code = "prediction_execution_lease_conflict"


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
        self._cleanup_lock = Lock()
        self._last_cleanup_at: float | None = None

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
            # Keep payloads private from other users while allowing the
            # explicitly configured job-volume group to run read-only privacy
            # validation under a host-mapped UID.
            temporary.mkdir(mode=0o750)
            (temporary / "input.dcm").write_bytes(payload)
            (temporary / "request.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "prediction_id": str(prediction_id),
                        "mode": request.mode.value,
                        "clinical_history": request.case.clinical_history,
                        "detector_score_threshold": (request.detector_score_threshold),
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
                item.chmod(0o640)
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
            metadata = json.loads((directory / "request.json").read_text(encoding="utf-8"))
            payload = (directory / "input.dcm").read_bytes()
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            raise JobPayloadNotFound("prediction request payload is unavailable") from None
        if (
            not isinstance(metadata, dict)
            or metadata.get("schema_version") != 1
            or metadata.get("prediction_id") != str(prediction_id)
        ):
            raise JobPayloadNotFound("prediction request payload is unavailable")
        try:
            mode = PredictionMode(metadata["mode"])
            history = metadata["clinical_history"]
            threshold = metadata.get("detector_score_threshold")
            expires_at = float(metadata["expires_at"])
            stored_fingerprint = metadata["request_fingerprint"]
            if history is not None and not isinstance(history, str):
                raise TypeError
            if not isinstance(stored_fingerprint, str):
                raise TypeError
        except (KeyError, TypeError, ValueError):
            raise JobPayloadNotFound("prediction request payload is unavailable") from None
        if self._clock() >= expires_at and not self.lease_active(locator):
            self.discard_job(locator)
            raise JobPayloadExpired("prediction request payload has expired")
        request = PredictionRequest(
            case=CaseInput(BytesIO(payload), history),
            mode=mode,
            detector_score_threshold=threshold,
        )
        if not hmac.compare_digest(
            request_fingerprint(request, payload),
            stored_fingerprint,
        ):
            raise JobPayloadNotFound("prediction request payload is unavailable")
        return request

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
        temporary.chmod(0o640)
        try:
            try:
                os.link(temporary, destination)
            except FileExistsError:
                try:
                    existing = json.loads(destination.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    raise JobResultConflict("stored prediction result is inconsistent") from None
                if not isinstance(existing, dict) or existing.get("result") != envelope["result"]:
                    raise JobResultConflict("stored prediction result is inconsistent")
        finally:
            temporary.unlink(missing_ok=True)
        self.purge_request(locator)

    def load_result(self, locator: str) -> PredictionResult:
        expired = False
        try:
            envelope = json.loads(
                (self._directory(locator) / "result.json").read_text(encoding="utf-8")
            )
            if not isinstance(envelope, dict) or envelope.get("schema_version") != 1:
                raise ValueError
            if self._clock() >= float(envelope["expires_at"]):
                expired = True
                raise ValueError
            return prediction_from_dict(envelope["result"])
        except (FileNotFoundError, OSError, KeyError, TypeError, ValueError):
            if expired:
                self.discard_job(locator)
            raise JobResultNotFound("prediction result payload is unavailable") from None

    def cleanup_expired(self) -> int:
        removed = 0
        now = self._clock()
        try:
            entries = list(self._root.iterdir())
        except OSError:
            return 0
        for entry in entries:
            if entry.is_symlink():
                continue
            if entry.name.startswith(_STAGING_PREFIXES):
                removed += self._remove_stale_staging_entry(entry)
                continue
            if not entry.is_dir() or not _LOCATOR_PATTERN.fullmatch(entry.name):
                continue
            metadata_path = (
                entry / "result.json"
                if (entry / "result.json").is_file()
                else entry / "request.json"
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
                removed += int(self._discard_directory_if_unleased(entry))
        return removed

    def maybe_cleanup(self, *, interval_seconds: float = 60.0) -> int:
        """Run one rate-limited cleanup sweep; safe to call on any hot path."""

        now = self._clock()
        with self._cleanup_lock:
            last = self._last_cleanup_at
            if last is not None and now - last < interval_seconds:
                return 0
            self._last_cleanup_at = now
        try:
            return self.cleanup_expired()
        except Exception:  # noqa: BLE001 - the janitor must never fail a request
            return 0

    def _remove_stale_staging_entry(self, entry: Path) -> int:
        try:
            # Orphan age is judged with the OS clock because st_mtime comes
            # from it, independent of the injected logical clock.
            if time() - entry.stat().st_mtime < _STAGING_ORPHAN_SECONDS:
                return 0
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
            return 1
        except OSError:
            return 0

    def acquire_lease(self, locator: str, *, ttl_seconds: float) -> None:
        """Atomically protect one locator from physical reclamation."""

        if ttl_seconds <= 0:
            raise ValueError("execution lease TTL must be positive")
        directory = self._directory(locator)
        payload = json.dumps(
            {
                "schema_version": 1,
                "expires_at": self._clock() + ttl_seconds,
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        lease = directory / _LEASE_NAME
        for _attempt in range(2):
            try:
                descriptor = os.open(lease, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
            except FileExistsError:
                if self._lease_active_in_directory(directory):
                    raise JobLeaseConflict("prediction payload already has an active lease")
                try:
                    lease.unlink()
                except FileNotFoundError:
                    pass
                continue
            except FileNotFoundError:
                raise JobPayloadNotFound("prediction payload locator is unavailable") from None
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
            return
        raise JobLeaseConflict("prediction payload already has an active lease")

    def release_lease(self, locator: str) -> None:
        try:
            (self._directory(locator) / _LEASE_NAME).unlink()
        except FileNotFoundError:
            pass

    def lease_active(self, locator: str) -> bool:
        return self._lease_active_in_directory(self._directory(locator))

    def discard_job(self, locator: str, *, force: bool = False) -> bool:
        directory = self._directory(locator)
        if directory.is_symlink():
            raise JobPayloadNotFound("prediction payload locator is invalid")
        if force:
            try:
                shutil.rmtree(directory)
                return True
            except FileNotFoundError:
                return False
        return self._discard_directory_if_unleased(directory)

    def _discard_directory_if_unleased(self, directory: Path) -> bool:
        tombstone = self._root / f".gc-{directory.name}-{secrets.token_hex(8)}"
        try:
            directory.replace(tombstone)
        except FileNotFoundError:
            return False
        except OSError:
            return False
        if self._lease_active_in_directory(tombstone):
            try:
                tombstone.replace(directory)
            except OSError:
                # Preserve live work under the tombstone rather than deleting it.
                pass
            return False
        try:
            shutil.rmtree(tombstone)
            return True
        except OSError:
            return False

    def _lease_active_in_directory(self, directory: Path) -> bool:
        try:
            value = json.loads((directory / _LEASE_NAME).read_text(encoding="utf-8"))
            return (
                isinstance(value, dict)
                and set(value) == {"schema_version", "expires_at"}
                and value["schema_version"] == 1
                and self._clock() < float(value["expires_at"])
            )
        except (
            FileNotFoundError,
            OSError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ):
            return False

    def _directory(self, locator: str) -> Path:
        if not isinstance(locator, str) or not _LOCATOR_PATTERN.fullmatch(locator):
            raise JobPayloadNotFound("prediction payload locator is invalid")
        directory = self._root / locator
        if directory.is_symlink():
            raise JobPayloadNotFound("prediction payload locator is invalid")
        return directory
