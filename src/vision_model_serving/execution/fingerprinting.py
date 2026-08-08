"""Privacy-preserving request identity for idempotent prediction submission."""

from __future__ import annotations

from hashlib import sha256

from .contracts import PredictionRequest


def request_fingerprint(request: PredictionRequest, payload: bytes) -> str:
    digest = sha256()
    for value in (
        request.mode.value.encode("utf-8"),
        (request.case.clinical_history or "").encode("utf-8"),
        payload,
    ):
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return digest.hexdigest()
