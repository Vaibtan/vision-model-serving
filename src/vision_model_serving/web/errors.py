"""Stable, sanitized HTTP error envelopes."""

from __future__ import annotations

import secrets
from collections.abc import Callable

from rest_framework import status
from rest_framework.exceptions import APIException, ValidationError
from rest_framework.response import Response
from rest_framework.views import exception_handler as drf_exception_handler


class ClinicalHistoryRequired(APIException):
    status_code = 422
    default_detail = "Clinical history is required for full mode."
    default_code = "clinical_history_required"


class EncodedDicomTooLarge(APIException):
    status_code = 413
    default_detail = "The DICOM file exceeds the encoded-size limit."
    default_code = "dicom_encoded_size_exceeded"


class RequestIdMiddleware:
    def __init__(self, get_response: Callable[[object], object]):
        self._get_response = get_response

    def __call__(self, request: object) -> object:
        request.request_id = secrets.token_urlsafe(16)  # type: ignore[attr-defined]
        return self._get_response(request)


def exception_handler(error: Exception, context: dict[str, object]) -> Response:
    response = drf_exception_handler(error, context)
    request = context.get("request")
    if response is None:
        return public_error(
            request,
            "internal_error",
            "The request could not be completed.",
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    if isinstance(error, ValidationError):
        code = "invalid_request"
        message = "The request fields are invalid."
        details = response.data
    else:
        code = str(getattr(error, "default_code", "request_failed"))
        message = str(getattr(error, "default_detail", "The request failed."))
        details = {}
    response.data = error_payload(request, code, message, details)
    return response


def public_error(
    request: object,
    code: str,
    message: str,
    http_status: int,
    *,
    details: object | None = None,
) -> Response:
    return Response(
        error_payload(request, code, message, details),
        status=http_status,
    )


def error_payload(
    request: object,
    code: str,
    message: str,
    details: object | None = None,
) -> dict[str, object]:
    return {
        "error": {
            "code": code,
            "message": message,
            "request_id": str(getattr(request, "request_id", "unavailable")),
            "details": {} if details is None else details,
        }
    }
