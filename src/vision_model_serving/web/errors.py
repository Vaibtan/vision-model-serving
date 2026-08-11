"""Stable, sanitized HTTP error envelopes."""

from __future__ import annotations

import secrets
from collections.abc import Callable
from time import perf_counter

from rest_framework import serializers, status
from rest_framework.exceptions import APIException, ValidationError
from rest_framework.response import Response
from rest_framework.views import exception_handler as drf_exception_handler

from vision_model_serving.observability import (
    record_http_exception,
    record_http_response,
)


class ErrorDetailSerializer(serializers.Serializer):
    code = serializers.CharField()
    message = serializers.CharField()
    request_id = serializers.CharField()
    details = serializers.JSONField()


class ErrorEnvelopeSerializer(serializers.Serializer):
    error = ErrorDetailSerializer()


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
        started = perf_counter()
        try:
            response = self._get_response(request)
        except Exception as error:
            record_http_response(
                request,
                status_code=500,
                duration_seconds=perf_counter() - started,
                exception_class=type(error).__name__,
            )
            raise
        record_http_response(
            request,
            status_code=int(getattr(response, "status_code", 500)),
            duration_seconds=perf_counter() - started,
        )
        return response


def exception_handler(error: Exception, context: dict[str, object]) -> Response:
    response = drf_exception_handler(error, context)
    request = context.get("request")
    if response is None:
        record_http_exception(error)
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
    payload = {
        "error": {
            "code": code,
            "message": message,
            "request_id": str(getattr(request, "request_id", "unavailable")),
            "details": {} if details is None else details,
        }
    }
    serializer = ErrorEnvelopeSerializer(data=payload)
    serializer.is_valid(raise_exception=True)
    return dict(serializer.data)
