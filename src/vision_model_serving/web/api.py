"""Versioned HTTP adapter for prediction submission and polling."""

from __future__ import annotations

from datetime import UTC, datetime
from io import BytesIO

from django.conf import settings
from django.urls import reverse
from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import extend_schema
from rest_framework import serializers, status
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from vision_model_serving.execution import (
    GatewayUnavailable,
    IdempotencyConflict,
    PredictionFailed,
    PredictionHandle,
    PredictionNotFound,
    PredictionRequest,
    QueueSaturated,
    ResultExpired,
    ResultNotReady,
)
from vision_model_serving.pipeline import CaseInput, PredictionMode
from vision_model_serving.pipeline.serialization import prediction_to_dict

from .errors import ClinicalHistoryRequired, EncodedDicomTooLarge, public_error
from .runtime import prediction_gateway


class PredictionSubmissionSerializer(serializers.Serializer):
    dicom = serializers.FileField(write_only=True)
    mode = serializers.ChoiceField(
        choices=tuple(mode.value for mode in PredictionMode),
        default=PredictionMode.FULL.value,
    )
    clinical_history = serializers.CharField(
        allow_blank=True,
        max_length=4_000,
        required=False,
        trim_whitespace=True,
        write_only=True,
    )
    detector_score_threshold = serializers.FloatField(
        min_value=0.0,
        max_value=1.0,
        required=False,
        write_only=True,
    )

    def validate(self, attrs: dict[str, object]) -> dict[str, object]:
        mode = PredictionMode(str(attrs["mode"]))
        history = str(attrs.get("clinical_history", ""))
        if mode is PredictionMode.FULL and not history.strip():
            raise ClinicalHistoryRequired
        upload = attrs["dicom"]
        size = getattr(upload, "size", None)
        if not isinstance(size, int) or size < 1:
            raise serializers.ValidationError({"dicom": "The DICOM file is empty."})
        if size > 64 * 1024 * 1024:
            raise EncodedDicomTooLarge
        return attrs


class PredictionCollectionView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    @extend_schema(
        request=PredictionSubmissionSerializer,
        responses={
            200: OpenApiTypes.OBJECT,
            202: OpenApiTypes.OBJECT,
            400: OpenApiTypes.OBJECT,
            409: OpenApiTypes.OBJECT,
            413: OpenApiTypes.OBJECT,
            415: OpenApiTypes.OBJECT,
            422: OpenApiTypes.OBJECT,
            429: OpenApiTypes.OBJECT,
            503: OpenApiTypes.OBJECT,
        },
    )
    def post(self, request: Request) -> Response:
        serializer = PredictionSubmissionSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        values = serializer.validated_data
        upload = values["dicom"]
        payload = upload.read()
        if not isinstance(payload, bytes):
            return public_error(
                request,
                "invalid_dicom",
                "The DICOM upload is unreadable.",
                400,
            )
        idempotency_key = request.headers.get("Idempotency-Key")
        if idempotency_key is not None and not (1 <= len(idempotency_key) <= 200):
            return public_error(
                request,
                "invalid_idempotency_key",
                "The idempotency key must contain 1 to 200 characters.",
                400,
            )
        prediction_request = PredictionRequest(
            case=CaseInput(
                BytesIO(payload),
                str(values.get("clinical_history", "")) or None,
            ),
            mode=PredictionMode(str(values["mode"])),
            idempotency_key=idempotency_key,
        )
        gateway = prediction_gateway()
        try:
            handle = gateway.submit(prediction_request)
        except QueueSaturated:
            return public_error(
                request,
                "prediction_queue_full",
                "The prediction queue is full.",
                429,
            )
        except IdempotencyConflict:
            return public_error(
                request,
                "prediction_idempotency_conflict",
                "The idempotency key was used for different input.",
                409,
            )
        except GatewayUnavailable:
            return public_error(
                request,
                "prediction_gateway_unavailable",
                "Prediction submission is temporarily unavailable.",
                503,
            )
        if request.headers.get("Prefer", "").strip().lower() == "respond-async":
            return Response(_handle_payload(handle), status=status.HTTP_202_ACCEPTED)
        completed = gateway.wait(
            handle.prediction_id,
            timeout_seconds=settings.VMS_SYNC_WAIT_SECONDS,
        )
        if isinstance(completed, PredictionHandle):
            return Response(_handle_payload(completed), status=status.HTTP_202_ACCEPTED)
        return Response(
            {
                "prediction_id": str(handle.prediction_id),
                "result": prediction_to_dict(completed),
            },
            status=status.HTTP_200_OK,
        )


class PredictionStatusView(APIView):
    @extend_schema(responses={200: OpenApiTypes.OBJECT, 404: OpenApiTypes.OBJECT})
    def get(self, request: Request, prediction_id: str) -> Response:
        try:
            prediction = prediction_gateway().status(prediction_id)
        except PredictionNotFound:
            return public_error(
                request,
                "prediction_not_found",
                "Prediction was not found.",
                404,
            )
        except GatewayUnavailable:
            return public_error(
                request,
                "prediction_gateway_unavailable",
                "Prediction status is temporarily unavailable.",
                503,
            )
        return Response(
            {
                "prediction_id": str(prediction.prediction_id),
                "state": prediction.state.value,
                "submitted_at": _timestamp(prediction.submitted_at),
                "started_at": _optional_timestamp(prediction.started_at),
                "completed_at": _optional_timestamp(prediction.completed_at),
                "expires_at": _timestamp(prediction.expires_at),
                "queue_wait_ms": prediction.queue_wait_ms,
                "failure": (
                    None
                    if prediction.failure is None
                    else {
                        "code": prediction.failure.code,
                        "message": prediction.failure.detail,
                        "retryable": prediction.failure.retryable,
                    }
                ),
            }
        )


class PredictionResultView(APIView):
    @extend_schema(responses={200: OpenApiTypes.OBJECT, 404: OpenApiTypes.OBJECT})
    def get(self, request: Request, prediction_id: str) -> Response:
        try:
            result = prediction_gateway().result(prediction_id)
        except PredictionNotFound:
            return public_error(
                request,
                "prediction_not_found",
                "Prediction was not found.",
                404,
            )
        except ResultNotReady:
            return public_error(
                request,
                "prediction_result_not_ready",
                "Prediction has not completed successfully.",
                409,
            )
        except ResultExpired:
            return public_error(
                request,
                "prediction_result_expired",
                "Prediction result has expired.",
                410,
            )
        except PredictionFailed:
            return public_error(
                request,
                "prediction_failed",
                "Prediction execution failed.",
                422,
            )
        except GatewayUnavailable:
            return public_error(
                request,
                "prediction_gateway_unavailable",
                "Prediction result is temporarily unavailable.",
                503,
            )
        return Response(
            {
                "prediction_id": prediction_id,
                "result": prediction_to_dict(result),
            }
        )


def _handle_payload(handle: PredictionHandle) -> dict[str, object]:
    prediction_id = str(handle.prediction_id)
    return {
        "prediction_id": prediction_id,
        "state": handle.state.value,
        "submitted_at": _timestamp(handle.submitted_at),
        "expires_at": _timestamp(handle.expires_at),
        "status_url": reverse("prediction-status", args=(prediction_id,)),
        "result_url": reverse("prediction-result", args=(prediction_id,)),
        "idempotent_replay": handle.idempotent_replay,
    }


def _timestamp(value: float) -> str:
    return datetime.fromtimestamp(value, tz=UTC).isoformat()


def _optional_timestamp(value: float | None) -> str | None:
    return None if value is None else _timestamp(value)
