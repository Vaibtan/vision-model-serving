"""Versioned HTTP adapter for prediction submission and polling."""

from __future__ import annotations

from datetime import UTC, datetime
from io import BytesIO

from django.conf import settings
from django.http import HttpResponse
from django.urls import reverse
from drf_spectacular.utils import extend_schema
from PIL import Image
from rest_framework import serializers, status
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.renderers import JSONRenderer
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from vision_model_serving.dicom import (
    DicomCanonicalizationError,
    DicomCanonicalizer,
    EncodedSizeLimitError,
    InvalidDicomError,
    UnsupportedPhotometricInterpretationError,
    UnsupportedTransferSyntaxError,
)
from vision_model_serving.execution import (
    GatewayUnavailable,
    IdempotencyConflict,
    PredictionFailed,
    PredictionHandle,
    PredictionNotFound,
    PredictionRequest,
    PredictionRuntimeUnavailable,
    PredictionTimedOut,
    QueueSaturated,
    ResultExpired,
    ResultNotReady,
)
from vision_model_serving.observability import record_dicom
from vision_model_serving.pipeline import CaseInput, PredictionMode
from vision_model_serving.pipeline.serialization import prediction_to_public_dict

from .errors import (
    ClinicalHistoryRequired,
    EncodedDicomTooLarge,
    ErrorEnvelopeSerializer,
    public_error,
)
from .protection import browser_origin_rejection
from .prediction_serializers import (
    PredictionHandleResponseSerializer,
    PredictionResultResponseSerializer,
    PredictionSyncResponseSerializer,
)
from .renderers import PngRenderer
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


class DicomPreviewSerializer(serializers.Serializer):
    dicom = serializers.FileField(write_only=True)

    def validate_dicom(self, upload: object) -> object:
        size = getattr(upload, "size", None)
        if not isinstance(size, int) or size < 1:
            raise serializers.ValidationError("The DICOM file is empty.")
        if size > 64 * 1024 * 1024:
            raise EncodedDicomTooLarge
        return upload


class PredictionFailureSerializer(serializers.Serializer):
    code = serializers.CharField()
    message = serializers.CharField()
    retryable = serializers.BooleanField()


class PredictionStatusSerializer(serializers.Serializer):
    prediction_id = serializers.CharField()
    state = serializers.CharField()
    submitted_at = serializers.DateTimeField()
    started_at = serializers.DateTimeField(allow_null=True)
    completed_at = serializers.DateTimeField(allow_null=True)
    expires_at = serializers.DateTimeField()
    queue_wait_ms = serializers.FloatField(allow_null=True)
    failure = PredictionFailureSerializer(allow_null=True)


class NoStoreResponseMixin:
    """Prediction payloads carry medical data; forbid caching on every response."""

    def finalize_response(self, request, response, *args, **kwargs):
        response = super().finalize_response(request, response, *args, **kwargs)
        response["Cache-Control"] = "no-store"
        return response


class DicomPreviewView(NoStoreResponseMixin, APIView):
    """Return an ephemeral metadata-minimized rendering of sensitive pixels."""

    parser_classes = (MultiPartParser, FormParser)
    renderer_classes = (JSONRenderer, PngRenderer)
    throttle_scope = "preview"

    @extend_schema(
        request=DicomPreviewSerializer,
        responses={
            (200, "image/png"): bytes,
            400: ErrorEnvelopeSerializer,
            413: ErrorEnvelopeSerializer,
            415: ErrorEnvelopeSerializer,
            422: ErrorEnvelopeSerializer,
        },
    )
    def post(self, request: Request) -> HttpResponse | Response:
        rejection = browser_origin_rejection(request)
        if rejection is not None:
            return rejection
        serializer = DicomPreviewSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        upload = serializer.validated_data["dicom"]
        payload = upload.read()
        if not isinstance(payload, bytes):
            return public_error(
                request,
                "dicom_invalid",
                "The DICOM upload is unreadable.",
                400,
            )
        try:
            canonical = DicomCanonicalizer().decode(BytesIO(payload))
        except DicomCanonicalizationError as error:
            return _dicom_error(request, error)

        encoded = BytesIO()
        Image.fromarray(canonical.pixels).save(encoded, format="PNG", optimize=True)
        response = HttpResponse(encoded.getvalue(), content_type="image/png")
        response["Cache-Control"] = "no-store"
        response["Content-Disposition"] = 'inline; filename="mammogram-preview.png"'
        return response


class PredictionCollectionView(NoStoreResponseMixin, APIView):
    parser_classes = (MultiPartParser, FormParser)
    throttle_scope = "predictions"

    @extend_schema(
        request=PredictionSubmissionSerializer,
        responses={
            200: PredictionSyncResponseSerializer,
            202: PredictionHandleResponseSerializer,
            400: ErrorEnvelopeSerializer,
            409: ErrorEnvelopeSerializer,
            410: ErrorEnvelopeSerializer,
            413: ErrorEnvelopeSerializer,
            415: ErrorEnvelopeSerializer,
            422: ErrorEnvelopeSerializer,
            429: ErrorEnvelopeSerializer,
            500: ErrorEnvelopeSerializer,
            503: ErrorEnvelopeSerializer,
            504: ErrorEnvelopeSerializer,
        },
    )
    def post(self, request: Request) -> Response:
        rejection = browser_origin_rejection(request)
        if rejection is not None:
            return rejection
        serializer = PredictionSubmissionSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        values = serializer.validated_data
        upload = values["dicom"]
        payload = upload.read()
        if not isinstance(payload, bytes):
            return public_error(
                request,
                "dicom_invalid",
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
        try:
            # Structural gate only: pixel decode happens once, in the GPU
            # executor; pixel-level failures surface as async case failures.
            header = DicomCanonicalizer().validate_header(BytesIO(payload))
        except DicomCanonicalizationError as error:
            record_dicom("rejected")
            return _dicom_error(request, error)
        modality_rejection = _modality_rejection(request, header.modality)
        if modality_rejection is not None:
            record_dicom("rejected")
            return modality_rejection
        record_dicom("accepted")
        prediction_request = PredictionRequest(
            case=CaseInput(
                BytesIO(payload),
                str(values.get("clinical_history", "")) or None,
            ),
            mode=PredictionMode(str(values["mode"])),
            idempotency_key=idempotency_key,
            detector_score_threshold=values.get("detector_score_threshold"),
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
        request_echo = {
            "detector_score_threshold": values.get("detector_score_threshold"),
        }
        if request.headers.get("Prefer", "").strip().lower() == "respond-async":
            payload = {**_handle_payload(handle), "request": request_echo}
            return Response(
                _validated_payload(PredictionHandleResponseSerializer, payload),
                status=status.HTTP_202_ACCEPTED,
            )
        try:
            completed = gateway.wait(
                handle.prediction_id,
                timeout_seconds=settings.VMS_SYNC_WAIT_SECONDS,
            )
        except (
            PredictionTimedOut,
            PredictionRuntimeUnavailable,
            PredictionFailed,
            ResultExpired,
            GatewayUnavailable,
        ) as error:
            return _completion_error(request, error)
        if isinstance(completed, PredictionHandle):
            payload = {**_handle_payload(completed), "request": request_echo}
            return Response(
                _validated_payload(PredictionHandleResponseSerializer, payload),
                status=status.HTTP_202_ACCEPTED,
            )
        payload = {
            "prediction_id": str(handle.prediction_id),
            "result": prediction_to_public_dict(completed),
            "request": request_echo,
        }
        return Response(
            _validated_payload(PredictionSyncResponseSerializer, payload),
            status=status.HTTP_200_OK,
        )


class PredictionStatusView(NoStoreResponseMixin, APIView):
    throttle_scope = "polling"

    @extend_schema(
        responses={
            200: PredictionStatusSerializer,
            404: ErrorEnvelopeSerializer,
            503: ErrorEnvelopeSerializer,
        }
    )
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
        payload = {
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
        return Response(_validated_payload(PredictionStatusSerializer, payload))


class PredictionResultView(NoStoreResponseMixin, APIView):
    throttle_scope = "polling"

    @extend_schema(
        responses={
            200: PredictionResultResponseSerializer,
            404: ErrorEnvelopeSerializer,
            409: ErrorEnvelopeSerializer,
            410: ErrorEnvelopeSerializer,
            500: ErrorEnvelopeSerializer,
            503: ErrorEnvelopeSerializer,
            504: ErrorEnvelopeSerializer,
        }
    )
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
        except PredictionTimedOut as error:
            return _completion_error(request, error)
        except PredictionRuntimeUnavailable as error:
            return _completion_error(request, error)
        except PredictionFailed as error:
            return _completion_error(request, error)
        except GatewayUnavailable:
            return public_error(
                request,
                "prediction_gateway_unavailable",
                "Prediction result is temporarily unavailable.",
                503,
            )
        payload = {
            "prediction_id": prediction_id,
            "result": prediction_to_public_dict(result),
            "request": {
                "detector_score_threshold": result.detector_score_threshold,
            },
        }
        return Response(_validated_payload(PredictionResultResponseSerializer, payload))


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


def _validated_payload(
    serializer_type: type[serializers.Serializer],
    payload: dict[str, object],
) -> dict[str, object]:
    serializer = serializer_type(data=payload)
    serializer.is_valid(raise_exception=True)
    return dict(serializer.data)


def _timestamp(value: float) -> str:
    return datetime.fromtimestamp(value, tz=UTC).isoformat()


def _optional_timestamp(value: float | None) -> str | None:
    return None if value is None else _timestamp(value)


def _modality_rejection(request: Request, modality: str | None) -> Response | None:
    allowed = tuple(getattr(settings, "VMS_ALLOWED_MODALITIES", ()) or ())
    if not allowed:
        return None
    if (modality or "").strip().upper() in allowed:
        return None
    return public_error(
        request,
        "dicom_modality_rejected",
        "The DICOM modality is not accepted by this deployment.",
        422,
    )


def _dicom_error(request: Request, error: DicomCanonicalizationError) -> Response:
    if isinstance(error, EncodedSizeLimitError):
        http_status = 413
        message = "The DICOM file exceeds the encoded-size limit."
    elif isinstance(error, InvalidDicomError):
        http_status = 400
        message = "The uploaded file is not a valid DICOM object."
    elif isinstance(
        error,
        (UnsupportedTransferSyntaxError, UnsupportedPhotometricInterpretationError),
    ):
        http_status = 415
        message = "The DICOM encoding is not supported."
    else:
        http_status = 422
        message = "The DICOM object cannot be processed."
    return public_error(request, error.code, message, http_status)


def _completion_error(request: Request, error: Exception) -> Response:
    if isinstance(error, PredictionTimedOut):
        return public_error(
            request,
            "prediction_timeout",
            "Prediction execution timed out.",
            504,
        )
    if isinstance(error, (PredictionRuntimeUnavailable, GatewayUnavailable)):
        return public_error(
            request,
            "prediction_runtime_unavailable",
            "Prediction execution is temporarily unavailable.",
            503,
        )
    if isinstance(error, ResultExpired):
        return public_error(
            request,
            "prediction_result_expired",
            "Prediction result has expired.",
            410,
        )
    return public_error(
        request,
        "prediction_failed",
        "Prediction execution failed.",
        500,
    )
