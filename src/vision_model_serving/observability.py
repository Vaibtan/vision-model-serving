"""Bounded Prometheus metrics and privacy-safe structured events."""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime

from prometheus_client import Counter, Gauge, Histogram

from vision_model_serving.model_ids import MODEL_STAGE_BY_ID
from vision_model_serving.pipeline.contracts import PredictionMode, PredictionResult

_HTTP_REQUESTS = Counter(
    "vms_http_requests",
    "HTTP responses by bounded route, method, and outcome.",
    ("route", "method", "outcome"),
)
_HTTP_DURATION = Histogram(
    "vms_http_request_duration_seconds",
    "HTTP response duration by bounded route and method.",
    ("route", "method"),
)
_DICOM = Counter(
    "vms_dicom_inputs",
    "DICOM preflight outcomes.",
    ("outcome",),
)
_PREDICTIONS = Counter(
    "vms_predictions",
    "Executor prediction outcomes by mode.",
    ("mode", "outcome"),
)
_QUEUE_WAIT = Histogram(
    "vms_queue_wait_seconds",
    "Time spent waiting for an RQ worker.",
)
_LIFECYCLE = Counter(
    "vms_model_lifecycle",
    "Model lifecycle events.",
    ("model", "event"),
)
_MODEL_LOAD = Histogram(
    "vms_model_load_seconds",
    "Model load and warmup duration.",
    ("model",),
)
_MODEL_INFERENCE = Histogram(
    "vms_model_inference_seconds",
    "Model inference duration.",
    ("model",),
)
_PIPELINE_STAGE = Histogram(
    "vms_pipeline_stage_seconds",
    "Pipeline duration by bounded stage.",
    ("stage",),
)
_CUDA_MEMORY = Gauge(
    "vms_cuda_memory_bytes",
    "CUDA allocator bytes by model and bounded memory kind.",
    ("model", "kind"),
    multiprocess_mode="max",
)
_CPU_RSS = Gauge(
    "vms_process_rss_bytes",
    "Resident memory by bounded process role.",
    ("process",),
    multiprocess_mode="max",
)
_CLASSIFIER_ROIS = Histogram(
    "vms_classifier_rois",
    "Classifier ROI count per completed prediction.",
)
_ROI_FALLBACKS = Counter(
    "vms_roi_fallbacks",
    "Padded classifier ROI selections.",
)
_OOM = Counter(
    "vms_cuda_oom",
    "CUDA out-of-memory failures.",
)

_ROUTES = {
    "inspection-workbench",
    "monitoring-console",
    "dicom-preview",
    "prediction-collection",
    "prediction-status",
    "prediction-result",
    "model-inventory",
    "operations-snapshot",
    "liveness",
    "readiness",
    "metrics",
    "schema",
    "docs",
}


def configure_structured_logging(level: str = "INFO") -> None:
    logger = logging.getLogger("vision_model_serving.telemetry")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level.upper())
    logger.propagate = False


def record_http_response(
    request: object,
    *,
    status_code: int,
    duration_seconds: float,
    exception_class: str | None = None,
) -> None:
    route = getattr(getattr(request, "resolver_match", None), "url_name", None)
    route = route if route in _ROUTES else "unmatched"
    method = str(getattr(request, "method", "OTHER")).upper()
    method = method if method in {"GET", "POST"} else "OTHER"
    outcome = _http_outcome(status_code)
    _HTTP_REQUESTS.labels(route, method, outcome).inc()
    _HTTP_DURATION.labels(route, method).observe(max(0.0, duration_seconds))
    _CPU_RSS.labels("web").set(_rss_bytes())
    if (
        route in {"prediction-status", "operations-snapshot"}
        and outcome == "success"
    ):
        return
    _event(
        "http_response",
        route=route,
        method=method,
        outcome=outcome,
        status_code=status_code,
        duration_ms=round(max(0.0, duration_seconds) * 1_000.0, 3),
        request_id=str(getattr(request, "request_id", "unavailable")),
        exception_class=exception_class,
    )


def record_http_exception(error: Exception) -> None:
    """Record only the bounded exception type at the DRF sanitization boundary."""

    _event("http_exception", exception_class=type(error).__name__)


def record_dicom(outcome: str) -> None:
    _DICOM.labels(outcome if outcome in {"accepted", "rejected"} else "rejected").inc()


def record_queue_wait(wait_seconds: float) -> None:
    _QUEUE_WAIT.observe(max(0.0, wait_seconds))
    _CPU_RSS.labels("worker").set(_rss_bytes())
    _event(
        "queue_started",
        outcome="started",
        queue_wait_ms=round(max(0.0, wait_seconds) * 1_000.0, 3),
        rss_bytes=_rss_bytes(),
    )


def record_prediction_success(result: PredictionResult) -> None:
    mode = _mode(result.mode)
    _PREDICTIONS.labels(mode, "succeeded").inc()
    _CPU_RSS.labels("executor").set(_rss_bytes())
    _PIPELINE_STAGE.labels("decode").observe(result.timings.decode_ms / 1_000.0)
    _PIPELINE_STAGE.labels("total").observe(result.timings.total_ms / 1_000.0)
    stages = (("detector", result.timings.detector),)
    if result.timings.classifier is not None:
        stages += (("classifier", result.timings.classifier),)
    for model, stage in stages:
        runtime = stage.runtime
        event = "reuse" if runtime.reused else "load"
        _LIFECYCLE.labels(model, event).inc()
        if runtime.switch_ms > 0:
            _LIFECYCLE.labels(model, "switch").inc()
        if runtime.load_ms > 0:
            _MODEL_LOAD.labels(model).observe(runtime.load_ms / 1_000.0)
        _MODEL_INFERENCE.labels(model).observe(runtime.inference_ms / 1_000.0)
        for kind, value in (
            ("allocated", stage.memory.allocated_bytes),
            ("reserved", stage.memory.reserved_bytes),
            ("peak_allocated", stage.memory.peak_allocated_bytes),
            ("peak_reserved", stage.memory.peak_reserved_bytes),
        ):
            _CUDA_MEMORY.labels(model, kind).set(value)
    roi_count = len(result.detector.classifier_rois)
    padded = sum(proposal.padded for proposal in result.detector.classifier_rois)
    _CLASSIFIER_ROIS.observe(roi_count)
    if padded:
        _ROI_FALLBACKS.inc(padded)
    classifier = result.timings.classifier
    _event(
        "prediction_completed",
        mode=mode,
        outcome="succeeded",
        rows=result.input.rows,
        columns=result.input.columns,
        frames=result.input.frames,
        transfer_syntax=result.input.transfer_syntax_uid,
        detector_artifact=result.provenance.detector.sha256[:12],
        classifier_artifact=(
            None
            if result.provenance.classifier is None
            else result.provenance.classifier.sha256[:12]
        ),
        classifier_rois=roi_count,
        roi_fallbacks=padded,
        detector_lifecycle=_lifecycle_event(result.timings.detector.runtime),
        classifier_lifecycle=(
            None if classifier is None else _lifecycle_event(classifier.runtime)
        ),
        detector_cuda_allocated_bytes=(result.timings.detector.memory.allocated_bytes),
        detector_cuda_reserved_bytes=(result.timings.detector.memory.reserved_bytes),
        classifier_cuda_allocated_bytes=(
            None if classifier is None else classifier.memory.allocated_bytes
        ),
        classifier_cuda_reserved_bytes=(
            None if classifier is None else classifier.memory.reserved_bytes
        ),
        decode_ms=round(result.timings.decode_ms, 3),
        detector_ms=round(result.timings.detector.runtime.inference_ms, 3),
        classifier_ms=(
            None if classifier is None else round(classifier.runtime.inference_ms, 3)
        ),
        total_ms=round(result.timings.total_ms, 3),
        rss_bytes=_rss_bytes(),
    )


def record_prediction_failure(
    mode: PredictionMode | None,
    error: Exception,
) -> None:
    exception_class = type(error).__name__
    _PREDICTIONS.labels(_mode(mode), "failed").inc()
    _LIFECYCLE.labels("unknown", "failure").inc()
    if "outofmemory" in f"{exception_class}{error}".replace("_", "").lower():
        _OOM.inc()
    _CPU_RSS.labels("executor").set(_rss_bytes())
    _event(
        "prediction_completed",
        mode=_mode(mode),
        outcome="failed",
        exception_class=exception_class,
        rss_bytes=_rss_bytes(),
    )


def record_executor_cleanup(resident_models: tuple[str, ...]) -> None:
    cleaned = 0
    for model_id in resident_models:
        model = MODEL_STAGE_BY_ID.get(model_id)
        if model is not None:
            _LIFECYCLE.labels(model, "unload").inc()
            _LIFECYCLE.labels(model, "cleanup").inc()
            cleaned += 1
    _event("executor_cleanup", outcome="succeeded", resident_models=cleaned)


def _http_outcome(status_code: int) -> str:
    if 200 <= status_code < 300:
        return "success"
    if 400 <= status_code < 500:
        return "client_error"
    if status_code >= 500:
        return "server_error"
    return "other"


def _lifecycle_event(runtime: object) -> str:
    if bool(getattr(runtime, "reused", False)):
        return "reuse"
    if float(getattr(runtime, "switch_ms", 0.0)) > 0:
        return "switch"
    return "load"


def _mode(value: PredictionMode | None) -> str:
    return (
        value.value
        if value in {PredictionMode.DETECTION, PredictionMode.FULL}
        else "unknown"
    )


def _rss_bytes() -> int:
    try:
        with open("/proc/self/statm", encoding="ascii") as statm:
            resident_pages = int(statm.read().split()[1])
        return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, IndexError, OSError, ValueError):
        return 0


def _event(name: str, **fields: object) -> None:
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "level": "INFO",
        "event": name,
        **{key: value for key, value in fields.items() if value is not None},
    }
    logging.getLogger("vision_model_serving.telemetry").info(
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
    )
