"""Named OpenAPI response contracts for the operational HTTP surface."""

from __future__ import annotations

from rest_framework import serializers


class LivenessResponseSerializer(serializers.Serializer):
    status = serializers.CharField()


class ReadinessChecksSerializer(serializers.Serializer):
    redis = serializers.BooleanField()
    rq_worker = serializers.BooleanField()
    executor_artifact_ready = serializers.BooleanField()
    verified_artifacts = serializers.BooleanField()
    runtime_initialized = serializers.BooleanField()
    device_available = serializers.BooleanField()
    native_operator_available = serializers.BooleanField()
    manifest_available = serializers.BooleanField()
    telemetry_available = serializers.BooleanField()


class ReadinessRuntimeSerializer(serializers.Serializer):
    initialized = serializers.BooleanField()
    state = serializers.CharField()
    inference_warm = serializers.BooleanField()
    warm_model = serializers.CharField(allow_null=True)


class ReadinessResponseSerializer(serializers.Serializer):
    schema_version = serializers.IntegerField()
    status = serializers.CharField()
    readiness_scope = serializers.CharField()
    checks = ReadinessChecksSerializer()
    reasons = serializers.ListField(child=serializers.CharField())
    runtime = ReadinessRuntimeSerializer()


class ArtifactInventorySerializer(serializers.Serializer):
    id = serializers.CharField()
    role = serializers.CharField()
    sha256 = serializers.CharField()
    strict_load_verified = serializers.BooleanField()
    semantics_status = serializers.CharField()
    class_names = serializers.ListField(
        child=serializers.CharField(),
        allow_null=True,
    )
    decision_threshold = serializers.FloatField(allow_null=True)


class StartupTimingSerializer(serializers.Serializer):
    artifact_verification_seconds = serializers.FloatField()
    runtime_initialization_seconds = serializers.FloatField()
    process_start_to_artifact_ready_seconds = serializers.FloatField()


class ModelRuntimeSerializer(serializers.Serializer):
    state = serializers.CharField()
    initialized = serializers.BooleanField()
    artifact_ready = serializers.BooleanField()
    inference_warm = serializers.BooleanField()
    warm_model = serializers.CharField(allow_null=True)
    startup = StartupTimingSerializer(allow_null=True)
    active_model = serializers.CharField(allow_null=True)
    resident_models = serializers.ListField(child=serializers.CharField())
    device = serializers.CharField(allow_null=True)
    last_error = serializers.CharField(allow_null=True)


class ModelInventoryResponseSerializer(serializers.Serializer):
    manifest_id = serializers.CharField(allow_null=True)
    models = ArtifactInventorySerializer(many=True)
    runtime = ModelRuntimeSerializer()


class QueueSnapshotSerializer(serializers.Serializer):
    available = serializers.BooleanField()
    capacity = serializers.IntegerField()
    active = serializers.IntegerField()
    queued = serializers.IntegerField()
    running = serializers.IntegerField()
    admitted_total = serializers.IntegerField()
    rejected_total = serializers.IntegerField()
    succeeded_total = serializers.IntegerField()
    failed_total = serializers.IntegerField()
    worker_lost_total = serializers.IntegerField()
    wait_accumulated_seconds = serializers.FloatField()


class ExecutorSnapshotSerializer(serializers.Serializer):
    available = serializers.BooleanField()
    artifact_ready = serializers.BooleanField()
    runtime_initialized = serializers.BooleanField()
    inference_warm = serializers.BooleanField()
    warm_model = serializers.CharField(allow_null=True)
    state = serializers.CharField()
    active_model = serializers.CharField(allow_null=True)
    resident_models = serializers.ListField(child=serializers.CharField())
    device = serializers.CharField(allow_null=True)
    device_available = serializers.BooleanField()
    artifacts_verified = serializers.BooleanField()
    native_operator_available = serializers.BooleanField()
    startup = StartupTimingSerializer(allow_null=True)
    failure_code = serializers.CharField(allow_null=True)
    failure_present = serializers.BooleanField()
    precision = serializers.CharField()


class HistogramSerializer(serializers.Serializer):
    count = serializers.IntegerField()
    mean = serializers.FloatField(allow_null=True)
    p50 = serializers.FloatField(allow_null=True)
    p95 = serializers.FloatField(allow_null=True)


class HttpTrafficSerializer(serializers.Serializer):
    success = serializers.IntegerField()
    client_error = serializers.IntegerField()
    server_error = serializers.IntegerField()


class PredictionOutcomesSerializer(serializers.Serializer):
    succeeded = serializers.IntegerField()
    failed = serializers.IntegerField()


class PredictionTrafficSerializer(serializers.Serializer):
    detection = PredictionOutcomesSerializer()
    full = PredictionOutcomesSerializer()


class TrafficSerializer(serializers.Serializer):
    http = HttpTrafficSerializer()
    predictions = PredictionTrafficSerializer()


class LatencySnapshotSerializer(serializers.Serializer):
    http = HistogramSerializer()
    queue_wait = HistogramSerializer()
    pipeline_total = HistogramSerializer()
    dicom_decode = HistogramSerializer()
    detector_inference = HistogramSerializer()
    classifier_inference = HistogramSerializer()
    detector_load = HistogramSerializer()
    classifier_load = HistogramSerializer()


class CudaModelMemorySerializer(serializers.Serializer):
    allocated = serializers.IntegerField()
    reserved = serializers.IntegerField()
    peak_allocated = serializers.IntegerField()
    peak_reserved = serializers.IntegerField()


class CudaMemorySerializer(serializers.Serializer):
    detector = CudaModelMemorySerializer()
    classifier = CudaModelMemorySerializer()


class ProcessRssSerializer(serializers.Serializer):
    web = serializers.IntegerField()
    worker = serializers.IntegerField()
    executor = serializers.IntegerField()


class MemorySnapshotSerializer(serializers.Serializer):
    cuda = CudaMemorySerializer()
    process_rss = ProcessRssSerializer()


class ModelLifecycleSerializer(serializers.Serializer):
    load = serializers.IntegerField()
    reuse = serializers.IntegerField()
    switch = serializers.IntegerField()
    unload = serializers.IntegerField()
    failure = serializers.IntegerField()


class LifecycleSnapshotSerializer(serializers.Serializer):
    detector = ModelLifecycleSerializer()
    classifier = ModelLifecycleSerializer()


class EventSnapshotSerializer(serializers.Serializer):
    cuda_oom_total = serializers.IntegerField()
    roi_fallbacks_total = serializers.IntegerField()
    lifecycle = LifecycleSnapshotSerializer()


class TelemetrySnapshotSerializer(serializers.Serializer):
    scope = serializers.CharField()
    traffic = TrafficSerializer()
    latency_seconds = LatencySnapshotSerializer()
    memory_bytes = MemorySnapshotSerializer()
    events = EventSnapshotSerializer()


class OperationsSnapshotSerializer(serializers.Serializer):
    schema_version = serializers.IntegerField()
    captured_at = serializers.DateTimeField()
    status = serializers.CharField()
    checks = ReadinessChecksSerializer()
    reasons = serializers.ListField(child=serializers.CharField())
    queue = QueueSnapshotSerializer()
    executor = ExecutorSnapshotSerializer()
    manifest_id = serializers.CharField(allow_null=True)
    models = ArtifactInventorySerializer(many=True)
    telemetry = TelemetrySnapshotSerializer()
    validation_boundary = serializers.CharField()
