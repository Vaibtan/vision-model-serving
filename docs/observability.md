# Privacy-safe observability

The service emits Prometheus metrics and newline-delimited JSON events without
placing request, prediction, patient, DICOM-instance, tokenizer, filename, or
filesystem values in metric labels. Labels are limited to repository-owned
enums: HTTP route/method/outcome, prediction mode, model stage, lifecycle event,
CUDA memory kind, and process role.

Metrics are disabled by default. Enable them only on a trusted internal network:

```text
VMS_METRICS_ENABLED=true
VMS_METRICS_ALLOWED_NETWORKS=127.0.0.0/8,::1/128
VMS_METRICS_DIR=/run/vision-model-serving/metrics
PROMETHEUS_MULTIPROC_DIR=/run/vision-model-serving/metrics
```

`VMS_METRICS_DIR` and `PROMETHEUS_MULTIPROC_DIR` must name the same directory.
Create and empty it before starting Django, RQ, or the persistent executor; all
three processes must share it. Never clean it while those processes are alive.
The endpoint returns `503` when disabled or when Redis/executor observations are
unavailable, and `403` outside `VMS_METRICS_ALLOWED_NETWORKS`.

`GET /metrics` combines process-safe instrumentation with snapshots from the
existing Redis/RQ and executor-status contracts. It covers:

- HTTP, DICOM, prediction, queue, and worker-loss outcomes;
- queue depth, active work, and queue-wait duration;
- model load/warmup, reuse, switch, cleanup, inference, and failure events;
- decode and total pipeline latency;
- allocated, reserved, peak-allocated, and peak-reserved CUDA bytes;
- classifier ROI counts, padding fallbacks, CUDA OOMs, and process RSS; and
- executor readiness, artifact/operator/device checks, and model residency.

The JSON event stream uses explicit safe fields. HTTP events contain only a
bounded route name, method, outcome, status, duration, and opaque request ID.
Sanitized internal-error events contain only the Python exception class, not
its message or traceback.
Executor completion events contain mode, non-identifying image shape and transfer
syntax, short artifact hash prefixes, timing/memory observations, and ROI counts.
They never serialize request bodies, clinical history, filenames, filesystem
paths, DICOM patient/study/instance identifiers, token content, or prediction IDs.

Readiness returns bounded internal reason codes such as `redis_unavailable` or
`native_operator_unavailable`; exception text and filesystem details remain
private. Metrics are operational telemetry, not a clinical audit log.
