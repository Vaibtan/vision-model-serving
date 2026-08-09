# Privacy-safe observability

`GET /monitoring` is a built-in operations console for the live inference
path. It shows Django, Redis, RQ, executor, detector, and classifier state;
queue pressure; cumulative latency estimates and outcomes; CUDA/process memory;
and bounded artifact identity. It polls `GET /api/v1/operations` every five
seconds and retains no server-side or browser-side history.

The JSON snapshot is schema-versioned and returns `Cache-Control: no-store`.
The same snapshot contract supplies `/readyz`, `/api/v1/models`, and the
Redis/RQ/executor metrics described below, so those views share one observation
of current state. Redis and executor probes have explicit timeouts.

The service emits Prometheus metrics and newline-delimited JSON events without
placing request, prediction, patient, DICOM-instance, tokenizer, filename, or
filesystem values in metric labels. Labels are limited to repository-owned
enums: HTTP route/method/outcome, prediction mode, model stage, lifecycle event,
CUDA memory kind, and process role.

The Prometheus `/metrics` export is disabled by default. Its toggle does not
disable the bounded telemetry that feeds `/api/v1/operations` and
`/monitoring`; those private operational views remain available without an
external scrape endpoint.

For Compose, enable the export only on a trusted internal network:

```text
VMS_METRICS_ENABLED=true
```

Compose already shares its private metrics directory and sets
`PROMETHEUS_MULTIPROC_DIR` before Django, RQ, or the executor imports
instrumentation. For a custom deployment, set both directory variables to the
same pre-created shared directory:

```text
VMS_METRICS_ALLOWED_NETWORKS=127.0.0.0/8,::1/128
VMS_METRICS_DIR=/run/vision-model-serving/metrics
PROMETHEUS_MULTIPROC_DIR=/run/vision-model-serving/metrics
```

For a custom deployment, create and empty that shared directory before starting
Django, RQ, or the persistent executor. Never clean it while those processes
are alive. The endpoint returns `503` when disabled or when Redis/executor
observations are unavailable, and `403` outside
`VMS_METRICS_ALLOWED_NETWORKS`.

`GET /metrics` combines process-safe instrumentation with snapshots from the
existing Redis/RQ and executor-status contracts. It covers:

- HTTP, DICOM, prediction, queue, and worker-loss outcomes;
- queue depth, active work, and queue-wait duration;
- model load/warmup, reuse, switch, cleanup, inference, and failure events;
- decode and total pipeline latency;
- allocated, reserved, peak-allocated, and peak-reserved CUDA bytes;
- classifier ROI counts, padding fallbacks, CUDA OOMs, and process RSS; and
- executor readiness, artifact/operator/device checks, and model residency.

The console derives p50 and p95 estimates from Prometheus histogram buckets.
Those values and counters are cumulative since process start; they are not a
durable time series, service-level objective, or alerting system. Use the
restricted `/metrics` endpoint with an external Prometheus/Grafana deployment
when retained history, alert rules, or cross-instance aggregation is required.

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
private. The operations snapshot and page additionally exclude prediction
values and request identifiers. These surfaces are operational telemetry, not
a clinical audit log or evidence of accuracy, calibration, or medical fitness.
