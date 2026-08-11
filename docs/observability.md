# Privacy-safe observability

`GET /monitoring` is a built-in operations console for the live inference
path. It shows Django, Redis, RQ, executor, detector, and classifier state;
queue pressure; cumulative latency estimates and outcomes; CUDA/process memory;
and bounded artifact identity. It polls `GET /api/v1/operations` every five
seconds and retains no server-side or browser-side history.

The JSON snapshot is schema-versioned and returns `Cache-Control: no-store`.
The same snapshot contract supplies `/readyz`, `/api/v1/models`, and the
Redis/RQ/executor metrics described below, so those views share one observation
of current state. Redis and executor probes have explicit timeouts. Readiness
declares `readiness_scope: artifact_ready`; cold `unloaded` is ready when the
manifest, telemetry collector, artifacts, controller, device, and native
operator are available.
`inference_warm` and `warm_model` separately describe the sole resident model.

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

Forked RQ work-horses no longer write Prometheus multiprocess files: queue-wait
accounting is recorded in Redis and surfaced through the gateway observations,
so per-job child processes leave no shards behind. Gunicorn's `child_exit`
hook marks the worker dead and then removes every exact-PID counter, histogram,
and gauge shard for that process. Every
telemetry write is additionally exception-guarded, so a full or corrupt
metrics directory degrades observability without failing requests, and the
readiness gate no longer depends on the telemetry collector (its state is
reported informationally). CUDA and process-RSS gauges use `livemostrecent`
multiprocess semantics, so only live-process observations contribute.

`GET /metrics` combines process-safe instrumentation with snapshots from the
existing Redis/RQ and executor-status contracts. It covers:

- HTTP, DICOM, prediction, queue, and worker-loss outcomes;
- queue depth, active work, and queue-wait duration;
- model load/warmup, reuse, switch, cleanup, inference, and failure events;
- decode and executor-pipeline latency;
- allocated, reserved, peak-allocated, and peak-reserved CUDA bytes;
- classifier ROI counts, padding fallbacks, CUDA OOMs, and process RSS; and
- executor artifact readiness, controller initialization, model-specific
  inference warmth, artifact/operator/device checks, model residency, and
  active-task age/deadline state.

The console derives p50 and p95 estimates from histogram buckets. Queue-wait
buckets are Redis-owned and cumulative for the Redis lifecycle. Process-local
web/executor metrics describe the live-process epoch; exited Gunicorn shards
are removed, so they are not a durable cumulative history across worker churn.
None of these values is a durable time series, service-level objective, or
alerting system.
Clear the directory only while every instrumented process is stopped. Use the
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
The packaged gunicorn configuration disables the access log for the same
reason: default access-log request lines would print capability-style
prediction IDs that this event stream deliberately withholds.

Readiness returns bounded internal reason codes such as `redis_unavailable` or
`native_operator_unavailable`. Manifest and collector failures similarly return
`manifest_unavailable` or `telemetry_unavailable` with an empty inventory or
zero-valued telemetry shape; exception text and filesystem details remain
private. The operations snapshot and page additionally exclude prediction
values and request identifiers. These surfaces are operational telemetry, not a
clinical audit log or evidence of accuracy, calibration, or medical fitness.
