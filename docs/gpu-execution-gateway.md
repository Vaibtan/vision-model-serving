# GPU execution gateway

`GpuExecutionGateway` protects the single GPU owner behind a bounded prediction
job lifecycle. Django will call the same narrow interface for asynchronous
polling and bounded synchronous compatibility:

```python
handle = gateway.submit(request)
status = gateway.status(handle.prediction_id)
result = gateway.result(handle.prediction_id)
result_or_handle = gateway.wait(handle.prediction_id, timeout_seconds=5)
```

`wait` never cancels accepted work. It returns a completed `PredictionResult`,
raises the same terminal error as `result`, or returns a current
`PredictionHandle` that the HTTP layer can map to a healthy `202` response.

## Adapters and process isolation

The module has two adapters at the execution seam:

- `InMemoryGpuExecutionGateway` owns one background thread by default and is
  suitable for local and interface tests.
- `CeleryRedisGpuExecutionGateway` uses Redis Lua transitions for cross-process
  admission/state and Celery only for opaque task dispatch.

The production web-side import does not import PyTorch or the local CUDA
pipeline composition root. `register_prediction_task` constructs the worker
through a lazy factory on the first task execution in the Celery child process.
The web process therefore has no interface through which to initialize CUDA or
create a model copy.

The required Celery settings are applied and asserted in tests:

- worker concurrency is one;
- worker prefetch multiplier is one;
- late acknowledgement is enabled;
- worker-loss rejection/requeue and automatic task retry are disabled;
- JSON is the only accepted task serialization format; and
- Redis visibility timeout must exceed the configured worker-loss lease.

The task has `max_retries=0`. A hard-terminated prefork child is replaced with
a clean child whose lazy worker factory has not constructed CUDA state. The
failed job is not automatically replayed after GPU execution may have begun.

## Private payload and broker contract

Submission reads the caller stream once and writes it under a random locator in
the ephemeral jobs root. The request metadata file contains mode and clinical
history; the DICOM is a separate private file. Neither value enters Redis job
metadata or the Celery message.

The complete broker argument list is:

```json
["<unguessable-prediction-id>", "<opaque-storage-locator>"]
```

Idempotency keys are SHA-256 digested before state storage. Request identity is
a length-delimited SHA-256 over mode, clinical history, and DICOM bytes. The
digest detects reuse of one idempotency key for different input without
placing the input in Redis.

Successful result JSON is atomically created and an identical repeated write
is accepted. A conflicting second result fails closed. After the result is
durable, request metadata and DICOM bytes are deleted. Startup cleanup and
`cleanup_expired()` remove expired or corrupt job directories; symlinked job
paths are rejected.

## Admission and lifecycle

Redis admission is one Lua operation across all web processes. It expires
abandoned leases, checks an existing idempotency binding, checks the combined
queued-plus-running capacity, creates job state, reserves capacity, and updates
observations atomically.

```text
queued -> running -> succeeded -> expired
   |         |            |
   |         +-> failed ---+
   +------------> failed --+
```

Queued lease expiry becomes `prediction_reservation_expired`. Running lease
expiry becomes `prediction_worker_lost`. A result or failure remains available
for the result TTL, then becomes an `expired` tombstone for the shorter
tombstone TTL. Unknown IDs, not-ready results, expired results, idempotency
conflicts, queue saturation, and state/dispatch unavailability have distinct
stable exception codes.

`observations()` exposes active, queued, and running depth; admitted and
rejected totals; succeeded, failed, and worker-loss totals; and accumulated
queue wait milliseconds. It contains no prediction IDs, locators, input data,
or clinical text.

## Deterministic failure outcomes

| Failure | Outcome |
| --- | --- |
| Queue full | Submission fails with `prediction_queue_full`; no broker message is sent and the staged payload is deleted. |
| Celery dispatch failure | Atomic admission is cancelled, the staged payload is deleted, and submission fails with `prediction_gateway_unavailable`. |
| Redis unavailable before execution | The worker does not enter the prediction pipeline; the task fails without automatic retry. |
| Pipeline failure | Request files are deleted and state becomes `failed` with sanitized `prediction_execution_failed`. |
| Worker loss or hard termination | The running lease expires to sanitized `prediction_worker_lost`; deployment must replace the worker process to obtain a clean CUDA runtime. |
| Result written but state completion fails | The result file is left for TTL cleanup; state remains authoritative and eventually fails by lease. GPU execution is not automatically repeated. |
| Synchronous wait timeout | The job remains valid and a pollable handle is returned. |
| Redis state loss | Persistence is intentionally disabled for the assessment profile; affected IDs become unknown even if an orphan result file exists, and cleanup removes that file by TTL. |

## Composition

Install the optional CPU-side dependencies with the project:

```bash
python -m pip install -e ".[gateway]"
```

Construct `CeleryRedisGpuExecutionGateway` from a Celery app, a Redis client,
an ephemeral shared jobs path, and `GpuExecutionConfig`. Register the task with
`register_prediction_task(app, build_prediction_worker_factory(...))`. The
pipeline factory passed to the worker factory is the only place that should
call `build_local_cuda_pipeline`.

The broker and state Redis instance must be network-isolated and configured
without persistence for the assessment profile. The jobs root must be mounted
only into the web and inference-worker containers. Model artifacts and the GPU
must be mounted only into the inference worker.

## Validation boundary

The local suite exercises 32 gateway cases covering concurrent admission,
idempotency, saturation, status/result transitions, TTL expiry, abandoned
reservations, worker loss, bounded waiting, cleanup, opaque Celery messages,
lazy worker construction, Redis Lua contracts, and lifecycle observations:

```powershell
$env:PYTHONPATH = "src"
python -m unittest tests.test_gpu_execution_gateway -v
python -m unittest discover -s tests -v
```

These tests need no GPU. The pinned Celery and Redis dependencies resolve on
Python 3.12, but this ticket does not claim a live Redis/Celery container smoke
test; that belongs to the reproducible deployment work. Real model execution
remains covered by the separate NVIDIA L4 prediction-pipeline gate.
