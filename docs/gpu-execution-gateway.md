# GPU execution gateway

`GpuExecutionGateway` protects the single GPU owner behind `submit`, `status`,
`result`, and bounded `wait` operations. RQ owns the queued, started, finished,
and failed job lifecycle; the gateway adds only the serving rules that RQ does
not know about.

The queue choice is recorded in
[ADR 0001](adr/0001-use-rq-for-gpu-job-execution.md).

## Execution contract

- One standard RQ worker consumes `gpu-inference` one job at a time.
- The worker parent configures a lazy pipeline factory but never initializes
  CUDA. RQ forks an isolated work-horse that constructs the pipeline and runs
  one prediction.
- Automatic retry is disabled. An unexpected work-horse exit is terminal for
  that job; the next job receives a clean child process.
- The queue and worker use RQ's JSON serializer.
- The only job arguments are the opaque prediction ID and storage locator.

The default forked worker deliberately trades process-startup overhead for
clean CUDA failure isolation. That overhead must be measured on the L4 before
claiming API latency; the ADR explains why `SimpleWorker` is not the production
default.

## Privacy and storage

DICOM bytes and clinical history are written beneath a random per-job locator
in the ephemeral jobs directory. Redis contains opaque identifiers, request
fingerprints, TTLs, and lifecycle metadata, but no private request or result
payloads. The worker writes the prediction result atomically to the same
private directory and removes the request files.

RQ retains only job status for the configured status TTL. The task returns no
prediction value to Redis. Expired and corrupt job directories are removed at
gateway and worker startup and by `cleanup_expired()`.

## Admission and idempotency

Redis `WATCH`/`MULTI` reserves capacity and enqueues the RQ job in one
transaction. The reservation counts pending plus running work across every web
process. When capacity is full, submission fails with
`prediction_queue_full` and the staged private payload is deleted.

Idempotency keys are SHA-256 digested. A duplicate request with the same input
returns the existing job; reuse for different input fails with
`prediction_idempotency_conflict`. Once a result expires, its binding can be
replaced by a new submission.

## Failure outcomes

| Failure | Outcome |
| --- | --- |
| Queue full | Reject before creating another RQ job. |
| Redis or enqueue unavailable | Delete the staged payload and return `prediction_gateway_unavailable`. |
| Pipeline failure | RQ records one terminal failed job with a sanitized public error and no retry. |
| Work-horse termination | RQ records failure; the gateway maps an abandoned execution to `prediction_worker_lost`. |
| Synchronous timeout | Return a healthy pollable handle without cancelling the job. |
| Result TTL elapsed | Return `prediction_result_expired` while short-lived RQ status remains available. |

`observations()` reports active, queued, and running counts plus admission,
rejection, success, failure, worker-loss, and accumulated queue-wait metrics.
It exposes no prediction identifiers or private payload data.

## Local validation

Use the repository's uv environment:

```powershell
uv sync --extra gateway
uv run --extra gateway python -m unittest tests.test_rq_execution_gateway -v
uv run --extra gateway python -m unittest discover -s tests -v
```

These tests require no GPU. The production RQ worker's process-startup and
end-to-end inference latency still require the separate NVIDIA L4 validation
gate.
