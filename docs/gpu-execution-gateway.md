# GPU execution gateway

`GpuExecutionGateway` protects the single GPU owner behind `submit`, `status`,
`result`, and bounded `wait` operations. RQ owns the queued, started, finished,
and failed job lifecycle; the gateway adds only the serving rules that RQ does
not know about.

The queue choice is recorded in
[ADR 0001](adr/0001-use-rq-for-gpu-job-execution.md).
The persistent CUDA-owner topology is recorded in
[ADR 0002](adr/0002-use-a-persistent-gpu-executor.md).

## Execution contract

- One standard RQ worker consumes `gpu-inference` one job at a time.
- The worker parent and its isolated work-horses never construct the pipeline
  or initialize CUDA. A work-horse sends the two opaque job tokens over an
  owner-only Unix socket and waits for a generic success or failure response.
- One long-lived executor owns CUDA and the pipeline. Artifact verification
  completes before the socket becomes ready; detector and classifier remain
  resident after their first successful loads.
- Automatic retry is disabled. An unexpected work-horse exit is terminal for
  that job; the next job receives a clean child process.
- The queue and worker use RQ's JSON serializer.
- The only job arguments are the opaque prediction ID and storage locator.

RQ retains clean work-horse failure isolation without forcing CUDA or model
construction into every job process. The executor is deliberately not an RQ
`SimpleWorker`: its narrow interface keeps queue lifecycle and GPU ownership
separate.

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
| Executor unavailable or execution failure | The current RQ job fails once with a sanitized error; no retry is scheduled. |
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

## Production processes

The processes share the same private socket directory and ephemeral job root.
Set `PYTHONPATH` because this repository is not packaged as an installed wheel:

```bash
export PYTHONPATH="$PWD/src"

uv run --extra gateway python -m vision_model_serving.execution.executor_cli \
  --socket-path /run/vision-model-serving/executor.sock \
  --job-root /var/lib/vision-model-serving/jobs \
  --result-ttl-seconds 900 \
  --artifact-root /models \
  --tokenizer-root /assets/roberta-tokenizer \
  --focalnet-root /sources/FocalNet-DINO \
  --mmbcd-root /sources/MMBCD \
  --dino-root /sources/dino

uv run --extra gateway python -m vision_model_serving.execution.rq_cli \
  --redis-url redis://127.0.0.1:6379/0 \
  --queue-name gpu-inference \
  --executor-socket /run/vision-model-serving/executor.sock \
  --executor-timeout-seconds 180
```

The executor socket is the readiness signal. Start the RQ worker only after it
exists. The executor timeout must not exceed the RQ job timeout, and the result
TTL must match the gateway configuration.
