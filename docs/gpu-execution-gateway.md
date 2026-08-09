# GPU execution gateway

`GpuExecutionGateway` protects the single GPU owner behind `submit`, `status`,
`result`, and bounded `wait` operations. RQ owns the queued, started, finished,
and failed job lifecycle; the gateway adds only the serving rules that RQ does
not know about.

The queue choice is recorded in
[ADR 0001](adr/0001-use-rq-for-gpu-job-execution.md).
The persistent CUDA-owner topology is recorded in
[ADR 0002](adr/0002-use-a-persistent-gpu-executor.md).
The model-residency correction is recorded in
[ADR 0003](adr/0003-enforce-single-model-residency.md).

## Execution contract

- One standard RQ worker consumes `gpu-inference` one job at a time.
- The worker parent and its isolated work-horses never construct the pipeline
  or initialize CUDA. A work-horse sends the two opaque job tokens over an
  owner-only Unix socket and waits for a generic success or failure response.
- One long-lived executor owns CUDA and the pipeline. Artifact verification
  plus pinned runtime, device, and native-operator verification complete before
  the socket becomes artifact-ready. The controller may remain unloaded;
  exactly one model loads and warms on demand, and cross-model requests unload
  it before loading the other model.
- The same socket exposes a bounded, sanitized status operation used by Django
  readiness and model inventory. It reports no paths, artifact filenames,
  prediction identifiers, or failure details.
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
payloads. The persistent executor writes the prediction result atomically to
the same private directory and removes the request files.

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
| Pipeline failure | RQ records one terminal failed job; result retrieval returns sanitized HTTP 500 and no retry. |
| Work-horse termination | RQ records failure and result retrieval returns sanitized HTTP 503. |
| Executor unavailable or failed runtime | The current RQ job fails once and result retrieval returns sanitized HTTP 503. |
| RQ execution timeout | RQ records terminal failure and result retrieval returns sanitized HTTP 504. |
| Bounded synchronous wait elapsed | Return a healthy pollable handle without cancelling the job. |
| Result TTL elapsed | Return `prediction_result_expired` while short-lived RQ status remains available. |

`observations()` reports active, queued, and running counts plus admission,
rejection, success, failure, worker-loss, and accumulated queue-wait metrics.
It exposes no prediction identifiers or private payload data.
Prometheus multiprocess setup, bounded labels, and the JSON event contract are
documented in [`observability.md`](observability.md).

## Real-infrastructure validation

Use the repository's uv environment:

```powershell
uv sync --extra gateway --extra web
$env:VMS_TEST_REDIS_URL = "redis://127.0.0.1:6379/15"
$env:VMS_TEST_DICOM_PATH = "$PWD\fixtures\cbis-ddsm\1.3.6.1.4.1.9590.100.1.2.100131208110604806117271735422083351547\1-1.dcm"
uv run --extra gateway --extra web python tests/real_infra/test_django_api.py
```

This gate requires real Redis but no GPU. The separate
`tests/real_infra/test_django_l4_inference.py` gate requires the real executor,
RQ worker, model artifacts, and NVIDIA L4.

## Production processes

The processes share the same private socket directory and ephemeral job root.
Set `PYTHONPATH` because this repository is not packaged as an installed wheel:

```bash
export PYTHONPATH="$PWD/src"

uv run --extra gateway python -m vision_model_serving.execution.executor \
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

Start the RQ worker only after the executor socket exists. Django readiness
uses the socket's status operation rather than file existence. The executor
timeout must not exceed the RQ job timeout, and the result TTL must match the
gateway configuration.

## L4 acceptance evidence

The checked-in 2026-08-08/09 packaged records describe the superseded
dual-resident policy and remain historical only. The standalone
single-residency record proves real adapter unload/switch behavior but predates
the corrected packaged topology. Run current GPU smoke, browser,
destructive-restart, and schema-v3 benchmark gates on one clean revision before
claiming current packaged L4 acceptance.
