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
- Django's request-path Redis client bounds connection establishment with
  `VMS_REDIS_CONNECT_TIMEOUT_SECONDS` and each socket operation with
  `VMS_REDIS_SOCKET_TIMEOUT_SECONDS`; the shorter operational probe deadline is
  configured separately.

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
prediction value to Redis. Retention is enforced physically as well as
logically: `load_request()` fails closed and deletes the directory once
`expires_at` passes, and recomputes the stored request fingerprint with a
constant-time comparison so a tampered or truncated payload can never reach
the model; `load_result()` deletes expired result directories on access; a
rate-limited janitor sweep (`maybe_cleanup()`) runs on status polls and after
every executor job, reclaiming expired locator directories and orphaned
`.tmp-*`/`.result-*` staging entries left by killed processes; and an expired
queued reservation is converted into a terminal FAILED state whose payload
directory is discarded immediately.

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
| Case rejected by the pipeline (bad DICOM pixels, no valid proposals, unusable ROIs) | Terminal `prediction_case_failed`, non-retryable; the model stays resident and the runtime stays healthy. |
| Pipeline failure | RQ records one terminal failed job; result retrieval returns sanitized HTTP 500 and no retry. |
| Work-horse termination | A terminal marker records `prediction_runtime_unavailable` (retryable) so the state survives RQ job-hash expiry; result retrieval returns sanitized HTTP 503. |
| Executor unavailable, busy, or failed runtime | The current RQ job fails once and result retrieval returns sanitized HTTP 503 (retryable). |
| RQ execution timeout | RQ records terminal failure and result retrieval returns sanitized HTTP 504. |
| Bounded synchronous wait elapsed | Return a healthy pollable handle without cancelling the job. |
| Reservation TTL elapsed while queued | The job is cancelled, its payload directory is discarded, and a Redis terminal marker pins the FAILED `prediction_reservation_expired` state monotonically — the job can no longer transition to RUNNING afterwards or decay into a 404. |
| Result TTL elapsed | Return `prediction_result_expired`; the access path deletes the stored directory. |

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
uses the socket's status operation rather than file existence. Compose now
enforces a strict timeout hierarchy: executor socket wait 170 s < RQ job
timeout 180 s < worker shutdown grace 190 s < executor shutdown grace 210 s.
On SIGTERM the executor stops accepting connections, finishes the in-flight
case (its response is still delivered), and latches the runtime closed so a
queued handler thread cannot reload a model mid-shutdown; Docker's grace is
the outer bound on that drain. The executor also fails fast with a retryable
busy signal if a second concurrent execute arrives, and bounds every
connection read so a dead peer cannot pin a handler thread. The result TTL is
shared with the web tier through `VMS_RESULT_TTL_SECONDS`.

## L4 acceptance evidence

The 2026-08-10 exact-revision
[resolution](validation/spec-resolution-l4-20260810.md),
[single-residency](validation/single-residency-l4-20260810.json), and
[destructive-restart](validation/compose-restart-l4-20260810.json) records prove
the corrected topology only for their embedded revisions. HEAD contains later
runtime, dependency, and validation changes. Run current GPU smoke, browser,
destructive-restart, and schema-v4 benchmark gates on one clean revision before
claiming current packaged L4 acceptance.
