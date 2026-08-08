---
status: accepted
---

# Use a persistent GPU executor behind RQ

The standard RQ worker will retain its forked work-horse model for queue
lifecycle, heartbeats, timeouts, and failure isolation. Each work-horse will
send only the opaque prediction ID and storage locator over a private Unix
socket to one long-lived GPU executor. The executor verifies artifacts before
opening the socket, owns the prediction pipeline for its lifetime, serializes
all inference, and retains both detector and classifier residents after their
first successful loads.

## Context

The first live RQ/L4 validation completed correctly but took 40.529 seconds.
Only 0.578 milliseconds elapsed between the RQ job start and task entry;
approximately 15 seconds were instead spent reconstructing and re-verifying
the pipeline in the per-job work-horse, while the pipeline lifecycle took
24.896 seconds. RQ's fork was not the material latency source.

## Why pair RQ with a separate executor

Model residency is deliberately owned by the GPU executor rather than by the
queue worker. Once CUDA and both models live behind that process boundary,
neither RQ nor Celery determines warm inference latency; the queue worker only
delivers a job and waits for the executor. Keeping RQ therefore preserves its
smaller job-lifecycle surface and per-job work-horse isolation without paying
per-job model construction.

A concurrency-one Celery worker with resident models would also be technically
possible, but it would couple CUDA lifetime to Celery's pool lifecycle and
reintroduce orchestration features this single-queue service does not use. An
RQ `SimpleWorker` was also rejected because same-process execution gives up the
standard worker's work-horse isolation and changes heartbeat and hard-failure
behavior. The selected split keeps queue lifecycle and GPU lifecycle explicit
and independently restartable.

## Consequences

- RQ remains the only durable job-lifecycle authority; the Unix socket is a
  local synchronous execution interface, not another queue.
- The RQ parent and work-horses do not import model factories or initialize
  CUDA. Losing a work-horse remains a terminal RQ job failure.
- DICOM bytes, clinical history, and prediction results stay in the private
  ephemeral job directory. The socket carries two 32-character opaque tokens.
- Artifact verification moves to executor startup. The first request still
  pays model load and warmup; later requests reuse both resident models.
- Executor failure makes the current RQ job fail without automatic retry. A
  process supervisor must restart the executor before later jobs can succeed.
- The socket directory and file are owner-only. RQ worker and executor must
  run as the same operating-system identity and share the ephemeral job root.
- The socket also provides a sanitized status operation so the CPU-only web
  process can report executor, artifact, device, native-operator, and model
  residency state without importing model code or CUDA.
- Dual residency and warm latency require a real NVIDIA L4 acceptance run;
  CPU substitutes do not establish this decision's performance or memory fit.

## Validation

The real Redis/RQ/L4 acceptance run at commit `775c52b` reproduced both exact
golden prediction hashes and classifier logits. The cold request completed in
25.871 seconds; the next request completed in 1.211 seconds with zero model
reload time. Both models occupied 2,334 MiB while resident. The complete
bounded evidence is in
[`persistent-rq-executor-l4-20260808.json`](../validation/persistent-rq-executor-l4-20260808.json).
