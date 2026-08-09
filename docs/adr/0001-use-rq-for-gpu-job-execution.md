---
status: accepted
---

# Use RQ for GPU job execution

The process topology consequence below is refined by
[ADR 0002](0002-use-a-persistent-gpu-executor.md).

The inference gateway will use RQ over Redis instead of Celery because RQ already owns the job lifecycle, status registries, failure handling, and TTLs required by this single-queue deployment. The gateway will retain only project-specific admission and idempotency rules, enqueue opaque prediction and storage identifiers, disable automatic retries, and keep private request and result payloads outside Redis.

## Considered options

- Celery is technically capable of running this workload, including a
  concurrency-one worker whose child process can persist across tasks. It was
  rejected because this deployment has one serialized GPU queue and does not
  need Celery's multi-queue routing, task graphs, scheduling, pool selection,
  or retry orchestration. Those facilities would add another configuration
  and operational surface without replacing the gateway's required admission,
  idempotency, private-payload storage, or result-retention rules.
- The earlier Celery adapter also required a separate project-owned Redis job
  state machine alongside Celery. RQ's jobs and registries directly provide
  the queued, started, finished, failed, timeout, TTL, and worker-loss
  lifecycle needed here, leaving the project to own only its serving-specific
  rules.
- A raw Redis list would require rebuilding job lifecycle and worker-loss behavior already provided by RQ.

This is a scope and ownership decision, not a claim that RQ executes model
inference faster than Celery. Reconsider Celery if the service gains multiple
independently scaled queues, task graphs, scheduled workflows, or routing
requirements that justify its broader orchestration model.

## Consequences

The production worker processes one job at a time using RQ's standard forked execution model, which provides clean failure isolation but recreates the job process for every prediction. Its startup overhead must be measured on the L4 before making an API-latency claim; a same-process worker is not adopted without separately addressing its heartbeat and CUDA-failure behavior.
