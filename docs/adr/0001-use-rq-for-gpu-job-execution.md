---
status: accepted
---

# Use RQ for GPU job execution

The inference gateway will use RQ over Redis instead of Celery because RQ already owns the job lifecycle, status registries, failure handling, and TTLs required by this single-queue deployment. The gateway will retain only project-specific admission and idempotency rules, enqueue opaque prediction and storage identifiers, disable automatic retries, and keep private request and result payloads outside Redis.

## Considered options

- Celery provides broader routing and worker controls than this single-GPU deployment needs.
- A raw Redis list would require rebuilding job lifecycle and worker-loss behavior already provided by RQ.

## Consequences

The production worker processes one job at a time using RQ's standard forked execution model, which provides clean failure isolation but recreates the job process for every prediction. Its startup overhead must be measured on the L4 before making an API-latency claim; a same-process worker is not adopted without separately addressing its heartbeat and CUDA-failure behavior.
