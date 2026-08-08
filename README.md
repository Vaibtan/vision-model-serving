# Vision Model Serving

A Django/DRF service for the pinned FocalNet-DINO → MMBCD mammography
pipeline. The deployed topology keeps Django and RQ CPU-only and gives one
long-lived process exclusive CUDA ownership behind an owner-only Unix socket.

This is a research-serving implementation, not a diagnostic system. Model
class semantics, a medical decision threshold, clinical accuracy, calibration,
robustness, and checkpoint redistribution rights are not established.

## What is implemented

- bounded DICOM decode and deterministic 1024×1024 canonicalization;
- strict, checksum-pinned offline detector and classifier adapters;
- deterministic top-300, strict `IoU > 0.1` NMS, and exactly eight MMBCD ROIs;
- detection-only and full multipart prediction endpoints with polling/results;
- Redis Queue admission, idempotency, TTLs, sanitized failures, and private
  tmpfs request/result storage;
- a persistent serialized L4 executor with both models reused after first load;
- liveness, fail-closed readiness, model inventory, OpenAPI, safe metrics/logs;
- pinned non-root/read-only Docker images, smoke/benchmark/restart profiles; and
- an attributed, checksum-pinned public CBIS-DDSM fixture fetcher.

## Quick verification

```powershell
uv sync --frozen --extra gateway --extra web
$env:PYTHONPATH = "src"
uv lock --check
uv run python -m vision_model_serving.artifacts config/model-artifacts.json
uv run python -m unittest discover -s tests
uv run python manage.py check
```

The unit suite validates contracts without proprietary weights. Real service
claims require the separate Redis/DICOM or L4/Compose gates; see
[fresh-machine reproduction](docs/reproduction.md).

## API surface

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/v1/predictions` | Submit multipart DICOM, mode, and full-mode history |
| `GET` | `/api/v1/predictions/{id}` | Poll bounded job status |
| `GET` | `/api/v1/predictions/{id}/result` | Retrieve a completed typed result |
| `GET` | `/api/v1/models` | Manifest and sanitized executor inventory |
| `GET` | `/livez` | Web-process liveness only |
| `GET` | `/readyz` | Redis, RQ, executor, artifacts, device, operator readiness |
| `GET` | `/api/schema/`, `/api/docs/` | OpenAPI schema and browser view |
| `GET` | `/metrics` | Trusted-network operational metrics when enabled |

The service binds to `127.0.0.1` in Compose and has no authentication layer.
Put authenticated TLS ingress in front of it before any remote exposure.

## Architecture decision to notice

The assignment asks for one model loaded at a time. The repository contains a
validated strict unload/switch implementation, but the accepted deployed ADR
retains both models after first use because repeated reconstruction dominated
latency and both fit on the L4. Inference remains serialized under one GPU
owner, but this is not literal compliance with the load/unload sentence. The
tradeoff and rollback path are explicit in
[the architecture guide](docs/architecture.md#gpu-lifecycle-active-is-not-resident).

## Documentation

- [Fresh-machine reproduction, curl examples, benchmark, troubleshooting](docs/reproduction.md)
- [Architecture, model flow, lifecycle, queue, privacy, operations](docs/architecture.md)
- [Assignment traceability and validation evidence index](docs/traceability.md)
- [Container build and profile runbook](docs/containers.md)
- [Artifact inventory, trust, licensing, and semantic boundaries](docs/model-artifact-inventory.md)
- [DICOM contract and supported transfer syntaxes](docs/dicom-canonicalization.md)
- [Detector adapter](docs/detector-adapter.md) and [classifier adapter](docs/classifier-adapter.md)
- [RQ/executor gateway and failure semantics](docs/gpu-execution-gateway.md)
- [Privacy-safe metrics and structured logs](docs/observability.md)
- [Detailed L4 upstream reproduction](docs/validation/lightning-l4-fp32-reproduction.md)

The source requirements are in [ASSIGNMENT.md](ASSIGNMENT.md) and the
evidence-backed status of each one is in
[docs/traceability.md](docs/traceability.md).
