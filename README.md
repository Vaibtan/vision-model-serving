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
- a server-rendered upload/inspection workbench with an ephemeral mammogram
  preview, ROI overlay/gallery, timings, warnings, and sanitized exports;
- Redis Queue admission, idempotency, TTLs, sanitized failures, and private
  tmpfs request/result storage;
- a persistent serialized L4 executor with an enforced maximum of one resident
  model, same-model reuse, and unload-before-switch behavior;
- scoped artifact readiness, model-specific inference-warm state, liveness,
  model inventory, OpenAPI, and safe metrics/logs;
- pinned non-root/read-only Docker images, smoke/restart profiles, and
  schema-v3 benchmark, optimization/TensorRT, and browser acceptance tooling; and
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

The browser lane additionally runs `uv sync --group browser`,
`uv run playwright install chromium`, and
`uv run --group browser python tests/browser/test_inspection_workbench.py`.

The unit suite validates contracts without proprietary weights. Real service
claims require the separate Redis/DICOM or L4/Compose gates; see
[fresh-machine reproduction](docs/reproduction.md).

The current NVIDIA L4 run passed strict single residency, artifact-scoped
readiness, the schema-v3 concurrency benchmark, destructive restart, and the
packaged browser workflow. The optimization matrix retained eager FP32 and the
TensorRT lane concluded a measured STOP. See the
[five-finding resolution record](docs/validation/spec-resolution-l4-20260810.md).

## API surface

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/v1/predictions` | Submit multipart DICOM, mode, and full-mode history |
| `GET` | `/` | Open the local DICOM and ROI inspection workbench |
| `POST` | `/api/v1/dicom-preview` | Render an ephemeral metadata-free canonical PNG |
| `GET` | `/api/v1/predictions/{id}` | Poll bounded job status |
| `GET` | `/api/v1/predictions/{id}/result` | Retrieve a completed typed result |
| `GET` | `/api/v1/models` | Manifest and sanitized executor inventory |
| `GET` | `/monitoring` | Privacy-safe live inference operations console |
| `GET` | `/api/v1/operations` | Versioned bounded operational snapshot |
| `GET` | `/livez` | Web-process liveness only |
| `GET` | `/readyz` | Redis, RQ, executor, artifacts, device, operator readiness |
| `GET` | `/api/schema/`, `/api/docs/` | OpenAPI schema and browser view |
| `GET` | `/metrics` | Trusted-network operational metrics when enabled |

The service binds to `127.0.0.1` in Compose and has no authentication layer.
Put authenticated TLS ingress in front of it before any remote exposure.

## Architecture decision to notice

The executor process is long-lived, but model residency is strict. It starts
artifact-ready and unloaded. A same-model request reuses the sole resident;
requesting the other stage unloads the active model before loading the next.
Artifact readiness and model-specific inference warmth are separate facts. See
[the architecture guide](docs/architecture.md#gpu-lifecycle-strict-single-residency).

## Documentation

- [Fresh-machine reproduction, curl examples, benchmark, troubleshooting](docs/reproduction.md)
- [Architecture, model flow, lifecycle, queue, privacy, operations](docs/architecture.md)
- [Assignment traceability and validation evidence index](docs/traceability.md)
- [Container build and profile runbook](docs/containers.md)
- [Artifact inventory, trust, licensing, and semantic boundaries](docs/model-artifact-inventory.md)
- [DICOM contract and supported transfer syntaxes](docs/dicom-canonicalization.md)
- [Detector adapter](docs/detector-adapter.md) and [classifier adapter](docs/classifier-adapter.md)
- [RQ/executor gateway and failure semantics](docs/gpu-execution-gateway.md)
- [Privacy-safe monitoring, metrics, and structured logs](docs/observability.md)
- [PyTorch optimization and strict TensorRT gates](docs/acceleration.md)
- [Detailed L4 upstream reproduction](docs/validation/lightning-l4-fp32-reproduction.md)

The source requirements are in [ASSIGNMENT.md](ASSIGNMENT.md) and the
evidence-backed status of each one is in
[docs/traceability.md](docs/traceability.md).
