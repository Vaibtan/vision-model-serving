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
  preview, ROI overlay/gallery, timings, warnings, and metadata-minimized
  exports that remain sensitive derived data;
- Redis Queue admission, idempotency, lease-aware physical TTLs, sanitized
  failures, and private tmpfs request/result storage;
- a persistent serialized L4 executor with an enforced maximum of one resident
  model, same-model reuse, and unload-before-switch behavior;
- scoped artifact readiness, model-specific inference-warm state, liveness,
  model inventory, OpenAPI, and safe metrics/logs;
- pinned non-root/read-only Docker images, smoke/restart profiles, and
  schema-v4 benchmark, optimization/TensorRT, and browser acceptance tooling; and
- an attributed, checksum-pinned public CBIS-DDSM fixture fetcher.

## Quick verification

```powershell
$env:UV_PYTHON = "3.12"
uv sync --frozen --python 3.12 --extra gateway --extra web --group dev
$env:PYTHONPATH = "src"
uv lock --check
uv run python -m vision_model_serving.artifacts config/model-artifacts.json
uv run python -m unittest discover -s tests
uvx --from ruff==0.14.13 ruff check src tests scripts manage.py config_cfg.py main.py
uv run python manage.py check
```

The tracked [CI workflow](.github/workflows/ci.yml) additionally checks Ruff
formatting only on changed Python files, audits the complete Ubuntu/Python 3.12
`uv.lock` selection plus the direct TensorRT pins, runs `manage.py check
--deploy` with hardened settings, validates generated OpenAPI and every Compose
profile, and executes the browser suite on pinned Chromium. It also builds both
the CPU web image and the CUDA executor image, so a broken
Docker build context or `COPY` path fails before L4 validation. The exact PyTorch
and ONNX findings in the reviewable
[audit baseline](.github/dependency-audit-baseline.json) remain risks, not clean
results; their pins require L4-compatible upgrades and renewed validation.
TensorRT-only transitive packages are outside the claim because their
requirements file is not a transitive lock. See
[fresh-machine reproduction](docs/reproduction.md) for the audit command and
boundary. The browser lane locally runs `uv sync --frozen --extra gateway
--extra web --group dev --group browser`, `uv run --no-sync playwright install
chromium`, and `uv run --no-sync python -m unittest discover -s tests/browser
-v`.

The unit suite validates contracts without proprietary weights. Real service
claims require the separate Redis/DICOM or L4/Compose gates; see
[fresh-machine reproduction](docs/reproduction.md).

The previous exact-revision NVIDIA L4 run passed strict single residency,
artifact-scoped readiness, the schema-v3 concurrency benchmark, destructive
restart, and the packaged browser workflow. The optimization matrix retained
eager FP32 and the TensorRT lane concluded a measured STOP. Current code and
dependency changes invalidated that release evidence. The 2026-08-11
[worktree L4 run](docs/validation/worktree-l4-20260811.md) subsequently passed
real packaged inference, destructive restart, Chromium, and observability, but
it is not clean-revision evidence and did not run the schema-v4 benchmark or
long switch/resource soak.

## Current operational gaps

This is an assessment-grade, production-oriented implementation, not a
production-validated service. The operational gaps identified in the
2026-08-11 review are now closed in source: request/result TTLs are enforced
physically (fail-closed loads, deletion on access, an independent lease-aware
janitor, and fingerprint verification); worker claims are atomic and an active
lease prevents cleanup from racing a running case; POST routes reject
cross-site browser requests via `Sec-Fetch-Site`/`Origin` checks and use shared
Redis-backed rate limits; work-horses no longer write per-PID Prometheus files
(queue-wait histograms live in Redis, and a gunicorn `child_exit` hook removes
every shard for the exited web worker); readiness requires exactly one fresh RQ
worker; and the timeout hierarchy is strict (executor task deadline 160 s <
socket wait 170 s < RQ timeout 180 s < worker grace 190 s < executor grace
210 s). Admission remains reserved after worker loss until the executor-owned
deadline restarts the CUDA owner. These closures are proven by the CPU suite
and bounded 2026-08-11 worktree L4 gates. A clean-revision L4 benchmark and long
switch/resource soak are still required
before any production claim. Authentication and TLS remain intentionally
absent from this loopback deployment. See
[the architecture guide](docs/architecture.md#current-operational-limitations)
and [traceability matrix](docs/traceability.md#current-review-gaps).

The preview removes DICOM metadata but does not inspect or redact burned-in
pixel annotations. Preview/overlay PNGs, the source-file hash, and prediction
JSON are not certified de-identified and must remain local and access-controlled.

## API surface

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/v1/predictions` | Submit multipart DICOM, mode, and full-mode history |
| `GET` | `/` | Open the local DICOM and ROI inspection workbench |
| `POST` | `/api/v1/dicom-preview` | Render an ephemeral canonical PNG without copied DICOM metadata; not guaranteed de-identified |
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
Loopback prevents direct remote connections, and the POST routes now reject
cross-site browser requests (`Sec-Fetch-Site`/`Origin` enforcement) and apply
per-scope rate limits. Authentication and TLS ingress are still required
before any shared-user or remote exposure.

## External model weights

The weights are never stored in Git or copied into either image. Compose mounts
`${VMS_ARTIFACT_ROOT:-../vision-model-serving-artifacts}` read-only at
`/models`. From this checkout on the current Windows machine, the default
resolves to `D:\SWE_DEV_NEW\vision-model-serving-artifacts`; set
`VMS_ARTIFACT_ROOT` to override it. Exact filenames, sizes, and SHA-256 values
are defined by [`config/model-artifacts.json`](config/model-artifacts.json).

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
- [2026-08-11 architecture and implementation review](docs/architecture-implementation-review-20260811.md)
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
