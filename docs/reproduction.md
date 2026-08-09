# Fresh-machine reproduction and API operation

This guide reproduces the validated FP32 service from source. It requires the
two evaluator-supplied checkpoints; their redistribution is not authorized, so
they are never downloaded by this repository or copied into an image.

## 1. Prerequisites

- Git, Docker Engine with Compose, `uv` 0.8.4, `curl`, `jq`, and OpenSSL;
- for inference, an NVIDIA L4 with driver 580.173.02-compatible CUDA support
  and the NVIDIA Container Toolkit;
- enough disk for the 8.6 GB executor image and 3.3 GB of external weights;
- outbound network during source/image/tokenizer/fixture preparation only.

Clone and synchronize the CPU development environment:

```bash
git clone https://github.com/Vaibtan/vision-model-serving.git
cd vision-model-serving
git switch main
uv sync --frozen --extra gateway --extra web
uv lock --check
export PYTHONPATH="$PWD/src"
uv run python -m unittest discover -s tests
uv run python manage.py check
```

The assessed implementation is on `main`. Use a reviewed commit rather than a
moving branch for an assessed deployment.

## 2. Fetch the attributed public fixture

The fetcher accepts only the manifest-owned TCIA host, bounds download and ZIP
expansion, rejects unsafe members, verifies exact bytes, preserves `LICENSE`,
and writes `ATTRIBUTION.md` plus a bounded report.

```bash
uv run python scripts/fetch_public_fixture.py \
  --manifest config/public-fixtures.json \
  --fixture cbis-ddsm-l4-reference \
  --output-root fixtures

uv run python scripts/fetch_public_fixture.py \
  --manifest config/public-fixtures.json \
  --fixture cbis-ddsm-l4-reference \
  --output-root fixtures \
  --offline
```

Both commands must report DICOM SHA-256
`9f70081672a460f29231bb471e8a9e26dd3ed26a2ebbd91c064e575e7842a19c`.
The second command proves the installed copy without network access.

## 3. Prepare external runtime assets

Use this layout, or export the corresponding Compose variables to other
absolute paths:

```text
external/
├── artifacts/
│   ├── focalnet-dino-finetuned.pth
│   └── mmbcd_best.pt
├── assets/roberta-base-tokenizer-e2da8e2f811d1448a5b465c236feacd80ffbac7b/
└── sources/
    ├── MMBCD/  # 14ac5e099c79253b01e0885d2ebefa6f86cfd8f0
    └── dino/   # 7c446df5b9f45747937fb0d72314eb9f7b66930a
```

The weight filenames, sizes, and hashes are authoritative in
[`config/model-artifacts.json`](../config/model-artifacts.json). Do not rename,
modify, or deserialize a checkpoint until its registry verification passes.

Prepare the two source mounts and tokenizer while outbound access is allowed:

```bash
mkdir -p external/sources external/assets
git clone https://github.com/adsbansal/MMBCD.git external/sources/MMBCD
git -C external/sources/MMBCD checkout --detach \
  14ac5e099c79253b01e0885d2ebefa6f86cfd8f0
git clone https://github.com/facebookresearch/dino.git external/sources/dino
git -C external/sources/dino checkout --detach \
  7c446df5b9f45747937fb0d72314eb9f7b66930a

uv sync --frozen --extra gpu --extra gateway --extra web
uv run python scripts/l4_validation/10_download_roberta_tokenizer.py \
  --output-dir external/assets/roberta-base-tokenizer-e2da8e2f811d1448a5b465c236feacd80ffbac7b
```

## 4. Build and run the packaged L4 smoke test

```bash
export VMS_ARTIFACT_ROOT="$PWD/external/artifacts"
export VMS_TOKENIZER_ROOT="$PWD/external/assets/roberta-base-tokenizer-e2da8e2f811d1448a5b465c236feacd80ffbac7b"
export VMS_MMBCD_ROOT="$PWD/external/sources/MMBCD"
export VMS_DINO_ROOT="$PWD/external/sources/dino"
export VMS_DICOM_PATH="$PWD/fixtures/cbis-ddsm/1.3.6.1.4.1.9590.100.1.2.100131208110604806117271735422083351547/1-1.dcm"
export VMS_SECRET_KEY="$(openssl rand -hex 32)"

docker compose --profile gpu up --build \
  --abort-on-container-exit --exit-code-from gpu-smoke
docker compose --profile gpu down --volumes --remove-orphans
```

The smoke container sends a real multipart request through Gunicorn, Django,
Redis, a standard RQ worker, the Unix socket, and the persistent L4 executor.
It fails unless the detector and classifier reproduce the pinned serving
hashes. The longer two-lifecycle gate is documented in
[`containers.md`](containers.md#destructive-restart-validation-profile).

## 5. Operate the API

Open `http://127.0.0.1:8000/` for the server-rendered inspection workbench. It
submits through the same versioned prediction API shown below and displays the
canonical mammogram, ROI overlays/crops, non-causal attention weights, timings,
runtime residency, warnings, and downloadable sanitized JSON/PNG. The preview
is returned with `Cache-Control: no-store`; the browser keeps no job history.

Start only the long-running services:

```bash
docker compose --profile gpu up --build -d redis executor rq-worker web
curl --fail http://127.0.0.1:8000/livez
curl --fail http://127.0.0.1:8000/readyz
curl --fail http://127.0.0.1:8000/api/v1/models
```

Detection needs only a DICOM. Full mode also requires non-blank clinical
history; that text is formatted as `Indication: {history}` and is not returned.

```bash
wait_for_prediction() {
  local prediction_id="$1"
  while true; do
    payload="$(curl --fail --silent \
      "http://127.0.0.1:8000/api/v1/predictions/${prediction_id}")"
    state="$(jq -r .state <<<"${payload}")"
    case "${state}" in
      succeeded) return 0 ;;
      failed|expired) echo "${payload}" >&2; return 1 ;;
    esac
    sleep 0.2
  done
}

curl --fail --silent --show-error \
  -H 'Prefer: respond-async' \
  -F mode=detection \
  -F "dicom=@${VMS_DICOM_PATH};type=application/dicom" \
  http://127.0.0.1:8000/api/v1/predictions > detection-handle.json

PREDICTION_ID="$(jq -r .prediction_id detection-handle.json)"
wait_for_prediction "${PREDICTION_ID}"
curl --fail --silent \
  "http://127.0.0.1:8000/api/v1/predictions/${PREDICTION_ID}/result" \
  > detection-result.json

curl --fail --silent --show-error \
  -H 'Prefer: respond-async' \
  -F mode=full \
  -F 'clinical_history=real public mammogram acceptance.' \
  -F "dicom=@${VMS_DICOM_PATH};type=application/dicom" \
  http://127.0.0.1:8000/api/v1/predictions > full-handle.json

PREDICTION_ID="$(jq -r .prediction_id full-handle.json)"
wait_for_prediction "${PREDICTION_ID}"
curl --fail --silent \
  "http://127.0.0.1:8000/api/v1/predictions/${PREDICTION_ID}/result" \
  > full-result.json
```

A submission returns this stable shape; token and timestamps vary:

```json
{
  "prediction_id": "<32-character opaque token>",
  "state": "queued",
  "submitted_at": "<ISO-8601 timestamp>",
  "expires_at": "<ISO-8601 timestamp>",
  "status_url": "/api/v1/predictions/<token>",
  "result_url": "/api/v1/predictions/<token>/result",
  "idempotent_replay": false
}
```

The raw result intentionally includes bounded detector tensors, boxes, timing,
and provenance. This `jq` view is a compact rendering of a real full result,
not a substitute response schema:

```bash
jq '.result | {
  mode,
  input_sha256: .input.source_sha256,
  detector_sha256: .detector.prediction_sha256,
  classifier_sha256: .classification.prediction_sha256,
  classifier_rois: (.detector.classifier_rois | length),
  warnings: [.warnings[].code],
  provenance,
  disclaimer
}' full-result.json
```

For the pinned fixture and history it reports detector hash `4cdd09d9…a9a5`,
classifier hash `f994ccfa…f1a3`, eight ROIs, unverified-semantics warnings, and
`Research use only; not a medical diagnosis.` It does not expose a class name
or medical decision threshold. Browse `/api/docs/` for the generated OpenAPI
view and `/api/schema/?format=json` for the machine-readable schema.

## 6. Browser acceptance, benchmark, and cleanup

```bash
docker compose --profile gpu down --volumes --remove-orphans
docker compose --profile browser up --build --pull never \
  --abort-on-container-exit --exit-code-from browser-acceptance
docker compose --profile browser down --volumes --remove-orphans
```

The browser gate drives real upload, polling, overlay/crop/attention
inspection, and sanitized JSON/PNG export through Chromium. For the schema-v3
host benchmark and strict TensorRT/PyTorch L4 lanes, use
[`containers.md`](containers.md#benchmark-profile) and
[`acceleration.md`](acceleration.md). The benchmark requires a clean exact
revision, fresh unloaded executor, concurrency 1/2/4, Docker identity, and
`nvidia-smi` sampling; it writes both JSON and Markdown or fails.

Always remove the stack volumes after assessment work; they are tmpfs-backed
but can contain bounded results until their TTL expires:

```bash
docker compose --profile gpu down --volumes --remove-orphans
```

## Failure recovery

| Symptom | Action |
| --- | --- |
| `/readyz` says `verified_artifacts_unavailable` | Recheck mounted filenames, sizes, hashes, source revisions, tokenizer files, and read permissions. Do not bypass verification. |
| `/readyz` says `native_operator_unavailable` | Rebuild the executor for compute capability 8.9 and inspect the pinned CUDA/PyTorch build; do not fall back silently. |
| `prediction_queue_full` | Wait for the one running or queued job; capacity is intentionally one. |
| `prediction_runtime_unavailable` | Restart the executor, wait for readiness, then submit a new request. Automatic inference retry is disabled. |
| `prediction_result_expired` | Resubmit the source request; the result TTL elapsed by design. |
| A golden hash changes | Stop, preserve the report, compare commit/config/artifact/runtime identities, and explain drift before updating any baseline. |
| DICOM is rejected | Check the supported syntax/pixel limits in [`dicom-canonicalization.md`](dicom-canonicalization.md); do not convert it silently. |

TensorRT, FP16/BF16, TF32, and `torch.compile` are not selected in production.
The repository implements isolated fail-closed L4 evaluation lanes; promotion
still requires generated same-revision parity, memory, reliability, restart,
and end-to-end performance evidence. There is no eager fallback inside the
TensorRT engine verifier.
