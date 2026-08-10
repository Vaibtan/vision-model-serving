# Evidence-gated acceleration

Eager FP32 with TF32 disabled is the only selected serving backend. Candidate
code lives in validation/acceleration modules and scripts; the executor has no
runtime fallback chain or backend toggle.

## PyTorch matrix

Run the real strict-residency gate first, then evaluate each model with exactly
one changed policy: FP32 baseline, TF32, FP16 autocast, BF16 autocast, and
full-graph `torch.compile`.

```bash
export PYTHONPATH="$PWD/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
REVISION="$(git rev-parse HEAD)"

uv run --extra gpu python scripts/l4_validation/16_validate_single_residency.py \
  --cycles 2 --output /tmp/single-residency.json

uv run --extra gpu python scripts/l4_validation/17_evaluate_pytorch_optimizations.py \
  --dicom "$VMS_DICOM_PATH" \
  --single-residency-evidence /tmp/single-residency.json \
  --output /tmp/pytorch-optimization.json \
  --revision "$REVISION"
```

The matrix always publishes all five candidates for both models. Parity and
CUDA/model-switch reliability are mandatory. A candidate needs at least 15%
lower warm p50, 15% higher throughput, or 20% lower peak memory. The report
does not change production selection; a retained candidate would also need the
packaged restart and schema-v3 benchmark gates.

## TensorRT lane

The isolated L4 lane pins Torch-TensorRT 2.8.0, TensorRT 10.12.0.36, ONNX
1.16.0, Polygraphy 0.49.24, and CUDA Python 12.8.0. Build it only after the
normal executor image:

```bash
docker build -f docker/tensorrt.Dockerfile \
  -t vision-model-serving-tensorrt:local .
```

Run the builder on the target L4 with the same read-only model/source mounts.
It performs:

1. strict MMBCD `torch.export` at `[1,8,3,224,224]`, first probing the
   production token-width profile of 2 through 90 and then building one exact
   static-width-5 diagnostic engine for the pinned fixture;
2. Torch-TensorRT dry-run analysis with full compilation required;
3. raw serialized FP32 plan build with TF32 disabled;
4. execution in a fresh verifier that imports TensorRT/CUDA Python/NumPy but
   not PyTorch, Torch-TensorRT, Transformers, or model source;
5. output parity and the 15% warm-p50 gate; and
6. a separate strict detector capture/coverage probe for
   `MultiScaleDeformableAttention`.

```bash
uv pip install --python .venv/bin/python -r requirements/tensorrt-l4.txt
uv run python scripts/l4_validation/18_build_tensorrt_candidate.py \
  --dicom "$VMS_DICOM_PATH" \
  --optimization-evidence /tmp/pytorch-optimization.json \
  --output-dir /tmp/tensorrt-spike \
  --revision "$REVISION"
```

The static classifier plan is always removed after the TensorRT-only, parity,
and performance measurements because it accepts only the public fixture's
token width. Production remains STOP until one strict 2-through-90 engine
passes the same gates. The detector likewise reports STOP unless it reaches
full coverage. No result changes the executor or enables an eager fallback.

The detailed compatibility and plugin rationale is in
[`research/tensorrt-serving-feasibility-20260810.md`](research/tensorrt-serving-feasibility-20260810.md).
