# Evidence-gated acceleration

Eager FP32 with TF32 disabled is the only selected serving backend. Candidate
code lives in validation/acceleration modules and scripts; the executor has no
runtime fallback chain or backend toggle.

## Current L4 decision

The 2026-08-10 matrix retained eager FP32 for both models. TF32 and FP16 were
faster but failed the frozen-output parity gates; detector BF16 is unsupported,
classifier BF16 failed parity, and both strict full-graph compile candidates
failed capture. See the [machine-readable matrix](validation/pytorch-optimization-l4-20260810.json)
and [summary](validation/pytorch-optimization-l4-20260810.md).

That matrix predates the schema-v3 acceptance contract. It remains useful as
historical screening, but it cannot promote a candidate because it used
index-aligned detector differences and did not measure candidate release or
candidate-specific repeated switches. The next L4 run records raw-output,
post-NMS, selected-ROI, release, and VRAM evidence; until then eager FP32 with
TF32 disabled remains selected.

Detector parity independently matches three semantic sets: every raw
`pred_logits`/`pred_boxes` query, the proposals retained after NMS, and the
exactly eight classifier ROIs. Each set is permutation-aware and requires class
identity plus bounded raw-logit, score, box-coordinate, and IoU differences.
IoU must be 1.0 for the unchanged baseline or at least 0.99 for a changed
candidate. Classifier tensors retain explicit per-output absolute tolerances.

TensorRT concluded **STOP**. The static token-width-5 MMBCD diagnostic engine
used zero PyTorch partitions, ran in the PyTorch-free verifier, passed parity,
and improved warm p50 by 27.8%. It was still deleted because the required
token-width 2-through-90 profile failed strict export. FocalNet-DINO stopped at
an earlier upstream `NestedTensor` tensor/string comparison during strict
capture. Its custom `MultiScaleDeformableAttention` CUDA operation remains the
expected next coverage blocker and needs a real TensorRT plugin or a separately
proven decomposition. The full evidence is in the [TensorRT report](validation/tensorrt-l4-20260810/tensorrt-spike.json),
with the measured-versus-expected failure distinction preserved in the
[failure analysis](validation/tensorrt-l4-20260810/failure-analysis.md).

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

The matrix attempts all five candidates for both models, but aborts before the
next load if the current resident is not proven released; it never completes a
matrix by measuring over a leaked predecessor. Standalone weak-reference and
CUDA-allocator release is recorded as release evidence, never as a model
switch. Each policy also runs two actual detector-to-classifier cycles. Those
cycles record singleton/empty residency snapshots, loaded and post-release
allocated/reserved VRAM, and same-model reserved-memory growth; promotion fails
closed if the cycles are unmeasured, co-residency or retained VRAM appears, or
growth exceeds 20%. Parity and repeated-switch reliability are mandatory. A
candidate also needs at least 15%
lower warm p50, 15% higher throughput, or 20% lower peak memory. The report
does not change production selection; a retained candidate would also need the
packaged restart and schema-v4 benchmark gates.

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
The report-ready explanation of the measured failures is in
[`validation/tensorrt-l4-20260810/failure-analysis.md`](validation/tensorrt-l4-20260810/failure-analysis.md).
