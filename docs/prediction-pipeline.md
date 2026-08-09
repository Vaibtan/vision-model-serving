# Prediction pipeline

`PredictionPipeline` is the repository-owned orchestration boundary for one
DICOM mammogram. Its public interface is intentionally narrow:

```python
result = pipeline.infer(case, mode)
```

The module decodes the caller-owned stream, executes the detector through the
accelerator lifecycle runtime, and optionally executes MMBCD. That runtime can
strictly switch residents; the deployed executor enables its accepted
two-resident retention policy. The persistent executor and tests consume the
same interface; neither reimplements model ordering, ROI selection, or
result shaping.

## Input and mode contract

`CaseInput` contains a readable binary DICOM stream and optional clinical
history. The caller owns the stream. The pipeline neither closes nor persists
it; the stored-request processor removes the temporary upload in a `finally`
path.

`PredictionMode.DETECTION` runs DICOM canonicalization and FocalNet-DINO only.
`PredictionMode.FULL` runs detector then MMBCD with the detector's exact eight
classifier ROIs. The packaged full-serving path always requires non-blank
clinical history; this does not change the label-free `Indication:` prompt
contract.

## Result contract

`PredictionResult` is immutable and contains only host-side, serialization-safe
values:

- whitelisted DICOM facts and the source SHA-256;
- original/crop/canonical geometry and scale factors;
- raw detector logits, sigmoid scores, and normalized center-format boxes with
  their original tensor shapes;
- top candidates, post-NMS candidates, and exactly eight classifier ROIs with
  normalized, canonical-pixel, and original-pixel coordinates;
- raw MMBCD class indices, logits, probabilities, predicted index, ROI
  attention inspection values, and prediction hashes in full mode;
- exact detector, classifier, tokenizer, revision, precision, offline-load, and
  strict-load provenance where applicable;
- decoder, adapter, runtime load/inference/switch, memory, and end-to-end timing
  records; and
- structured DICOM, detector, and classifier warnings plus the research-use
  disclaimer.

No class name or decision threshold is exposed because neither is verified by
the supplied artifact metadata. Clinical history and the formatted prompt are
also excluded from the result. `prediction_to_dict()` converts the result to
plain JSON-compatible values without retaining DICOM pixels, model inputs,
PyTorch tensors, or NumPy arrays.

Decoder failures, artifact/runtime failures, and adapter failures preserve
their existing stable typed errors. Pipeline-owned validation and adapter
contract failures use `PredictionInputError` and `PredictionContractError`.

## Validation

The CPU suite exercises both modes through decoder/runtime contract adapters,
JSON serialization, typed failures, success/failure cleanup, and the actual
accelerator lifecycle state machine:

```powershell
$env:PYTHONPATH = "src"
uv run python -m unittest tests.test_prediction_pipeline -v
uv run python -m unittest discover -s tests -v
```

The NVIDIA L4 gate uses the real public DICOM, external weights, pinned source
trees, offline tokenizer, production adapters, and repeated full cycles:

```bash
export PYTHONPATH="$PWD/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
uv run python scripts/l4_validation/17_validate_prediction_pipeline.py --cycles 2
```

The required terminal marker is `PREDICTION PIPELINE L4 PASSED`. This validates
reproducible execution and lifecycle behavior on one public fixture, not model
accuracy, calibration, class semantics, or clinical fitness.

The two-cycle gate passed on an NVIDIA L4 on 2026-08-08. Both detector cycles
reproduced prediction SHA-256
`4cdd09d986702e8839acff8d7517a63f263ca2a01b0607d78d6b2086c886a9a5`,
both classifier cycles reproduced
`43ec1c4593c0549510098ea082ea7092c7fd5631c95d8b912ecf31633185899b`,
and the logits were identical. The committed evidence is
[`docs/validation/prediction-pipeline-l4-20260808.json`](validation/prediction-pipeline-l4-20260808.json).
