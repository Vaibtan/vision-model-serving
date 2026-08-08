# Prediction pipeline

`PredictionPipeline` is the repository-owned orchestration boundary for one
DICOM mammogram. Its public interface is intentionally narrow:

```python
result = pipeline.infer(case, mode)
```

The module decodes the caller-owned stream, executes the detector through the
single-residency runtime, and optionally switches to MMBCD. HTTP workers, the
reference CLI, and tests consume the same interface; none reimplement model
ordering, ROI selection, or result shaping.

## Input and mode contract

`CaseInput` contains a readable binary DICOM stream and optional clinical
history. The caller owns the stream. The pipeline neither closes nor persists
it; the CLI uses a `with` block and the later request gateway must delete its
temporary upload in a `finally` path.

`PredictionMode.DETECTION` runs DICOM canonicalization and FocalNet-DINO only.
`PredictionMode.FULL` runs detector then MMBCD with the detector's exact eight
classifier ROIs. Full mode requires non-blank clinical history by default. The
composition root may explicitly disable that provisional requirement for the
archived empty-history golden fixture; this does not change the label-free
`Indication:` prompt contract.

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

## Local reference CLI

The CLI verifies the external checkpoints and local source/tokenizer trees,
constructs the real CUDA adapters, and calls `PredictionPipeline.infer`:

```bash
export PYTHONPATH="$PWD/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m vision_model_serving.pipeline /path/to/case.dcm \
  --mode full \
  --clinical-history-file /path/to/history.txt \
  --artifact-root /models \
  --tokenizer-root /assets/roberta-base-tokenizer-e2da8e2f811d1448a5b465c236feacd80ffbac7b \
  --focalnet-root /src/FocalNet-DINO \
  --mmbcd-root /src/MMBCD \
  --dino-root /src/dino \
  --output prediction.json
```

Supplying history via a file avoids placing clinical text in the process list
or shell history. The output is intentional caller-owned data; no preview or
temporary request file is created by this command.

## Validation

The CPU suite exercises both modes through decoder/runtime fakes, JSON
serialization, typed failures, success/failure cleanup, and the actual
single-residency state machine:

```powershell
$env:PYTHONPATH = "src"
python -m unittest tests.test_prediction_pipeline -v
python -m unittest discover -s tests -v
```

The remaining NVIDIA L4 gate uses the real public DICOM, external weights,
pinned source trees, offline tokenizer, production adapters, and repeated full
cycles. It must reproduce both archived hashes before T09 is complete:

```bash
export PYTHONPATH="$PWD/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python scripts/l4_validation/17_validate_prediction_pipeline.py --cycles 2
```

The required terminal marker is `PREDICTION PIPELINE L4 PASSED`. This validates
reproducible execution and lifecycle behavior on one public fixture, not model
accuracy, calibration, class semantics, or clinical fitness.
