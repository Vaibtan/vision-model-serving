# Prediction pipeline

`PredictionPipeline` is the repository-owned orchestration boundary for one
DICOM mammogram. Its public interface is intentionally narrow:

```python
result = pipeline.infer(case, mode)
```

The module decodes the caller-owned stream, executes the detector through the
accelerator lifecycle runtime, and optionally executes MMBCD. The deployed
runtime reuses only the current model and unloads it before a cross-model
switch. The persistent executor and tests consume the same interface; neither
reimplements model ordering, ROI selection, or result shaping.

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
values. Its internal/storage contract includes:

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
- decoder, adapter, runtime load/inference/switch, memory, and executor-pipeline
  timing records; and
- structured DICOM, detector, and classifier warnings plus the research-use
  disclaimer.

No class name or decision threshold is exposed because neither is verified by
the supplied artifact metadata. Clinical history and the formatted prompt are
also excluded from the result. `prediction_to_dict()` converts the complete
internal result to
plain JSON-compatible values without retaining DICOM pixels, model inputs,
PyTorch tensors, or NumPy arrays. HTTP uses `prediction_to_public_dict()`, which
omits the full raw detector logits/scores/box tensors while retaining the
bounded candidates, eight ROIs, hashes, provenance, and all user-relevant
outputs. The request echo includes the detector display threshold used for the
submission. Timings call the executor-only aggregate `pipeline_ms`; it excludes
upload, queueing, IPC, persistence, network transfer, and polling.

The source SHA-256 is stable and correlatable, and predictions are sensitive
derived health data. Excluding direct DICOM identifiers does not certify the
serialized result as de-identified.

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

Real GPU validation runs only through the packaged executor topology described
in [the reproduction guide](reproduction.md#4-build-and-run-the-packaged-l4-smoke-test).
That gate uses the public DICOM, external weights, pinned source trees, offline
tokenizer, standard RQ worker, and the private executor composition root. The
2026-08-10 exact-revision destructive-restart evidence reproduced detector SHA-256
`4cdd09d986702e8839acff8d7517a63f263ca2a01b0607d78d6b2086c886a9a5` and
served-request classifier SHA-256
`f994ccfad2e1894f95b487cf1068b5c0038b4bb12c7d49f5e0dc396afc83f1a3`.
See
[`compose-restart-l4-20260810.json`](validation/compose-restart-l4-20260810.json)
and the [resolution record](validation/spec-resolution-l4-20260810.md). This
validates reproducible execution and lifecycle behavior on one public fixture
for the embedded revision, not current HEAD, model accuracy, calibration, class
semantics, or clinical fitness.
