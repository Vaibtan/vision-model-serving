# FocalNet-DINO detector adapter

`FocalNetDinoAdapter` is the repository-owned boundary between a canonical
DICOM mammogram and MMBCD's fixed eight-proposal ROI contract. It owns model
construction, checkpoint loading, device tensors, inference, and proposal
selection. Neither the CUDA model nor a PyTorch tensor is returned.

```python
from vision_model_serving.detector import FocalNetDinoAdapter

artifact = registry.resolve("focalnet-dino-detector")
adapter = FocalNetDinoAdapter.from_local_source(
    artifact,
    repository_root="/opt/FocalNet-DINO",
    project_root="/app/vision-model-serving",
)
result = adapter.predict(canonical_mammogram)
```

Before construction, production startup should require the artifact registry's
aggregate readiness gate. `from_local_source` additionally requires the exact
pinned FocalNet-DINO revision, all checked-in compatibility patches in their
applied state, the checksum-pinned `config_cfg.py`, and modules that resolve
inside that source checkout. It performs no clone, download, hub, or URL
operation.

## Detector input

The preprocessor reproduces the official evaluation transform:

1. Convert the canonical unsigned 8-bit image to RGB.
2. Preserve aspect ratio while resizing the shorter edge to 800 pixels.
3. Cap the longer edge at 1333 pixels using the upstream integer-size rule.
4. Convert to a contiguous CHW float32 array in the range zero to one.
5. Normalize with ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation
   `[0.229, 0.224, 0.225]`.

The default 1024 by 1024 canonical mammogram becomes a 3 by 800 by 800 model
input. The NumPy preprocessing output is immutable. The runtime makes one
private writable host copy before `torch.from_numpy` and device transfer.

## Verified model load and execution

The runtime accepts only the manifest-owned `focalnet-dino-detector` artifact
with strict-load evidence. `VerifiedArtifact.open_checkpoint()` rechecks the
same file identity and SHA-256 immediately before loading. The adapter then:

- establishes the validated deterministic FP32 policy before model
  construction: seed zero, deterministic algorithms, cuDNN benchmarking off,
  TF32 off, and highest float32 matmul precision;
- requires `CUBLAS_WORKSPACE_CONFIG=:4096:8` to be set before CUDA is
  initialized and rejects a conflicting or late configuration;
- loads on CPU with `weights_only=True` and the one required safe global;
- requires the wrapped `model` state-dict root;
- calls `load_state_dict(..., strict=True)` and rejects any key difference;
- moves the private model to the requested device as float32;
- calls `eval()` once; and
- runs each forward pass under `torch.inference_mode()` with CUDA
  synchronization around the timed region.

Construction and execution failures return stable typed errors without paths,
state-key names, checkpoint exception text, or model objects.

## Proposal contract

Raw outputs must match the verified shapes exactly: logits `[1, 900, 1]` and
normalized center-format boxes `[1, 900, 4]`, with only finite values. The
adapter retains immutable float32 logits, sigmoid scores, and center-format
boxes. Their prediction SHA-256 is calculated from logits followed by boxes as
contiguous little-endian float32 bytes, matching the L4 evidence harness.

Proposal selection is fixed:

1. Sigmoid and flatten query/class scores.
2. Sort descending with a stable flattened-index tie break and select at most
   300 candidates.
3. Convert center-format boxes to XYXY, clamp to `[0, 1]`, and reject boxes
   that become degenerate.
4. Apply greedy NMS in confidence order. Suppression occurs only when IoU is
   strictly greater than `0.1`; equality is retained.
5. Keep the first eight survivors for MMBCD.
6. If one to seven survive, duplicate them round-robin in confidence order and
   mark each duplicate. If none survive, fail with
   `detector_no_valid_proposals`.

Each proposal contains its raw logit, sigmoid score, normalized CXCYWH and
XYXY boxes, canonical-pixel XYXY box, and original-DICOM-pixel XYXY box mapped
through `GeometryLedger`. Structured warnings report rejected degenerates and
ROI padding. Display-score filtering is a read-only view of post-NMS proposals
and cannot change `classifier_rois`.

Detector scores are proposal-ranking values only. They are not benign or
malignant labels, calibrated probabilities, or an authorized medical display
threshold.

## Validation boundary and missing local inputs

The immutable Lightning evidence records raw detector prediction SHA-256
`4cdd09d986702e8839acff8d7517a63f263ca2a01b0607d78d6b2086c886a9a5` on
an NVIDIA L4. The checkpoint is intentionally outside this checkout. The
sibling read-only artifact directory contains `focalnet-dino-finetuned.pth` at
exactly 2,731,092,364 bytes, and its SHA-256 was rechecked as
`67a7b0cd787a3aaba199cf1ff82ed2934c33ffe37544473379d7a837ab1637b4`.
This workstation still cannot rerun that forward pass because it lacks:

- a FocalNet-DINO checkout at revision
  `23901e021dc6ec8f66bad47983f45a25574452cc` with all repository patches
  applied and its native CUDA operator built; and
- the pinned NVIDIA L4 / PyTorch 2.8.0+cu128 environment.

The evidence archive intentionally does not contain the multi-gigabyte model
checkpoint or source checkout. Do not treat CPU unit tests or the archived hash
as a fresh detector run. After the external checkpoint, source checkout, and
L4 runtime are available together, the explicit success gate remains
`REAL DICOM DETECTOR INFERENCE PASSED` with an exact raw prediction-hash match.

That gate passed on an NVIDIA L4 on 2026-08-08 as part of two complete
detector-to-classifier residency cycles. The committed runtime evidence is
`docs/validation/single-residency-l4-20260808.json`. This proves serving-path
execution and exact output parity for the one checksum-pinned public fixture;
it is not medical-performance or clinical validation.

Run the locally available coverage with:

```powershell
$env:PYTHONPATH = "src"
uv run python -m unittest tests.test_detector_adapter -v
uv run python -m unittest discover -s tests -v
```
