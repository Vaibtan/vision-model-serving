# MMBCD classifier adapter

`MmbcdClassifierAdapter` is the repository-owned boundary between a canonical
mammogram plus exactly eight detector proposals and the pinned multimodal
classifier. It owns crop construction, label-free prompt formatting, offline
tokenization, model loading, FP32 inference, and result interpretation. It
never returns a PyTorch model or device tensor.

```python
from vision_model_serving.classifier import MmbcdClassifierAdapter

artifact = registry.resolve("mmbcd-classifier")
classifier = MmbcdClassifierAdapter.from_local_assets(
    artifact,
    tokenizer_root="/models/roberta-base-tokenizer-e2da8e2f",
    dino_root="/opt/dino",
    mmbcd_root="/opt/MMBCD",
    project_root="/app/vision-model-serving",
)
result = classifier.predict(
    canonical_mammogram,
    detector_result.proposals.classifier_rois,
    clinical_history,
)
```

Production startup should first require the artifact registry's aggregate
readiness gate. `from_local_assets` also verifies the DINO and MMBCD checkout
heads against `config/l4-fp32-environment.json`. It performs no clone, hub,
URL, or model-download operation.

## Input contract

The adapter requires exactly eight proposals in detector order. Each proposal's
canonical-pixel XYXY box must be finite, non-degenerate, and inside the 1024 by
1024 canonical image. For every proposal, the adapter:

1. crops the canonical unsigned 8-bit grayscale image using unclipped
   integer-truncated box corners; pixels outside the frame are zero-padded by
   the crop (the archived MMBCD reference behavior), a border overhang adds a
   `roi_extends_beyond_canonical_image_zero_padded` warning, and a sub-pixel
   box is expanded to one pixel with a `roi_expanded_to_minimum_extent`
   warning instead of failing the case;
2. converts it to RGB;
3. resizes directly to 224 by 224 with Pillow bilinear interpolation;
4. converts it to contiguous CHW float32 in the range zero to one; and
5. applies ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation
   `[0.229, 0.224, 0.225]`.

The resulting immutable tensor shape is `[1, 8, 3, 224, 224]`. The adapter does
not reorder, filter, or pad proposals; those decisions belong to the detector
adapter's fixed top-eight contract.

Clinical history is whitespace-normalized and formatted as
`Indication: {history}`, or exactly `Indication:` when empty. No label,
pathology, target class, or other ground-truth field is accepted by the public
prediction method. The pinned local RoBERTa snapshot tokenizes with padding,
truncation, and a maximum length of 90.

The HTTP layer accepts up to 4,000 characters while the tokenizer keeps at
most 90 tokens. The result now reports `clinical_text_truncated` in the
classifier input summary and adds a `clinical_text_truncated` warning whenever
tail text was discarded, so callers can detect the loss.

## Offline model and tokenizer loading

The tokenizer loader revalidates the manifest and the exact size and SHA-256 of
all five local files before calling `RobertaTokenizer.from_pretrained` with
`local_files_only=True` and `trust_remote_code=False`. Both Hugging Face offline
environment flags are set before the import.

The classifier runtime reconstructs the validated architecture from the pinned
local DINO `vision_transformer.py` and an in-memory RoBERTa configuration. It
temporarily binds the pinned sibling `utils.py` while executing that module,
then restores any prior generic `utils` module binding. This prevents a
detector or application import from changing the classifier architecture. It
uses no pretrained-model loader. The architecture is fixed to:

- DINO ViT-small, patch size 8, producing 384 image features per ROI;
- a 384-to-256 image projection and max pool across eight ROIs;
- a 12-layer, 768-hidden RoBERTa encoder and 768-to-256 text projection;
- one-head text-to-ROI multihead attention; and
- a 768-to-2 classifier over attention, text, and max-pooled image features.

The manifest-owned raw state dictionary is reopened through
`VerifiedArtifact.open_checkpoint()`, loaded on CPU with `weights_only=True`,
and normalized only by removing one leading `module.` prefix. Prefix collisions
and unequal released alias tensors fail closed. `load_state_dict(...,
strict=True)` must report no key differences before the model is moved to the
device and placed in evaluation mode.

The runtime sets the validated deterministic FP32 controls: seed zero,
deterministic algorithms, cuDNN benchmarking off, TF32 off, highest float32
matmul precision, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. A conflicting CUBLAS
configuration fails startup. Each forward pass uses `torch.inference_mode()`
and CUDA synchronization around the measured region.

## Result contract

Results expose only raw class indices `(0, 1)`, two logits, numerically stable
softmax probabilities, and the argmax index. They do not invent class names or
a medical decision threshold. The fused embedding is immutable and shaped
`[1, 768]`.

ROI attention must be finite, non-negative, shaped `[1, 1, 8]`, and sum to one.
It is labeled `model_inspection_not_causal_or_clinical_evidence`. Structured
warnings repeat that attention is inspection data and that class semantics and
the decision threshold are unverified.

For auditability, every result includes:

- artifact and tokenizer identities;
- strict-load/offline/FP32 provenance;
- crop, input-ID, attention-mask, and prediction SHA-256 values; and
- model-load, crop, tokenization, inference, and result-processing timings.

The prediction SHA-256 is the contiguous little-endian float32 logits followed
by the contiguous little-endian float32 fused embedding. This reproduces the
archived L4 prediction hash exactly.

## Validation boundary and missing local inputs

Local tests reproduce the archived crop tensor SHA-256
`89cda9694e3696f63eb70706a6ae4cc2dd16e9be214f27ba6f106479eac90155`
from the public DICOM and detector boxes. They also read the real archived L4
output bundle and reproduce prediction SHA-256
`43ec1c4593c0549510098ea082ea7092c7fd5631c95d8b912ecf31633185899b`
from its logits and fused embedding.

The checkpoint is intentionally outside this checkout. The manifest requires
`mmbcd_best.pt` at 587,689,457 bytes with SHA-256
`2264351216f9fb4945af35e300459ff4ce2e7f5445519348024f3bf1eec721a4`.
A fresh classifier forward pass requires all of these external inputs together:

- a DINO checkout at
  `7c446df5b9f45747937fb0d72314eb9f7b66930a`;
- an MMBCD checkout at
  `14ac5e099c79253b01e0885d2ebefa6f86cfd8f0`; and
- the pinned NVIDIA L4, PyTorch 2.8.0+cu128, and Transformers 5.14.1 runtime.

The evidence archive intentionally contains result bundles rather than the
unlicensed checkpoint or source trees. Unit tests and archived-byte hash
reproduction are not a fresh model run. Once the external checkpoint, pinned
sources, and L4 environment are mounted together, the success gate remains
`REAL DICOM MMBCD INFERENCE PASSED` with an exact prediction-hash match.

That gate passed on an NVIDIA L4 in archived revision-bound records. The latest
corrected topology is indexed by the
[`2026-08-10 resolution record`](validation/spec-resolution-l4-20260810.md).
This proves serving-path execution and exact output parity for one
checksum-pinned public fixture at the embedded revisions; it is not current-HEAD
or medical-performance/clinical validation.

Run locally available coverage with:

```powershell
$env:PYTHONPATH = "src"
uv run python -m unittest tests.test_classifier_adapter -v
uv run python -m unittest discover -s tests -v
```
