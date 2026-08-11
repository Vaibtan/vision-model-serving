# Model artifact inventory

The source of truth is
[`config/model-artifacts.json`](../config/model-artifacts.json). It records
the exact evaluator-supplied checkpoints, repository assets, tokenizer files,
source revisions, and the repository-defined detector-to-classifier handoff
exercised deterministically on an NVIDIA L4 on 2026-08-07.

This inventory is evidence about artifact identity and serving compatibility.
It is not evidence of medical accuracy, calibration, or clinical fitness.
The released source does not include the author proposal generator or an
inference-realistic golden text/proposal/logit bundle, so the handoff and
label-free prompt remain provisional repository contracts rather than proven
equivalence to the authors' complete inference path.

## Verification

The verifier uses only the Python standard library and reads manifest metadata
only. It does not locate, hash, or deserialize the named checkpoint files.

PowerShell:

```powershell
$env:PYTHONPATH = "src"
uv run python -m vision_model_serving.artifacts config/model-artifacts.json
uv run python -m unittest discover -s tests -v
```

POSIX shells:

```bash
PYTHONPATH=src uv run python -m vision_model_serving.artifacts config/model-artifacts.json
PYTHONPATH=src uv run python -m unittest discover -s tests -v
```

Pass `--json` to emit a machine-readable inventory summary. Schema tests use
tiny temporary fixtures and require neither model weights nor network access.

## Two-model contract

The service pipeline is fixed to:

1. `focalnet-dino-detector`: emit the ordered top 300 normalized `cx cy width
   height confidence` proposals without internal NMS.
2. Apply MMBCD-compatible NMS using strict `IoU > 0.1` suppression.
3. Retain eight proposals, duplicating existing proposals when fewer than eight
   remain and failing closed when none remain.
4. `mmbcd-classifier`: consume those eight ROIs plus tokenized clinical text.

The manifest validator rejects another stage order or an altered proposal
contract.

## Checkpoint inventory

| Role | File | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| Detector | `focalnet-dino-finetuned.pth` | 2,731,092,364 | `67a7b0cd787a3aaba199cf1ff82ed2934c33ffe37544473379d7a837ab1637b4` |
| Classifier | `mmbcd_best.pt` | 587,689,457 | `2264351216f9fb4945af35e300459ff4ce2e7f5445519348024f3bf1eec721a4` |

The detector checkpoint is a wrapped state dictionary under `model`. Its 839
tensors contain 228,987,809 elements and include the serving backbone, so the
training-only backbone preload is not required. Strict loading succeeded with
the pinned FocalNet-DINO source and produced `pred_logits [1, 900, 1]` and
`pred_boxes [1, 900, 4]` in the archived FP32 run.

The classifier checkpoint is a raw, `module.`-prefixed state dictionary. After
only that known prefix is stripped, all 375 keys strict-load against the pinned
MMBCD and DINO sources. The audit found 373 float32 tensors and two int64
tensors across the required image encoder, image projection, text encoder, text
projection, cross-attention, and classifier groups.

The derived detector inference-only checkpoint is not retained in Git, but its
archived identity is recorded in the manifest so the L4 result remains
traceable.

## Pinned source and tokenizer revisions

| Component | Revision |
| --- | --- |
| FocalNet-DINO | `23901e021dc6ec8f66bad47983f45a25574452cc` |
| MMBCD | `14ac5e099c79253b01e0885d2ebefa6f86cfd8f0` |
| DINO image encoder | `7c446df5b9f45747937fb0d72314eb9f7b66930a` |
| FacebookAI/roberta-base tokenizer | `e2da8e2f811d1448a5b465c236feacd80ffbac7b` |

The tokenizer record pins the filenames, byte sizes, and SHA-256 values for
`config.json`, `merges.txt`, `tokenizer.json`, `tokenizer_config.json`, and
`vocab.json`. Runtime use is local-only; fetching a mutable `main` revision is
not part of the serving contract.

The L4-validated detector config had no final newline and hashes to
`e40f26be40a5aaccb149ee09534ac61975c76df150bb93ef475c0a6b29af1f24`.
The repository copy adds one trailing LF and hashes to
`6bf3f0bee489209195879b71fbcff6d5f0368a8488e8cf533366e325fbc403db`.
The manifest records both identities and the exact equivalence rather than
hiding the byte-level difference.

## Trust, licensing, and semantic boundaries

PyTorch pickle-based checkpoints can execute code when loaded through unsafe
deserialization paths. Only the evaluator-supplied files with the pinned hashes
above are authorized for local loading. The public API must never accept a
user-uploaded model artifact. Runtime loading must start on CPU, use restricted
`weights_only=True` deserialization where the checkpoint format permits it,
normalize only explicitly tested key prefixes, and fail closed on strict-load
differences.

The following facts remain deliberately unresolved:

- redistribution of either evaluator-supplied checkpoint is not authorized;
- the pinned MMBCD repository and checkpoint have no verified license grant;
- detector proposal scores have no validated medical class name or threshold;
- upstream MMBCD code treats class index 1 as cancer, but the checkpoint does
  not independently prove that mapping; and
- no calibrated medical decision threshold was supplied.

For both models, the manifest therefore requires `semantics.status` to remain
`unverified`, class names and decision thresholds to remain `null`, and medical
validation to remain `false`. The validator rejects silent promotion of any of
those fields.

## Runtime registry

`ArtifactRegistry` is the fail-closed runtime boundary for these records. It
accepts only the two manifest-owned model IDs, hashes and inspects checkpoints
from read-only CPU streams, verifies tokenizer and repository assets, checks the
exact runtime lane and native CUDA operator, and aggregates those checks into
startup readiness. Public reports contain stable error codes but no local paths
or full hashes.

```python
from vision_model_serving.artifacts import ArtifactRegistry

registry = ArtifactRegistry(
    "config/model-artifacts.json",
    artifact_root="/models",
    tokenizer_root="/tokenizer/roberta-base",
    repository_root=".",
)
report = registry.verify_all()
if not report.ready:
    raise RuntimeError(report.as_public_dict())

detector = registry.resolve("focalnet-dino-detector")
with detector.open_checkpoint() as stream:
    # The T08 adapter must deserialize from this stream and call
    # module.load_state_dict(..., strict=True) before retaining the model.
    pass
```

`open_checkpoint()` reopens the verified artifact without revealing its path,
rehashes it before yielding the stream, and rejects identity changes before or
during model loading. Runtime adapters remain responsible for constructing the
pinned architecture and performing the final strict state-dict load. The
registry verifies the exact archived strict-load evidence and live checkpoint
structure; it does not substitute a shape audit for adapter-level strict load.

The implementation performs no network access. Tokenizers must be opened with
`local_files_only=True` by the consuming adapter, and model/tokenizer roots can
be mounted read-only.

## Storage boundary

Weights and the L4 evidence archive stay outside Git and outside public image
layers. Their identities are represented by filenames, sizes, hashes, and the
checked-in reference record. Runtime discovery, live checksums, restricted
checkpoint inspection, and readiness are implemented by `ArtifactRegistry`.
Model construction and GPU lifecycle remain outside this metadata module.
