# TensorRT failure analysis

## Purpose and decision

This note records why the 2026-08-10 NVIDIA L4 TensorRT feasibility run did
not produce a deployable backend. It is intended to support the final
assignment report without conflating a successful fixed-shape diagnostic with
production input coverage.

The measured decision was **STOP**. Eager PyTorch FP32 with TF32 disabled
remains the only selected serving backend. No TensorRT engine, backend switch,
or eager fallback was added to the executor.

The validation used one NVIDIA L4, one checksum-pinned public DICOM, PyTorch
2.8.0 with CUDA 12.8, Torch-TensorRT 2.8.0, and TensorRT 10.12.0.36. It is
feasibility evidence, not clinical, multi-input-corpus, or cross-hardware
evidence.

## What the experiment implemented

The isolated validation lane required:

1. strict `torch.export` graph capture;
2. FP32 compilation with TF32 disabled;
3. `require_full_compilation=True` and zero PyTorch partitions;
4. raw TensorRT plan serialization on the target L4;
5. plan execution in a verifier that imports TensorRT, CUDA Python, and NumPy,
   but not PyTorch, Torch-TensorRT, Transformers, or model source;
6. numerical and discrete output parity against eager FP32;
7. at least 15% lower warm p50 for the isolated model forward pass; and
8. complete coverage of the input shapes accepted by the service.

Failure of any mandatory gate stopped promotion. Unsupported operations,
failed dynamic shapes, or plan-load failures were not allowed to select eager
PyTorch automatically.

## MMBCD classifier

### Fixed-width diagnostic result

The public request tokenized to width 5. For the exact input contract

```text
roi_crops:       float32 [1, 8, 3, 224, 224]
input_ids:       int64   [1, 5]
attention_mask:  int64   [1, 5]
```

the MMBCD classifier passed the graph, runtime, parity, and isolated
performance gates:

- strict export succeeded;
- the dry run reported 802 of 802 supported operators in one TensorRT engine;
- the engine contained zero PyTorch partitions;
- engine construction took 19,380.69 ms;
- the TensorRT-only verifier deserialized and executed the plan without
  importing a forbidden framework;
- the predicted class matched eager FP32;
- maximum absolute differences were `5.96e-6` for fused embeddings,
  `2.86e-6` for logits, and `1.49e-7` for ROI attention, all below the
  `1e-4` acceptance tolerance; and
- warm classifier-forward p50 improved from 94.96 ms to 68.57 ms, a 27.79%
  improvement.

The different exact output SHA-256 values do not represent a parity failure:
TensorRT selected numerically different kernels, while the measured numeric
differences stayed within the declared tolerance and the predicted class was
unchanged.

This timing covers only the classifier forward pass for the frozen input. It
does not measure tokenization, detector execution, model switching, Django/RQ,
or end-to-end request latency.

### Production-profile failure

The service preserves unpadded clinical-history tokens and accepts tied
`input_ids` and `attention_mask` widths from 2 through 90. Width 5 is only the
optimization point for the public fixture, not the production contract.

Strict export of a symbolic token-width 2-through-90 graph failed with a
PyTorch `UserError`. The exporter generated an attention/SDPA stride and
alignment guard that it could not prove for every width in the declared range:

```text
Constraints violated (token_width)
Not all values of token_width in the specified range satisfy the generated guard
```

The failure occurred during `torch.export`, before a dynamic TensorRT engine
could be built. It was not an incorrect TensorRT classification.

The fixed-width plan was not promotable because it would support only one
clinical-history length. The rejected alternatives were:

- rejecting otherwise valid requests whose token width is not 5;
- falling back to eager PyTorch for unsupported widths; or
- padding every request to width 90 without checked-in corpus parity evidence.

The first option violates the input contract, the second violates the
no-fallback policy, and the third remains unproven. Consequently, the static
plan was retained only long enough to record its hash and complete runtime,
parity, and performance measurements, then it was deleted.

## FocalNet-DINO detector

### Exact measured failure

The detector did not reach TensorRT engine construction. Strict `torch.export`
failed in the upstream FocalNet-DINO `NestedTensor` constructor at:

```python
if mask == "auto":
```

During graph capture, `mask` was a traced tensor-like value. Comparing it with
the string `"auto"` produced a non-tensor `NotImplemented` result that Dynamo
could not represent in the FX graph. The captured error reported:

```text
torch.* op returned non-Tensor
example_value type: NotImplementedType
target: __eq__
```

Therefore the measured detector gates ended as follows:

| Gate | Result |
| --- | --- |
| Strict export | Failed |
| Torch-TensorRT dry run | Not reached |
| Full-coverage engine build | Not reached |
| TensorRT-only execution | Not reached |
| Output parity | Not measured |
| Performance | Not measured |

### Expected downstream custom-operator blocker

The measured export stopped before it reached
`MultiScaleDeformableAttention`. The deformable-attention boundary must
therefore be described as a separately established architectural blocker, not
as the exact exception observed in this run.

Erratum: the immutable schema-v1 machine-readable report records
`failure_code: strict_coverage_failed:Unsupported` and
`plugin_required: true`. Those values came from the old builder's
stage-agnostic fallback and overstate what this run measured. The preserved
coverage transcript and `strict_export: false` establish that coverage and
plugin analysis were not reached. Future schema-v2 reports record an explicit
`plugin_requirement_status: not_reached` with a null `plugin_required` value.
This correction does not rewrite the historical result or claim a new L4 run.

FocalNet-DINO invokes deformable attention through a custom PyBind/CUDA
extension. The pinned implementation has no exportable `torch.library` schema
and fake/meta implementation, no Torch-TensorRT converter, and no TensorRT
`IPluginV3` implementation. Correcting the earlier `NestedTensor` capture issue
would not by itself make the detector deployable.

A complete detector TensorRT path would next require either:

1. an inference-only export wrapper followed by an exact `IPluginV3` and
   Torch-TensorRT converter for deformable attention; or
2. a separately validated full-graph decomposition that matches the eager
   checkpoint numerically and satisfies latency and accelerator-residency
   gates.

Either approach would require operator-level shape, type, workspace, CUDA
stream, serialization, ABI, boundary-shape, and numeric-parity tests before
full detector validation. A graph break or PyTorch partition would not satisfy
the no-fallback contract.

## Why the overall result was STOP

The decision levels were defined as:

- **GO:** both models pass complete strict production coverage and every
  runtime, parity, performance, packaging, and residency gate;
- **PARTIAL:** MMBCD passes its complete token-width 2-through-90 production
  profile, but detector TensorRT coverage remains blocked; and
- **STOP:** the classifier production profile fails, strict graph coverage
  fails, or another mandatory gate fails.

MMBCD proved that a static-width TensorRT engine could be correct and faster,
but it did not cover the service's accepted token widths. FocalNet-DINO failed
strict capture before compilation and retained an additional unimplemented
custom-operator boundary. The evidence therefore did not justify even a
PARTIAL production claim.

## Report-ready summary

> TensorRT was evaluated through a strict, no-fallback NVIDIA L4 validation
> lane rather than enabled as a production runtime option. The MMBCD
> classifier successfully compiled as one FP32 TensorRT engine for the frozen
> width-5 fixture, with zero PyTorch partitions, TensorRT-only execution,
> output parity, and a 27.79% improvement in isolated warm p50. It was not
> promoted because the service accepts unpadded token widths from 2 through
> 90, and strict export of that bounded dynamic profile failed an exporter
> shape guard. The static plan was therefore deleted. FocalNet-DINO failed
> strict export earlier in its upstream `NestedTensor` mask handling, before
> TensorRT coverage analysis or engine construction. Even after correcting
> that capture boundary, its custom `MultiScaleDeformableAttention` CUDA
> operation still requires a validated TensorRT plugin or full-graph
> decomposition. Production consequently remains eager FP32 with TF32
> disabled, and no TensorRT-to-PyTorch fallback is enabled.

## Evidence index

- [Machine-readable decision](tensorrt-spike.json)
- [Short generated summary](tensorrt-spike.md)
- [Classifier static dry-run coverage](mmbcd-tensorrt-dryrun.txt)
- [Classifier dynamic-export failure](mmbcd-dynamic-export.txt)
- [TensorRT-only runtime record](mmbcd-tensorrt-runtime.json)
- [Classifier parity measurements](mmbcd-tensorrt-parity.json)
- [Classifier performance measurements](mmbcd-tensorrt-performance.json)
- [Detector strict-capture failure](focalnet-dino-tensorrt-coverage.txt)
- [Implementation and compatibility rationale](../../research/tensorrt-serving-feasibility-20260810.md)
