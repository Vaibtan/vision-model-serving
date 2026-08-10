# TensorRT Serving Feasibility for the L4 Pipeline

**Status:** research and fail-closed implementation complete; no L4 TensorRT engine has been built or run
**Date:** 2026-08-10
**Scope:** the pinned PyTorch 2.8/CUDA 12.8 FocalNet-DINO and MMBCD implementations in this repository, strict no-fallback compilation, target-L4 validation, and promotion criteria

## Executive decision

TensorRT is feasible for this project, but it is not yet an evidence-backed property of either model.

- **MMBCD is the first implementation target.** Its reconstructed DINO ViT, RoBERTa, pooling, attention, and classifier are composed from ordinary PyTorch/Transformers operations. That makes a full-engine build plausible, not guaranteed. A strict `torch.export` capture and Torch-TensorRT dry run must determine actual converter coverage.
- **FocalNet-DINO is expected to require a TensorRT plugin.** Its `MultiScaleDeformableAttention` is a PyBind C++/CUDA extension, not a registered `torch.library` custom operator, and upstream provides neither a fake/meta implementation nor an ONNX symbolic. Its current wrapper cannot simply execute inside a TensorRT plan. The CUDA kernel must be exposed through a real TensorRT plugin, or an alternative pure-PyTorch decomposition must separately prove full compilation, numerical parity, and adequate performance.
- **The production path must be fail closed.** Default Torch-TensorRT behavior can partition a graph and run unsupported regions in PyTorch. That is useful for exploration but does not satisfy a full-engine claim. Promotion requires strict export, zero unsupported operations, `require_full_compilation=True`, a raw serialized TensorRT plan, and a TensorRT-only runtime smoke test.
- **Initial precision is FP32 with TF32 disabled.** FP16 is a later, independently gated optimization. The current eager FP32 L4 evidence does not validate TensorRT, TensorRT FP32, or FP16.
- **No L4 acceleration claim can be made from this workstation.** The plan must be built and validated with the real checkpoints on an NVIDIA L4 in the pinned environment. Until then, the correct decision is **GO for implementation, STOP for promotion**.

This is intentionally a single TensorRT deployment path, not a chain of optional execution fallbacks. ONNX and Polygraphy are diagnostic and comparison tools unless the ONNX parser route is explicitly selected as the sole production builder.

## 1. Verified baseline and exact compatibility lane

The repository pins PyTorch `2.8.0+cu128`, torchvision `0.23.0+cu128`, Python 3.12, CUDA 12.8, and target compute capability 8.9. NVIDIA L4 is an Ada GPU with compute capability 8.9, so it is inside TensorRT's supported hardware range ([TensorRT support matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html)).

Torch-TensorRT 2.8.0 is the matching integration release. Its release targets PyTorch 2.8 and TensorRT 10.12 and lists CUDA 12.6, 12.8, and 12.9 support ([Torch-TensorRT 2.8 release](https://github.com/pytorch/TensorRT/releases/tag/v2.8.0)). The published package metadata constrains `torch>=2.8,<2.9`, `tensorrt>=10.12,<10.13`, and the corresponding CUDA 12 TensorRT bindings and libraries ([Torch-TensorRT 2.8.0 package metadata](https://pypi.org/project/torch-tensorrt/2.8.0/)). TensorRT 10.12 was tested with ONNX 1.16 and supports CUDA 12.8 Update 1 ([TensorRT 10.12 release notes](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/release-notes-10/10.12.0.html)).

The reproducible acceleration environment should therefore add these exact pins without changing the existing PyTorch/CUDA lane:

```text
torch==2.8.0+cu128                  # already pinned
torchvision==0.23.0+cu128          # already pinned
torch-tensorrt==2.8.0
tensorrt==10.12.0.36
onnx==1.16.0                       # builder/diagnostic dependency only
polygraphy==0.49.24                # tool version shipped on TensorRT release/10.12
cuda-python==12.8.0                # TensorRT-only verifier device buffers/stream
```

Polygraphy 0.49.24 is the version declared by NVIDIA's TensorRT 10.12 source branch ([Polygraphy version in TensorRT 10.12](https://github.com/NVIDIA/TensorRT/blob/release/10.12/tools/Polygraphy/polygraphy/__init__.py)). The service runtime does not need ONNX or Polygraphy when it loads a serialized plan directly.

The environment lock and the generated engine manifest must record the resolved TensorRT Python package, C++ runtime, CUDA runtime, driver, Torch-TensorRT, PyTorch, plugin-library, GPU name, and compute capability versions. `trtexec --version` and the TensorRT Python `__version__` must both report 10.12.x. A newer `trtexec` must not be used to characterize a 10.12 plan.

TensorRT's Python wheel does not provide all C++ headers, samples, and tools; NVIDIA directs users who need those components to the Debian/RPM or tar installation ([TensorRT installation guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html)). The build image should install a matching 10.12 `trtexec` and plugin SDK from an official 10.12 distribution rather than borrow a CLI from another TensorRT release.

### Compatibility decision

| Component | Decision | Reason |
|---|---|---|
| PyTorch/CUDA | Keep `2.8.0+cu128` | Already validated for eager L4 execution and accepted by Torch-TensorRT 2.8 metadata |
| Torch-TensorRT | Pin `2.8.0` | Exact PyTorch 2.8 integration line |
| TensorRT | Pin `10.12.0.36` | Inside Torch-TensorRT's required 10.12 range; an exact published build |
| ONNX | Pin `1.16.0` if used | Version tested by TensorRT 10.12 |
| Polygraphy | Pin `0.49.24` in validation image | Version in NVIDIA's TensorRT 10.12 source branch |
| GPU build target | Build on L4/SM 8.9 | A normal serialized plan is not a portable source artifact |

TensorRT 10.12 documentation is archived because newer TensorRT generations now exist. That is not a reason to mix major versions into this lane: upgrading PyTorch, Torch-TensorRT, TensorRT, CUDA, or the plugin ABI is a separate revalidation project.

## 2. Frontend choice: direct Torch-TensorRT first, ONNX as an independent route

### Recommended production builder

Use the Torch-TensorRT Dynamo/AOT path:

1. Wrap each model so its inputs and outputs are tensors with an explicit serving contract.
2. Capture it with `torch.export.export(..., strict=True)` using the exact fixed-shape sample inputs.
3. Run Torch-TensorRT compilation analysis and persist its dry-run report.
4. Convert the `ExportedProgram` to raw TensorRT engine bytes with full compilation required.
5. Store the `.plan` and a manifest; load it with the TensorRT runtime in the executor.

PyTorch documents `torch.export` as producing an ahead-of-time, normalized full graph and requiring explicit dynamic-shape declarations when shapes are not static ([PyTorch 2.8 export documentation](https://docs.pytorch.org/docs/2.8/export.html)). Torch-TensorRT exposes AOT compilation and `convert_exported_program_to_serialized_trt_engine`, while `require_full_compilation=True` prohibits a hybrid TensorRT/PyTorch result ([Torch-TensorRT 2.8 Dynamo API](https://docs.pytorch.org/TensorRT/v2.8.0/py_api/dynamo.html)).

Do not use `torch.compile` as the production proof. It can graph-break, specialize, or recompile at runtime. A successful call therefore does not establish that every operation executed in TensorRT.

### ONNX route

ONNX remains valuable for three purposes:

- producing a human-inspectable interchange graph;
- comparing ONNX Runtime and TensorRT outputs with Polygraphy;
- representing the detector custom operation as a named ONNX node that maps to a TensorRT plugin.

If this route is selected, first create the strict `ExportedProgram`; then pass that captured program to the PyTorch 2.8 ONNX exporter with `dynamo=True`, `fallback=False`, `report=True`, and `verify=True`. Supplying the already strict program avoids treating the exporter's progressively less strict capture strategies as proof. The PyTorch exporter documents the Dynamo path, verification/report options, custom translations, and legacy fallback control ([PyTorch 2.8 ONNX exporter](https://docs.pytorch.org/docs/2.8/onnx.html), [Dynamo ONNX exporter](https://docs.pytorch.org/docs/2.8/onnx_dynamo.html)).

The ONNX parser either builds a TensorRT network or reports an unsupported node; it does not execute unsupported nodes itself. A custom ONNX node still needs a registered TensorRT plugin whose name, version, and namespace match at parse/deserialization time ([TensorRT ONNX operator support](https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/onnx-opset-guide.html), [TensorRT plugin documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/extending-custom-layers.html)).

Do not ship both direct and ONNX builders as automatic production alternatives. Select one after the feasibility gates, make its manifest authoritative, and fail startup when its engine or plugin contract is invalid.

## 3. Model-specific feasibility

### 3.1 MMBCD: plausible full-engine candidate

The serving reconstruction consists of a DINO ViT image encoder, a RoBERTa text encoder, max pooling over the eight ROI features, multi-head cross-attention, and a final two-class head. Those operations are visible in the upstream MMBCD implementation ([MMBCD model source](https://github.com/adsbansal/MMBCD/blob/14ac5e099c79253b01e0885d2ebefa6f86cfd8f0/code/model.py)) and the pinned DINO ViT implementation ([DINO Vision Transformer source](https://github.com/facebookresearch/dino/blob/7c446df5cc3c7afc758fbf9c23f28f6f4012207d/vision_transformer.py)).

This is a **feasibility signal only**. Converter coverage depends on the exact graph emitted by PyTorch 2.8 and Transformers 5.14.1, including attention decompositions, indexing, masks, reshapes, and output selection. The acceptance evidence is the compiler report, not the fact that the source appears to use standard modules.

The first MMBCD engine contract should be:

```text
roi_crops:       float32 [1, 8, 3, 224, 224]
input_ids:       int64   [1, 2..90] (optimization point 5)
attention_mask:  int64   [1, 2..90] (tied to input_ids)
logits:          float32 [1, 2]
```

The repository already fixes eight crops and 224-by-224 images. Its tokenizer
does not pad to the 90-token limit, so the TensorRT lane must preserve the
observed width. RoBERTa's special tokens make 2 the smallest realizable width;
the backend therefore admits 2 through 90 and rejects every other shape before
enqueue. The eager parity reference must use the identical unpadded tokens.

If strict export or full compilation fails, record the exact unsupported operator and stop. Do not turn on eager fallback. A narrowly scoped converter is acceptable only when it has unit-level shape/type tests and end-to-end parity evidence.

### 3.2 FocalNet-DINO: plugin engineering is the expected critical path

The detector invokes `MultiScaleDeformableAttention` through a custom autograd function. Its pinned source imports a PyBind module and calls `ms_deform_attn_forward` directly ([custom autograd function](https://github.com/FocalNet/FocalNet-DINO/blob/23901e021dc6ec8f66bad47983f45a25574452cc/models/dino/ops/functions/ms_deform_attn_func.py)). The extension registers only the PyBind forward/backward entry points ([extension binding](https://github.com/FocalNet/FocalNet-DINO/blob/23901e021dc6ec8f66bad47983f45a25574452cc/models/dino/ops/src/vision.cpp)); its CUDA implementation is a custom kernel, not a TensorRT layer ([CUDA implementation](https://github.com/FocalNet/FocalNet-DINO/blob/23901e021dc6ec8f66bad47983f45a25574452cc/models/dino/ops/src/cuda/ms_deform_attn_cuda.cu)).

Consequences:

- strict `torch.export` is expected to stop at this opaque call because it has no `torch.library` schema and fake/meta implementation;
- if capture is forced around it, Torch-TensorRT is expected to classify it as unsupported and partition it back to PyTorch by default;
- exporting a custom ONNX node changes representation only; TensorRT still needs an implementation;
- the existing PyBind/ATen wrapper cannot be called as a TensorRT layer because a TensorRT plugin must operate on TensorRT descriptors, device pointers, formats, workspace, and CUDA stream.

This expectation must be confirmed by the actual strict-export exception and dry-run report. Torch-TensorRT explicitly documents that unsupported operations cause PyTorch fallback unless full compilation is required, and recommends dry-run analysis plus a converter or plugin for unsupported operators ([unsupported-operator guidance](https://docs.pytorch.org/TensorRT/user_guide/compilation/unsupported_ops.html), [dry-run report](https://docs.pytorch.org/TensorRT/tutorials/compilation_analysis/dryrun.html)). PyTorch requires a fake implementation for custom operators to participate in compile/export systems ([PyTorch custom C++ operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)).

#### Required plugin work

1. Define an inference-only custom operator schema for multi-scale deformable attention, plus a fake/meta implementation that computes output shape without accessing data.
2. Refactor or reuse the underlying CUDA forward kernel without routing through the Python/ATen extension boundary.
3. Implement a TensorRT `IPluginV3` layer or Torch-TensorRT AOT plugin for the exact input tensors, output tensor, data types, layouts, workspace, serialization fields, and CUDA stream behavior.
4. Register a Torch-TensorRT converter for the custom operator. If ONNX is the selected frontend, emit a stable custom-domain ONNX node and register the same plugin creator before parsing/deserialization.
5. Package the plugin shared library in the build and runtime images; record its SHA-256, ABI, plugin name/version/namespace, TensorRT version, CUDA architecture, and source revision in the engine manifest.
6. Compare the plugin against the eager extension on randomized valid tensors, boundary shapes, and real detector features before full detector parity testing.

NVIDIA's current plugin API is `IPluginV3`; plugins advertise capabilities, data types/formats, output shapes, serialization fields, and enqueue behavior ([TensorRT plugin API and migration guidance](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/plugins-api-migration.html), [TensorRT Python/plugin guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/python-api/pluginGuide.html)). A missing or mismatched plugin must be a build/startup error, not a reason to load the eager model.

The upstream module contains a pure-PyTorch core used for debug/testing, but upstream itself directs normal execution to the CUDA implementation ([deformable-attention module](https://github.com/FocalNet/FocalNet-DINO/blob/23901e021dc6ec8f66bad47983f45a25574452cc/models/dino/ops/modules/ms_deform_attn.py)). Replacing the CUDA operator with that decomposition is not automatically a simpler production solution: it must fully lower through TensorRT, match the real checkpoint numerically, and meet the latency/VRAM gate. Until those three facts are measured, the plugin is the honest design assumption.

## 4. Shape policy

Fix every dimension whose semantics are fixed. Admit a dynamic profile only
where the endpoint already produces a bounded variable dimension and changing
that dimension would change the model result.

### MMBCD

Use exactly batch 1 and eight 224-by-224 crops. Preserve the tokenizer's real
sequence width and admit only a tied `input_ids`/`attention_mask` TensorRT
profile from 2 through 90 tokens, with the pinned public request's width 5 as
the optimization point. Padding that request to width 90 was rejected on L4:
it changed the fused-embedding golden even with the attention mask present.
The exact-version strict exporter also rejected the bounded dynamic profile on
its generated SDPA stride guard. A static width-5 engine is useful only as a
converter/runtime diagnostic and must never be promoted as service coverage.

### Detector

The current preprocessing preserves aspect ratio with a short edge of 800 and maximum edge of 1333, so detector height and width vary by input. The first proof engine should use the exact preprocessed height and width of the frozen golden fixture, recorded in its manifest. That proves only that shape.

A production detector then needs one explicit policy:

1. **Static buckets:** a separately built and validated engine per accepted `(height, width)` bucket, with deterministic routing and no catch-all eager path.
2. **Bounded dynamic profile:** declared minimum/optimum/maximum dimensions after both the exported graph and custom plugin prove correct symbolic-shape handling.
3. **Fixed padded canvas:** one fixed tensor and mask, but only after corpus evidence shows padding does not change proposal scores, ordering, boxes, or top-eight crops beyond accepted tolerances.

Static buckets are the safest first production option. A single `1333 x 1333` canvas must not be adopted merely because 1333 is the configured long-edge cap; it changes memory use and may change model behavior. PyTorch export makes static shapes the default and requires explicit `Dim` constraints for dynamic inputs ([PyTorch 2.8 export semantics](https://docs.pytorch.org/docs/2.8/export.html)). TensorRT optimization profiles likewise define the admitted dynamic-shape range and must be benchmarked at its relevant points ([TensorRT dynamic-shape guidance](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-dynamic-shapes.html)).

## 5. Strict full-engine and no-fallback proof

Passing unit tests or receiving tensors from `torch.compile` is insufficient. Each promoted model needs all of the following evidence.

### Capture and compiler evidence

1. Run the eager wrapper once on frozen real inputs and store input/output tensors and hashes.
2. Capture with `torch.export.export(wrapper, example_args, strict=True)`. Do not suppress a constraint violation or graph break.
3. Persist a Torch-TensorRT dry-run report for the same exported program and input specification.
4. Require 100 percent TensorRT coverage: zero unsupported operators, zero PyTorch partitions, zero forced PyTorch operations, and no ignored build failure.
5. Serialize raw engine bytes using `convert_exported_program_to_serialized_trt_engine` with at least:

```text
require_full_compilation = True
pass_through_build_failures = True
enabled_precisions = {torch.float32}
disable_tf32 = True
version_compatible = False
hardware_compatible = False
```

The exact API and option meanings are defined by the versioned Torch-TensorRT 2.8 Dynamo API ([Torch-TensorRT 2.8 Dynamo API](https://docs.pytorch.org/TensorRT/v2.8.0/py_api/dynamo.html)). `version_compatible` and `hardware_compatible` remain disabled for the first target-specific plan so that compatibility modes cannot trade away target-specific tactics or widen an untested deployment claim.

### Runtime-only evidence

6. Compute the plan SHA-256 and copy only the plan, its manifest, and any required plugin library into a fresh validation image/process.
7. In that process, do not import the upstream models, Transformers, Torch-TensorRT, or PyTorch. Load the plugin library if required, deserialize with the TensorRT runtime, bind the recorded tensors, and execute.
8. Run `trtexec --loadEngine=<plan> --profilingVerbosity=detailed --dumpLayerInfo --exportLayerInfo=<json>` with the exact TensorRT 10.12 CLI. Archive stdout, stderr, and exported layer metadata.
9. Demonstrate that removing or corrupting the plan fails startup and that removing or corrupting a required plugin fails deserialization. Neither condition may select eager PyTorch.

This fresh-process TensorRT-only execution is the strongest practical proof that no hidden PyTorch partition or service fallback completed the request.

### ONNX proof, if that builder is selected

1. Run `onnx.checker` and archive the PyTorch exporter report.
2. Inspect and optionally constant-fold with Polygraphy; any graph surgery must preserve the frozen outputs.
3. Build with `trtexec --onnx=<model> --saveEngine=<plan> --skipInference` and fail on every parser/plugin error.
4. Apply the same raw-plan runtime-only evidence above.

TensorRT documents engine inspection and `trtexec` as the supported mechanisms for plan benchmarking and layer information ([TensorRT engine inspection tools](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-tools.html), [TensorRT benchmarking guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/benchmarking.html)).

## 6. Precision policy

The first engine for each model is FP32 with TF32 disabled. This matches the intent of the existing eager baseline closely enough to expose graph/plugin errors before adding reduced precision. It does not imply bitwise equality: TensorRT may select different but mathematically valid kernels or operation order. Acceptance tolerances must be derived from the frozen corpus and clinically relevant downstream invariants.

After the FP32 engine passes every gate, FP16 may be evaluated as a new artifact:

- compare raw detector logits and boxes, proposal ordering, threshold membership, NMS results, and top-eight crop coordinates;
- compare classifier logits, probability, predicted class, pooled features, and attention outputs used by the API;
- keep sensitive reductions, normalization, softmax, and accumulation in FP32 when measured error requires it;
- reject FP16 if it saves no meaningful end-to-end time or violates any corpus invariant.

TensorRT documents weakly and strongly typed precision control, precision constraints, and the accuracy risks around reduced-precision overflow and sensitive operations ([TensorRT precision control](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/precision-control.html), [TensorRT accuracy considerations](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/accuracy-considerations.html)).

The detector's existing wrapper casts deformable-attention inputs to FP32 in its half-precision branch and casts the result back. That is not evidence for a native FP16 TensorRT plugin. The plugin must declare and test its actual formats and accumulation policy. INT8 and FP8 are out of scope until representative calibration/evaluation data and explicit accuracy criteria exist.

## 7. Engine serialization, portability, and residency

A TensorRT plan is a compiled deployment artifact, not a portable model file. By default, serialized engines are tied to the TensorRT version and GPU compute capability used to build them. TensorRT provides version-compatible and hardware-compatible modes, but those widen compatibility under documented limitations and may reduce performance ([TensorRT engine compatibility](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html)).

For this assignment:

- build the release plan offline on an L4/SM 8.9 using the exact release image;
- deploy it with the same TensorRT 10.12 runtime, plugin build, and CUDA major lane;
- rebuild rather than copy a plan produced on the local RTX 3050;
- reject a plan when any manifest compatibility field differs;
- never build tactics during a user request;
- do not rely on an engine cache to choose among incompatible or stale plans.

The manifest should contain at least:

```text
model/checkpoint identity and SHA-256
wrapper source revision and contract version
plan SHA-256
TensorRT, Torch-TensorRT, PyTorch, CUDA, cuDNN, driver versions
GPU name, UUID class, and compute capability used for the build
builder flags, input/output names, dtypes, and shape/profile bounds
precision and TF32 policy
timing-cache SHA-256, if used
plugin name/version/namespace, library SHA-256, compiler and ABI metadata
frozen parity-corpus identity and validation-result SHA-256
```

TensorRT does not change the assignment's single-residency rule. The executor may keep both plan files on host storage, but only one model's engine, execution context, and device buffers may reside on the GPU at a time:

```text
load detector plan -> infer -> synchronize -> destroy detector context/engine/buffers
load MMBCD plan    -> infer -> synchronize -> destroy MMBCD context/engine/buffers
```

The L4 evidence bundle must include NVML sampling across that sequence and show no interval where allocations from both engines overlap. Holding a deserialized detector engine while loading the classifier, even if one is idle, violates the required residency contract.

## 8. Numerical, end-to-end, and performance gates

### Layer/backend comparison

Polygraphy can run the same frozen input data through ONNX Runtime and TensorRT and compare outputs; NVIDIA provides official examples for framework comparison and custom input data ([Polygraphy framework comparison](https://github.com/NVIDIA/TensorRT/tree/release/10.12/tools/Polygraphy/examples/cli/run/01_comparing_frameworks), [Polygraphy custom input data](https://github.com/NVIDIA/TensorRT/tree/release/10.12/tools/Polygraphy/examples/cli/run/05_comparing_with_custom_input_data)). A representative command shape is:

```powershell
polygraphy run model.onnx --onnxrt --trt `
  --load-inputs inputs.json `
  --atol <approved-absolute-tolerance> `
  --rtol <approved-relative-tolerance> `
  --check-error-stat elemwise --validate
```

Use frozen real inputs, not only Polygraphy's random data. Tolerances are acceptance-test data derived from observed FP32 differences and downstream invariants; they must not be loosened merely to obtain a pass. For a dynamic profile, compare minimum, optimum, maximum, and real interior shapes.

### Endpoint corpus

For every accepted DICOM in the corpus, compare eager FP32 and TensorRT through the complete endpoint:

- canonical image identity and detector tensor shape;
- raw detector tensors before postprocessing;
- proposal scores/classes/boxes, stable ordering, threshold membership, NMS output, and top-eight crop coordinates;
- classifier logits, probability, label, token/crop identities, and any returned attention data;
- warnings, structured API fields, failure behavior, and deterministic repeats.

Exact equality is required for discrete contract outcomes such as label, proposal count, ordering after stable tie-breaking, and error class. Numeric tensors use explicit per-output tolerances. Any borderline corpus example that crosses a score threshold, changes NMS, changes an ROI, or changes the final label is a STOP until the contract is adjudicated.

### Performance

Use `trtexec` to archive kernel-level timing with explicit warm-up, duration/iterations, detailed profiling verbosity, per-layer information, and profile exports. NVIDIA documents `--warmUp`, `--duration`, `--iterations`, `--profilingVerbosity=detailed`, `--dumpLayerInfo`, `--dumpProfile`, `--exportLayerInfo`, and `--exportProfile` ([TensorRT benchmarking guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/benchmarking.html)).

`trtexec` throughput is not the assignment result. Also measure on the target L4:

- cold plan load/deserialization and plugin registration;
- warm per-model inference;
- detector unload plus classifier load transition;
- complete DICOM endpoint latency, including preprocessing and postprocessing;
- peak and time-series GPU memory for each state;
- repeated-run determinism and failure/recovery behavior.

Compare against the same-revision eager FP32 baseline under the same driver, power state, input corpus, worker count, and warm-up. Promote only when the acceleration is meaningful end to end and does not break single residency.

## 9. Decision matrix

### GO

A model is **GO for TensorRT production** only when all are true:

- exact dependency lane installs reproducibly in build and runtime images;
- strict export succeeds with fixed documented inputs;
- compiler report shows no unsupported operation or PyTorch partition;
- target-L4 raw plan builds with full compilation required;
- a TensorRT-only process executes it with no model/PyTorch import;
- golden and representative-corpus numerical and discrete invariants pass;
- repeated runs are deterministic within the approved contract;
- plan/plugin/manifest compatibility and negative startup tests pass;
- single-residency NVML trace passes;
- end-to-end latency and VRAM improve enough to justify the extra artifact/plugin surface.

### PARTIAL

The project is **PARTIAL** when MMBCD passes every gate but the detector remains blocked on its custom plugin, or when only a fixed detector shape is validated. In that state it may accurately claim:

> The MMBCD classifier uses a strict full TensorRT engine for the validated
> fixed image/ROI and dynamic 2-through-90 token-width contract.

It may not claim that the whole pipeline uses TensorRT. The runtime must report the backend per stage, and the TensorRT-backed stage still has no eager fallback. Whether a mixed eager-detector/TensorRT-classifier release satisfies the assignment is a product decision, not something the runtime should conceal.

### STOP

Stop promotion when any of these occurs:

- strict export fails, falls back to another capture path, or relies on a graph break;
- dry run reports an unsupported operator, PyTorch partition, or converter/build failure;
- the plugin is absent, ABI-mismatched, numerically incorrect, or shape-unsafe;
- the plan was built for a different GPU/version lane without an explicitly validated compatibility mode;
- any accepted example changes threshold membership, NMS/top-eight selection, crop identity, final label, or API contract;
- the engine exceeds the single-residency/VRAM envelope, is nondeterministic beyond the contract, or is not materially faster end to end;
- evidence exists only on a non-L4 GPU.

## 10. What can and cannot be claimed now

### Supported now by primary sources and repository inspection

- Torch-TensorRT 2.8 is the official matching line for PyTorch 2.8 and TensorRT 10.12.
- TensorRT 10.12 supports the repository's CUDA 12.8 lane and L4 compute capability.
- MMBCD is the rational first compilation target, subject to an actual strict coverage report.
- FocalNet-DINO contains an opaque custom deformable-attention CUDA operation with no upstream export/plugin registration, so plugin work is the expected blocker.
- fixed image/ROI shapes, a bounded tied token-width profile, strict full compilation, raw-plan execution, and the validation gates above are an implementable design.

### Not supported until a target-L4 run with the real artifacts

- that either actual checkpoint exports or builds successfully;
- that either model is 100 percent TensorRT with no fallback;
- that the deformable-attention plugin is correct, required in exactly the predicted form, or faster than a decomposition;
- that FP32 TensorRT meets numerical tolerance, or that FP16 is safe;
- any latency, throughput, speedup, cold-load, memory, or determinism number;
- that a plan built elsewhere is portable to L4;
- that the Django endpoint is TensorRT accelerated or production ready.

The existing eager FP32 L4 validation remains valuable baseline evidence. It does not transfer to a different graph compiler, kernel set, precision policy, engine artifact, or runtime.

## 11. Recommended implementation order

1. Add the exact acceleration-tool pins to a separate locked builder/validation dependency group and record every resolved version.
2. Preserve MMBCD token width, bind both token tensors to the same 2-through-90
   dynamic profile, and reject every shape outside that profile.
3. Create an inference-only MMBCD tensor wrapper; run strict export and archive its first unsupported-operator report.
4. If coverage is complete, build the FP32/TF32-disabled raw plan on L4 and run TensorRT-only, parity, endpoint, residency, and performance gates.
5. Run the same strict detector export solely to capture the exact custom-operator failure and graph context.
6. Implement and unit-validate the deformable-attention custom operator plus `IPluginV3`/converter; then repeat fixed-golden-shape detector gates on L4.
7. Decide static buckets versus bounded dynamic profiles only after the fixed detector plan passes.
8. Evaluate FP16 only as new, separately manifested artifacts after both FP32 paths are accepted.

This sequence can yield a defensible MMBCD TensorRT result early while keeping the detector plugin work measurable and preventing silent fallback from being mistaken for acceleration.
