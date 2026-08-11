# Single-residency accelerator runtime

This document records the deployed strict switching policy. The private
executor composition selects `SingleResidencyRuntime`; no retention toggle or
second residency implementation exists. See
[ADR 0003](adr/0003-enforce-single-model-residency.md) and the
[current architecture](architecture.md#gpu-lifecycle-strict-single-residency).

`SingleResidencyRuntime` is the deep module that owns accelerator lifecycle
state. Its operational interface is deliberately limited to:

```python
outputs = runtime.execute(model_id, inputs)
status = runtime.status()
```

Callers receive immutable lifecycle records and model-specific host outputs.
They never receive the resident adapter, a PyTorch model, device tensors, or a
CUDA handle. Construction accepts `ModelBinding` adapters at the composition
root so detector/classifier loading can vary and tests can use non-CUDA fakes.

## State and concurrency contract

The runtime begins `Unloaded`. A first request transitions through `Loading`
to `Ready` with strict adapter construction; the cold request's own execute()
call is the warm pass (the composition-root warmup hook is a no-op so a
patient case never runs through the model twice). Requests for the ready
model reuse it. A different-model request marks the runtime `Draining` while
active inference finishes, then transitions through `Unloading`, `Unloaded`,
and `Loading` before the new model becomes `Ready`.

`close()` is terminal: it latches the runtime closed, waits on the execution
lock for the in-flight case to finish, and unloads. A caller that queued
behind `close()` is refused instead of reloading a model into a process that
is shutting down.

One execution lock serializes the accelerator critical section. A separate
status lock permits `status()` to observe loading, active inference, and
draining without gaining access to lifecycle implementation objects.

Every unload:

1. removes the runtime's strong resident reference;
2. synchronizes the accelerator;
3. runs Python garbage collection;
4. releases unused allocator cache;
5. synchronizes again and captures memory metrics; and
6. requires the old adapter's weak reference to be dead.

`empty_cache()` is therefore cleanup, not proof. A still-reachable adapter
causes a typed unload failure even if cache cleanup succeeds.

## Failure and recovery contract

Exceptions carrying the `case_input_error` marker (unusable proposals or ROI
inputs, rejected DICOM pixels) are data-dependent: they fail only that case
with a typed `runtime_case_input_invalid` error, keep the model resident, and
leave the runtime `Ready`. They never enter `Failed`, so one unusual image
cannot lock healthy models out of service.

Load, inference/OOM-style, and unload failures enter `Failed` with a stable
code, phase, model ID, and SHA-256 of the binding's cause token. Original
exception text, paths, state keys, and inputs are not retained. The runtime
rejects another attempt for the same model while that token is unchanged. A
changed artifact/config/source generation permits one new attempt; another
failure is bounded again by the new token.

Reachability failures have a different, directly observable cause. The runtime
retains only a weak reference to the leaked resident and rejects every model
load while that reference remains alive. Once the external owner releases it,
the same unchanged artifact may recover; changing an unrelated target artifact
is neither required nor treated as proof that the old resident disappeared.

An allocator synchronization, cache-cleanup, or memory-observation failure is
not treated as recoverable reachability. It latches the entire accelerator
process failed for every model because in-process cleanup can no longer prove a
safe CUDA state. Recovery requires replacing the executor process.

The cause token is supplied by the composition root and should cover the
verified checkpoint digest, pinned source/config revision, and any other
external state whose change authorizes recovery.

## CUDA and memory behavior

`TorchCudaLifecycle` owns the process-wide deterministic FP32 policy used by
the residency runtime and both standalone model adapters. Runtime construction
prepares PyTorch, verifies CUDA availability, seeds CPU/CUDA execution, disables
TF32 and cuDNN benchmarking, and enables deterministic algorithms before any
resident model is built. This construction happens only inside the dedicated
GPU executor, never in a pre-fork web or RQ work-horse process.

Each inference resets peak stats, synchronizes before and after the timed
region, and runs under `torch.inference_mode()`. The resident detector and
classifier adapters separately call `model.eval()` when they strict-load.
Status and outputs distinguish:

- current allocated tensor memory;
- current memory reserved by the caching allocator;
- peak allocated memory since reset; and
- peak reserved memory since reset.

Lifecycle metrics count successful loads, same-model reuse, switches,
successful unloads, and failures. Timings retain the latest load, inference,
switch, and unload durations.

## Local validation

All state transitions, serialization, same-model reuse, cross-model switching,
load failure, inference/OOM-style failure, reachability failure, cleanup order,
cause-token recovery, and lazy CUDA calls are tested with public-interface fake
adapters:

```powershell
$env:PYTHONPATH = "src"
uv run python -m unittest tests.test_single_residency_runtime -v
uv run python -m unittest discover -s tests -v
```

The real checkpoints remain outside Git. A fresh run requires externally
supplied checkpoints matching the manifest sizes/hashes plus the pinned source
trees, tokenizer, and L4 runtime. Their presence in any one checkout is mutable
host state and not repository evidence.

## NVIDIA L4 acceptance gate

The live harness uses the real public DICOM, production detector and classifier
adapters, external checkpoints, pinned source trees, and two full
detector-to-classifier cycles:

```bash
cd /teamspace/studios/this_studio/vision-model-serving
export PYTHONPATH="$PWD/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
uv run python scripts/l4_validation/16_validate_single_residency.py --cycles 2
```

The default Studio layout expects:

- weights in `/teamspace/studios/this_studio/vision-model-serving-artifacts`;
- FocalNet-DINO in `/teamspace/studios/this_studio/src/FocalNet-DINO`;
- DINO in `/teamspace/studios/this_studio/src/dino`;
- MMBCD in `/teamspace/studios/this_studio/src/MMBCD`; and
- the tokenizer under `/teamspace/studios/this_studio/assets` at the pinned
  revision-named directory.

Use command-line path overrides if the mounts differ. Do not accept the run
unless it ends with all three exact markers:

```text
REAL DICOM DETECTOR INFERENCE PASSED
REAL DICOM MMBCD INFERENCE PASSED
SINGLE RESIDENCY L4 PASSED
```

The generated runtime manifest records each stage's load/inference/switch
timings and allocated/reserved/peak memory. This is a serving correctness and
lifecycle smoke test on one public fixture, not medical validation.

The archived 2026-08-08 direct gate proved two real-model cycles for its
embedded revision. Corrected 2026-08-10
[`single-residency`](validation/single-residency-l4-20260810.json),
[`destructive-restart`](validation/compose-restart-l4-20260810.json), and
[`resolution`](validation/spec-resolution-l4-20260810.md) records cover the
corrected packaged topology for their own embedded revisions. HEAD has later
changes, so fresh same-revision packaged evidence remains required.
