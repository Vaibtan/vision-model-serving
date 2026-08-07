# Single-residency accelerator runtime

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
to `Ready`, including strict adapter construction and one deliberate warmup.
Requests for the ready model reuse it. A different-model request marks the
runtime `Draining` while active inference finishes, then transitions through
`Unloading`, `Unloaded`, and `Loading` before the new model becomes `Ready`.

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

Load, warmup, inference/OOM-style, and unload failures enter `Failed` with a
stable code, phase, model ID, and SHA-256 of the binding's cause token. Original
exception text, paths, state keys, and inputs are not retained. The runtime
rejects another attempt for the same model while that token is unchanged. A
changed artifact/config/source generation permits one new attempt; another
failure is bounded again by the new token.

The cause token is supplied by the composition root and should cover the
verified checkpoint digest, pinned source/config revision, and any other
external state whose change authorizes recovery.

## CUDA and memory behavior

`TorchCudaLifecycle` imports PyTorch and tests CUDA availability only on its
first execution-path method. Constructing the runtime and reading `status()` do
not initialize CUDA, allowing the object to be created inside the dedicated
worker child rather than a pre-fork web process.

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
python -m unittest tests.test_single_residency_runtime -v
python -m unittest discover -s tests -v
```

The real checkpoints remain outside Git. This workstation's read-only sibling
artifact directory has been rechecked against the manifest: the 2,731,092,364
byte FocalNet-DINO checkpoint and 587,689,457 byte MMBCD checkpoint both match
their required SHA-256 values.

## NVIDIA L4 acceptance gate

The live harness uses the real public DICOM, production detector and classifier
adapters, external checkpoints, pinned source trees, and two full
detector-to-classifier cycles:

```bash
cd /teamspace/studios/this_studio/vision-model-serving
export PYTHONPATH="$PWD/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python scripts/l4_validation/16_validate_single_residency.py --cycles 2
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
