# Assignment traceability and validation index

This matrix distinguishes implementation, evidence, intentional deviations,
and unvalidated claims. “Passed” never means clinically validated.

The linked L4 records remain authoritative for their embedded revisions, not
for the current worktree. Post-evidence runtime, validation-contract, Django,
CI, Django/Pillow dependency, and documentation changes have passed local CPU
and browser gates; current-revision L4 acceptance is pending a rerun.

## Requirement matrix

| Assignment requirement | Implementation | Executable evidence | Status |
| --- | --- | --- | --- |
| Understand both models, inputs, outputs, preprocessing, and dependencies | [Artifact manifest](../config/model-artifacts.json) plus [DICOM](../src/vision_model_serving/dicom), [detector](../src/vision_model_serving/detector), [classifier](../src/vision_model_serving/classifier), and [pipeline](../src/vision_model_serving/pipeline) modules | [Archived strict loads and raw FP32 reference](validation/reference-l4-fp32-20260807.json) | Passed for the pinned artifacts/runtime |
| Django inference service with structured JSON | [HTTP adapter](../src/vision_model_serving/web/api.py), [URL table](../src/vision_model_serving/web/urls.py), [typed serialization](../src/vision_model_serving/pipeline/serialization.py) | [Real Redis/Django acceptance](../tests/real_infra/test_django_api.py) and prior exact-revision [packaged L4 validation](validation/compose-restart-l4-20260810.json) | Previously passed; current L4 rerun pending |
| Accept appropriate inputs and run all preprocessing/inference | [Multipart validation](../src/vision_model_serving/web/api.py), required history for full, [canonicalization](../src/vision_model_serving/dicom/canonicalization.py), and [two-stage pipeline](../src/vision_model_serving/pipeline/pipeline.py) | Public fixture SHA/canonical hash and exact detector/classifier hashes in prior exact-revision [restart evidence](validation/compose-restart-l4-20260810.json) | Previously passed for one fixture; current L4 rerun pending |
| Load models once and reuse subsequent compatible requests | [Private GPU executor composition](../src/vision_model_serving/execution/_composition.py) keeps one controller and reuses only the current resident | Warm detection reuse in the prior exact-revision [schema-v3 benchmark](validation/benchmark-l4-20260810.json) and both [restart cycles](validation/compose-restart-l4-20260810.json) | Previously passed on the pinned L4 fixture; rerun pending |
| Use one model at a time and load/unload between the two models | [Only strict-switching runtime](../src/vision_model_serving/residency/runtime.py), fail-closed status contract, and [ADR 0003](adr/0003-enforce-single-model-residency.md) | Prior exact-revision [direct L4 lifecycle](validation/single-residency-l4-20260810.json), packaged max-one snapshots, and destructive restart evidence; old dual-resident records remain explicitly superseded | Previously passed on the pinned L4 fixture; rerun pending |
| Clean modular organization | Deep module boundaries listed in [architecture.md](architecture.md#module-seams) | CPU contract suite plus real integration gates | Passed |
| Robust production-oriented deployment | [Capacity/idempotency and failure semantics](gpu-execution-gateway.md), [readiness](../src/vision_model_serving/web/operational.py), [safe telemetry](observability.md), and [hardened Compose](../compose.yaml) | Prior exact-revision [schema-v3 benchmark](validation/benchmark-l4-20260810.json), [browser gate](validation/browser-workbench-l4-20260810.log), and [destructive restart](validation/compose-restart-l4-20260810.json) | Previously passed within the stated single-node scope; rerun pending |
| TensorRT appreciated | Pinned [TensorRT lane](../requirements/tensorrt-l4.txt), strict [manifest contract](../src/vision_model_serving/acceleration/tensorrt.py), package-owned [experiment contract](../src/vision_model_serving/validation/tensorrt_experiment.py), private GPU measurement adapter, thin [CLI](../scripts/l4_validation/18_build_tensorrt_candidate.py), PyTorch-free runtime verifier, and validation image | Historical [L4 TensorRT report](validation/tensorrt-l4-20260810/tensorrt-spike.json), dry-run/coverage reports, parity, performance, and runtime-verifier records | Historical measured STOP; current optimization acceptance rerun pending, eager FP32 remains selected |
| Docker packaging and minimal setup | Pinned [web](../docker/web.Dockerfile)/[executor](../docker/executor.Dockerfile) images and [Compose profiles](../compose.yaml) | Prior clean-revision image identity, packaged benchmark, browser inspection, restart, and cleanup in the [resolution record](validation/spec-resolution-l4-20260810.md) | Previously passed on NVIDIA L4; current image rerun pending |
| Public mammogram DICOM | Checksum-pinned [TCIA manifest](../config/public-fixtures.json)/[fetcher](../scripts/fetch_public_fixture.py) with license and attribution | [Live fetch gate](../tests/real_infra/test_public_fixture_fetch.py) plus canonical array hash in prior exact-revision [packaged validation](validation/compose-restart-l4-20260810.json) | Previously passed for the selected Secondary Capture object; rerun pending |
| Clear reproduction and examples | [Fresh-machine guide](reproduction.md), generated OpenAPI, [container runbook](containers.md) | Commands and links checked; prior exact-revision runs indexed in the [resolution record](validation/spec-resolution-l4-20260810.md) | Documentation checked; current L4 rerun pending and weights remain externally supplied |

## Validation layers

| Layer | Requires | Proves | Does not prove |
| --- | --- | --- | --- |
| CPU contract suite | uv environment | Schema, pure policies, typed failures, serialization, lifecycle logic | Real Redis, CUDA, checkpoint loading, latency |
| Real CPU integration | Redis plus pinned public DICOM | Django/DRF behavior, queue representation, fixture decoding, broker privacy | GPU/model correctness |
| Archive verification | Downloaded evidence archive and sidecar | Stored member identities, strict-load transcripts, archived hashes | A new live GPU run |
| Standalone L4 harness | L4, external weights/sources/tokenizer | Strict loads, native operator, raw FP32 outputs, original switching policy | Packaged HTTP topology |
| Packaged L4 smoke/schema-v4 benchmark | Docker, L4, all external mounts | Full real service path, image/runtime isolation, serialized output identities, exact hashes, lifecycle distributions, concurrency 1/2/4, bounded resource-sampling coverage | Clinical performance or multi-node scale |
| Destructive-restart L4 gate | Same plus bind-mounted report | Detection reuse, strict detector-to-classifier switching, max-one residency, privacy, identity after teardown | Accuracy, calibration, robustness, or clinical utility |
| Browser acceptance | Chromium plus packaged stack | Real upload, polling, result rendering, overlay/ROI/attention inspection, JSON/PNG export | Accessibility certification or clinical usability |

Existing CPU tests use small deterministic fixtures or interface fakes where a
real checkpoint would make the test non-hermetic. They are contract checks,
not substitutes for the separately recorded real-infrastructure gates. New
integration claims in this repository require real infrastructure.

## Evidence index

| Evidence | Scope |
| --- | --- |
| [`reference-l4-fp32-20260807.json`](validation/reference-l4-fp32-20260807.json) | Authoritative archive-backed environment, artifact, preprocessing, and per-model golden hashes |
| [`single-residency-l4-20260808.json`](validation/single-residency-l4-20260808.json) | Original detector unload/classifier unload switching lifecycle |
| [`prediction-pipeline-l4-20260808.json`](validation/prediction-pipeline-l4-20260808.json) | Archived pre-gateway typed pipeline run over real models and DICOM |
| [`persistent-rq-executor-l4-20260808.json`](validation/persistent-rq-executor-l4-20260808.json) | Historical dual-resident RQ/executor latency evidence; superseded policy |
| [`container-l4-20260809.json`](validation/container-l4-20260809.json) | Historical dual-resident image/smoke/benchmark evidence; not current revision proof |
| [`compose-restart-l4-20260809.json`](validation/compose-restart-l4-20260809.json) | Historical dual-resident restart evidence; superseded by ADR 0003 |
| [`spec-resolution-l4-20260810.md`](validation/spec-resolution-l4-20260810.md) | Exact-revision five-finding closure matrix and revision boundary; current rerun pending |
| [`benchmark-l4-20260810.json`](validation/benchmark-l4-20260810.json) and [summary](validation/benchmark-l4-20260810.md) | Exact-revision schema-v3 packaged lifecycle, latency, concurrency, resource, and golden-output evidence |
| [`single-residency-l4-20260810.json`](validation/single-residency-l4-20260810.json) | Exact-revision isolated real-model unload-before-switch lifecycle |
| [`compose-restart-l4-20260810.json`](validation/compose-restart-l4-20260810.json) | Exact-revision post-destruction packaged behavior, privacy, readiness, and baseline match |
| [`pytorch-optimization-l4-20260810.json`](validation/pytorch-optimization-l4-20260810.json) and [summary](validation/pytorch-optimization-l4-20260810.md) | Historical schema-v1 screening; eager FP32 remains selected, while raw/post-NMS/eight-ROI parity plus candidate-release and repeated-switch VRAM evidence await an L4 rerun |
| [`tensorrt-l4-20260810/tensorrt-spike.json`](validation/tensorrt-l4-20260810/tensorrt-spike.json), [summary](validation/tensorrt-l4-20260810/tensorrt-spike.md), and [failure analysis](validation/tensorrt-l4-20260810/failure-analysis.md) | Strict no-fallback TensorRT result: classifier static feasibility, production-profile STOP, measured detector capture failure, and expected downstream plugin boundary |
| [`browser-workbench-l4-20260810.log`](validation/browser-workbench-l4-20260810.log) | Exact-revision packaged Chromium upload/poll/inspect/export transcript |

The evidence archive and sidecar remain outside Git. The checked-in records are
bounded summaries tied to hashes and commits, not replacements for the source
archive or externally supplied model artifacts.
