# Assignment traceability and validation index

This matrix distinguishes implementation, evidence, intentional deviations,
and unvalidated claims. “Passed” never means clinically validated.

## Requirement matrix

| Assignment requirement | Implementation | Executable evidence | Status |
| --- | --- | --- | --- |
| Understand both models, inputs, outputs, preprocessing, and dependencies | [Artifact manifest](../config/model-artifacts.json) plus [DICOM](../src/vision_model_serving/dicom), [detector](../src/vision_model_serving/detector), [classifier](../src/vision_model_serving/classifier), and [pipeline](../src/vision_model_serving/pipeline) modules | [Archived strict loads and raw FP32 reference](validation/reference-l4-fp32-20260807.json) | Passed for the pinned artifacts/runtime |
| Django inference service with structured JSON | [HTTP adapter](../src/vision_model_serving/web/api.py), [URL table](../src/vision_model_serving/web/urls.py), [typed serialization](../src/vision_model_serving/pipeline/serialization.py) | [Real Redis/Django acceptance](../tests/real_infra/test_django_api.py) and [packaged L4 validation](validation/compose-restart-l4-20260809.json) | Passed |
| Accept appropriate inputs and run all preprocessing/inference | [Multipart validation](../src/vision_model_serving/web/api.py), required history for full, [canonicalization](../src/vision_model_serving/dicom/canonicalization.py), and [two-stage pipeline](../src/vision_model_serving/pipeline/pipeline.py) | Public fixture SHA/canonical hash and exact detector/classifier hashes in [restart evidence](validation/compose-restart-l4-20260809.json) | Passed for one fixture |
| Load models once and reuse subsequent requests | [Persistent GPU executor composition](../src/vision_model_serving/pipeline/factory.py) loads on first use and retains both residents | [Warm cycle](validation/compose-restart-l4-20260809.json) has detector/classifier `reused: true` | Passed after cold load |
| Use one model at a time and load/unload between the two models | [Strict-switching runtime](../src/vision_model_serving/residency/runtime.py) and earlier L4 evidence exist, but [ADR 0002](adr/0002-use-a-persistent-gpu-executor.md) retains both in deployment | [Single-residency evidence](validation/single-residency-l4-20260808.json) proves switching; [current Compose evidence](validation/compose-restart-l4-20260809.json) proves dual residency | Intentional deployed deviation; documented, not silently claimed compliant |
| Clean modular organization | Deep module boundaries listed in [architecture.md](architecture.md#module-seams) | CPU contract suite plus real integration gates | Passed |
| Robust production-oriented deployment | [Capacity/idempotency and failure semantics](gpu-execution-gateway.md), [readiness](../src/vision_model_serving/web/operational.py), [safe telemetry](observability.md), and [hardened Compose](../compose.yaml) | [Container](validation/container-l4-20260809.json) and [destructive-restart](validation/compose-restart-l4-20260809.json) evidence | Passed within stated single-node scope |
| TensorRT appreciated | No TensorRT/ONNX/FP16/compile path is enabled | None | Not implemented; future work requires parity and performance gates |
| Docker packaging and minimal setup | Pinned [web](../docker/web.Dockerfile)/[executor](../docker/executor.Dockerfile) images and [Compose profiles](../compose.yaml) | [Clean L4 image build, smoke, benchmark, inspection, cleanup](validation/container-l4-20260809.json) | Passed on NVIDIA L4 |
| Public mammogram DICOM | Checksum-pinned [TCIA manifest](../config/public-fixtures.json)/[fetcher](../scripts/fetch_public_fixture.py) with license and attribution | [Live fetch gate](../tests/real_infra/test_public_fixture_fetch.py) plus canonical array hash in [packaged validation](validation/compose-restart-l4-20260809.json) | Passed for the selected Secondary Capture object |
| Clear reproduction and examples | [Fresh-machine guide](reproduction.md), generated OpenAPI, [container runbook](containers.md) | Commands and links checked; exact L4 run recorded | Passed, subject to externally supplied weights |

## Validation layers

| Layer | Requires | Proves | Does not prove |
| --- | --- | --- | --- |
| CPU contract suite | uv environment | Schema, pure policies, typed failures, serialization, lifecycle logic | Real Redis, CUDA, checkpoint loading, latency |
| Real CPU integration | Redis plus pinned public DICOM | Django/DRF behavior, queue representation, fixture decoding, broker privacy | GPU/model correctness |
| Archive verification | Downloaded evidence archive and sidecar | Stored member identities, strict-load transcripts, archived hashes | A new live GPU run |
| Standalone L4 harness | L4, external weights/sources/tokenizer | Strict loads, native operator, raw FP32 outputs, original switching policy | Packaged HTTP topology |
| Packaged L4 smoke/benchmark | Docker, L4, all external mounts | Full real service path, image/runtime isolation, golden hashes, warm latency | Clinical performance or multi-node scale |
| Destructive-restart L4 gate | Same plus bind-mounted report | Two modes, finite/bounded results, exact ROI/provenance/warnings, warm reuse, privacy, identity after full teardown | Accuracy, calibration, robustness, or clinical utility |

Existing CPU tests use small deterministic fixtures or interface fakes where a
real checkpoint would make the test non-hermetic. They are contract checks,
not substitutes for the separately recorded real-infrastructure gates. New
integration claims in this repository require real infrastructure.

## Evidence index

| Evidence | Scope |
| --- | --- |
| [`reference-l4-fp32-20260807.json`](validation/reference-l4-fp32-20260807.json) | Authoritative archive-backed environment, artifact, preprocessing, and per-model golden hashes |
| [`single-residency-l4-20260808.json`](validation/single-residency-l4-20260808.json) | Original detector unload/classifier unload switching lifecycle |
| [`prediction-pipeline-l4-20260808.json`](validation/prediction-pipeline-l4-20260808.json) | Typed repository pipeline over real models and DICOM |
| [`persistent-rq-executor-l4-20260808.json`](validation/persistent-rq-executor-l4-20260808.json) | RQ plus long-lived dual-resident executor and warm latency |
| [`container-l4-20260809.json`](validation/container-l4-20260809.json) | Clean image build, smoke, benchmark, profile/security inspection, cleanup |
| [`compose-restart-l4-20260809.json`](validation/compose-restart-l4-20260809.json) | Current packaged two-lifecycle identity, behavior, privacy, and cleanup proof |

The evidence archive and sidecar remain outside Git. The checked-in records are
bounded summaries tied to hashes and commits, not replacements for the source
archive or externally supplied model artifacts.
