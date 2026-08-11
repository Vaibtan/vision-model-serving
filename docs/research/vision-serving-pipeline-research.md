# Vision Model Serving Pipeline: Technical Research and Planning Inputs

**Status:** Historical pre-implementation research snapshot from 2026-08-03.
Statements about missing implementation or unresolved execution gates describe
that snapshot, not the current repository. Use [`README.md`](../../README.md),
[`docs/architecture.md`](../architecture.md), and
[`docs/traceability.md`](../traceability.md) for current behavior and evidence.
**Scope:** assignment requirements, supplied detector configuration, official MMBCD/FocalNet-DINO source, DICOM handling, GPU inference, export/acceleration, serving, observability, and benchmarking

## Executive finding

The assignment is a **sequential two-stage medical-image inference pipeline**:

1. FocalNet-DINO detects candidate regions of interest (ROIs) in a mammogram.
2. MMBCD classifies one mammogram view using the top ROI crops plus clinical history.

That interpretation is supported by the MMBCD paper, which describes FocalNet-DINO as the ROI extractor and then applies ViT-DINO, RoBERTa, max pooling, cross-attention, and a final classifier to the selected regions ([MMBCD paper, Sections 3.1-3.3](https://papers.miccai.org/miccai-2024/paper/1311_paper.pdf)). It is also reflected in the released source: MMBCD loads detector proposals from text files, crops the image, encodes the crops and text, and returns two logits ([MMBCD data path](https://github.com/adsbansal/MMBCD/blob/14ac5e099c79253b01e0885d2ebefa6f86cfd8f0/code/data.py), [MMBCD model](https://github.com/adsbansal/MMBCD/blob/14ac5e099c79253b01e0885d2ebefa6f86cfd8f0/code/model.py)).

At the time of this research snapshot, credible implementation was blocked from
real inference by missing artifact and contract evidence:

- This repository contains the detector config but **no detector checkpoint and no MMBCD classifier checkpoint**.
- The upstream MMBCD release documents the detector proposal file shape only by example; it does not include the code that produces those files.
- The released evaluation dataset constructs some text prompts using ground-truth `cancer` and `all_views_cancer` columns. A live API must not require labels, so its inference prompt contract must be decided and validated.
- FocalNet-DINO calls a compiled `MultiScaleDeformableAttention` C++/CUDA operator directly. This is a real deployment and export constraint, not an ordinary pure-PyTorch graph ([custom operator call](https://github.com/FocalNet/FocalNet-DINO/blob/23901e021dc6ec8f66bad47983f45a25574452cc/models/dino/ops/modules/ms_deform_attn.py), [extension build](https://github.com/FocalNet/FocalNet-DINO/blob/23901e021dc6ec8f66bad47983f45a25574452cc/models/dino/ops/setup.py)).
- The FocalNet backbone builder also tries to load `./focalnet_large_lrf_384.pth` before the fine-tuned detector checkpoint is applied. That backbone artifact is absent and must either be supplied or proven unnecessary through a validated builder change plus strict full-checkpoint coverage ([backbone construction](https://github.com/FocalNet/FocalNet-DINO/blob/23901e021dc6ec8f66bad47983f45a25574452cc/models/dino/backbone.py#L204-L245)).

The senior-grade approach should therefore make **reproducible eager PyTorch correctness the first milestone**, use one dedicated GPU execution process with an explicit model-residency state machine, and treat mixed precision, `torch.compile`, ONNX Runtime, and TensorRT as progressively gated optimizations. No acceleration claim should be accepted until it passes numerical and end-to-end parity tests on the actual checkpoints and target GPU.

## 1. Assignment-required scope versus optional enhancement scope

### Required by `ASSIGNMENT.md`

- Understand both models, their input/output contracts, preprocessing, postprocessing, and dependencies.
- Build a Django inference service with structured JSON responses.
- Avoid loading both models on the GPU at the same time; load/unload them one at a time.
- Avoid re-reading/reconstructing artifacts unnecessarily; reuse initialized model state across requests where compatible with the one-model-at-a-time constraint.
- Keep routes, model loading, inference, configuration, and utilities modular.
- Containerize the application and provide reproducible local instructions plus example requests/responses.
- Validate the endpoint with a publicly available mammography DICOM.
- TensorRT is appreciated, not required.

### Optional enhancements that add senior-level engineering signal

- A durable job API and small frontend for DICOM upload, clinical-history entry, status, timing breakdown, ROI overlay, model-attention visualization, and downloadable JSON.
- Artifact manifests with SHA-256 checksums, source/license metadata, config hash, model version, preprocessing version, and startup compatibility checks.
- Admission control, bounded queueing, cancellation/timeouts, and explicit `429`/`503` behavior instead of allowing GPU OOM or unbounded web-worker blocking.
- A benchmark and numerical-parity harness that promotes an inference backend only after it beats the baseline within agreed accuracy tolerances.
- OpenTelemetry traces and Prometheus metrics for each pipeline stage, plus NVIDIA GPU telemetry where supported.
- An optional ONNX Runtime CUDA/TensorRT or native TensorRT experiment on Lightning AI.

These enhancements must remain subordinate to the reference pipeline. A polished UI around an unverified preprocessing path would weaken, not strengthen, the submission.

## 2. Reconstructed reference pipeline

### 2.1 Study semantics

The paper explicitly describes **single-view classification** and states that, for malignant patients during training, clinical history is not included for the unaffected breast ([MMBCD paper, Section 3.2](https://papers.miccai.org/miccai-2024/paper/1311_paper.pdf)). The released sample CSV has one row per view and includes `UHID`, `text`, `cancer`, `im_path`, and `all_views_cancer` ([sample schema](https://github.com/adsbansal/MMBCD/blob/14ac5e099c79253b01e0885d2ebefa6f86cfd8f0/sample_data/test.csv)).

Therefore, the smallest defensible API contract is one DICOM mammogram view plus one clinical-history string per inference. Multi-view/study-level aggregation would be a new model behavior unless the authors provide an aggregation rule or separate checkpoint.

### 2.2 DICOM to detector image

The released MMBCD preprocessing script:

1. reads `dataset.pixel_array`,
2. applies the VOI LUT,
3. inverts pixels for non-`MONOCHROME2`,
4. min-max normalizes to 8-bit,
5. writes PNG,
6. removes black space using the largest contour,
7. resizes the result to `1024 x 1024`.

See [DICOM conversion](https://github.com/adsbansal/MMBCD/blob/14ac5e099c79253b01e0885d2ebefa6f86cfd8f0/preprocess_img.py), [black-space crop](https://github.com/adsbansal/MMBCD/blob/14ac5e099c79253b01e0885d2ebefa6f86cfd8f0/preprocess/crop.py), and [resize](https://github.com/adsbansal/MMBCD/blob/14ac5e099c79253b01e0885d2ebefa6f86cfd8f0/preprocess/resize.py).

This code is a research reference, not yet a robust DICOM contract:

- `apply_voi_lut` requires any modality LUT/rescale operation to be applied first; the upstream script does not do that explicitly ([pydicom VOI LUT documentation](https://pydicom.github.io/pydicom/stable/reference/generated/pydicom.pixels.apply_voi_lut.html)).
- `pixel_array` may need additional decoding plugins for compressed transfer syntaxes, and pydicom 3.x changed the default decoding backend ([pydicom compressed pixel data guide](https://pydicom.github.io/pydicom/stable/guides/user/image_data_handlers.html)).
- DICOM defines `MONOCHROME1` as minimum sample value displayed white and `MONOCHROME2` as minimum displayed black after VOI transforms ([DICOM PS3.3 C.7.6.3.1.2](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_c.7.6.3.html)).
- Pixel padding is meant to be excluded before deriving display range; thresholding the post-windowed image at value `1` does not implement the DICOM padding semantics ([DICOM PS3.3 C.7.5.1.1.2](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.5.html)).
- The resize helper changes aspect ratio to a square. That may be the training-time contract and therefore must initially be preserved for parity, even if a more geometrically faithful resize looks preferable.

The implementation should first reproduce the upstream output on fixed fixtures, then harden error handling without silently changing pixel semantics. Every spatial transform must be recorded so detector boxes can be mapped back to original DICOM pixel coordinates.

### 2.3 Detector construction and preprocessing

The supplied `config_cfg.py` describes a one-class, four-feature-level DINO detector with a `focalnet_L_384_22k` backbone, 6 encoder layers, 6 decoder layers, 900 queries, 300 selected predictions, and no postprocessor NMS (`nms_iou_threshold = -1`). It is a flattened/fine-tuned variant of the upstream FocalNet-DINO configuration, not an MMDetection/MMCV config; the official repository does not depend on MMDetection or MMCV ([FocalNet-DINO requirements](https://github.com/FocalNet/FocalNet-DINO/blob/23901e021dc6ec8f66bad47983f45a25574452cc/requirements.txt)). It uses ImageNet normalization, an evaluation short-side scale of 800, and a maximum long side of 1333 through the upstream evaluation transform ([FocalNet-DINO COCO transforms](https://github.com/FocalNet/FocalNet-DINO/blob/23901e021dc6ec8f66bad47983f45a25574452cc/datasets/coco.py)).

Important serving implications of this config:

- `use_checkpoint = True` is a training memory-saving flag. The FocalNet forward path calls activation checkpointing whenever the flag is true, including in this source's forward implementation ([FocalNet checkpoint path](https://github.com/FocalNet/FocalNet-DINO/blob/23901e021dc6ec8f66bad47983f45a25574452cc/models/dino/focal.py)). Serving should test `use_checkpoint=False` for identical outputs and lower latency; it must not assume equivalence without a golden test.
- `num_select = 300` and `nms_iou_threshold = -1` mean upstream DINO postprocessing applies sigmoid to logits, selects the top 300 query/class scores, converts/scales boxes, and does **not** perform NMS ([DINO `PostProcess`](https://github.com/FocalNet/FocalNet-DINO/blob/23901e021dc6ec8f66bad47983f45a25574452cc/models/dino/dino.py#L655-L704)).
- `num_classes = 1` means each of the 900 queries contributes a single lesion score. The output class label alone carries no benign/malignant classification meaning; it is an ROI proposal score.
- The raw output contract is `pred_logits [B, 900, 1]` and normalized `pred_boxes [B, 900, 4]` in `cxcywh` order. No confidence threshold is encoded in the supplied config; thresholding would be an additional behavior that needs evidence.
- The official notebook loads `checkpoint['model']`, calls `eval()`, resizes to short side 800 with max size 1333, converts to tensor, normalizes with ImageNet mean/std, runs the model, and applies the bbox postprocessor ([official inference notebook](https://github.com/FocalNet/FocalNet-DINO/blob/23901e021dc6ec8f66bad47983f45a25574452cc/inference_and_visualization.ipynb)).

The actual checkpoint must be loaded strictly against this exact config. A filename or config resemblance is not evidence that the tensor shapes, class head, or checkpoint wrapper match.

### 2.4 Detector-to-MMBCD proposal handoff

MMBCD expects a text file per image containing rows of five floats: normalized center-x, center-y, width, height, and confidence. It assumes rows are already sorted by confidence, applies custom NMS at IoU `0.1`, keeps the first `topk`, and samples duplicate boxes if too few remain ([proposal ingestion and NMS](https://github.com/adsbansal/MMBCD/blob/14ac5e099c79253b01e0885d2ebefa6f86cfd8f0/code/data.py#L112-L221), [paper implementation details](https://papers.miccai.org/miccai-2024/paper/1311_paper.pdf)). The published training command uses `topk=8` and `img_size=224` ([training command](https://github.com/adsbansal/MMBCD/blob/14ac5e099c79253b01e0885d2ebefa6f86cfd8f0/models/mmbcd/train.sh)).

The missing link is the proposal-generation script. A plausible handoff is DINO's top scores plus its normalized `cxcywh` boxes, but that is an inference from the two repositories, not a documented fact. Before implementation, obtain either the authors' script or a known input plus its expected proposal file and reproduce it byte-for-byte/tolerance-for-tolerance.

The released loader also has edge cases that a service must define explicitly:

- `np.loadtxt` returns a one-dimensional array for a single row, but the NMS code expects rows.
- zero proposals make random duplication impossible;
- fewer than eight proposals cause nondeterministic random duplication;
- coordinates are not validated for finiteness, range, positive area, or crop validity;
- the loader assumes proposal ordering rather than sorting itself.

These cases need deterministic policies and tests. They should not be "fixed" in a way that changes normal-case outputs before a reference comparison exists.

### 2.5 MMBCD classifier input and output

For each proposal, MMBCD crops the whole image, resizes the crop to `224 x 224`, converts to RGB tensor, and applies ImageNet normalization. Eight crops therefore form an input shaped like `[batch, 8, 3, 224, 224]` ([MMBCD crop transform](https://github.com/adsbansal/MMBCD/blob/14ac5e099c79253b01e0885d2ebefa6f86cfd8f0/code/data.py#L143-L170)).

The released model:

- obtains a ViT-S/8 DINO image encoder through `torch.hub`;
- obtains `roberta-base` and its tokenizer from Hugging Face;
- projects image and text features to 256 dimensions;
- max-pools ROI image embeddings;
- uses the RoBERTa first-token embedding as the text representation;
- uses text as the query and ROI embeddings as key/value in one-head cross-attention;
- concatenates attention, text, and max-pooled image embeddings;
- returns two logits from a final linear layer.

See the [released MMBCD model](https://github.com/adsbansal/MMBCD/blob/14ac5e099c79253b01e0885d2ebefa6f86cfd8f0/code/model.py). The evaluation script tokenizes with padding/truncation and `max_length=90`, applies softmax, uses class index `1` as cancer probability, and predicts with argmax ([MMBCD evaluation](https://github.com/adsbansal/MMBCD/blob/14ac5e099c79253b01e0885d2ebefa6f86cfd8f0/code/test.py)).

Production must not depend on runtime internet. Pin and package the exact DINO architecture/revision, RoBERTa config/tokenizer assets, and all checkpoints, or mount them as versioned read-only artifacts. Loading arbitrary pickle checkpoints is unsafe; current PyTorch defaults `torch.load` to `weights_only=True` in modern versions, and the service should use state dictionaries plus explicit safe loading wherever the actual checkpoint format permits it ([PyTorch serialization source](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/serialization.py)).

### 2.6 Live text contract is unresolved

The research dataset's `create_valid_prompt` uses ground-truth `cancer` and `all_views_cancer` values to decide whether to emit `Indication: {text}` or blank the history for an unaffected breast. That is valid as a paper-specific train/evaluation transformation, but a live inference request cannot know the cancer label ([released prompt construction](https://github.com/adsbansal/MMBCD/blob/14ac5e099c79253b01e0885d2ebefa6f86cfd8f0/code/data.py#L89-L111)).

The provisional inference prompt should be `Indication: {normalized clinical_history}` with no target-derived fields. It must be compared with an author-provided inference example or known expected logits. The API must never ask a caller for `cancer` or `all_views_cancer` just to satisfy the research CSV format.

## 3. Artifact, compatibility, and licensing gates

Before product code, create an artifact manifest and answer the following:

| Artifact | Required evidence |
|---|---|
| FocalNet-DINO detector checkpoint | filename, byte size, SHA-256, source, checkpoint wrapper (`model` key or raw state dict), matching config hash, expected class mapping |
| FocalNet-L initialization checkpoint | exact `focalnet_large_lrf_384.pth` artifact or proof that a full strict detector state dict permits safely skipping the hardcoded pre-load |
| MMBCD classifier checkpoint | filename, byte size, SHA-256, source, state-dict key format, expected `topk`, image size, tokenizer/model revision |
| ViT-DINO dependency | exact code revision and whether the classifier checkpoint fully replaces pretrained weights |
| RoBERTa assets | exact model/tokenizer revision, vocab/merge/config hashes, offline loading proof |
| CUDA extension | compiler, PyTorch/torchvision, CUDA toolkit, driver, C++ ABI, GPU architecture list, build/test result |

The upstream MMBCD repository pins Python 3.10.13, PyTorch 2.1.2, torchvision 0.16.2, and CUDA 11.8 in its environment file ([MMBCD environment](https://github.com/adsbansal/MMBCD/blob/14ac5e099c79253b01e0885d2ebefa6f86cfd8f0/mmbcd.yml)). FocalNet-DINO is older and its custom CUDA extension is compiled against the local PyTorch/CUDA stack. A modern Python/PyTorch upgrade may be desirable for security and deployment, but compatibility must be proven on Lightning AI rather than presumed.

FocalNet-DINO carries Apache-2.0 via its repository. The inspected MMBCD repository has no top-level license file. Clarify permission to copy, modify, containerize, or redistribute MMBCD source and weights before embedding them in a public image. If rights are unclear, mount user-provided artifacts at runtime and avoid publishing them.

## 4. Serving architecture implications

### 4.1 Reconcile "load once" with "one model at a time"

The most useful interpretation is:

- parse artifact metadata and build reusable CPU-side model instances once per dedicated inference process;
- permit exactly one model to be resident on the GPU at a time;
- serialize the composite detector-to-classifier critical section;
- move or destroy GPU tensors at each transition, verify live/peak memory, and keep the request's small CPU intermediates (proposal boxes and crops) between stages.

If the evaluator instead requires destroying and reconstructing each Python model on every request, that should be stated explicitly because it materially changes latency and reliability.

`torch.cuda.empty_cache()` only releases **unoccupied cached** allocations and does not free memory held by live tensors ([PyTorch CUDA memory documentation](https://docs.pytorch.org/docs/main/generated/torch.cuda.memory.empty_cache.html)). Correct unload validation therefore requires deleting/moving every live module and tensor reference, synchronizing, optionally collecting Python garbage, releasing the allocator cache for the other model/process, and measuring both allocated and reserved memory.

### 4.2 One GPU owner, not one model per Django worker

A process-local singleton in Django is insufficient: multiple Gunicorn/Uvicorn workers would each have their own singleton, lock, CUDA context, and model copy. The robust boundary is:

- Django/API and frontend processes: parsing request envelopes, authentication if needed, job state, response formatting, and serving static assets;
- one dedicated GPU worker process per GPU: DICOM tensor pipeline, model-residency manager, and all CUDA calls;
- a bounded queue or explicit in-process RPC between them.

The residency manager should expose a small state machine such as `EMPTY -> LOADING_DETECTOR -> DETECTOR_READY -> UNLOADING -> LOADING_CLASSIFIER -> CLASSIFIER_READY -> UNLOADING`, with a failed state that prevents serving after a partial load. Health endpoints must distinguish process liveness from model/artifact readiness.

The composite inference operation must hold the GPU lease across both stages so another request cannot evict the detector/classifier between them. Queue wait, load, warmup, compute, postprocess, and unload time should be independently visible.

### 4.3 Backpressure and API shape

At minimum:

- `POST /api/v1/inferences` accepts one DICOM plus clinical history and creates/runs a composite inference;
- `GET /api/v1/inferences/{id}` is useful if jobs are asynchronous;
- `GET /health/live`, `GET /health/ready`, and `GET /api/v1/model-status` expose process, artifact, extension, GPU, and residency state without patient data;
- size/type limits are enforced before decoding;
- the service returns stable error codes for invalid DICOM, unsupported transfer syntax, empty/invalid proposals, queue full, timeout, artifact mismatch, CUDA OOM, and internal failure.

An asynchronous job API is preferable if measured model switching makes ordinary HTTP timeouts likely. If a synchronous endpoint is retained for assignment simplicity, it still needs a bounded semaphore/queue and a documented timeout.

## 5. GPU and acceleration plan

### 5.1 Baseline first

The baseline is eager PyTorch FP32 on the actual Lightning AI GPU with:

- `model.eval()` and `torch.inference_mode()`; inference mode reduces autograd/view/version-counter overhead but does **not** set evaluation mode ([PyTorch `inference_mode`](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/autograd/grad_mode.py));
- checkpoints mapped to CPU before controlled device transfer;
- no `DataParallel` for a single GPU;
- fixed, documented detector and classifier shapes;
- warmup before measurement;
- strict golden outputs from the original scripts.

### 5.2 Optimization ladder

Promote one change at a time:

1. **Serving cleanup:** disable activation checkpointing if parity passes; avoid repeated tokenizer/model downloads; reuse CPU-side state; preallocate stable buffers where useful.
2. **Mixed precision:** test CUDA autocast separately for each stage. The detector's custom deformable-attention module casts FP16 inputs back to FP32, and its CUDA source dispatches floating types rather than Half/BFloat16, so neither FP16 nor BF16 detector acceleration is a safe promise ([module path](https://github.com/FocalNet/FocalNet-DINO/blob/23901e021dc6ec8f66bad47983f45a25574452cc/models/dino/ops/modules/ms_deform_attn.py), [CUDA kernel](https://github.com/FocalNet/FocalNet-DINO/blob/23901e021dc6ec8f66bad47983f45a25574452cc/models/dino/ops/src/cuda/ms_deform_attn_cuda.cu#L60-L69)). Classifier AMP can be evaluated independently.
3. **`torch.compile`:** benchmark default and suitable modes after registering/isolating custom operators. PyTorch documents that `reduce-overhead` uses CUDA graphs where possible but can consume more memory and is not guaranteed to apply; custom operators need appropriate registration/fake kernels for compiler integration ([`torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html), [custom C++/CUDA operator guidance](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)). CUDA graph modes also require stable shapes and memory addresses; the aspect-preserving detector transform produces variable spatial shapes, so use validated shape buckets/static buffers or skip graph capture.
4. **ONNX Runtime CUDA:** export fixed-shape stage wrappers, validate with ONNX/checker and ONNX Runtime, inspect execution-provider assignment, and use I/O Binding so host/device copies are not accidentally counted as graph time ([ONNX Runtime execution providers](https://onnxruntime.ai/docs/execution-providers/), [I/O Binding](https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html)).
5. **TensorRT:** only after ONNX export and ORT CUDA parity. Use FP32 first, then FP16, engine/timing caches, explicit profiles for every dynamic input, and prove no unacceptable CPU fallback. ONNX Runtime's TensorRT provider can fall back by graph partition to CUDA/CPU, so a session that merely starts is not proof that TensorRT executed the whole graph ([ONNX Runtime TensorRT EP](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)).

### 5.3 TensorRT feasibility assessment

MMBCD is the more conventional export candidate if `topk=8`, crop size 224, and token length 90 are fixed. It still includes ViT, RoBERTa hidden-state output, multihead attention, batch normalization, and checkpoint key quirks, so export must be demonstrated.

FocalNet-DINO is higher risk because its core deformable attention invokes a compiled custom CUDA operation with no export registration in the inspected repository. Modern PyTorch recommends its `torch.export`-based ONNX exporter, but custom operators still need an ONNX translation or decomposition ([PyTorch ONNX exporter](https://docs.pytorch.org/docs/stable/onnx.html)). TensorRT then needs a supported ONNX graph or a maintained TensorRT plugin; unsupported nodes/plugins are explicit parser failures ([TensorRT ONNX parser troubleshooting](https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/troubleshooting-error-messages.html)).

Therefore, the spec should promise a **time-boxed TensorRT feasibility ticket and report**, not a successful detector engine. Success means an engine artifact tied to a TensorRT/CUDA/GPU compatibility manifest, numerical parity, provider/engine inspection, and repeatable performance improvement. Failure with a precise unsupported-operator report is still a good engineering result.

### 5.4 Why Triton is not the first serving layer

NVIDIA Triton offers dynamic batching, metrics, model repositories, and explicit model-control APIs ([Triton model management](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_management.html)). However, the assignment's strict one-model-at-a-time sequential residency is not a natural fit for an ensemble that keeps both stages loaded. Adding Triton before proving model export and switching behavior would introduce another control plane without resolving the core contract. Revisit it only if measured traffic and a stable ONNX/TensorRT backend justify the operational cost.

## 6. Benchmark and promotion methodology

### 6.1 Correctness before speed

Create immutable fixtures containing:

- source DICOM hash and selected non-identifying metadata;
- expected decoded/cropped/resized image hashes or tolerance summaries;
- expected detector logits/boxes/scores before and after postprocessing;
- expected top-eight proposal list and ROI crop tensors;
- expected tokenizer IDs/attention mask;
- expected MMBCD logits, softmax probabilities, label, and attention weights;
- final JSON and overlay coordinates in original image space.

Backend promotion gates should compare:

- tensor max/mean absolute and relative error;
- detector top-K agreement, box IoU, score/ranking agreement, and post-NMS proposal agreement;
- classifier logit/probability error and label agreement;
- end-to-end response/overlay equivalence;
- failure behavior on corrupt, compressed, single-proposal, zero-proposal, empty-text, and oversized inputs.

FP16/TF32/compiled/ONNX/TensorRT tolerances must be chosen from measured distributions and clinical-risk posture; no universal tolerance should be invented in the spec.

### 6.2 Performance dimensions

Record at least:

- cold process startup and artifact verification;
- cold detector load/warmup and cold classifier load/warmup;
- detector-to-classifier and classifier-to-detector switch latency;
- DICOM decode, VOI/windowing, crop/resize, host-to-device, detector compute, proposal postprocess, ROI creation, classifier compute, device-to-host, serialization, and unload durations;
- end-to-end p50/p95/p99 latency at concurrency 1 and the planned queue depth;
- sustained throughput, rejection rate, timeout rate, and queue wait;
- GPU allocated/reserved/peak memory per stage and after unload;
- host RAM, CPU utilization, GPU utilization, power/temperature/clocks, and artifact/cache sizes.

CUDA operations are asynchronous. Use CUDA events plus `torch.cuda.synchronize()` for isolated GPU timings, and wall-clock timing around the whole request for user-visible latency ([PyTorch CUDA semantics](https://docs.pytorch.org/docs/main/notes/cuda.html)). Warm up every backend and report warmup policy. NVIDIA's `trtexec` also separates warmup/iterations/duration and should be used for engine-only TensorRT measurements ([TensorRT benchmarking](https://docs.nvidia.com/deeplearning/tensorrt/11.1.0/performance/benchmarking.html)).

Every result must include Git commit, checkpoint/config/tokenizer hashes, backend and precision, input shape, batch/concurrency, GPU model and compute capability, driver, CUDA, cuDNN, PyTorch/torchvision, compiler, OS/container digest, power mode, and whether transfers/pre/postprocessing were included.

### 6.3 Public test data limitation

TCIA's CBIS-DDSM collection provides mammography DICOMs and pathology annotations and is a reasonable public functional fixture ([TCIA CBIS-DDSM](https://www.cancerimagingarchive.net/collection/cbis-ddsm/)). It is based on scanned-film DDSM images converted to DICOM, while MMBCD reports results on private AIIMS diagnostic/opportunistic digital-mammography cohorts ([MMBCD paper dataset section](https://papers.miccai.org/miccai-2024/paper/1311_paper.pdf)). Passing a CBIS-DDSM case therefore validates file handling and pipeline execution only. It does **not** reproduce the paper's accuracy, establish calibration on the public domain, or support clinical-use claims.

## 7. Observability and operational acceptance

Use a trace/span per inference with child spans for queue, decode, preprocessing, model load/warmup, detector, proposal postprocess, unload, classifier load/warmup, classifier, output mapping, and serialization. Follow OpenTelemetry HTTP semantic conventions for transport spans ([OpenTelemetry HTTP conventions](https://opentelemetry.io/docs/specs/semconv/http/)).

Recommended low-cardinality metrics:

- requests, successes, failures, rejections, timeouts by stable reason;
- queue depth and wait histogram;
- end-to-end and per-stage latency histograms;
- model transition count/duration/failure by model version;
- current residency state and readiness;
- allocated/reserved/peak GPU memory;
- proposal count before/after NMS and selected ROI count;
- backend/precision/version info as bounded labels.

Do not put patient ID, free text, filename, request ID, raw exception text, or DICOM UIDs in metric labels or ordinary logs. Keep per-request correlation in trace IDs and structured logs under an explicit retention policy. NVIDIA DCGM Exporter can expose GPU telemetry to Prometheus on supported Linux NVIDIA environments ([NVIDIA DCGM Exporter](https://docs.nvidia.com/datacenter/dcgm/latest/reference/command-line-reference/dcgm-exporter.html)).

Readiness should fail when artifacts/hashes mismatch, the CUDA extension is unavailable, the GPU cannot allocate the next required model, the residency manager is failed, or the queue cannot make progress. Liveness should remain simpler so orchestration does not restart a healthy-but-not-ready process in a loop.

## 8. Frontend that strengthens the engineering story

A compact frontend can expose the pipeline's verifiability rather than act as decoration:

- drag/drop DICOM with explicit validation feedback;
- clinical-history input with length/empty-state rules;
- safe, non-identifying DICOM metadata summary;
- rendered preprocessed image and ROI boxes mapped to original pixels;
- detector proposal score and MMBCD model-attention weight per selected ROI;
- final class probability, model/config/preprocessing version, and stage timings;
- a persistent statement that this is a research demonstration, not a clinical diagnosis.

The paper visualizes cross-attention over ROIs, so surfacing those weights is faithful to the model ([MMBCD paper, Figure 3](https://papers.miccai.org/miccai-2024/paper/1311_paper.pdf)). They should be labeled **model attention**, not a causal explanation or proof of lesion localization.

## 9. Technical questions that must become decisions or tickets

### P0: blocks faithful inference

1. What are the exact detector and classifier checkpoint files, hashes, sources, and state-dict wrappers?
2. Does `config_cfg.py` exactly match the detector checkpoint, including class ID convention and all inherited/default values?
3. Is the separate `focalnet_large_lrf_384.pth` initialization file available, or does the fine-tuned detector checkpoint have strict full coverage that permits a source change to skip it?
4. What code produced `*_preds.txt` from FocalNet-DINO outputs? Can the authors provide one golden image, proposal file, and final MMBCD logits?
5. Is live inference one image view plus clinical history, or must an API accept an entire bilateral study? If multi-view, what trained aggregation rule/checkpoint exists?
6. How should clinical history be handled for laterality without the ground-truth `cancer`/`all_views_cancer` columns used by the research dataset?
7. What exact positive-class mapping and decision threshold should the API use? Is the raw class-1 softmax sufficient, and is any calibration artifact available?
8. Which DICOM transfer syntaxes, frames, VOI alternatives, pixel padding, photometric interpretations, and mammography views are in scope?
9. Does "load/unload one model at a time" mean one GPU-resident model while both CPU objects remain initialized, or full destruction/reconstruction for every stage/request?

### P1: determines production design

10. What is the target GPU, VRAM, driver/CUDA baseline, deployment OS, and expected request rate/concurrency/latency SLO?
11. May model/code/tokenizer artifacts be redistributed inside the Docker image, or must they be mounted/downloaded under separate terms?
12. What are request size, text length, retention, authentication, and PHI/logging requirements for this assessment?
13. Is a synchronous response required, or is an asynchronous job API acceptable?
14. What is the failure policy when the detector produces zero or fewer than eight valid proposals?
15. Must the output include original-pixel boxes, detector confidence, ROI attention weights, raw logits, probability, class, and per-stage timings?

### P2: optional optimization/product choices

16. Which optimization target matters: cold latency, warm latency, throughput, VRAM, startup time, or image size?
17. Is TensorRT success required for both stages, or is a documented feasibility result/one accelerated stage sufficient?
18. Is the frontend intended only for demonstration, or must it support job history, comparison, and export?
19. Is Triton/ONNX Runtime operational complexity justified by measured traffic after the eager baseline?

## 10. Recommended ticket sequence derived from the evidence

1. **Artifact and license intake gate** — obtain files, hashes, provenance, permission, and offline assets.
2. **Golden reference notebook/CLI** — reproduce upstream DICOM-to-proposals-to-logits on Lightning AI before Django.
3. **DICOM contract and fixture suite** — implement reference-compatible decoding plus explicit unsupported/error cases.
4. **Deterministic proposal adapter** — prove exact DINO-to-MMBCD coordinate/score/NMS/top-K behavior.
5. **Model residency manager spike** — measure load/switch/unload memory and latency on target GPU; decide CPU caching and worker topology.
6. **Composite inference core** — framework-independent typed pipeline and versioned response schema.
7. **Django API and bounded execution** — validation, job lifecycle or synchronous semaphore, health/readiness, stable errors.
8. **Container and GPU build** — multi-stage/reproducible image, compiled extension test, read-only artifacts, non-root service where feasible.
9. **Correctness/performance harness** — eager FP32 baseline, golden parity, cold/warm/switch/concurrency measurements.
10. **Observability** — traces, metrics, structured redacted logs, GPU telemetry, basic dashboard/alerts.
11. **Frontend visualization** — upload/history form, stage state, overlay, attention, model metadata, disclaimer.
12. **Optimization ladder** — activation-checkpoint removal, AMP, compile, then ONNX Runtime/TensorRT feasibility with promotion gates.

## 11. Evidence boundaries at the time of research

- No checkpoint was present in this checkout at the time, so this research run
  loaded no model and validated no numerical output, VRAM use, latency,
  compilation compatibility, ONNX export, or TensorRT result. Later
  revision-bound L4 evidence is indexed in `docs/traceability.md`.
- File presence and source inspection establish the intended pipeline, not that the released checkpoints are compatible with a modern runtime.
- The exact detector proposal generation path remains an inference until a golden artifact or missing author script is obtained.
- The MMBCD paper's reported metrics are results on private in-house datasets; they must not be presented as reproduced or as clinical validation of this service.
- TensorRT, `torch.compile`, AMP, and CUDA graph recommendations are experiments to benchmark, not guaranteed improvements.

## Primary sources consulted

- [Assignment-linked MMBCD paper (MICCAI 2024)](https://papers.miccai.org/miccai-2024/paper/1311_paper.pdf)
- [Official MMBCD repository, inspected at commit `14ac5e0`](https://github.com/adsbansal/MMBCD/tree/14ac5e099c79253b01e0885d2ebefa6f86cfd8f0)
- [Official FocalNet-DINO repository, inspected at commit `23901e0`](https://github.com/FocalNet/FocalNet-DINO/tree/23901e021dc6ec8f66bad47983f45a25574452cc)
- [Official DINO repository](https://github.com/IDEA-Research/DINO)
- [PyTorch documentation/source](https://docs.pytorch.org/docs/stable/)
- [pydicom documentation](https://pydicom.github.io/pydicom/stable/)
- [DICOM Standard](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/)
- [ONNX Runtime documentation](https://onnxruntime.ai/docs/)
- [NVIDIA TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/)
- [NVIDIA Triton Inference Server documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/)
- [OpenTelemetry semantic conventions](https://opentelemetry.io/docs/specs/semconv/)
- [NVIDIA DCGM documentation](https://docs.nvidia.com/datacenter/dcgm/latest/)
- [TCIA CBIS-DDSM collection](https://www.cancerimagingarchive.net/collection/cbis-ddsm/)
