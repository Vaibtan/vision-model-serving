# Spec findings resolution on NVIDIA L4

This evidence closed five implementation gaps for the exact revisions below.
The implementation has since changed, including the optimization acceptance
contract and a Pillow security patch, so current-revision L4 acceptance is
pending a rerun. This record does not claim that pending validation is complete,
nor medical accuracy, calibration, robustness, clinical utility, or multi-GPU
scale.

## Revision boundary

- Packaged serving, browser, and restart acceptance ran from clean revision
  `f565d2fb56076091a1b560cf08edd54c81a742be`.
- The isolated residency, PyTorch optimization, and TensorRT reports ran from
  clean revision `204cff209fbf88f0aff3cb932274d1426850a682`. That child revision changes
  only acceleration-report wording/sanitization and their regression tests;
  production serving modules, Compose policy, and dependency pins are unchanged
  from the packaged revision.
- The evidence files are committed after the revisions they measure. Their
  embedded revision fields, rather than the evidence commit, define the tested
  source boundary.

## Resolution matrix

| Finding | Result | Evidence |
| --- | --- | --- |
| Dual model residency | Resolved. The only deployed runtime unloads before switching and every recorded inventory contains at most one model. The isolated two-cycle run ended with 4 loads, 3 switches, 3 unloads, and 0 failures. | [Single-residency report](single-residency-l4-20260810.json), [schema-v3 residency gate](benchmark-l4-20260810.md), and [restart report](compose-restart-l4-20260810.json) |
| Ambiguous unloaded readiness | Resolved. `/readyz` is explicitly artifact-scoped; runtime initialization is a readiness fact while model-specific `inference_warm` is reported separately. Cold unloaded startup passed artifact readiness before the first load. | [Schema-v3 benchmark](benchmark-l4-20260810.json) and [environment report](environment-l4-20260810.json) |
| Missing benchmark evidence | Resolved for one L4 and one public fixture. The schema-v3 run recorded full identity, startup/lifecycle/stage distributions, p50/p95/p99/mean/stddev, failure accounting, concurrency 1/2/4, NVIDIA utilization/memory/power/temperature sampling, and OOM state. All 140 throughput attempts succeeded. | [Benchmark JSON](benchmark-l4-20260810.json) and [Markdown summary](benchmark-l4-20260810.md) |
| Missing optimization/TensorRT result | The prior run measured a TensorRT STOP and eager FP32 remains selected, but optimization acceptance must be rerun under the schema-v3 raw/post-NMS/selected-ROI parity, candidate-release, and repeated-switch gates. The classifier static TensorRT diagnostic had zero PyTorch partitions, passed a PyTorch-free runtime and parity, and improved warm p50 by 27.8%, but the required token-width 2..90 profile failed strict export. Detector strict export failed earlier in upstream `NestedTensor` mask handling, before plugin analysis. The static plan was deleted. | [Historical PyTorch matrix](pytorch-optimization-l4-20260810.md), [TensorRT report](tensorrt-l4-20260810/tensorrt-spike.md), and [failure analysis](tensorrt-l4-20260810/failure-analysis.md) |
| No browser-level workbench test | Resolved. Packaged Chromium executed upload, async polling, exact full prediction, eight overlays/crops, ROI attention selection, privacy-safe JSON export, PNG export, and zero console errors. | [Executable browser test](../../scripts/validate_browser_workbench.py) and [exact-revision Compose transcript](browser-workbench-l4-20260810.log) |

## Packaged benchmark result

| Offered concurrency | Attempts | Successes | Failures | p50 seconds | p95 seconds | p99 seconds | Successful requests/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 20 | 20 | 0 | 3.065 | 3.163 | 3.238 | 0.323 |
| 2 | 40 | 40 | 0 | 3.175 | 3.304 | 4.703 | 0.614 |
| 4 | 80 | 80 | 0 | 5.491 | 5.671 | 7.454 | 0.715 |

The benchmark collected 3,462 `nvidia-smi` samples, observed zero CUDA OOMs,
matched both golden output hashes, and ended with only the classifier resident.
The destructive restart comparison then matched its baseline and found no
request content in Redis or the job volume after completion.

## TensorRT decision

TensorRT is implemented as one strict, no-fallback experimental lane. The
static classifier result proves feasibility for the pinned fixture's token
width only. It is not a deployable engine because production admits token
widths 2 through 90. The detector requires a genuine TensorRT plugin or a
separately proven full-graph decomposition for `MultiScaleDeformableAttention`,
after first correcting the earlier measured `NestedTensor` strict-capture
failure. The [failure analysis](tensorrt-l4-20260810/failure-analysis.md)
records the distinction and includes report-ready wording.
No plan file is published or selected by the executor.

The browser transcript SHA-256 is
`a29470a2c9f38ab48d7834326eded76909f6ff873f1e00c5f371d122236115b9`.
