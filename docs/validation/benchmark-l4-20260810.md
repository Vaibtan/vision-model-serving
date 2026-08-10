# Packaged NVIDIA L4 benchmark

Revision: `f565d2fb56076091a1b560cf08edd54c81a742be`<br>
Measured at: `2026-08-10T06:39:49.602113+00:00`<br>
Outcome: **PASSED**

## Throughput and failure accounting

| Offered load | Attempts | Successes | Failures | p50 s | p95 s | p99 s | successful requests/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Concurrency 1 | 20 | 20 | 0 | 3.065349 | 3.162522 | 3.237874 | 0.323484 |
| Concurrency 2 | 40 | 40 | 0 | 3.175423 | 3.304438 | 4.702661 | 0.613730 |
| Concurrency 4 | 80 | 80 | 0 | 5.490958 | 5.670687 | 7.453503 | 0.715322 |

## Promotion gates

| Gate | Result |
| --- | --- |
| environment complete | PASS |
| revision exact clean | PASS |
| identity exact | PASS |
| artifact ready | PASS |
| single residency all snapshots | PASS |
| golden outputs all successes | PASS |
| measurement matrix complete | PASS |
| failure accounting complete | PASS |
| concurrency 1 no failures | PASS |
| each concurrency has success | PASS |
| no cuda oom | PASS |
| resource sampling complete | PASS |

The `single residency all snapshots` gate proves that no recorded runtime
inventory contained more than one accelerator-resident model.

## Validation boundary

One serialized NVIDIA L4 executor, one checksum-pinned public Secondary Capture DICOM, pinned FP32 artifacts, and offered HTTP concurrency 1/2/4. This is not evidence of accuracy, calibration, robustness, clinical performance, or multi-GPU scaling.
