# PyTorch optimization matrix

Revision: `204cff209fbf88f0aff3cb932274d1426850a682`

| Model | Candidate | Status | Parity | Warm p50 ms | Peak reserved bytes | Accepted |
| --- | --- | --- | --- | ---: | ---: | --- |
| focalnet-dino-detector | fp32 | passed | true | 232.953 | 1530920960 | false |
| focalnet-dino-detector | tf32 | rejected | false | 151.273 | 1530920960 | false |
| focalnet-dino-detector | fp16 | rejected | false | 121.432 | 2040528896 | false |
| focalnet-dino-detector | bf16 | failed | false | n/a | n/a | false |
| focalnet-dino-detector | compile | failed | false | n/a | n/a | false |
| mmbcd-classifier | fp32 | passed | true | 92.669 | 994050048 | false |
| mmbcd-classifier | tf32 | rejected | false | 74.119 | 994050048 | false |
| mmbcd-classifier | fp16 | rejected | false | 64.738 | 1277165568 | false |
| mmbcd-classifier | bf16 | rejected | false | 82.933 | 1335885824 | false |
| mmbcd-classifier | compile | failed | false | n/a | n/a | false |

The production runtime remains eager FP32 until a candidate also passes the packaged benchmark and restart gates.

Optimization screening on one checksum-pinned public DICOM and one NVIDIA L4. A candidate is not selected for serving until the packaged single-residency benchmark also passes parity and reliability gates.
