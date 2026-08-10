# TensorRT spike

Revision: `204cff209fbf88f0aff3cb932274d1426850a682`
Decision: **STOP**

| Model | Strict export | Full compilation | Engine | Parity | Performance gate | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| focalnet-dino-detector | False | False | False | False | False | stop |
| mmbcd-classifier | True | True | True | True | True | stop |

TensorRT was not promoted because the classifier's required token-width 2..90 profile did not pass and detector full coverage did not pass. Production remains eager FP32 and no fallback is enabled.

FP32/TF32-disabled TensorRT feasibility with a static token-width-5 diagnostic engine and a separately gated 2..90 production profile on one NVIDIA L4 and one public DICOM. It is not clinical or cross-hardware evidence.
