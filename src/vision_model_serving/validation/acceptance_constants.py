"""Dependency-free identities shared by packaged validation lanes."""

from typing import Final


PACKAGED_ACCEPTANCE_HISTORY: Final = "real public mammogram acceptance."
PUBLIC_DICOM_SHA256: Final = "9f70081672a460f29231bb471e8a9e26dd3ed26a2ebbd91c064e575e7842a19c"
PUBLIC_CANONICAL_ARRAY_SHA256: Final = (
    "97fa0f80a696ce7f822c1681a8c3f7c072da9262b2bd91239c9f1637eaf68552"
)
SERVED_DETECTOR_OUTPUT_SHA256: Final = (
    "4cdd09d986702e8839acff8d7517a63f263ca2a01b0607d78d6b2086c886a9a5"
)
# The archive used an empty history. Packaged requests use the pinned history above,
# so these two classifier identities are intentionally different contracts.
ARCHIVED_CLASSIFIER_OUTPUT_SHA256: Final = (
    "43ec1c4593c0549510098ea082ea7092c7fd5631c95d8b912ecf31633185899b"
)
SERVED_CLASSIFIER_OUTPUT_SHA256: Final = (
    "f994ccfad2e1894f95b487cf1068b5c0038b4bb12c7d49f5e0dc396afc83f1a3"
)
PACKAGED_MANIFEST_ID: Final = "vision-model-serving-l4-fp32-20260807"
PACKAGED_MANIFEST_SHA256: Final = "9d949a0a7b8c64fce7109bfb7176b2986c1a895ee5fc5b037792679c0decff4f"
PACKAGED_DISCLAIMER: Final = "Research use only; not a medical diagnosis."
TOKENIZER_REVISION: Final = "e2da8e2f811d1448a5b465c236feacd80ffbac7b"
