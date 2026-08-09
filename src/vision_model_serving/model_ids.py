"""Canonical identities for the fixed two-model serving contract."""

from __future__ import annotations

from types import MappingProxyType
from typing import Final, Mapping


DETECTOR_MODEL_ID: Final = "focalnet-dino-detector"
CLASSIFIER_MODEL_ID: Final = "mmbcd-classifier"
MODEL_IDS: Final = (DETECTOR_MODEL_ID, CLASSIFIER_MODEL_ID)
MODEL_STAGE_BY_ID: Final[Mapping[str, str]] = MappingProxyType(
    {
        DETECTOR_MODEL_ID: "detector",
        CLASSIFIER_MODEL_ID: "classifier",
    }
)
