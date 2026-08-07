"""Compatibility-lane inspection and preparation interfaces."""

from .environment import (
    EnvironmentGateResult,
    EnvironmentSnapshot,
    L4EnvironmentSpec,
    evaluate_environment,
    load_environment_spec,
)
from .focalnet import PatchCheckError, PatchResult, prepare_focalnet_patches

__all__ = [
    "EnvironmentGateResult",
    "EnvironmentSnapshot",
    "L4EnvironmentSpec",
    "PatchCheckError",
    "PatchResult",
    "evaluate_environment",
    "load_environment_spec",
    "prepare_focalnet_patches",
]
