#!/usr/bin/env python3
"""Strip training-only state from the detector checkpoint using safe loading."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from _common import (
    FOCALNET_COMMIT,
    FOCALNET_INFERENCE_SHA256,
    FOCALNET_TRAINING_SHA256,
    default_paths,
    require_file,
    sha256_file,
    verify_sha256,
)


def main() -> None:
    defaults = default_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=defaults["artifact_dir"] / "focalnet-dino-finetuned.pth",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=defaults["artifact_dir"] / "focalnet-dino-inference.pth",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    import torch

    source = require_file(args.source)
    destination = args.destination.expanduser().resolve()
    verify_sha256(source, FOCALNET_TRAINING_SHA256)

    if destination.exists() and not args.overwrite:
        verify_sha256(destination, FOCALNET_INFERENCE_SHA256)
        print("Existing inference checkpoint verified:", destination)
        print("FOCALNET INFERENCE CHECKPOINT PASSED")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        raise RuntimeError(f"Temporary destination already exists: {temporary}")

    with torch.serialization.safe_globals([argparse.Namespace]):
        checkpoint = torch.load(source, map_location="cpu", weights_only=True)

    state_dict = checkpoint["model"]
    payload = {
        "model": state_dict,
        "metadata": {
            "source_checkpoint_sha256": FOCALNET_TRAINING_SHA256,
            "focalnet_dino_commit": FOCALNET_COMMIT,
            "num_state_tensors": len(state_dict),
            "purpose": "inference-only",
        },
    }
    torch.save(payload, temporary)
    os.replace(temporary, destination)

    output_hash = sha256_file(destination)
    print("Created:", destination)
    print("State tensors:", len(state_dict))
    print("Size bytes:", destination.stat().st_size)
    print("SHA256:", output_hash)
    if output_hash != FOCALNET_INFERENCE_SHA256:
        raise RuntimeError(
            "Inference checkpoint differs from the validated L4 artifact: "
            f"expected {FOCALNET_INFERENCE_SHA256}, observed {output_hash}"
        )
    print("FOCALNET INFERENCE CHECKPOINT PASSED")


if __name__ == "__main__":
    main()
