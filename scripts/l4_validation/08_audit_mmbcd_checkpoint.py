#!/usr/bin/env python3
"""Safely inspect the MMBCD checkpoint without constructing network-backed models."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

from _common import MMBCD_SHA256, default_paths, verify_sha256, write_json_atomic


def main() -> None:
    defaults = default_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=defaults["artifact_dir"] / "mmbcd_best.pt",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import torch

    checkpoint_path = args.checkpoint.expanduser().resolve()
    verify_sha256(checkpoint_path, MMBCD_SHA256)
    audit_path = (
        args.output.expanduser().resolve()
        if args.output
        else checkpoint_path.parent / "mmbcd-checkpoint-audit.json"
    )

    def is_state_dict(value) -> bool:
        return (
            isinstance(value, Mapping)
            and len(value) > 0
            and all(isinstance(key, str) for key in value)
            and all(torch.is_tensor(tensor) for tensor in value.values())
        )

    obj = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if is_state_dict(obj):
        state_dict, container = obj, "raw_state_dict"
    elif isinstance(obj, Mapping):
        state_dict = None
        container = None
        for candidate in ("state_dict", "model_state_dict", "model"):
            if candidate in obj and is_state_dict(obj[candidate]):
                state_dict, container = obj[candidate], candidate
                break
        if state_dict is None:
            raise RuntimeError(
                f"No recognized tensor state dict. Keys: {list(obj.keys())[:30]}"
            )
    else:
        raise RuntimeError(f"Unsupported checkpoint type: {type(obj)!r}")

    canonical = {}
    for key, tensor in state_dict.items():
        new_key = key[7:] if key.startswith("module.") else key
        if new_key in canonical:
            raise RuntimeError(f"Prefix stripping collision: {new_key}")
        canonical[new_key] = tensor

    required_groups = {
        "image_encoder": "image_encoder.",
        "image_projection": "img_fc_layer.",
        "text_encoder": "text_encoder.",
        "text_projection": "txt_fc_layer.",
        "cross_attention": "attention.",
        "classifier": "model_fc2.",
    }
    group_counts = {
        name: sum(key.startswith(prefix) for key in canonical)
        for name, prefix in required_groups.items()
    }
    missing_groups = [name for name, count in group_counts.items() if count == 0]
    prefix_counts = Counter(key.split(".", 1)[0] for key in canonical)
    dtype_counts = Counter(str(tensor.dtype) for tensor in canonical.values())
    tensor_elements = sum(tensor.numel() for tensor in canonical.values())
    tensor_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in canonical.values()
    )
    audit = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": MMBCD_SHA256,
        "container": container,
        "original_key_count": len(state_dict),
        "canonical_key_count": len(canonical),
        "module_prefixed_keys": sum(
            key.startswith("module.") for key in state_dict
        ),
        "state_tensor_elements_including_aliases": tensor_elements,
        "tensor_bytes": tensor_bytes,
        "tensor_mib": tensor_bytes / (1024**2),
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "top_level_prefix_counts": dict(sorted(prefix_counts.items())),
        "required_group_counts": group_counts,
        "missing_required_groups": missing_groups,
        "first_keys": list(canonical)[:15],
        "last_keys": list(canonical)[-15:],
    }
    write_json_atomic(audit_path, audit)
    import json

    print(json.dumps(audit, indent=2, sort_keys=True))
    if missing_groups:
        raise RuntimeError(f"Missing MMBCD groups: {missing_groups}")
    assert len(canonical) == 375
    print("MMBCD CHECKPOINT STRUCTURE PASSED")
    print("Audit:", audit_path)


if __name__ == "__main__":
    main()
