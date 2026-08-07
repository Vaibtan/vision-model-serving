#!/usr/bin/env python3
"""Strict-load MMBCD from pinned local architecture code without network access."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import (
    DINO_COMMIT,
    MMBCD_SHA256,
    default_paths,
    strip_module_prefix,
    verify_git_commit,
    verify_sha256,
)
from _mmbcd_model import build_mmbcd, verify_alias_values


def main() -> None:
    defaults = default_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dino-repo", type=Path, default=defaults["dino_repo"])
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=defaults["artifact_dir"] / "mmbcd_best.pt",
    )
    args = parser.parse_args()

    import torch
    import transformers

    verify_git_commit(args.dino_repo, DINO_COMMIT)
    verify_sha256(args.checkpoint, MMBCD_SHA256)
    model = build_mmbcd(args.dino_repo)
    raw_state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state_dict = strip_module_prefix(raw_state)
    verify_alias_values(state_dict)
    result = model.load_state_dict(state_dict, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    assert len(model.state_dict()) == 375
    print("PyTorch:", torch.__version__)
    print("Transformers:", transformers.__version__)
    print("Model state keys:", len(model.state_dict()))
    print("Checkpoint keys:", len(state_dict))
    print("Strict load:", result)
    print("MMBCD STRICT LOAD PASSED")


if __name__ == "__main__":
    main()
