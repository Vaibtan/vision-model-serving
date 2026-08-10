#!/usr/bin/env python3
"""Construct and strict-load the complete FocalNet-DINO detector checkpoint."""

from __future__ import annotations

import argparse
import gc
import runpy
import sys
from pathlib import Path
from _common import (
    FOCALNET_COMMIT,
    FOCALNET_TRAINING_SHA256,
    default_paths,
    require_file,
    verify_git_commit,
    verify_sha256,
)


def main() -> None:
    defaults = default_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=defaults["focalnet_repo"])
    parser.add_argument(
        "--config",
        type=Path,
        default=defaults["project_repo"] / "config_cfg.py",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=defaults["artifact_dir"] / "focalnet-dino-finetuned.pth",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    repo = args.repo.resolve()
    config_path = require_file(args.config)
    checkpoint_path = require_file(args.checkpoint)
    verify_git_commit(repo, FOCALNET_COMMIT)
    verify_sha256(checkpoint_path, FOCALNET_TRAINING_SHA256)

    backbone_source = repo / "models" / "dino" / "backbone.py"
    if "checkpoint = {}" not in backbone_source.read_text(encoding="utf-8"):
        raise RuntimeError(
            "Serving preload patch is missing; apply patches/"
            "focalnet-serving-no-backbone-preload.patch"
        )
    sys.path[:0] = [str(repo), str(repo / "models" / "dino" / "ops")]
    import torch
    from models.dino import build_dino

    raw_config = runpy.run_path(str(config_path))
    config = {key: value for key, value in raw_config.items() if not key.startswith("__")}
    config["device"] = "cpu"
    config["use_checkpoint"] = False
    print("Constructing detector...")
    model, _, _ = build_dino(argparse.Namespace(**config))
    unsafe_globals = torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint_path)
    print("Checkpoint unsafe globals:", unsafe_globals)
    with torch.serialization.safe_globals([argparse.Namespace]):
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    state_dict = checkpoint["model"]
    load_result = model.load_state_dict(state_dict, strict=True)
    print("Strict load:", load_result)
    print("Checkpoint tensors:", len(state_dict))
    print("Model parameters:", f"{sum(p.numel() for p in model.parameters()):,}")
    assert not load_result.missing_keys
    assert not load_result.unexpected_keys
    del state_dict
    del checkpoint
    gc.collect()
    model.eval()

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model.to("cuda")
        torch.cuda.synchronize()
        print("GPU:", torch.cuda.get_device_name(0))
        print("Allocated GiB:", round(torch.cuda.memory_allocated() / 1024**3, 3))
        print("Peak allocated GiB:", round(torch.cuda.max_memory_allocated() / 1024**3, 3))
    print("FOCALNET STRICT LOAD PASSED")


if __name__ == "__main__":
    main()
