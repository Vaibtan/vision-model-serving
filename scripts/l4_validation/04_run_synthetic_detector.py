#!/usr/bin/env python3
"""Run the pre-DICOM structural detector forward gate on the L4."""

from __future__ import annotations

import argparse
import gc
import runpy
import sys
from pathlib import Path

from _common import (
    FOCALNET_COMMIT,
    FOCALNET_INFERENCE_SHA256,
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
        "--config", type=Path, default=defaults["artifact_dir"] / "config_cfg.py"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=defaults["artifact_dir"] / "focalnet-dino-inference.pth",
    )
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    repo = args.repo.resolve()
    verify_git_commit(repo, FOCALNET_COMMIT)
    config_path = require_file(args.config)
    checkpoint_path = require_file(args.checkpoint)
    verify_sha256(checkpoint_path, FOCALNET_INFERENCE_SHA256)
    sys.path[:0] = [str(repo), str(repo / "models" / "dino" / "ops")]

    config = {
        key: value
        for key, value in runpy.run_path(str(config_path)).items()
        if not key.startswith("__")
    }
    config["device"] = "cpu"
    config["use_checkpoint"] = False

    from models.dino import build_dino

    model, _, postprocessors = build_dino(argparse.Namespace(**config))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    load_result = model.load_state_dict(checkpoint["model"], strict=True)
    assert not load_result.missing_keys and not load_result.unexpected_keys
    del checkpoint
    gc.collect()

    device = torch.device("cuda:0")
    model.eval().to(device)
    torch.manual_seed(20260805)
    torch.cuda.manual_seed_all(20260805)
    torch.backends.cudnn.benchmark = False

    height, width = 800, 1024
    image = torch.randn(3, height, width, dtype=torch.float32, device=device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        output = model([image])
        torch.cuda.synchronize()
        timings_ms = []
        for _ in range(3):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = model([image])
            end.record()
            torch.cuda.synchronize()
            timings_ms.append(start.elapsed_time(end))

        target_sizes = torch.tensor([[height, width]], dtype=torch.float32, device=device)
        result = postprocessors["bbox"](output, target_sizes)[0]

    assert output["pred_logits"].shape == (1, 900, 1)
    assert output["pred_boxes"].shape == (1, 900, 4)
    assert result["scores"].shape == (300,)
    assert result["boxes"].shape == (300, 4)
    for tensor in (output["pred_logits"], output["pred_boxes"], result["scores"], result["boxes"]):
        assert torch.isfinite(tensor).all()

    print("Strict load:", load_result)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Input shape:", tuple(image.shape))
    print("pred_logits:", tuple(output["pred_logits"].shape))
    print("pred_boxes:", tuple(output["pred_boxes"].shape))
    print("selected scores:", tuple(result["scores"].shape))
    print("selected boxes:", tuple(result["boxes"].shape))
    print("FP32 timings ms:", [round(value, 3) for value in timings_ms])
    print("Mean FP32 latency ms:", round(sum(timings_ms) / len(timings_ms), 3))
    print("Peak allocated GiB:", round(torch.cuda.max_memory_allocated() / 1024**3, 3))
    print("SYNTHETIC DETECTOR INFERENCE PASSED")


if __name__ == "__main__":
    main()
