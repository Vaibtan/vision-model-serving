#!/usr/bin/env python3
"""Run deterministic FP32 FocalNet-DINO inference on the golden DICOM image."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import random
import runpy
import statistics
import sys
from pathlib import Path

from _common import (
    DETECTOR_PREDICTION_SHA256,
    FOCALNET_COMMIT,
    FOCALNET_INFERENCE_SHA256,
    default_paths,
    load_json,
    require_file,
    sha256_file,
    verify_git_commit,
    verify_sha256,
    write_json_atomic,
)


def calculate_iou(box1, box2) -> float:
    cx1, cy1, width1, height1, _ = map(float, box1)
    cx2, cy2, width2, height2, _ = map(float, box2)
    x1, y1 = cx1 - width1 / 2.0, cy1 - height1 / 2.0
    x2, y2 = cx2 - width2 / 2.0, cy2 - height2 / 2.0
    intersection_x, intersection_y = max(x1, x2), max(y1, y2)
    intersection_width = max(0.0, min(x1 + width1, x2 + width2) - intersection_x)
    intersection_height = max(
        0.0, min(y1 + height1, y2 + height2) - intersection_y
    )
    intersection = intersection_width * intersection_height
    union = width1 * height1 + width2 * height2 - intersection
    return 0.0 if union <= 0 else intersection / union


def upstream_mmbcd_nms(boxes, threshold: float):
    boxes_copy = boxes.copy()
    while True:
        selected_indices: list[int] = []
        removed_indices: list[int] = []
        for index in range(len(boxes_copy)):
            if index in selected_indices or index in removed_indices:
                continue
            selected_indices.append(index)
            for candidate in range(index + 1, len(boxes_copy)):
                if candidate in selected_indices or candidate in removed_indices:
                    continue
                if calculate_iou(boxes_copy[index], boxes_copy[candidate]) > threshold:
                    removed_indices.append(candidate)
        selected_indices = sorted(selected_indices)
        if len(selected_indices) == len(boxes_copy):
            return boxes_copy
        boxes_copy = boxes_copy[selected_indices]


def save_text_atomic(path: Path, array) -> None:
    import numpy as np

    temporary = path.with_suffix(path.suffix + ".tmp")
    np.savetxt(temporary, array, fmt="%.18e")
    temporary.replace(path)


def save_image_atomic(image, path: Path) -> None:
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    image.save(temporary, format="PNG")
    temporary.replace(path)


def sha256_predictions(logits, boxes) -> str:
    digest = hashlib.sha256()
    for tensor in (logits, boxes):
        array = tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
        digest.update(array.tobytes())
    return digest.hexdigest()


def draw_boxes(image, boxes, *, coordinate_space: str, geometry: dict):
    from PIL import ImageDraw

    rendered = image.convert("RGB")
    draw = ImageDraw.Draw(rendered)
    colors = [
        "#ff3b30",
        "#ff9500",
        "#ffcc00",
        "#34c759",
        "#00c7be",
        "#007aff",
        "#5856d6",
        "#af52de",
    ]
    image_width, image_height = rendered.size
    scale_x, scale_y = geometry["resize"]["scale_x"], geometry["resize"]["scale_y"]
    crop_x0, crop_y0, _, _ = geometry["crop"]["crop_box_xyxy_original"]

    for index, row in enumerate(boxes):
        cx, cy, width, height, score = map(float, row)
        x0_1024 = (cx - width / 2.0) * 1024.0
        y0_1024 = (cy - height / 2.0) * 1024.0
        x1_1024 = (cx + width / 2.0) * 1024.0
        y1_1024 = (cy + height / 2.0) * 1024.0
        if coordinate_space == "original":
            x0, y0 = x0_1024 / scale_x + crop_x0, y0_1024 / scale_y + crop_y0
            x1, y1 = x1_1024 / scale_x + crop_x0, y1_1024 / scale_y + crop_y0
            line_width = 12
        else:
            x0, y0, x1, y1 = x0_1024, y0_1024, x1_1024, y1_1024
            line_width = 3
        x0 = max(0.0, min(x0, image_width - 1.0))
        y0 = max(0.0, min(y0, image_height - 1.0))
        x1 = max(0.0, min(x1, image_width - 1.0))
        y1 = max(0.0, min(y1, image_height - 1.0))
        color = colors[index % len(colors)]
        draw.rectangle((x0, y0, x1, y1), outline=color, width=line_width)
        if coordinate_space == "1024":
            draw.text(
                (x0 + 3, max(0, y0 - 12)),
                f"{index + 1}: {score:.4f}",
                fill=color,
            )
    return rendered


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
    parser.add_argument("--preprocess-dir", type=Path, default=defaults["preprocess_dir"])
    parser.add_argument("--output-dir", type=Path, default=defaults["detector_dir"])
    parser.add_argument("--benchmark-runs", type=int, default=5)
    parser.add_argument("--expected-prediction-sha256", default=DETECTOR_PREDICTION_SHA256)
    args = parser.parse_args()

    import numpy as np
    import torch
    from PIL import Image

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    repo = args.repo.expanduser().resolve()
    verify_git_commit(repo, FOCALNET_COMMIT)
    config_path = require_file(args.config)
    checkpoint_path = require_file(args.checkpoint)
    verify_sha256(checkpoint_path, FOCALNET_INFERENCE_SHA256)
    preprocess_dir = args.preprocess_dir.expanduser().resolve()
    image_path = require_file(preprocess_dir / "upstream-1024.png")
    geometry_path = require_file(preprocess_dir / "preprocess-manifest.json")
    normalized_source_path = require_file(preprocess_dir / "upstream-normalized.png")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sys.path[:0] = [str(repo), str(repo / "models" / "dino" / "ops")]
    import MultiScaleDeformableAttention as extension
    from models.dino import build_dino

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    transform_spec = importlib.util.spec_from_file_location(
        "focalnet_transforms", repo / "datasets" / "transforms.py"
    )
    transforms = importlib.util.module_from_spec(transform_spec)
    assert transform_spec.loader is not None
    transform_spec.loader.exec_module(transforms)
    transform = transforms.Compose(
        [
            transforms.RandomResize([800], max_size=1333),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ]
    )

    input_image = Image.open(image_path).convert("RGB")
    assert input_image.size == (1024, 1024)
    input_tensor, _ = transform(input_image, None)
    assert tuple(input_tensor.shape) == (3, 800, 800)
    device = torch.device("cuda:0")
    input_tensor = input_tensor.to(device)

    config_values = {
        key: value
        for key, value in runpy.run_path(str(config_path)).items()
        if not key.startswith("__")
    }
    model_args = argparse.Namespace(**config_values)
    model_args.device = "cuda"
    model_args.use_checkpoint = False
    model_args.pretrain_model_path = ""
    model_args.nms_iou_threshold = -1
    assert (model_args.num_classes, model_args.num_queries, model_args.num_select) == (
        1,
        900,
        300,
    )

    model, criterion, postprocessors = build_dino(model_args)
    del criterion
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    load_result = model.load_state_dict(state_dict, strict=True)
    assert not load_result.missing_keys and not load_result.unexpected_keys
    state_tensor_count = len(state_dict)
    del state_dict, checkpoint
    gc.collect()

    model = model.to(device=device, dtype=torch.float32)
    model.eval()
    torch.cuda.empty_cache()
    with torch.inference_mode():
        warmup_output = model([input_tensor])
        assert tuple(warmup_output["pred_logits"].shape) == (1, 900, 1)
        assert tuple(warmup_output["pred_boxes"].shape) == (1, 900, 4)
        del warmup_output
    torch.cuda.synchronize()
    baseline_allocated = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)

    latencies_ms = []
    reference_logits = reference_boxes = last_output = None
    determinism_max_abs_diff = 0.0
    with torch.inference_mode():
        for run_index in range(args.benchmark_runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = model([input_tensor])
            end.record()
            torch.cuda.synchronize()
            latencies_ms.append(float(start.elapsed_time(end)))
            logits_cpu = output["pred_logits"].detach().cpu()
            boxes_cpu = output["pred_boxes"].detach().cpu()
            assert torch.isfinite(logits_cpu).all() and torch.isfinite(boxes_cpu).all()
            if reference_logits is None:
                reference_logits, reference_boxes = logits_cpu.clone(), boxes_cpu.clone()
            else:
                determinism_max_abs_diff = max(
                    determinism_max_abs_diff,
                    (logits_cpu - reference_logits).abs().max().item(),
                    (boxes_cpu - reference_boxes).abs().max().item(),
                )
            if run_index == args.benchmark_runs - 1:
                last_output = output
            else:
                del output

    assert reference_logits is not None and reference_boxes is not None
    assert last_output is not None and determinism_max_abs_diff <= 1e-6
    peak_allocated = torch.cuda.max_memory_allocated(device)
    target_sizes = torch.tensor([[1024, 1024]], dtype=torch.float32, device=device)
    postprocessed = postprocessors["bbox"](last_output, target_sizes)[0]
    assert len(postprocessed["scores"]) == 300
    assert torch.all(postprocessed["labels"] == 0)

    probabilities = last_output["pred_logits"].sigmoid()
    top_scores, top_indexes = torch.topk(
        probabilities.view(1, -1), model_args.num_select, dim=1
    )
    top_box_indexes = top_indexes // model_args.num_classes
    normalized_cxcywh = torch.gather(
        last_output["pred_boxes"],
        1,
        top_box_indexes.unsqueeze(-1).repeat(1, 1, 4),
    )
    assert torch.allclose(postprocessed["scores"], top_scores[0], atol=0, rtol=0)
    top_scores_np = top_scores[0].detach().cpu().numpy()
    normalized_boxes_np = normalized_cxcywh[0].detach().cpu().numpy()
    assert np.all(top_scores_np[:-1] >= top_scores_np[1:])

    mmbcd_top300 = np.concatenate(
        [normalized_boxes_np, top_scores_np[:, None]], axis=1
    ).astype(np.float32)
    nms_boxes = upstream_mmbcd_nms(mmbcd_top300, 0.1)
    if len(nms_boxes) < 8:
        raise RuntimeError(
            f"Only {len(nms_boxes)} boxes survived NMS; random duplication is forbidden"
        )
    top8 = nms_boxes[:8].copy()

    top300_path = output_dir / "detections-top300.txt"
    nms_path = output_dir / "detections-nms-iou0.1.txt"
    top8_path = output_dir / "detections-top8.txt"
    save_text_atomic(top300_path, mmbcd_top300)
    save_text_atomic(nms_path, nms_boxes)
    save_text_atomic(top8_path, top8)

    geometry = load_json(geometry_path)
    overlay_1024_path = output_dir / "overlay-top8-1024.png"
    overlay_original_path = output_dir / "overlay-top8-original.png"
    save_image_atomic(
        draw_boxes(input_image, top8, coordinate_space="1024", geometry=geometry),
        overlay_1024_path,
    )
    original_image = Image.open(normalized_source_path).convert("RGB")
    save_image_atomic(
        draw_boxes(
            original_image, top8, coordinate_space="original", geometry=geometry
        ),
        overlay_original_path,
    )

    prediction_hash = sha256_predictions(reference_logits, reference_boxes)
    golden_match = prediction_hash == args.expected_prediction_sha256
    manifest = {
        "pipeline": "focalnet-dino-real-dicom-fp32-v1",
        "semantic_validation": False,
        "semantic_validation_note": (
            "CBIS-DDSM validates pipeline execution only. Detector scores must "
            "not be interpreted as calibrated medical findings."
        ),
        "model": {
            "repository_commit": FOCALNET_COMMIT,
            "config_file": config_path.name,
            "config_sha256": sha256_file(config_path),
            "checkpoint_file": checkpoint_path.name,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "state_tensor_count": state_tensor_count,
            "strict_load": True,
            "dtype": "float32",
            "num_classes": model_args.num_classes,
            "num_queries": model_args.num_queries,
            "num_select": model_args.num_select,
            "internal_nms_iou_threshold": model_args.nms_iou_threshold,
            "custom_extension": str(extension.__file__),
        },
        "input": {
            "source_file": image_path.name,
            "source_sha256": sha256_file(image_path),
            "source_shape_hw": [1024, 1024],
            "model_input_shape_nchw": [1, 3, 800, 800],
            "resize_short_side": 800,
            "resize_max_side": 1333,
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
        "outputs": {
            "pred_logits_shape": list(reference_logits.shape),
            "pred_boxes_shape": list(reference_boxes.shape),
            "prediction_sha256": prediction_hash,
            "expected_prediction_sha256": args.expected_prediction_sha256,
            "golden_prediction_match": golden_match,
            "determinism_max_abs_diff": determinism_max_abs_diff,
            "top_score": float(mmbcd_top300[0, 4]),
            "lowest_top300_score": float(mmbcd_top300[-1, 4]),
            "pre_nms_count": int(len(mmbcd_top300)),
            "post_nms_count": int(len(nms_boxes)),
            "mmbcd_topk_count": int(len(top8)),
            "external_nms": {
                "implementation": "MMBCD code/data.py parity",
                "iou_threshold": 0.1,
                "comparison": "strictly greater than threshold",
            },
        },
        "benchmark": {
            "scope": (
                "Model forward only; excludes DICOM preprocessing, host-to-device "
                "transfer, postprocessing, NMS and rendering"
            ),
            "warmup_runs": 1,
            "measured_runs": args.benchmark_runs,
            "latencies_ms": latencies_ms,
            "mean_ms": statistics.fmean(latencies_ms),
            "median_ms": statistics.median(latencies_ms),
            "model_baseline_allocated_mib": baseline_allocated / 1024**2,
            "peak_allocated_mib": peak_allocated / 1024**2,
            "incremental_peak_mib": (peak_allocated - baseline_allocated) / 1024**2,
        },
        "environment": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "deterministic_algorithms_warn_only": True,
            "tf32": False,
        },
        "artifacts": {
            "top300": top300_path.name,
            "post_nms": nms_path.name,
            "top8": top8_path.name,
            "overlay_1024": overlay_1024_path.name,
            "overlay_original": overlay_original_path.name,
        },
    }
    manifest_path = output_dir / "inference-manifest.json"
    write_json_atomic(manifest_path, manifest)
    print("Strict load:", load_result)
    print("Input tensor:", (1, *input_tensor.shape))
    print("Prediction SHA256:", prediction_hash)
    print("Golden prediction match:", golden_match)
    print("Determinism max abs diff:", determinism_max_abs_diff)
    print("Latencies ms:", [round(value, 3) for value in latencies_ms])
    print("Median latency ms:", round(statistics.median(latencies_ms), 3))
    print("Peak allocated MiB:", round(peak_allocated / 1024**2, 2))
    print("Top score:", float(mmbcd_top300[0, 4]))
    print("Boxes before/after NMS:", len(mmbcd_top300), len(nms_boxes))
    print("Top-8 detections:", top8_path)
    print("Manifest:", manifest_path)
    if not golden_match:
        raise RuntimeError("Detector prediction does not match the validated L4 golden hash")
    print("REAL DICOM DETECTOR INFERENCE PASSED")


if __name__ == "__main__":
    main()
