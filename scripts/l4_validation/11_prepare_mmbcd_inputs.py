#!/usr/bin/env python3
"""Freeze MMBCD's eight ROI crops and label-free tokenizer input."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from _common import (
    MMBCD_INPUT_TENSOR_SHA256,
    ROBERTA_REVISION,
    default_paths,
    sha256_array,
    sha256_file,
    write_json_atomic,
)


def main() -> None:
    defaults = default_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-image",
        type=Path,
        default=defaults["preprocess_dir"] / "upstream-1024.png",
    )
    parser.add_argument(
        "--detections",
        type=Path,
        default=defaults["detector_dir"] / "detections-top8.txt",
    )
    parser.add_argument("--tokenizer-dir", type=Path, default=defaults["tokenizer_dir"])
    parser.add_argument("--output-dir", type=Path, default=defaults["mmbcd_input_dir"])
    parser.add_argument("--prompt", default="Indication:")
    parser.add_argument("--expected-tensor-sha256", default=MMBCD_INPUT_TENSOR_SHA256)
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import numpy as np
    import PIL
    import torch
    import torchvision
    import transformers
    from PIL import Image
    from torchvision import transforms
    from transformers import RobertaTokenizer

    source_image_path = args.source_image.expanduser().resolve()
    detections_path = args.detections.expanduser().resolve()
    tokenizer_dir = args.tokenizer_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for path in (source_image_path, detections_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not tokenizer_dir.is_dir():
        raise NotADirectoryError(tokenizer_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    crop_dir = output_dir / "raw-crops"
    crop_dir.mkdir(exist_ok=True)

    detections = np.loadtxt(detections_path, dtype=np.float32, ndmin=2)
    assert detections.shape == (8, 5), detections.shape
    assert np.isfinite(detections).all()
    assert np.all((detections[:, :2] >= 0) & (detections[:, :2] <= 1))
    assert np.all((detections[:, 2:4] > 0) & (detections[:, 2:4] <= 1))

    image = Image.open(source_image_path).convert("RGB")
    width, height = image.size
    assert (width, height) == (1024, 1024)
    transform = transforms.Compose(
        [
            transforms.Resize(
                (224, 224),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    crop_tensors = []
    crop_records = []
    montage_crops = []
    for index, detection in enumerate(detections):
        cx, cy, box_width, box_height, confidence = map(float, detection)
        x1 = int((cx - box_width / 2) * width)
        y1 = int((cy - box_height / 2) * height)
        x2 = int((cx + box_width / 2) * width)
        y2 = int((cy + box_height / 2) * height)
        assert x2 > x1 and y2 > y1
        raw_crop = image.crop((x1, y1, x2, y2))
        crop_path = crop_dir / f"roi-{index + 1:02d}.png"
        raw_crop.save(crop_path)
        crop_tensors.append(transform(raw_crop))
        montage_crops.append(
            raw_crop.resize((224, 224), resample=Image.Resampling.BILINEAR)
        )
        crop_records.append(
            {
                "rank": index + 1,
                "normalized_cxcywh": [cx, cy, box_width, box_height],
                "confidence": confidence,
                "pixel_box_xyxy": [x1, y1, x2, y2],
                "extends_beyond_source": (
                    x1 < 0 or y1 < 0 or x2 > width or y2 > height
                ),
                "raw_crop_size_wh": list(raw_crop.size),
                "file": str(crop_path.relative_to(output_dir)),
                "file_sha256": sha256_file(crop_path),
            }
        )

    crop_tensor = torch.stack(crop_tensors)
    assert crop_tensor.shape == (8, 3, 224, 224)
    assert torch.isfinite(crop_tensor).all()
    crop_tensor_hash = sha256_array(crop_tensor.numpy())
    if crop_tensor_hash != args.expected_tensor_sha256:
        raise RuntimeError(
            f"ROI tensor mismatch: expected {args.expected_tensor_sha256}, "
            f"observed {crop_tensor_hash}"
        )

    montage = Image.new("RGB", (224 * 8, 224))
    for index, crop in enumerate(montage_crops):
        montage.paste(crop, (index * 224, 0))
    montage_path = output_dir / "roi-montage.png"
    montage.save(montage_path)

    tokenizer = RobertaTokenizer.from_pretrained(
        str(tokenizer_dir), local_files_only=True
    )
    tokens = tokenizer(
        [args.prompt],
        padding=True,
        truncation=True,
        max_length=90,
        return_tensors="pt",
    )
    input_ids, attention_mask = tokens["input_ids"], tokens["attention_mask"]
    assert input_ids.shape == attention_mask.shape and input_ids.shape[0] == 1
    assert input_ids.shape[1] <= 90

    bundle_path = output_dir / "mmbcd-inputs.npz"
    np.savez_compressed(
        bundle_path,
        crops=crop_tensor.numpy(),
        boxes=detections,
        input_ids=input_ids.numpy(),
        attention_mask=attention_mask.numpy(),
    )
    tokenizer_files = {
        str(path.relative_to(tokenizer_dir)): sha256_file(path)
        for path in sorted(tokenizer_dir.rglob("*"))
        if path.is_file() and ".cache" not in path.parts
    }
    manifest = {
        "pipeline": "mmbcd-input-contract-v1",
        "source": {
            "image": str(source_image_path),
            "image_sha256": sha256_file(source_image_path),
            "detections": str(detections_path),
            "detections_sha256": sha256_file(detections_path),
            "source_shape_wh": [width, height],
        },
        "proposals": {
            "count": len(crop_records),
            "format": "normalized cx cy width height confidence",
            "records": crop_records,
        },
        "image_transform": {
            "output_shape": [8, 3, 224, 224],
            "resize": [224, 224],
            "interpolation": "Pillow bilinear",
            "rgb_conversion": True,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "tensor_sha256": crop_tensor_hash,
        },
        "text": {
            "prompt": args.prompt,
            "prompt_source": "fixture_has_no_clinical_history",
            "label_information_used": False,
            "max_length": 90,
            "input_ids": input_ids.tolist(),
            "attention_mask": attention_mask.tolist(),
            "input_ids_sha256": sha256_array(input_ids.numpy()),
            "attention_mask_sha256": sha256_array(attention_mask.numpy()),
        },
        "tokenizer": {
            "repository": "FacebookAI/roberta-base",
            "revision": ROBERTA_REVISION,
            "local_files_only": True,
            "files": tokenizer_files,
        },
        "artifacts": {
            "bundle": bundle_path.name,
            "bundle_sha256": sha256_file(bundle_path),
            "montage": montage_path.name,
            "montage_sha256": sha256_file(montage_path),
        },
        "environment": {
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "transformers": transformers.__version__,
        },
    }
    manifest_path = output_dir / "mmbcd-input-manifest.json"
    write_json_atomic(manifest_path, manifest)
    print("Crop tensor:", tuple(crop_tensor.shape))
    print("Crop tensor SHA256:", crop_tensor_hash)
    print("Prompt:", repr(args.prompt))
    print("Input IDs:", input_ids.tolist())
    print("Attention mask:", attention_mask.tolist())
    print("Bundle:", bundle_path)
    print("Montage:", montage_path)
    print("Manifest:", manifest_path)
    print("MMBCD INPUT CONTRACT PASSED")


if __name__ == "__main__":
    main()
