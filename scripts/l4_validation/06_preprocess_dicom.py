#!/usr/bin/env python3
"""Reproduce MMBCD DICOM preprocessing and persist the geometry ledger."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import (
    MMBCD_COMMIT,
    PREPROCESSED_ARRAY_SHA256,
    default_paths,
    load_json,
    sha256_array,
    sha256_file,
    write_json_atomic,
)


def main() -> None:
    defaults = default_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-dir", type=Path, default=defaults["fixture_dir"])
    parser.add_argument("--output-dir", type=Path, default=defaults["preprocess_dir"])
    parser.add_argument("--expected-array-sha256", default=PREPROCESSED_ARRAY_SHA256)
    args = parser.parse_args()

    import cv2
    import numpy as np
    import pydicom
    from PIL import Image, __version__ as pillow_version
    from pydicom.pixels import apply_modality_lut, apply_voi_lut

    fixture_dir = args.fixture_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_manifest = load_json(fixture_dir / "manifest.json")
    dicom_files = sorted(fixture_dir.glob("*.dcm"))
    assert len(dicom_files) == 1, dicom_files
    dicom_path = dicom_files[0]

    def write_png_cv2_atomic(path: Path, array) -> None:
        temporary = path.with_name(path.stem + ".tmp" + path.suffix)
        if not cv2.imwrite(str(temporary), array):
            raise RuntimeError(f"OpenCV failed to write {temporary}")
        temporary.replace(path)

    def write_png_pillow_atomic(path: Path, array) -> None:
        temporary = path.with_name(path.stem + ".tmp" + path.suffix)
        Image.fromarray(array).save(temporary, format="PNG")
        temporary.replace(path)

    def normalize_like_upstream(image, photometric_interpretation: str):
        normalized = (
            image - np.min(image)
            if photometric_interpretation == "MONOCHROME2"
            else np.amax(image) - image
        )
        denominator = np.max(normalized)
        if denominator <= 0:
            raise ValueError("Cannot normalize a constant-valued DICOM image")
        return ((normalized / denominator) * 255).astype(np.uint8)

    actual_dicom_hash = sha256_file(dicom_path)
    assert actual_dicom_hash == source_manifest["sha256"]
    ds = pydicom.dcmread(dicom_path)
    raw = ds.pixel_array
    assert raw.ndim == 2
    assert raw.shape == (source_manifest["rows"], source_manifest["columns"])

    has_modality_transform = any(
        name in ds for name in ("ModalityLUTSequence", "RescaleSlope", "RescaleIntercept")
    )
    has_voi_transform = any(
        name in ds for name in ("VOILUTSequence", "WindowCenter", "WindowWidth")
    )
    assert not has_modality_transform
    assert not has_voi_transform

    upstream_values = apply_voi_lut(raw, ds)
    upstream_uint8 = normalize_like_upstream(
        upstream_values, ds.PhotometricInterpretation
    )
    standard_values = apply_voi_lut(apply_modality_lut(raw, ds), ds)
    standard_uint8 = normalize_like_upstream(
        standard_values, ds.PhotometricInterpretation
    )
    assert np.array_equal(upstream_uint8, standard_uint8)

    threshold_value = 1
    padding = 15
    _, binary = cv2.threshold(upstream_uint8, threshold_value, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise RuntimeError("No foreground contour found")
    x, y, width, height = cv2.boundingRect(max(contours, key=cv2.contourArea))

    original_height, original_width = upstream_uint8.shape
    crop_x0 = max(0, x - padding)
    crop_y0 = max(0, y - padding)
    padded_width = min(original_width, width + 2 * padding)
    padded_height = min(original_height, height + 2 * padding)
    cropped = upstream_uint8[
        crop_y0 : crop_y0 + padded_height,
        crop_x0 : crop_x0 + padded_width,
    ]
    crop_height, crop_width = cropped.shape
    crop_x1, crop_y1 = crop_x0 + crop_width, crop_y0 + crop_height

    source_image = Image.fromarray(cropped)
    default_resize = np.asarray(source_image.resize((1024, 1024)))
    bicubic_resize = np.asarray(
        source_image.resize((1024, 1024), resample=Image.Resampling.BICUBIC)
    )
    assert np.array_equal(default_resize, bicubic_resize)
    resized = bicubic_resize
    resized_hash = sha256_array(resized)
    if resized_hash != args.expected_array_sha256:
        raise RuntimeError(
            f"Preprocessed array mismatch: expected {args.expected_array_sha256}, "
            f"observed {resized_hash}"
        )

    normalized_path = output_dir / "upstream-normalized.png"
    cropped_path = output_dir / "upstream-cropped-pad15.png"
    resized_path = output_dir / "upstream-1024.png"
    write_png_cv2_atomic(normalized_path, upstream_uint8)
    write_png_cv2_atomic(cropped_path, cropped)
    write_png_pillow_atomic(resized_path, resized)

    scale_x, scale_y = 1024.0 / crop_width, 1024.0 / crop_height
    manifest = {
        "pipeline": "mmbcd-upstream-parity-v1",
        "upstream": {
            "repository": "https://github.com/adsbansal/MMBCD",
            "commit": MMBCD_COMMIT,
            "crop_padding": padding,
            "crop_threshold": threshold_value,
            "resize": [1024, 1024],
            "resize_resampling": "Pillow BICUBIC",
        },
        "source": {
            "dicom_file": dicom_path.name,
            "dicom_sha256": actual_dicom_hash,
            "shape_hw": [original_height, original_width],
            "dtype": str(raw.dtype),
            "minimum": int(raw.min()),
            "maximum": int(raw.max()),
            "photometric_interpretation": ds.PhotometricInterpretation,
            "sop_class_uid": str(ds.SOPClassUID),
            "secondary_capture_storage": (
                str(ds.SOPClassUID) == "1.2.840.10008.5.1.4.1.1.7"
            ),
        },
        "transform_decisions": {
            "modality_transform_present": has_modality_transform,
            "voi_transform_present": has_voi_transform,
            "photometric_inversion_applied": ds.PhotometricInterpretation != "MONOCHROME2",
            "standard_path_matches_upstream": True,
        },
        "crop": {
            "largest_contour_box_xywh": [int(x), int(y), int(width), int(height)],
            "crop_box_xyxy_original": [crop_x0, crop_y0, crop_x1, crop_y1],
            "cropped_shape_hw": [crop_height, crop_width],
        },
        "resize": {
            "output_shape_hw": [1024, 1024],
            "scale_x": scale_x,
            "scale_y": scale_y,
            "distorts_aspect_ratio": not np.isclose(scale_x, scale_y),
        },
        "coordinate_mapping": {
            "original_to_1024": {
                "x": "(x_original - crop_x0) * scale_x",
                "y": "(y_original - crop_y0) * scale_y",
            },
            "1024_to_original": {
                "x": "x_1024 / scale_x + crop_x0",
                "y": "y_1024 / scale_y + crop_y0",
            },
        },
        "artifacts": {
            "normalized": {
                "file": normalized_path.name,
                "shape_hw": list(upstream_uint8.shape),
                "array_sha256": sha256_array(upstream_uint8),
                "file_sha256": sha256_file(normalized_path),
            },
            "cropped": {
                "file": cropped_path.name,
                "shape_hw": list(cropped.shape),
                "array_sha256": sha256_array(cropped),
                "file_sha256": sha256_file(cropped_path),
            },
            "resized": {
                "file": resized_path.name,
                "shape_hw": list(resized.shape),
                "array_sha256": resized_hash,
                "file_sha256": sha256_file(resized_path),
            },
        },
        "environment": {
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "pydicom": pydicom.__version__,
            "pillow": pillow_version,
        },
        "validation_limit": (
            "This Secondary Capture fixture validates the released pixel path, "
            "but does not cover native mammography SOP classes, MONOCHROME1, "
            "compressed transfer syntaxes, padding tags, modality LUTs, or VOI LUTs."
        ),
    }
    manifest_path = output_dir / "preprocess-manifest.json"
    write_json_atomic(manifest_path, manifest)
    print("Original shape:", (original_height, original_width))
    print("Largest contour xywh:", (x, y, width, height))
    print("Crop box xyxy:", (crop_x0, crop_y0, crop_x1, crop_y1))
    print("Cropped shape:", cropped.shape)
    print("Resize scale x/y:", scale_x, scale_y)
    print("1024 array SHA256:", resized_hash)
    print("Output image:", resized_path)
    print("Geometry manifest:", manifest_path)
    print("DICOM PREPROCESSING PARITY PASSED")


if __name__ == "__main__":
    main()
