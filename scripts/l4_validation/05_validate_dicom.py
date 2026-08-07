#!/usr/bin/env python3
"""Decode the pinned public CBIS-DDSM DICOM and write its manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import DICOM_SHA256, SERIES_UID, default_paths, sha256_file, write_json_atomic


def main() -> None:
    defaults = default_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-dir", type=Path, default=defaults["fixture_dir"])
    parser.add_argument("--series-uid", default=SERIES_UID)
    parser.add_argument("--expected-sha256", default=DICOM_SHA256)
    args = parser.parse_args()

    import numpy as np
    import pydicom
    from pydicom.uid import UID

    fixture_dir = args.fixture_dir.expanduser().resolve()
    dicom_files = sorted(
        path
        for path in fixture_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".dcm"
    )
    if len(dicom_files) != 1:
        raise RuntimeError(
            f"Expected exactly one DICOM, found {len(dicom_files)}: {dicom_files}"
        )

    dicom_path = dicom_files[0]
    ds = pydicom.dcmread(dicom_path, force=False)
    transfer_syntax = UID(str(ds.file_meta.TransferSyntaxUID))
    modality = str(ds.get("Modality", "")).strip()
    description = str(ds.get("SeriesDescription", "")).strip()
    actual_uid = str(ds.get("SeriesInstanceUID", "")).strip()
    frames = int(ds.get("NumberOfFrames", 1))

    assert modality == "MG", modality
    assert description.lower() == "full mammogram images", description
    assert actual_uid == args.series_uid, (actual_uid, args.series_uid)
    assert frames == 1, frames

    pixels = ds.pixel_array
    assert pixels.ndim == 2, pixels.shape
    assert pixels.shape == (int(ds.Rows), int(ds.Columns))
    assert np.isfinite(pixels).all()
    assert float(pixels.max()) > float(pixels.min())

    actual_hash = sha256_file(dicom_path)
    if actual_hash != args.expected_sha256:
        raise RuntimeError(
            f"DICOM hash mismatch: expected {args.expected_sha256}, observed {actual_hash}"
        )

    def text_value(name: str):
        value = ds.get(name)
        return None if value is None else str(value)

    manifest = {
        "source": "TCIA CBIS-DDSM",
        "series_instance_uid": args.series_uid,
        "series_description": description,
        "modality": modality,
        "sop_class_uid": text_value("SOPClassUID"),
        "transfer_syntax_uid": str(transfer_syntax),
        "transfer_syntax_name": transfer_syntax.name,
        "compressed": bool(transfer_syntax.is_compressed),
        "rows": int(ds.Rows),
        "columns": int(ds.Columns),
        "number_of_frames": frames,
        "samples_per_pixel": int(ds.get("SamplesPerPixel", 1)),
        "photometric_interpretation": text_value("PhotometricInterpretation"),
        "bits_allocated": int(ds.BitsAllocated),
        "bits_stored": int(ds.BitsStored),
        "high_bit": int(ds.HighBit),
        "pixel_representation": int(ds.PixelRepresentation),
        "pixel_padding_value": text_value("PixelPaddingValue"),
        "rescale_slope": text_value("RescaleSlope"),
        "rescale_intercept": text_value("RescaleIntercept"),
        "window_center": text_value("WindowCenter"),
        "window_width": text_value("WindowWidth"),
        "decoded_dtype": str(pixels.dtype),
        "decoded_min": float(pixels.min()),
        "decoded_max": float(pixels.max()),
        "file_size_bytes": dicom_path.stat().st_size,
        "sha256": actual_hash,
        "pydicom_version": pydicom.__version__,
    }
    manifest_path = fixture_dir / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    print("DICOM:", dicom_path)
    print("Shape:", tuple(pixels.shape))
    print("SHA256:", actual_hash)
    print("Manifest:", manifest_path)
    print("REAL DICOM DECODE PASSED")


if __name__ == "__main__":
    main()
