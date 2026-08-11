# DICOM canonicalization

`DicomCanonicalizer` is the repository-owned module that converts one encoded
DICOM stream into the exact 1024 by 1024 grayscale mammogram contract consumed
by the detector pipeline. The interface returns data and writes no input,
preview, or derived files.

```python
from vision_model_serving.dicom import DicomCanonicalizer

with open("mammogram.dcm", "rb") as stream:
    mammogram = DicomCanonicalizer().decode(stream)

original_box = mammogram.geometry.to_original_box(canonical_box)
```

The returned `CanonicalMammogram` contains:

- an immutable 1024 by 1024 `numpy.uint8` array;
- a geometry ledger for points and XYXY boxes;
- a strict non-identifying metadata record;
- structured warnings; and
- the source SHA-256 for provenance. The public typed result currently exposes
  this stable, correlatable value, so result JSON remains sensitive.

## Pixel contract

The processing order is fixed:

1. Read at most 64 MiB and reject an oversized stream before DICOM parsing.
2. Require one frame, at most 12,000 rows or columns, and at most 80 million
   declared decoded pixels.
3. Decode pixels through pydicom's registered decoder for the pinned runtime.
4. Apply the modality LUT or rescale transform.
5. Apply VOI LUT/window alternative index zero. Multiple alternatives produce
   a warning rather than an environment-dependent choice.
6. Support only `MONOCHROME1` and `MONOCHROME2`, then apply an optional
   `IDENTITY` or `INVERSE` Presentation LUT Shape.
7. Exclude `PixelPaddingValue` and an optional inclusive
   `PixelPaddingRangeLimit` from normalization and cropping.
8. Min-max normalize non-padding finite pixels to unsigned 8-bit. A zero range
   fails closed.
9. Select the largest external contour above value 1, clamp the crop origin to
   the frame, and extend the crop by the full `2 * 15` padding pixels before
   clamping the far edge — matching the archived MMBCD reference formula, so an
   edge-flush contour (the chest-wall side of a real mammogram) keeps the same
   crop extent as an interior one. Fail closed if no contour exists.
10. Resize the crop to 1024 by 1024 using Pillow bicubic resampling. This
    mirrors the repository's archived reference preprocessing and may
    intentionally distort aspect ratio. Author-golden parity for LUT, padding,
    and inversion variants remains unvalidated.

A cheap `validate_header()` entry point parses once, runs the same structural gates
(parseability, transfer syntax, photometric interpretation, frame count, and
declared-dimension limits) without touching `PixelData`, so the web tier can
reject unsupported uploads before queue admission without paying for a full
pixel decode. Pixel-level failures on accepted uploads surface asynchronously
as case failures from the executor. It returns a typed `ValidatedDicomHeader`;
the web tier uses its modality instead of parsing the upload a second time.

The first-window and no-contour policies are deliberate. They must not be
silently changed by an HTTP caller or decoder plugin.

## Transfer syntaxes

The module accepts these uncompressed dataset syntaxes:

- Implicit VR Little Endian;
- Explicit VR Little Endian.

The pinned dependencies provide these compressed pixel paths:

| Transfer syntax | Required decoder |
| --- | --- |
| RLE Lossless | pydicom native decoder |

Runtime packaging must retain all four exact versions pinned in
`pyproject.toml`. A listed compressed syntax still fails with
`dicom_transfer_syntax_unsupported` if its registered decoder is unavailable.
JPEG, JPEG-LS, JPEG 2000, HTJ2K, MPEG, Deflated Explicit VR, Explicit VR Big
Endian, and other undeclared syntaxes are rejected before pixel decoding until
a checksum-pinned fixture covers them.

## Geometry ledger

The crop box is stored as half-open original-pixel coordinates
`(x0, y0, x1, y1)`. For a canonical output width `W` and height `H`:

```text
scale_x = W / (x1 - x0)
scale_y = H / (y1 - y0)

x_canonical = (x_original - x0) * scale_x
y_canonical = (y_original - y0) * scale_y

x_original = x_canonical / scale_x + x0
y_original = y_canonical / scale_y + y0
```

Point and box methods clip to their declared coordinate spaces by default.
Tests exercise round trips across deterministic randomized points and verify
that invalid crop ledgers cannot be constructed.

## Privacy and failures

Metadata is limited to dimensions, frame count, modality, photometric and
Presentation LUT interpretation, transfer syntax, SOP class, bit layout, and
whether modality/VOI transforms ran. Patient, study, series, instance,
accession, institution, filename, and clinical-text fields are never copied to
the result. Errors contain a stable code and sanitized implementation detail;
raw pydicom exception text is not returned.

Metadata minimization is not de-identification. The decoder does not inspect the
`BurnedInAnnotation` tag, run OCR, or redact text embedded in pixel data. A
preview/overlay can therefore contain patient information even though no DICOM
metadata is copied. The service also does not require `Modality == "MG"` or a
mammography SOP class; any supported single-frame grayscale object can reach the
model, including the assessment Secondary Capture fixture.

Representative codes include:

| Code | Meaning |
| --- | --- |
| `dicom_invalid` | Not a conforming DICOM file or required header is absent |
| `dicom_encoded_size_exceeded` | Encoded input is above the configured limit |
| `dicom_multiframe_unsupported` | More than one frame was declared |
| `dicom_pixel_limit_exceeded` | Declared dimensions/pixels exceed limits |
| `dicom_transfer_syntax_unsupported` | Syntax or decoder plugin is unsupported |
| `dicom_photometric_unsupported` | Photometric/Presentation LUT is unsupported |
| `dicom_pixel_data_invalid` | Pixel decode or grayscale transform failed |
| `dicom_pixel_range_zero` | Valid pixels have no intensity range |
| `dicom_foreground_absent` | No foreground contour was found |

## Validation boundary

The checksum-pinned CBIS-DDSM fixture reproduces canonical array SHA-256
`97fa0f80a696ce7f822c1681a8c3f7c072da9262b2bd91239c9f1637eaf68552`.
It is an uncompressed unsigned 16-bit `MONOCHROME2` Secondary Capture image
whose contour crop is the full 3826 by 6601 frame. It does not validate native
mammography SOP classes, compression, padding tags, LUT variants, medical
accuracy, or clinical fitness. Synthetic tests cover those pixel-path contracts
without expanding that claim.

Run the focused and full suites with:

```powershell
$env:PYTHONPATH = "src"
uv run python -m unittest tests.test_dicom_canonicalization -v
uv run python -m unittest discover -s tests -v
```
