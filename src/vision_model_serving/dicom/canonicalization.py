"""Side-effect-free DICOM canonicalization and geometry tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from io import BytesIO
from typing import BinaryIO

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image
import pydicom
from pydicom.errors import InvalidDicomError as PydicomInvalidDicomError
from pydicom.pixels import apply_modality_lut, apply_voi_lut, get_decoder
from pydicom.uid import (
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    RLELossless,
)


_SECONDARY_CAPTURE_STORAGE = "1.2.840.10008.5.1.4.1.1.7"
_SUPPORTED_UNCOMPRESSED_TRANSFER_SYNTAXES = frozenset(
    str(uid)
    for uid in (
        ImplicitVRLittleEndian,
        ExplicitVRLittleEndian,
    )
)
_SUPPORTED_COMPRESSED_TRANSFER_SYNTAXES = frozenset(
    {str(RLELossless)}
)


class DicomCanonicalizationError(RuntimeError):
    """Base class for stable, non-identifying DICOM failures."""

    code = "dicom_canonicalization_failed"

    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(f"{self.code}: {detail}")


class EncodedSizeLimitError(DicomCanonicalizationError):
    code = "dicom_encoded_size_exceeded"


class InvalidDicomError(DicomCanonicalizationError):
    code = "dicom_invalid"


class MultiFrameNotSupportedError(DicomCanonicalizationError):
    code = "dicom_multiframe_unsupported"


class DicomPixelLimitError(DicomCanonicalizationError):
    code = "dicom_pixel_limit_exceeded"


class UnsupportedTransferSyntaxError(DicomCanonicalizationError):
    code = "dicom_transfer_syntax_unsupported"


class UnsupportedPhotometricInterpretationError(DicomCanonicalizationError):
    code = "dicom_photometric_unsupported"


class PixelDataError(DicomCanonicalizationError):
    code = "dicom_pixel_data_invalid"


class ConstantPixelDataError(PixelDataError):
    code = "dicom_pixel_range_zero"


class NoForegroundError(PixelDataError):
    code = "dicom_foreground_absent"


@dataclass(frozen=True, slots=True)
class DicomLimits:
    max_encoded_bytes: int = 64 * 1024 * 1024
    max_rows: int = 12_000
    max_columns: int = 12_000
    max_decoded_pixels: int = 80_000_000

    def __post_init__(self) -> None:
        for name in (
            "max_encoded_bytes",
            "max_rows",
            "max_columns",
            "max_decoded_pixels",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class DicomWarning:
    code: str
    detail: str


@dataclass(frozen=True, slots=True)
class DicomMetadata:
    rows: int
    columns: int
    frames: int
    modality: str | None
    photometric_interpretation: str
    presentation_lut_shape: str
    transfer_syntax_uid: str
    transfer_syntax_name: str
    compressed: bool
    sop_class_uid: str | None
    bits_allocated: int | None
    bits_stored: int | None
    pixel_representation: int | None
    modality_transform_applied: bool
    voi_transform_applied: bool
    voi_index: int


@dataclass(frozen=True, slots=True)
class GeometryLedger:
    original_width: int
    original_height: int
    crop_box: tuple[int, int, int, int]
    canonical_width: int = 1024
    canonical_height: int = 1024

    def __post_init__(self) -> None:
        if self.original_width <= 0 or self.original_height <= 0:
            raise ValueError("original dimensions must be positive")
        if self.canonical_width <= 0 or self.canonical_height <= 0:
            raise ValueError("canonical dimensions must be positive")
        x0, y0, x1, y1 = self.crop_box
        if not (0 <= x0 < x1 <= self.original_width):
            raise ValueError("crop x coordinates must lie within the original image")
        if not (0 <= y0 < y1 <= self.original_height):
            raise ValueError("crop y coordinates must lie within the original image")

    @property
    def scale_x(self) -> float:
        return self.canonical_width / (self.crop_box[2] - self.crop_box[0])

    @property
    def scale_y(self) -> float:
        return self.canonical_height / (self.crop_box[3] - self.crop_box[1])

    def to_canonical_point(
        self,
        x: float,
        y: float,
        *,
        clip: bool = True,
    ) -> tuple[float, float]:
        _validate_point(x, y)
        mapped_x = (float(x) - self.crop_box[0]) * self.scale_x
        mapped_y = (float(y) - self.crop_box[1]) * self.scale_y
        if clip:
            mapped_x = min(max(mapped_x, 0.0), float(self.canonical_width))
            mapped_y = min(max(mapped_y, 0.0), float(self.canonical_height))
        return mapped_x, mapped_y

    def to_original_point(
        self,
        x: float,
        y: float,
        *,
        clip: bool = True,
    ) -> tuple[float, float]:
        _validate_point(x, y)
        if clip:
            x = min(max(float(x), 0.0), float(self.canonical_width))
            y = min(max(float(y), 0.0), float(self.canonical_height))
        mapped_x = float(x) / self.scale_x + self.crop_box[0]
        mapped_y = float(y) / self.scale_y + self.crop_box[1]
        return mapped_x, mapped_y

    def to_canonical_box(
        self,
        box: tuple[float, float, float, float],
        *,
        clip: bool = True,
    ) -> tuple[float, float, float, float]:
        _validate_box(box)
        x0, y0 = self.to_canonical_point(box[0], box[1], clip=clip)
        x1, y1 = self.to_canonical_point(box[2], box[3], clip=clip)
        return x0, y0, x1, y1

    def to_original_box(
        self,
        box: tuple[float, float, float, float],
        *,
        clip: bool = True,
    ) -> tuple[float, float, float, float]:
        _validate_box(box)
        x0, y0 = self.to_original_point(box[0], box[1], clip=clip)
        x1, y1 = self.to_original_point(box[2], box[3], clip=clip)
        return x0, y0, x1, y1


@dataclass(frozen=True, slots=True)
class CanonicalMammogram:
    pixels: NDArray[np.uint8] = field(repr=False)
    geometry: GeometryLedger
    metadata: DicomMetadata
    warnings: tuple[DicomWarning, ...]
    source_sha256: str


class DicomCanonicalizer:
    """Decode one DICOM stream into the MMBCD-compatible canonical image."""

    def __init__(
        self,
        *,
        limits: DicomLimits | None = None,
        output_size: tuple[int, int] = (1024, 1024),
        crop_threshold: int = 1,
        crop_padding: int = 15,
    ):
        self._limits = limits or DicomLimits()
        self._output_width, self._output_height = output_size
        for name, value in (
            ("output width", self._output_width),
            ("output height", self._output_height),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(crop_padding, bool)
            or not isinstance(crop_padding, int)
            or crop_padding < 0
        ):
            raise ValueError("crop padding must be a non-negative integer")
        self._crop_threshold = crop_threshold
        self._crop_padding = crop_padding

    def decode(self, stream: BinaryIO) -> CanonicalMammogram:
        """Return a canonical image without persisting input or derived data."""

        encoded = self._read_bounded(stream)
        source_sha256 = hashlib.sha256(encoded).hexdigest()
        try:
            dataset = pydicom.dcmread(BytesIO(encoded), force=False)
        except (PydicomInvalidDicomError, EOFError, OSError, ValueError) as error:
            raise InvalidDicomError(
                f"DICOM parsing failed ({type(error).__name__})"
            ) from None

        transfer_syntax = dataset.file_meta.get("TransferSyntaxUID")
        if transfer_syntax is None:
            raise InvalidDicomError("DICOM file meta has no Transfer Syntax UID")
        self._verify_transfer_syntax(transfer_syntax)
        photometric = str(dataset.get("PhotometricInterpretation", ""))
        if photometric not in {"MONOCHROME1", "MONOCHROME2"}:
            raise UnsupportedPhotometricInterpretationError(
                "only MONOCHROME1 and MONOCHROME2 are supported"
            )
        presentation_lut_shape = str(
            dataset.get("PresentationLUTShape", "IDENTITY")
        ).upper()
        if presentation_lut_shape not in {"IDENTITY", "INVERSE"}:
            raise UnsupportedPhotometricInterpretationError(
                "unsupported Presentation LUT Shape"
            )

        rows = _positive_int(dataset.get("Rows"), "Rows")
        columns = _positive_int(dataset.get("Columns"), "Columns")
        frames = _positive_int(dataset.get("NumberOfFrames", 1), "NumberOfFrames")
        if frames != 1:
            raise MultiFrameNotSupportedError("only one DICOM frame is supported")
        if rows > self._limits.max_rows or columns > self._limits.max_columns:
            raise DicomPixelLimitError(
                "declared image dimensions exceed configured limits"
            )
        if rows * columns > self._limits.max_decoded_pixels:
            raise DicomPixelLimitError(
                "declared decoded pixel count exceeds configured limits"
            )

        try:
            raw = dataset.pixel_array
        except Exception as error:
            raise PixelDataError(
                f"pixel decoding failed ({type(error).__name__})"
            ) from None
        if raw.ndim != 2 or raw.shape != (rows, columns):
            raise PixelDataError("decoded pixels do not match one declared two-dimensional frame")

        modality_transform_applied = any(
            name in dataset
            for name in ("ModalityLUTSequence", "RescaleSlope", "RescaleIntercept")
        )
        voi_transform_applied = any(
            name in dataset
            for name in ("VOILUTSequence", "WindowCenter", "WindowWidth")
        )
        try:
            modality_values = apply_modality_lut(raw, dataset)
            display_values = apply_voi_lut(modality_values, dataset, index=0)
        except Exception as error:
            raise PixelDataError(
                f"grayscale transformation failed ({type(error).__name__})"
            ) from None

        inverted = (photometric == "MONOCHROME1") ^ (
            presentation_lut_shape == "INVERSE"
        )
        padding_mask = _pixel_padding_mask(raw, dataset)
        normalized = _normalize(
            display_values,
            inverted=inverted,
            valid_mask=~padding_mask,
        )
        crop_box = self._find_crop(normalized)
        x0, y0, x1, y1 = crop_box
        cropped = normalized[y0:y1, x0:x1]
        resized = np.asarray(
            Image.fromarray(cropped).resize(
                (self._output_width, self._output_height),
                resample=Image.Resampling.BICUBIC,
            )
        )
        resized.setflags(write=False)

        geometry = GeometryLedger(
            original_width=columns,
            original_height=rows,
            crop_box=crop_box,
            canonical_width=self._output_width,
            canonical_height=self._output_height,
        )
        metadata = DicomMetadata(
            rows=rows,
            columns=columns,
            frames=frames,
            modality=_optional_text(dataset.get("Modality")),
            photometric_interpretation=photometric,
            presentation_lut_shape=presentation_lut_shape,
            transfer_syntax_uid=str(transfer_syntax),
            transfer_syntax_name=transfer_syntax.name,
            compressed=bool(transfer_syntax.is_compressed),
            sop_class_uid=_optional_text(dataset.get("SOPClassUID")),
            bits_allocated=_optional_int(dataset.get("BitsAllocated")),
            bits_stored=_optional_int(dataset.get("BitsStored")),
            pixel_representation=_optional_int(dataset.get("PixelRepresentation")),
            modality_transform_applied=modality_transform_applied,
            voi_transform_applied=voi_transform_applied,
            voi_index=0,
        )
        warnings: list[DicomWarning] = []
        if padding_mask.any():
            warnings.append(
                DicomWarning(
                    "pixel_padding_excluded",
                    "declared pixel padding was excluded from normalization and cropping",
                )
            )
        if _has_multiple_voi_alternatives(dataset):
            warnings.append(
                DicomWarning(
                    "multiple_voi_alternatives_first_selected",
                    "the first declared VOI LUT or window was selected deterministically",
                )
            )
        if metadata.sop_class_uid == _SECONDARY_CAPTURE_STORAGE:
            warnings.append(
                DicomWarning(
                    "secondary_capture_storage",
                    "fixture compatibility does not establish native mammography support",
                )
            )
        if not np.isclose(geometry.scale_x, geometry.scale_y):
            warnings.append(
                DicomWarning(
                    "aspect_ratio_distorted",
                    "canonical resizing uses the validated non-uniform 1024 by 1024 contract",
                )
            )
        return CanonicalMammogram(
            pixels=resized,
            geometry=geometry,
            metadata=metadata,
            warnings=tuple(warnings),
            source_sha256=source_sha256,
        )

    @staticmethod
    def _verify_transfer_syntax(transfer_syntax: pydicom.uid.UID) -> None:
        value = str(transfer_syntax)
        if value in _SUPPORTED_UNCOMPRESSED_TRANSFER_SYNTAXES:
            return
        if value not in _SUPPORTED_COMPRESSED_TRANSFER_SYNTAXES:
            raise UnsupportedTransferSyntaxError(
                "declared pixel transfer syntax is not supported"
            )
        try:
            decoder = get_decoder(transfer_syntax)
        except NotImplementedError:
            raise UnsupportedTransferSyntaxError(
                "no decoder is implemented for the declared transfer syntax"
            ) from None
        if not decoder.is_available:
            raise UnsupportedTransferSyntaxError(
                "no installed decoder plugin supports the declared transfer syntax"
            )

    def _read_bounded(self, stream: BinaryIO) -> bytes:
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = stream.read(min(1024 * 1024, self._limits.max_encoded_bytes + 1 - total))
            if not chunk:
                break
            if not isinstance(chunk, bytes):
                raise InvalidDicomError("input stream must yield bytes")
            chunks.append(chunk)
            total += len(chunk)
            if total > self._limits.max_encoded_bytes:
                raise EncodedSizeLimitError("encoded DICOM exceeds configured byte limit")
        if not chunks:
            raise InvalidDicomError("input stream is empty")
        return b"".join(chunks)

    def _find_crop(self, image: NDArray[np.uint8]) -> tuple[int, int, int, int]:
        _, binary = cv2.threshold(
            image,
            self._crop_threshold,
            255,
            cv2.THRESH_BINARY,
        )
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            raise NoForegroundError("no foreground contour exceeds the crop threshold")
        x, y, width, height = cv2.boundingRect(max(contours, key=cv2.contourArea))
        image_height, image_width = image.shape
        x0 = max(0, x - self._crop_padding)
        y0 = max(0, y - self._crop_padding)
        x1 = min(image_width, x + width + self._crop_padding)
        y1 = min(image_height, y + height + self._crop_padding)
        return x0, y0, x1, y1


def _normalize(
    values: NDArray[np.generic],
    *,
    inverted: bool,
    valid_mask: NDArray[np.bool_],
) -> NDArray[np.uint8]:
    numeric = np.asarray(values, dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise PixelDataError("transformed pixels contain non-finite values")
    if valid_mask.shape != numeric.shape or not valid_mask.any():
        raise PixelDataError("no non-padding pixels remain")
    valid_values = numeric[valid_mask]
    if inverted:
        shifted = np.max(valid_values) - numeric
    else:
        shifted = numeric - np.min(valid_values)
    maximum = float(np.max(shifted[valid_mask]))
    if maximum <= 0:
        raise ConstantPixelDataError("transformed pixel range is zero")
    normalized = np.clip((shifted / maximum) * 255, 0, 255).astype(np.uint8)
    normalized[~valid_mask] = 0
    return normalized


def _pixel_padding_mask(
    raw: NDArray[np.generic],
    dataset: pydicom.dataset.Dataset,
) -> NDArray[np.bool_]:
    value = dataset.get("PixelPaddingValue")
    if value is None:
        return np.zeros(raw.shape, dtype=np.bool_)
    padding_value = int(value)
    range_limit = dataset.get("PixelPaddingRangeLimit")
    if range_limit is None:
        return np.asarray(raw == padding_value, dtype=np.bool_)
    lower, upper = sorted((padding_value, int(range_limit)))
    return np.asarray((raw >= lower) & (raw <= upper), dtype=np.bool_)


def _has_multiple_voi_alternatives(dataset: pydicom.dataset.Dataset) -> bool:
    sequence = dataset.get("VOILUTSequence")
    if sequence is not None and len(sequence) > 1:
        return True
    for name in ("WindowCenter", "WindowWidth"):
        value = dataset.get(name)
        if value is not None and not isinstance(value, (str, bytes)):
            try:
                if len(value) > 1:
                    return True
            except TypeError:
                pass
    return False


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise InvalidDicomError(f"{name} is invalid")
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise InvalidDicomError(f"{name} is absent or invalid") from None
    if result <= 0:
        raise InvalidDicomError(f"{name} must be positive")
    return result


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _validate_box(box: tuple[float, float, float, float]) -> None:
    if len(box) != 4:
        raise ValueError("a box must contain x0, y0, x1, y1")
    if not all(np.isfinite(float(value)) for value in box):
        raise ValueError("box coordinates must be finite")
    if box[2] < box[0] or box[3] < box[1]:
        raise ValueError("box coordinates must be ordered")


def _validate_point(x: float, y: float) -> None:
    if not np.isfinite(float(x)) or not np.isfinite(float(y)):
        raise ValueError("point coordinates must be finite")
