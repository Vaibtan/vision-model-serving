from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
import random
import sys
import unittest

import numpy as np
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.encaps import encapsulate
from pydicom.sequence import Sequence
from pydicom.uid import (
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    MPEG2MPML,
    RLELossless,
    SecondaryCaptureImageStorage,
    generate_uid,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from vision_model_serving.dicom import (  # noqa: E402
    DicomCanonicalizationError,
    DicomCanonicalizer,
    DicomLimits,
    GeometryLedger,
)


SERIES_UID = "1.3.6.1.4.1.9590.100.1.2.100131208110604806117271735422083351547"
GOLDEN_DICOM_SHA256 = (
    "9f70081672a460f29231bb471e8a9e26dd3ed26a2ebbd91c064e575e7842a19c"
)
GOLDEN_ARRAY_SHA256 = (
    "97fa0f80a696ce7f822c1681a8c3f7c072da9262b2bd91239c9f1637eaf68552"
)


def dicom_bytes(
    pixels: np.ndarray,
    *,
    photometric: str = "MONOCHROME2",
    frames: int = 1,
    patient_name: str | None = None,
    pixel_data: bytes | None = None,
    rescale: tuple[float, float] | None = None,
    window: tuple[object, object] | None = None,
    pixel_padding_value: int | None = None,
    presentation_lut_shape: str | None = None,
    rle_compress: bool = False,
    unsupported_compressed_syntax: bool = False,
    implicit_vr: bool = False,
    modality_lut: list[int] | None = None,
) -> bytes:
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = generate_uid()
    dataset = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.Modality = "MG"
    dataset.Rows = pixels.shape[-2]
    dataset.Columns = pixels.shape[-1]
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = photometric
    dataset.NumberOfFrames = frames
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.PixelData = (
        pixel_data
        if pixel_data is not None
        else pixels.astype("<u2", copy=False).tobytes()
    )
    if rescale is not None:
        dataset.RescaleSlope, dataset.RescaleIntercept = rescale
    if modality_lut is not None:
        item = Dataset()
        item.LUTDescriptor = [len(modality_lut), 0, 16]
        item.LUTData = np.asarray(modality_lut, dtype="<u2").tobytes()
        item.ModalityLUTType = "US"
        dataset.ModalityLUTSequence = Sequence([item])
    if window is not None:
        dataset.WindowCenter, dataset.WindowWidth = window
    if pixel_padding_value is not None:
        dataset.PixelPaddingValue = pixel_padding_value
    if presentation_lut_shape is not None:
        dataset.PresentationLUTShape = presentation_lut_shape
    if patient_name is not None:
        dataset.PatientName = patient_name
        dataset.PatientID = "never-return-this-id"
    if rle_compress:
        dataset.compress(RLELossless)
    elif unsupported_compressed_syntax:
        dataset.file_meta.TransferSyntaxUID = MPEG2MPML
        dataset.PixelData = encapsulate([dataset.PixelData])
        dataset["PixelData"].is_undefined_length = True
    elif implicit_vr:
        dataset.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    stream = BytesIO()
    dataset.save_as(stream, enforce_file_format=True)
    return stream.getvalue()


class GoldenDicomTests(unittest.TestCase):
    def test_public_fixture_reproduces_the_archived_canonical_array(self) -> None:
        fixture = (
            REPOSITORY_ROOT
            / "fixtures"
            / "cbis-ddsm"
            / SERIES_UID
            / "1-1.dcm"
        )
        if not fixture.is_file():
            self.skipTest("checksum-pinned CBIS-DDSM fixture is not installed")

        with fixture.open("rb") as stream:
            result = DicomCanonicalizer().decode(stream)

        self.assertEqual(result.pixels.shape, (1024, 1024))
        self.assertEqual(result.pixels.dtype, np.uint8)
        self.assertFalse(result.pixels.flags.writeable)
        self.assertEqual(result.source_sha256, GOLDEN_DICOM_SHA256)
        self.assertEqual(
            hashlib.sha256(result.pixels.tobytes()).hexdigest(),
            GOLDEN_ARRAY_SHA256,
        )
        self.assertEqual(result.metadata.rows, 6601)
        self.assertEqual(result.metadata.columns, 3826)
        self.assertEqual(result.metadata.photometric_interpretation, "MONOCHROME2")
        self.assertNotIn("patient", repr(result.metadata).lower())
        self.assertEqual(result.geometry.crop_box, (0, 0, 3826, 6601))

        canonical = result.geometry.to_canonical_point(1913.0, 3300.5)
        original = result.geometry.to_original_point(*canonical)
        self.assertAlmostEqual(original[0], 1913.0)
        self.assertAlmostEqual(original[1], 3300.5)


class DicomInputContractTests(unittest.TestCase):
    def test_invalid_limits_and_output_geometry_fail_at_configuration_time(self) -> None:
        with self.assertRaises(ValueError):
            DicomLimits(max_encoded_bytes=0)
        with self.assertRaises(ValueError):
            DicomLimits(max_decoded_pixels=-1)
        with self.assertRaises(ValueError):
            DicomCanonicalizer(output_size=(0, 1024))
        with self.assertRaises(ValueError):
            DicomCanonicalizer(crop_padding=-1)

    def test_non_dicom_bytes_fail_with_a_stable_typed_error(self) -> None:
        with self.assertRaises(DicomCanonicalizationError) as raised:
            DicomCanonicalizer().decode(BytesIO(b"not a DICOM file"))

        self.assertEqual(raised.exception.code, "dicom_invalid")

    def test_multiframe_input_fails_with_a_stable_typed_error(self) -> None:
        encoded = dicom_bytes(
            np.array(
                [
                    [[0, 100], [200, 300]],
                    [[400, 500], [600, 700]],
                ],
                dtype=np.uint16,
            ),
            frames=2,
        )

        with self.assertRaises(DicomCanonicalizationError) as raised:
            DicomCanonicalizer().decode(BytesIO(encoded))

        self.assertEqual(raised.exception.code, "dicom_multiframe_unsupported")

    def test_encoded_size_limit_fails_before_dicom_parsing(self) -> None:
        with self.assertRaises(DicomCanonicalizationError) as raised:
            DicomCanonicalizer(
                limits=DicomLimits(max_encoded_bytes=4)
            ).decode(BytesIO(b"12345"))

        self.assertEqual(raised.exception.code, "dicom_encoded_size_exceeded")

    def test_declared_pixel_limits_fail_before_pixel_decoding(self) -> None:
        encoded = dicom_bytes(
            np.array([[0, 1], [2, 3]], dtype=np.uint16),
            pixel_data=b"corrupt but must not be decoded",
        )

        with self.assertRaises(DicomCanonicalizationError) as raised:
            DicomCanonicalizer(
                limits=DicomLimits(max_rows=1)
            ).decode(BytesIO(encoded))

        self.assertEqual(raised.exception.code, "dicom_pixel_limit_exceeded")

    def test_corrupt_pixel_payload_has_a_stable_redacted_error(self) -> None:
        encoded = dicom_bytes(
            np.array([[0, 1], [2, 3]], dtype=np.uint16),
            pixel_data=b"\x00\x00",
            patient_name="Sensitive^Person",
        )

        with self.assertRaises(DicomCanonicalizationError) as raised:
            DicomCanonicalizer().decode(BytesIO(encoded))

        self.assertEqual(raised.exception.code, "dicom_pixel_data_invalid")
        self.assertNotIn("Sensitive", str(raised.exception))
        self.assertNotIn("never-return", str(raised.exception))
        self.assertIsNone(raised.exception.__cause__)
        self.assertTrue(raised.exception.__suppress_context__)

    def test_returned_result_contains_only_whitelisted_metadata(self) -> None:
        encoded = dicom_bytes(
            np.array([[0, 100], [200, 300]], dtype=np.uint16),
            patient_name="Sensitive^Person",
        )

        result = DicomCanonicalizer(
            output_size=(2, 2),
            crop_threshold=-1,
        ).decode(BytesIO(encoded))

        public = repr(result)
        self.assertNotIn("Sensitive", public)
        self.assertNotIn("never-return", public)
        self.assertEqual(result.metadata.modality, "MG")

    def test_unsupported_transfer_syntax_fails_before_pixel_decoding(self) -> None:
        encoded = dicom_bytes(
            np.array([[0, 1], [2, 3]], dtype=np.uint16),
            unsupported_compressed_syntax=True,
        )

        with self.assertRaises(DicomCanonicalizationError) as raised:
            DicomCanonicalizer().decode(BytesIO(encoded))

        self.assertEqual(raised.exception.code, "dicom_transfer_syntax_unsupported")

    def test_unsupported_photometric_interpretation_is_typed(self) -> None:
        encoded = dicom_bytes(
            np.array([[0, 1], [2, 3]], dtype=np.uint16),
            photometric="PALETTE COLOR",
        )

        with self.assertRaises(DicomCanonicalizationError) as raised:
            DicomCanonicalizer().decode(BytesIO(encoded))

        self.assertEqual(raised.exception.code, "dicom_photometric_unsupported")


class DicomTransformTests(unittest.TestCase):
    def test_monochrome_photometric_interpretations_have_explicit_polarity(self) -> None:
        pixels = np.array([[0, 100], [200, 300]], dtype=np.uint16)
        canonicalizer = DicomCanonicalizer(
            output_size=(2, 2),
            crop_threshold=-1,
        )

        mono2 = canonicalizer.decode(
            BytesIO(dicom_bytes(pixels, photometric="MONOCHROME2"))
        )
        mono1 = canonicalizer.decode(
            BytesIO(dicom_bytes(pixels, photometric="MONOCHROME1"))
        )

        np.testing.assert_array_equal(
            mono2.pixels,
            np.array([[0, 85], [170, 255]], dtype=np.uint8),
        )
        np.testing.assert_array_equal(
            mono1.pixels,
            np.array([[255, 170], [85, 0]], dtype=np.uint8),
        )

    def test_presentation_lut_inverse_is_applied_after_monochrome_polarity(self) -> None:
        encoded = dicom_bytes(
            np.array([[0, 100], [200, 300]], dtype=np.uint16),
            photometric="MONOCHROME2",
            presentation_lut_shape="INVERSE",
        )

        result = DicomCanonicalizer(
            output_size=(2, 2),
            crop_threshold=-1,
        ).decode(BytesIO(encoded))

        np.testing.assert_array_equal(
            result.pixels,
            np.array([[255, 170], [85, 0]], dtype=np.uint8),
        )

    def test_modality_transform_precedes_first_voi_window(self) -> None:
        encoded = dicom_bytes(
            np.array([[0, 50, 100, 150]], dtype=np.uint16),
            rescale=(2.0, -100.0),
            window=([50.0, 100.0], [100.0, 300.0]),
        )

        result = DicomCanonicalizer(
            output_size=(4, 1),
            crop_threshold=-1,
        ).decode(BytesIO(encoded))

        np.testing.assert_array_equal(
            result.pixels,
            np.array([[0, 0, 255, 255]], dtype=np.uint8),
        )
        self.assertIn(
            "multiple_voi_alternatives_first_selected",
            {warning.code for warning in result.warnings},
        )

    def test_modality_lut_precedes_voi_window(self) -> None:
        encoded = dicom_bytes(
            np.array([[0, 1, 2, 3]], dtype=np.uint16),
            modality_lut=[1000, 2000, 3000, 4000],
            window=(2500.0, 2000.0),
        )

        result = DicomCanonicalizer(
            output_size=(4, 1),
            crop_threshold=-1,
        ).decode(BytesIO(encoded))

        np.testing.assert_array_equal(
            result.pixels,
            np.array([[0, 63, 191, 255]], dtype=np.uint8),
        )
        self.assertTrue(result.metadata.modality_transform_applied)
        self.assertTrue(result.metadata.voi_transform_applied)

    def test_pixel_padding_is_excluded_before_normalization_and_crop(self) -> None:
        pixels = np.full((6, 8), 500, dtype=np.uint16)
        pixels[2:4, 3:6] = np.array(
            [[100, 200, 300], [150, 250, 400]],
            dtype=np.uint16,
        )
        encoded = dicom_bytes(pixels, pixel_padding_value=500)

        result = DicomCanonicalizer(
            output_size=(5, 4),
            crop_padding=1,
        ).decode(BytesIO(encoded))

        self.assertEqual(result.geometry.crop_box, (2, 1, 7, 5))
        self.assertIn(
            "pixel_padding_excluded",
            {warning.code for warning in result.warnings},
        )

    def test_constant_pixels_and_missing_foreground_fail_closed(self) -> None:
        constant = dicom_bytes(np.full((2, 2), 7, dtype=np.uint16))
        with self.assertRaises(DicomCanonicalizationError) as zero_range:
            DicomCanonicalizer().decode(BytesIO(constant))
        self.assertEqual(zero_range.exception.code, "dicom_pixel_range_zero")

        varying = dicom_bytes(np.array([[0, 1], [2, 3]], dtype=np.uint16))
        with self.assertRaises(DicomCanonicalizationError) as no_foreground:
            DicomCanonicalizer(crop_threshold=255).decode(BytesIO(varying))
        self.assertEqual(no_foreground.exception.code, "dicom_foreground_absent")

    def test_rle_lossless_and_uncompressed_inputs_canonicalize_identically(self) -> None:
        pixels = np.array([[0, 100], [200, 300]], dtype=np.uint16)
        canonicalizer = DicomCanonicalizer(
            output_size=(2, 2),
            crop_threshold=-1,
        )

        uncompressed = canonicalizer.decode(BytesIO(dicom_bytes(pixels)))
        compressed = canonicalizer.decode(
            BytesIO(dicom_bytes(pixels, rle_compress=True))
        )

        self.assertFalse(uncompressed.metadata.compressed)
        self.assertTrue(compressed.metadata.compressed)
        np.testing.assert_array_equal(compressed.pixels, uncompressed.pixels)

    def test_implicit_and_explicit_vr_little_endian_are_equivalent(self) -> None:
        pixels = np.array([[0, 100], [200, 300]], dtype=np.uint16)
        canonicalizer = DicomCanonicalizer(
            output_size=(2, 2),
            crop_threshold=-1,
        )

        explicit = canonicalizer.decode(BytesIO(dicom_bytes(pixels)))
        implicit = canonicalizer.decode(
            BytesIO(dicom_bytes(pixels, implicit_vr=True))
        )

        np.testing.assert_array_equal(implicit.pixels, explicit.pixels)


class GeometryLedgerTests(unittest.TestCase):
    def test_non_finite_points_are_rejected_before_mapping(self) -> None:
        ledger = GeometryLedger(
            original_width=200,
            original_height=100,
            crop_box=(20, 10, 180, 90),
        )

        for point in ((float("nan"), 1.0), (1.0, float("inf"))):
            with self.assertRaises(ValueError):
                ledger.to_canonical_point(*point)
            with self.assertRaises(ValueError):
                ledger.to_original_point(*point)

    def test_forward_inverse_point_mapping_round_trips_inside_the_crop(self) -> None:
        ledger = GeometryLedger(
            original_width=4000,
            original_height=6000,
            crop_box=(125, 250, 3625, 5750),
        )
        generator = random.Random(20260808)

        for _ in range(250):
            point = (
                generator.uniform(ledger.crop_box[0], ledger.crop_box[2]),
                generator.uniform(ledger.crop_box[1], ledger.crop_box[3]),
            )
            canonical = ledger.to_canonical_point(*point)
            restored = ledger.to_original_point(*canonical)
            self.assertAlmostEqual(restored[0], point[0], places=9)
            self.assertAlmostEqual(restored[1], point[1], places=9)

    def test_box_mapping_clips_and_round_trips_declared_coordinates(self) -> None:
        ledger = GeometryLedger(
            original_width=200,
            original_height=100,
            crop_box=(20, 10, 180, 90),
            canonical_width=1024,
            canonical_height=1024,
        )
        original_box = (40.0, 20.0, 160.0, 80.0)

        canonical_box = ledger.to_canonical_box(original_box)
        restored_box = ledger.to_original_box(canonical_box)

        for restored, expected in zip(restored_box, original_box, strict=True):
            self.assertAlmostEqual(restored, expected, places=9)
        self.assertEqual(
            ledger.to_canonical_box((-50.0, -50.0, 250.0, 150.0)),
            (0.0, 0.0, 1024.0, 1024.0),
        )

    def test_invalid_geometry_cannot_escape_in_a_result(self) -> None:
        with self.assertRaises(ValueError):
            GeometryLedger(
                original_width=100,
                original_height=100,
                crop_box=(50, 50, 50, 75),
            )
        with self.assertRaises(ValueError):
            GeometryLedger(
                original_width=100,
                original_height=100,
                crop_box=(-1, 0, 100, 100),
            )


if __name__ == "__main__":
    unittest.main()
