from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts" / "l4_validation"))

from _common import decode_dicom_file  # noqa: E402


class _RecordingCanonicalizer:
    def decode(self, stream: object) -> bytes:
        if not callable(getattr(stream, "read", None)):
            raise TypeError("canonicalizer requires a binary stream")
        return stream.read()


class L4ValidationHelperTests(unittest.TestCase):
    def test_decode_dicom_file_opens_a_binary_stream(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.dcm"
            path.write_bytes(b"dicom")

            decoded = decode_dicom_file(path, _RecordingCanonicalizer())

        self.assertEqual(decoded, b"dicom")


if __name__ == "__main__":
    unittest.main()
