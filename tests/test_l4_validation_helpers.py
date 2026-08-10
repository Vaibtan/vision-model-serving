from __future__ import annotations

from pathlib import Path
import runpy
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from vision_model_serving.validation.tensorrt_experiment import (
    load_classifier_inputs,
)


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

    def test_optimization_inputs_preserve_pinned_token_width(self) -> None:
        module = runpy.run_path(
            str(
                REPOSITORY_ROOT
                / "scripts"
                / "l4_validation"
                / "17_evaluate_pytorch_optimizations.py"
            )
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "inputs.npz"
            np.savez_compressed(
                path,
                crops=np.zeros((8, 3, 2, 2), dtype=np.float32),
                input_ids=np.ones((1, 5), dtype=np.int64),
                attention_mask=np.ones((1, 5), dtype=np.int64),
            )

            _, input_ids, attention_mask = module["_classifier_inputs"](path)

        self.assertEqual(input_ids.shape, (1, 5))
        self.assertEqual(attention_mask.shape, (1, 5))

    def test_tensorrt_inputs_preserve_pinned_token_width(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "inputs.npz"
            np.savez_compressed(
                path,
                crops=np.zeros((8, 3, 2, 2), dtype=np.float32),
                input_ids=np.ones((1, 5), dtype=np.int64),
                attention_mask=np.ones((1, 5), dtype=np.int64),
            )

            _, input_ids, attention_mask = load_classifier_inputs(path)

        self.assertEqual(input_ids.shape, (1, 5))
        self.assertEqual(attention_mask.shape, (1, 5))


if __name__ == "__main__":
    unittest.main()
