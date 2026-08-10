from __future__ import annotations

from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
import unittest

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODULE = runpy.run_path(
    str(REPOSITORY_ROOT / "scripts" / "tensorrt_runtime_verify.py")
)


class _Engine:
    def __init__(
        self,
        profile: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    ) -> None:
        self.profile = profile

    def get_tensor_dtype(self, name: str) -> np.dtype:
        return np.dtype(np.int64)

    def get_tensor_profile_shape(
        self,
        name: str,
        profile_index: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        return self.profile

    def get_tensor_shape(self, name: str) -> tuple[int, ...]:
        return self.profile[1]


class _Context:
    def __init__(self, shape: tuple[int, ...] = (1, 5)) -> None:
        self.shape = shape

    def set_input_shape(self, name: str, shape: tuple[int, ...]) -> bool:
        self.shape = shape
        return True

    def get_tensor_shape(self, name: str) -> tuple[int, ...]:
        return self.shape


class _TensorRt:
    @staticmethod
    def nptype(value: object) -> object:
        return value


class TensorRtRuntimeVerifyTests(unittest.TestCase):
    def test_exact_static_fixture_profile_is_accepted(self) -> None:
        profile = ((1, 5), (1, 5), (1, 5))
        context = _Context()

        MODULE["_verify_input_contract"](
            _Engine(profile),
            context,
            "input_ids",
            np.dtype(np.int64),
            *profile,
            (1, 5),
            _TensorRt(),
        )

        self.assertEqual(context.shape, (1, 5))

    def test_engine_with_dynamic_profile_is_rejected(self) -> None:
        expected = ((1, 5), (1, 5), (1, 5))
        with self.assertRaises(RuntimeError):
            MODULE["_verify_input_contract"](
                _Engine(((1, 2), (1, 5), (1, 90))),
                _Context(),
                "input_ids",
                np.dtype(np.int64),
                *expected,
                (1, 5),
                _TensorRt(),
            )

    def test_input_bundle_preserves_width_and_rejects_mismatched_mask(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "inputs.npz"
            np.savez_compressed(
                path,
                crops=np.zeros((8, 3, 224, 224), dtype=np.float32),
                input_ids=np.ones((1, 5), dtype=np.int64),
                attention_mask=np.ones((1, 5), dtype=np.int64),
            )
            values = MODULE["_load_inputs"](path)
            self.assertEqual(values["input_ids"].shape, (1, 5))

            np.savez_compressed(
                path,
                crops=np.zeros((8, 3, 224, 224), dtype=np.float32),
                input_ids=np.ones((1, 5), dtype=np.int64),
                attention_mask=np.ones((1, 4), dtype=np.int64),
            )
            with self.assertRaises(RuntimeError):
                MODULE["_load_inputs"](path)


if __name__ == "__main__":
    unittest.main()
