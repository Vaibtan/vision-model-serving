from __future__ import annotations

from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from vision_model_serving.validation.evidence import (  # noqa: E402
    sanitize_error_detail,
)


class EvidenceSanitizationTests(unittest.TestCase):
    def test_host_root_is_replaced_in_native_and_forward_slash_forms(self) -> None:
        root = (REPOSITORY_ROOT / "private-runtime").resolve()
        native = str(root)
        forward = native.replace("\\", "/")

        observed = sanitize_error_detail(
            f"first {native}/model.py then {forward}/weights.bin",
            (root,),
        )

        self.assertEqual(
            observed,
            "first <private-runtime>/model.py then "
            "<private-runtime>/weights.bin",
        )
        self.assertNotIn(native, observed)
        self.assertNotIn(forward, observed)

    def test_error_detail_is_bounded(self) -> None:
        self.assertEqual(
            sanitize_error_detail("sensitive" * 20, (), limit=9),
            "sensitive",
        )


if __name__ == "__main__":
    unittest.main()
