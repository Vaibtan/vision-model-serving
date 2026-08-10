from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPOSITORY_ROOT / "scripts" / "l4_validation"
sys.path.insert(0, str(SCRIPT_ROOT))
spec = importlib.util.spec_from_file_location(
    "tensorrt_builder",
    SCRIPT_ROOT / "18_build_tensorrt_candidate.py",
)
assert spec is not None and spec.loader is not None
tensorrt_builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tensorrt_builder)


class TensorRtReportTests(unittest.TestCase):
    def test_stop_reason_names_failed_dynamic_profile_and_detector_coverage(self) -> None:
        reason = tensorrt_builder._production_reason(
            "stop",
            {"decision": "stop"},
            {"decision": "stop", "shape_coverage": {"passed": False}},
        )

        self.assertIn("token-width 2..90 profile did not pass", reason)
        self.assertIn("detector full coverage did not pass", reason)
        self.assertIn("no fallback is enabled", reason)
        self.assertNotIn("classifier engine GO", reason)

    def test_partial_reason_does_not_claim_whole_pipeline_tensorrt(self) -> None:
        reason = tensorrt_builder._production_reason(
            "partial",
            {"decision": "stop"},
            {"decision": "go", "shape_coverage": {"passed": True}},
        )

        self.assertIn("PARTIAL", reason)
        self.assertIn("not whole-pipeline TensorRT", reason)
        self.assertIn("no fallback is enabled", reason)


if __name__ == "__main__":
    unittest.main()
