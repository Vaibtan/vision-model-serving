from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from vision_model_serving.validation.benchmark import BenchmarkContractError  # noqa: E402
from vision_model_serving.validation.benchmark_environment import (  # noqa: E402
    benchmark_environment_identity,
)


class BenchmarkEnvironmentTests(unittest.TestCase):
    def test_environment_evidence_must_match_the_current_exact_lane(self) -> None:
        evidence_path = REPOSITORY_ROOT / "docs" / "validation" / "environment-l4-20260810.json"
        with self.assertRaisesRegex(BenchmarkContractError, "current L4 lane"):
            benchmark_environment_identity(
                REPOSITORY_ROOT,
                evidence_path,
                executor_image_id="sha256:" + "a" * 64,
            )

        lane_path = REPOSITORY_ROOT / "config" / "l4-fp32-environment.json"
        lane = json.loads(lane_path.read_text(encoding="utf-8"))
        current = deepcopy(json.loads(evidence_path.read_text(encoding="utf-8")))
        current["snapshot"]["packages"] = lane["packages"]

        with TemporaryDirectory() as temporary_directory:
            current_path = Path(temporary_directory) / "environment.json"
            current_path.write_text(json.dumps(current), encoding="utf-8")
            identity = benchmark_environment_identity(
                REPOSITORY_ROOT,
                current_path,
                executor_image_id="sha256:" + "a" * 64,
            )

        self.assertEqual(identity["lane"]["id"], lane["lane_id"])
        self.assertEqual(identity["lane"]["packages"], lane["packages"])
        self.assertEqual(
            identity["lane"]["config_sha256"],
            hashlib.sha256(lane_path.read_bytes()).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
