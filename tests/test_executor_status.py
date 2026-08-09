from __future__ import annotations

from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from vision_model_serving.execution import GpuExecutorStatus  # noqa: E402
from vision_model_serving.model_ids import (  # noqa: E402
    CLASSIFIER_MODEL_ID,
    DETECTOR_MODEL_ID,
)
from vision_model_serving.web.operations import OperationalSnapshot  # noqa: E402


def status(**overrides: object) -> GpuExecutorStatus:
    values: dict[str, object] = {
        "verified_artifacts": True,
        "runtime_initialized": True,
        "device_available": True,
        "native_operator_available": True,
        "runtime_state": "unloaded",
        "active_model": None,
        "resident_models": (),
        "device_name": "NVIDIA L4",
        "last_error": None,
    }
    values.update(overrides)
    return GpuExecutorStatus(**values)


class GpuExecutorStatusTests(unittest.TestCase):
    def test_cold_initialized_executor_is_artifact_ready_but_not_warm(self) -> None:
        observed = status()

        self.assertTrue(observed.artifact_ready)
        self.assertFalse(observed.inference_warm)
        self.assertIsNone(observed.warm_model)

    def test_one_ready_resident_is_warm_for_that_model_only(self) -> None:
        observed = status(
            runtime_state="ready",
            active_model=DETECTOR_MODEL_ID,
            resident_models=(DETECTOR_MODEL_ID,),
        )

        self.assertTrue(observed.artifact_ready)
        self.assertTrue(observed.inference_warm)
        self.assertEqual(observed.warm_model, DETECTOR_MODEL_ID)

    def test_failed_runtime_is_not_artifact_ready(self) -> None:
        observed = status(runtime_state="failed", last_error="runtime_failed")

        self.assertFalse(observed.artifact_ready)
        self.assertFalse(observed.inference_warm)

    def test_impossible_residency_states_fail_closed(self) -> None:
        invalid = (
            {
                "runtime_state": "ready",
                "active_model": DETECTOR_MODEL_ID,
                "resident_models": (),
            },
            {
                "runtime_state": "ready",
                "active_model": DETECTOR_MODEL_ID,
                "resident_models": (CLASSIFIER_MODEL_ID,),
            },
            {
                "runtime_state": "ready",
                "active_model": DETECTOR_MODEL_ID,
                "resident_models": (DETECTOR_MODEL_ID, CLASSIFIER_MODEL_ID),
            },
            {
                "runtime_state": "ready",
                "active_model": "unknown-model",
                "resident_models": ("unknown-model",),
            },
            {
                "runtime_state": "unloaded",
                "active_model": DETECTOR_MODEL_ID,
                "resident_models": (DETECTOR_MODEL_ID,),
            },
        )

        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                status(**values)

    def test_transient_switch_state_is_never_reported_warm(self) -> None:
        observed = status(
            runtime_state="draining",
            active_model=CLASSIFIER_MODEL_ID,
            resident_models=(CLASSIFIER_MODEL_ID,),
        )

        self.assertTrue(observed.artifact_ready)
        self.assertFalse(observed.inference_warm)


class OperationalReadinessContractTests(unittest.TestCase):
    def test_cold_readiness_names_its_scope_and_warmth(self) -> None:
        snapshot = OperationalSnapshot(
            captured_at="2026-08-10T00:00:00+00:00",
            status="ready",
            checks={
                "redis": True,
                "rq_worker": True,
                "executor_artifact_ready": True,
                "runtime_initialized": True,
            },
            reasons=(),
            queue={},
            executor={
                "available": True,
                "artifact_ready": True,
                "runtime_initialized": True,
                "state": "unloaded",
                "inference_warm": False,
                "warm_model": None,
                "active_model": None,
                "resident_models": (),
            },
            manifest_id="manifest",
            models=(),
            telemetry={},
        )

        readiness = snapshot.readiness_dict()
        operations = snapshot.as_dict()

        self.assertEqual(readiness["schema_version"], 2)
        self.assertEqual(readiness["readiness_scope"], "artifact_ready")
        self.assertEqual(
            readiness["runtime"],
            {
                "initialized": True,
                "state": "unloaded",
                "inference_warm": False,
                "warm_model": None,
            },
        )
        self.assertEqual(operations["schema_version"], 2)
        self.assertEqual(operations["executor"]["resident_models"], [])


if __name__ == "__main__":
    unittest.main()
