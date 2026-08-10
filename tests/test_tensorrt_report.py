from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from vision_model_serving.model_ids import CLASSIFIER_MODEL_ID, DETECTOR_MODEL_ID
from vision_model_serving.validation.tensorrt_experiment import (
    TENSORRT_REPORT_SCHEMA_VERSION,
    TensorRtReportContext,
    run_tensorrt_experiment,
)


def detector_result(*, decision: str = "stop") -> dict[str, object]:
    eligible = decision == "go"
    return {
        "strict_export": eligible,
        "dryrun_completed": eligible,
        "require_full_compilation": True,
        "pytorch_partition_count": 0 if eligible else None,
        "unsupported_operators": [],
        "engine_built": eligible,
        "tensorrt_only_runtime_passed": eligible,
        "parity_passed": eligible,
        "performance_threshold_passed": eligible,
        "plugin_requirement_status": "measured" if eligible else "not_reached",
        "plugin_required": False if eligible else None,
        "decision": decision,
    }


def classifier_result(*, decision: str = "stop", shape_coverage: bool = False) -> dict[str, object]:
    eligible = decision == "go"
    return {
        "strict_export": True,
        "dryrun_completed": True,
        "require_full_compilation": True,
        "pytorch_partition_count": 0,
        "unsupported_operators": [],
        "engine_built": True,
        "tensorrt_only_runtime_passed": eligible,
        "parity": {"passed": eligible},
        "performance": {"promotion_threshold_passed": eligible},
        "shape_coverage": {
            "passed": shape_coverage,
            "required_profile": {
                "min_token_width": 2,
                "opt_token_width": 5,
                "max_token_width": 90,
            },
            "static_fixture_token_width": 5,
            "dynamic_strict_export": {"passed": shape_coverage},
            "failure_code": None if shape_coverage else "dynamic_profile_engine_not_built",
        },
        "decision": decision,
    }


class SuccessfulMeasurements:
    def __init__(
        self,
        detector: dict[str, object],
        classifier: dict[str, object],
    ) -> None:
        self.detector = detector
        self.classifier = classifier
        self.detector_released = False

    def measure_detector(self) -> dict[str, object]:
        return self.detector

    def release_detector(self) -> None:
        self.detector_released = True

    def measure_classifier(self) -> dict[str, object]:
        if not self.detector_released:
            raise RuntimeError("classifier measurement overlapped detector residency")
        return self.classifier


class FailingClassifierMeasurements(SuccessfulMeasurements):
    def __init__(self, output_dir: Path, error_detail: str) -> None:
        super().__init__(detector_result(), classifier_result())
        self.output_dir = output_dir
        self.error_detail = error_detail

    def measure_classifier(self) -> dict[str, object]:
        super().measure_classifier()
        (self.output_dir / "mmbcd-fp32.candidate.plan").write_bytes(b"candidate")
        raise RuntimeError(self.error_detail)


class FailingDetectorMeasurements:
    def measure_detector(self) -> dict[str, object]:
        raise RuntimeError("detector measurement failed")

    def release_detector(self) -> None:
        raise AssertionError("release is unreachable after detector failure")

    def measure_classifier(self) -> dict[str, object]:
        raise AssertionError("classifier is unreachable after detector failure")


def report_context() -> TensorRtReportContext:
    return TensorRtReportContext(
        revision="a" * 40,
        measured_at="2026-08-10T00:00:00+00:00",
        environment={"torch": "2.8.0+cu128"},
    )


class TensorRtReportTests(unittest.TestCase):
    def test_environment_is_captured_after_measurements_complete(self) -> None:
        measurements = SuccessfulMeasurements(detector_result(), classifier_result())

        def environment() -> dict[str, object]:
            if not measurements.detector_released:
                raise RuntimeError("environment captured before detector release")
            return {"torch": "2.8.0+cu128"}

        context = TensorRtReportContext(
            revision="a" * 40,
            measured_at="2026-08-10T00:00:00+00:00",
            environment=environment,
        )
        with TemporaryDirectory() as directory:
            result = run_tensorrt_experiment(
                context,
                measurements,
                output_dir=Path(directory),
                failure_roots=(Path(directory),),
            )

        self.assertEqual(result.report["environment"], {"torch": "2.8.0+cu128"})

    def test_stop_report_preserves_schema_reason_and_plugin_not_reached_truth(self) -> None:
        with TemporaryDirectory() as directory:
            result = run_tensorrt_experiment(
                report_context(),
                SuccessfulMeasurements(detector_result(), classifier_result()),
                output_dir=Path(directory),
                failure_roots=(Path(directory),),
            )

        self.assertEqual(TENSORRT_REPORT_SCHEMA_VERSION, 2)
        self.assertEqual(result.report["schema_version"], 2)
        self.assertEqual(result.report["decision"], "stop")
        self.assertIn(
            "token-width 2..90 profile did not pass",
            result.report["production_selection"]["reason"],
        )
        self.assertIn(
            "detector full coverage did not pass",
            result.report["production_selection"]["reason"],
        )
        detector = result.report["models"][DETECTOR_MODEL_ID]
        self.assertEqual(detector["plugin_requirement_status"], "not_reached")
        self.assertIsNone(detector["plugin_required"])
        self.assertIn("plugin requirement was not reached", result.markdown)

    def test_classifier_only_success_is_explicitly_partial(self) -> None:
        with TemporaryDirectory() as directory:
            result = run_tensorrt_experiment(
                report_context(),
                SuccessfulMeasurements(
                    detector_result(),
                    classifier_result(decision="go", shape_coverage=True),
                ),
                output_dir=Path(directory),
                failure_roots=(Path(directory),),
            )

        self.assertEqual(result.report["decision"], "partial")
        self.assertIn("PARTIAL", result.report["production_selection"]["reason"])
        self.assertIn(
            "not whole-pipeline TensorRT",
            result.report["production_selection"]["reason"],
        )
        self.assertIn("no fallback is enabled", result.report["production_selection"]["reason"])

    def test_both_model_successes_report_go_without_selecting_tensorrt(self) -> None:
        with TemporaryDirectory() as directory:
            result = run_tensorrt_experiment(
                report_context(),
                SuccessfulMeasurements(
                    detector_result(decision="go"),
                    classifier_result(decision="go", shape_coverage=True),
                ),
                output_dir=Path(directory),
                failure_roots=(Path(directory),),
            )

        self.assertEqual(result.report["decision"], "go")
        self.assertEqual(result.report["production_selection"]["backend"], "pytorch-eager")
        self.assertIn(
            "Production remains eager FP32",
            result.report["production_selection"]["reason"],
        )

    def test_adapter_go_flags_cannot_override_failed_measured_gates(self) -> None:
        detector = detector_result(decision="go")
        detector["engine_built"] = False
        classifier = classifier_result(decision="go", shape_coverage=True)
        classifier["tensorrt_only_runtime_passed"] = False
        with TemporaryDirectory() as directory:
            result = run_tensorrt_experiment(
                report_context(),
                SuccessfulMeasurements(detector, classifier),
                output_dir=Path(directory),
                failure_roots=(Path(directory),),
            )

        self.assertEqual(result.report["decision"], "stop")
        self.assertEqual(result.report["models"][DETECTOR_MODEL_ID]["decision"], "stop")
        self.assertEqual(result.report["models"][CLASSIFIER_MODEL_ID]["decision"], "stop")

    def test_classifier_failure_is_sanitized_and_candidate_plan_is_always_deleted(self) -> None:
        with TemporaryDirectory() as directory, TemporaryDirectory() as external_directory:
            output_dir = Path(directory)
            external_root = Path(external_directory)
            output_secret = output_dir / "private" / "engine.plan"
            secret_path = external_root / "private" / "checkpoint.pth"
            result = run_tensorrt_experiment(
                report_context(),
                FailingClassifierMeasurements(
                    output_dir,
                    f"failed at {output_secret} using {secret_path}",
                ),
                output_dir=output_dir,
                failure_roots=(external_root,),
            )

            failure_text = (output_dir / "mmbcd-tensorrt-failure.txt").read_text(encoding="utf-8")
            self.assertFalse((output_dir / "mmbcd-fp32.candidate.plan").exists())

        classifier = result.report["models"][CLASSIFIER_MODEL_ID]
        self.assertEqual(classifier["decision"], "stop")
        self.assertEqual(classifier["failure_code"], "classifier_tensorrt_failed:RuntimeError")
        self.assertEqual(len(classifier["failure_report_sha256"]), 64)
        self.assertNotIn(str(output_dir), failure_text)
        self.assertNotIn(str(external_root), failure_text)

    def test_stale_candidate_plan_is_deleted_even_when_detector_measurement_fails(self) -> None:
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            candidate_plan = output_dir / "mmbcd-fp32.candidate.plan"
            candidate_plan.write_bytes(b"stale")

            with self.assertRaisesRegex(RuntimeError, "detector measurement failed"):
                run_tensorrt_experiment(
                    report_context(),
                    FailingDetectorMeasurements(),
                    output_dir=output_dir,
                    failure_roots=(output_dir,),
                )

            self.assertFalse(candidate_plan.exists())


if __name__ == "__main__":
    unittest.main()
