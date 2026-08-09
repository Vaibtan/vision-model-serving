from __future__ import annotations

import os
import re
import subprocess
import sys
import unittest
from unittest.mock import patch
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier, Thread
from time import time

import fakeredis
from rq import Queue, SimpleWorker
from rq.job import Job
from rq.serializers import JSONSerializer
from rq.timeouts import TimerDeathPenalty

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from vision_model_serving.execution import (
    GatewayUnavailable,
    IdempotencyConflict,
    PredictionFailed,
    PredictionId,
    PredictionJobState,
    PredictionNotFound,
    PredictionRequest,
    QueueSaturated,
    ResultExpired,
    ResultNotReady,
    RqExecutionConfig,
    RqGpuExecutionGateway,
)
from vision_model_serving.execution.job_processor import StoredPredictionProcessor
from vision_model_serving.execution.rq_worker import create_prediction_rq_worker
from vision_model_serving.pipeline import CaseInput, PredictionMode


class FakeClock:
    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def request(
    *,
    idempotency_key: str | None = None,
    dicom_bytes: bytes = b"private-dicom",
) -> PredictionRequest:
    return PredictionRequest(
        case=CaseInput(BytesIO(dicom_bytes), "private history"),
        mode=PredictionMode.FULL,
        idempotency_key=idempotency_key,
    )


def prediction_result() -> object:
    from tests.test_prediction_pipeline import (
        DecoderFake,
        FullRuntimeFake,
        canonical_mammogram,
        classifier_result,
        detector_result,
    )
    from vision_model_serving.pipeline import PredictionPipeline

    mammogram = canonical_mammogram()
    return PredictionPipeline(
        decoder=DecoderFake(mammogram),
        runtime=FullRuntimeFake(
            detector_result(mammogram),
            classifier_result(),
            mammogram,
        ),
    ).infer(
        CaseInput(BytesIO(b"source"), "  prior   surgery  "),
        PredictionMode.FULL,
    )


class WindowsSimpleWorker(SimpleWorker):
    death_penalty_class = TimerDeathPenalty


class PipelineStub:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[tuple[bytes, str | None, PredictionMode]] = []

    def infer(self, case: CaseInput, mode: PredictionMode) -> object:
        self.calls.append((case.dicom_stream.read(), case.clinical_history, mode))
        return self.result


class FailingPipelineStub:
    def infer(self, case: CaseInput, mode: PredictionMode) -> object:
        raise RuntimeError(r"C:\patients\Alice\scan.dcm")


class RqGatewayTests(unittest.TestCase):
    def test_submission_enqueues_only_opaque_job_tokens(self) -> None:
        clock = FakeClock()
        redis = fakeredis.FakeRedis()
        with TemporaryDirectory() as directory:
            gateway = RqGpuExecutionGateway(
                redis_client=redis,
                job_root=Path(directory),
                config=RqExecutionConfig(
                    capacity=2,
                    reservation_ttl_seconds=30,
                    job_timeout_seconds=600,
                    result_ttl_seconds=60,
                    status_ttl_seconds=90,
                    queue_name="gpu-inference",
                    key_prefix="test:predictions",
                ),
                clock=clock,
            )

            handle = gateway.submit(request(idempotency_key="private-key"))
            job = Job.fetch(
                str(handle.prediction_id),
                connection=redis,
                serializer=JSONSerializer,
            )

        self.assertEqual(handle.state, PredictionJobState.QUEUED)
        self.assertRegex(str(handle.prediction_id), re.compile(r"^[A-Za-z0-9_-]{32}$"))
        self.assertEqual(
            job.func_name,
            "vision_model_serving.execution.rq_worker.execute_prediction_job",
        )
        self.assertEqual(job.args[0], str(handle.prediction_id))
        self.assertRegex(job.args[1], re.compile(r"^[A-Za-z0-9_-]{32}$"))
        self.assertNotIn(b"private-dicom", job.data)
        self.assertNotIn(b"private history", job.data)
        self.assertNotIn(b"private-key", job.data)

    def test_duplicate_idempotency_key_reuses_the_existing_rq_job(self) -> None:
        redis = fakeredis.FakeRedis()
        with TemporaryDirectory() as directory:
            gateway = RqGpuExecutionGateway(
                redis_client=redis,
                job_root=Path(directory),
                config=RqExecutionConfig(
                    capacity=1,
                    reservation_ttl_seconds=30,
                    job_timeout_seconds=600,
                    result_ttl_seconds=60,
                    status_ttl_seconds=90,
                ),
            )

            first = gateway.submit(request(idempotency_key="same-upload"))
            replay = gateway.submit(request(idempotency_key="same-upload"))

        self.assertEqual(replay.prediction_id, first.prediction_id)
        self.assertTrue(replay.idempotent_replay)
        self.assertEqual(redis.llen("rq:queue:gpu-inference"), 1)

    def test_conflicting_idempotency_key_is_rejected_without_another_job(self) -> None:
        redis = fakeredis.FakeRedis()
        with TemporaryDirectory() as directory:
            gateway = RqGpuExecutionGateway(
                redis_client=redis,
                job_root=Path(directory),
                config=RqExecutionConfig(
                    capacity=2,
                    reservation_ttl_seconds=30,
                    job_timeout_seconds=600,
                    result_ttl_seconds=60,
                    status_ttl_seconds=90,
                ),
            )
            gateway.submit(request(idempotency_key="same-upload"))

            with self.assertRaises(IdempotencyConflict):
                gateway.submit(
                    request(
                        idempotency_key="same-upload",
                        dicom_bytes=b"different-dicom",
                    )
                )

        self.assertEqual(redis.llen("rq:queue:gpu-inference"), 1)

    def test_wait_timeout_keeps_the_job_pollable_and_unknown_ids_are_stable(
        self,
    ) -> None:
        redis = fakeredis.FakeRedis()
        with TemporaryDirectory() as directory:
            gateway = RqGpuExecutionGateway(
                redis_client=redis,
                job_root=Path(directory),
                config=RqExecutionConfig(
                    capacity=1,
                    reservation_ttl_seconds=30,
                    job_timeout_seconds=600,
                    result_ttl_seconds=60,
                    status_ttl_seconds=90,
                ),
            )
            submitted = gateway.submit(request())

            timed_out = gateway.wait(submitted.prediction_id, timeout_seconds=0)
            with self.assertRaises(PredictionNotFound):
                gateway.status(PredictionId("unknown"))

        self.assertEqual(timed_out.prediction_id, submitted.prediction_id)
        self.assertEqual(timed_out.state, PredictionJobState.QUEUED)
        self.assertEqual(redis.llen("rq:queue:gpu-inference"), 1)

    def test_rq_worker_completes_a_job_through_the_gateway_interface(self) -> None:
        redis = fakeredis.FakeRedis()
        expected = prediction_result()
        pipeline = PipelineStub(expected)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            processor = StoredPredictionProcessor(
                job_root=root,
                pipeline=pipeline,
                result_ttl_seconds=60,
            )
            gateway = RqGpuExecutionGateway(
                redis_client=redis,
                job_root=root,
                config=RqExecutionConfig(
                    capacity=1,
                    reservation_ttl_seconds=30,
                    job_timeout_seconds=600,
                    result_ttl_seconds=60,
                    status_ttl_seconds=90,
                ),
            )
            with patch(
                "vision_model_serving.execution.rq_worker.GpuExecutorClient",
                return_value=processor,
            ):
                create_prediction_rq_worker(
                    redis_client=redis,
                    queue_name="gpu-inference",
                    executor_socket_path=root / "executor.sock",
                    executor_timeout_seconds=60,
                )
                handle = gateway.submit(request(idempotency_key="completed-upload"))

                with self.assertRaises(ResultNotReady):
                    gateway.result(handle.prediction_id)
                queue = Queue(
                    "gpu-inference",
                    connection=redis,
                    serializer=JSONSerializer,
                )
                worker = WindowsSimpleWorker(
                    [queue],
                    connection=redis,
                    serializer=JSONSerializer,
                )
                worker.work(burst=True, logging_level="WARNING")

                status = gateway.status(handle.prediction_id)
                result = gateway.result(handle.prediction_id)
                replay = gateway.submit(request(idempotency_key="completed-upload"))

        self.assertEqual(status.state, PredictionJobState.SUCCEEDED)
        self.assertEqual(result, expected)
        self.assertEqual(replay.prediction_id, handle.prediction_id)
        self.assertEqual(replay.state, PredictionJobState.SUCCEEDED)
        self.assertEqual(
            pipeline.calls,
            [(b"private-dicom", "private history", PredictionMode.FULL)],
        )

    def test_worker_failure_is_terminal_sanitized_and_not_retried(self) -> None:
        redis = fakeredis.FakeRedis()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            processor = StoredPredictionProcessor(
                job_root=root,
                pipeline=FailingPipelineStub(),
                result_ttl_seconds=60,
            )
            gateway = RqGpuExecutionGateway(
                redis_client=redis,
                job_root=root,
                config=RqExecutionConfig(
                    capacity=1,
                    reservation_ttl_seconds=30,
                    job_timeout_seconds=600,
                    result_ttl_seconds=60,
                    status_ttl_seconds=90,
                ),
            )
            with patch(
                "vision_model_serving.execution.rq_worker.GpuExecutorClient",
                return_value=processor,
            ):
                create_prediction_rq_worker(
                    redis_client=redis,
                    queue_name="gpu-inference",
                    executor_socket_path=root / "executor.sock",
                    executor_timeout_seconds=60,
                )
                handle = gateway.submit(request())
                queue = Queue(
                    "gpu-inference",
                    connection=redis,
                    serializer=JSONSerializer,
                )
                WindowsSimpleWorker(
                    [queue],
                    connection=redis,
                    serializer=JSONSerializer,
                ).work(burst=True, logging_level="CRITICAL")

                status = gateway.status(handle.prediction_id)
                job = Job.fetch(
                    str(handle.prediction_id),
                    connection=redis,
                    serializer=JSONSerializer,
                )
                with self.assertRaises(PredictionFailed) as raised:
                    gateway.result(handle.prediction_id)
                observations = gateway.observations()

        self.assertEqual(status.state, PredictionJobState.FAILED)
        self.assertEqual(status.failure.code, "prediction_execution_failed")
        self.assertNotIn("Alice", status.failure.detail)
        self.assertNotIn("Alice", str(raised.exception))
        self.assertIsNone(job.retries_left)
        self.assertEqual(observations.active_jobs, 0)
        self.assertEqual(observations.failed_total, 1)

    def test_expired_result_is_reported_and_releases_idempotency(self) -> None:
        redis = fakeredis.FakeRedis()
        clock = FakeClock(time())
        with TemporaryDirectory() as directory:
            root = Path(directory)
            processor = StoredPredictionProcessor(
                job_root=root,
                pipeline=PipelineStub(prediction_result()),
                result_ttl_seconds=5,
                clock=clock,
            )
            gateway = RqGpuExecutionGateway(
                redis_client=redis,
                job_root=root,
                config=RqExecutionConfig(
                    capacity=1,
                    reservation_ttl_seconds=30,
                    job_timeout_seconds=600,
                    result_ttl_seconds=5,
                    status_ttl_seconds=30,
                ),
                clock=clock,
            )
            with patch(
                "vision_model_serving.execution.rq_worker.GpuExecutorClient",
                return_value=processor,
            ):
                create_prediction_rq_worker(
                    redis_client=redis,
                    queue_name="gpu-inference",
                    executor_socket_path=root / "executor.sock",
                    executor_timeout_seconds=60,
                )
                first = gateway.submit(request(idempotency_key="expiring-upload"))
                queue = Queue(
                    "gpu-inference",
                    connection=redis,
                    serializer=JSONSerializer,
                )
                WindowsSimpleWorker(
                    [queue],
                    connection=redis,
                    serializer=JSONSerializer,
                ).work(burst=True, logging_level="WARNING")
                clock.advance(6)

                status = gateway.status(first.prediction_id)
                with self.assertRaises(ResultExpired) as raised:
                    gateway.result(first.prediction_id)
                replacement = gateway.submit(
                    request(idempotency_key="expiring-upload")
                )

        self.assertEqual(status.state, PredictionJobState.EXPIRED)
        self.assertEqual(
            getattr(raised.exception, "code", None), "prediction_result_expired"
        )
        self.assertNotEqual(replacement.prediction_id, first.prediction_id)

    def test_standard_rq_worker_configures_only_the_executor_socket(self) -> None:
        redis = fakeredis.FakeRedis()

        with TemporaryDirectory() as directory:
            with patch(
                "vision_model_serving.execution.rq_worker.GpuExecutorClient"
            ) as executor_client:
                worker = create_prediction_rq_worker(
                    redis_client=redis,
                    queue_name="gpu-inference",
                    executor_socket_path=Path(directory) / "executor.sock",
                    executor_timeout_seconds=60,
                )

        executor_client.assert_not_called()
        self.assertEqual([queue.name for queue in worker.queues], ["gpu-inference"])
        self.assertIs(worker.serializer, JSONSerializer)

    def test_redis_loss_is_sanitized_and_discards_the_staged_payload(self) -> None:
        server = fakeredis.FakeServer()
        redis = fakeredis.FakeRedis(server=server)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            gateway = RqGpuExecutionGateway(
                redis_client=redis,
                job_root=root,
                config=RqExecutionConfig(
                    capacity=1,
                    reservation_ttl_seconds=30,
                    job_timeout_seconds=600,
                    result_ttl_seconds=60,
                    status_ttl_seconds=90,
                ),
            )
            submitted = gateway.submit(request())
            staged_before_failure = list(root.iterdir())
            server.connected = False

            with self.assertRaises(GatewayUnavailable) as status_error:
                gateway.status(submitted.prediction_id)
            with self.assertRaises(GatewayUnavailable) as raised:
                gateway.submit(request(dicom_bytes=b"another-dicom"))

            staged_directories = list(root.iterdir())

        self.assertNotIn("redis", str(raised.exception).lower())
        self.assertNotIn("redis", str(status_error.exception).lower())
        self.assertEqual(staged_directories, staged_before_failure)

    def test_rq_worker_import_cannot_initialize_cuda_or_import_the_processor(self) -> None:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(REPOSITORY_ROOT / "src")
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "import vision_model_serving.execution.rq_worker; "
                    "assert 'torch' not in sys.modules; "
                    "assert 'vision_model_serving.execution._composition' not in sys.modules; "
                    "assert 'vision_model_serving.execution.job_processor' not in sys.modules"
                ),
            ],
            cwd=REPOSITORY_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_capacity_is_reserved_atomically_across_gateway_instances(self) -> None:
        redis = fakeredis.FakeRedis()
        config = RqExecutionConfig(
            capacity=1,
            reservation_ttl_seconds=30,
            job_timeout_seconds=600,
            result_ttl_seconds=60,
            status_ttl_seconds=90,
        )
        with TemporaryDirectory() as directory:
            gateways = tuple(
                RqGpuExecutionGateway(
                    redis_client=redis,
                    job_root=Path(directory),
                    config=config,
                )
                for _ in range(2)
            )
            barrier = Barrier(3)
            handles: list[object] = []
            errors: list[Exception] = []

            def submit(gateway: RqGpuExecutionGateway) -> None:
                barrier.wait()
                try:
                    handles.append(gateway.submit(request()))
                except Exception as error:  # noqa: BLE001 - surface thread failures
                    errors.append(error)

            threads = [Thread(target=submit, args=(gateway,)) for gateway in gateways]
            for thread in threads:
                thread.start()
            barrier.wait()
            for thread in threads:
                thread.join(2)

        self.assertEqual(len(handles), 1)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], QueueSaturated)
        self.assertEqual(redis.llen("rq:queue:gpu-inference"), 1)
        observations = gateways[0].observations()
        self.assertEqual(observations.active_jobs, 1)
        self.assertEqual(observations.queued_jobs, 1)
        self.assertEqual(observations.admitted_total, 1)
        self.assertEqual(observations.rejected_total, 1)


if __name__ == "__main__":
    unittest.main()
