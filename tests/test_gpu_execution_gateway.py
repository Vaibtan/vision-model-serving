from __future__ import annotations

from concurrent.futures import Future
from io import BytesIO
import json
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Barrier, Event, Thread
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from vision_model_serving.execution import (  # noqa: E402
    IdempotencyConflict,
    EphemeralJobStore,
    CeleryTaskDispatcher,
    CeleryRedisGpuExecutionGateway,
    GpuExecutionConfig,
    GPU_TASK_NAME,
    InMemoryPredictionStateRepository,
    GatewayUnavailable,
    InMemoryGpuExecutionGateway,
    JobPayloadNotFound,
    JobResultNotFound,
    PredictionId,
    PredictionJobState,
    PredictionJobWorker,
    PredictionNotFound,
    PredictionFailed,
    PredictionRequest,
    QueueSaturated,
    RedisPredictionStateRepository,
    ResultExpired,
    ResultNotReady,
    StoredGpuExecutionGateway,
    configure_gpu_worker,
    register_prediction_task,
)
from vision_model_serving.pipeline import CaseInput, PredictionMode  # noqa: E402


class CapturingExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[object, tuple[object, ...]]] = []

    def submit(self, function: object, *args: object) -> Future[object]:
        self.calls.append((function, args))
        return Future()

    def run_next(self) -> None:
        function, args = self.calls.pop(0)
        function(*args)


class CapturingDispatcher:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def dispatch(self, prediction_id: PredictionId, locator: str) -> None:
        self.messages.append((str(prediction_id), locator))


class FailingDispatcher(CapturingDispatcher):
    def __init__(self) -> None:
        super().__init__()
        self.fail = True

    def dispatch(self, prediction_id: PredictionId, locator: str) -> None:
        if self.fail:
            raise RuntimeError("redis://secret@private-host")
        super().dispatch(prediction_id, locator)


class CeleryAppFake:
    def __init__(self) -> None:
        self.conf: dict[str, object] = {}
        self.sent: list[dict[str, object]] = []
        self.task_options: dict[str, object] | None = None

    def send_task(self, name: str, **options: object) -> None:
        self.sent.append({"name": name, **options})

    def task(self, **options: object):
        self.task_options = options

        def decorate(function: object) -> object:
            return function

        return decorate


class RedisClientFake:
    def __init__(self, *results: object) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, int, tuple[object, ...]]] = []

    def eval(self, script: str, key_count: int, *values: object) -> object:
        self.calls.append((script, key_count, values))
        result = self.results.pop(0)
        return result(values) if callable(result) else result


class PostExecutionStateFailure(InMemoryPredictionStateRepository):
    def mark_succeeded(self, *args: object, **kwargs: object) -> bool:
        raise GatewayUnavailable("prediction state store is unavailable")


class PipelineStub:
    def __init__(self, result: object | None = None) -> None:
        self.result = result
        self.calls: list[tuple[bytes, str | None, PredictionMode]] = []

    def infer(self, case: CaseInput, mode: PredictionMode) -> object:
        self.calls.append((case.dicom_stream.read(), case.clinical_history, mode))
        return self.result


class FailingPipelineStub(PipelineStub):
    def infer(self, case: CaseInput, mode: PredictionMode) -> object:
        raise RuntimeError(r"C:\patients\Alice\scan.dcm")


class FakeClock:
    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class BlockingPipelineStub(PipelineStub):
    def __init__(self, entered: Event, release: Event, result: object) -> None:
        super().__init__(result)
        self.entered = entered
        self.release = release

    def infer(self, case: CaseInput, mode: PredictionMode) -> object:
        self.entered.set()
        if not self.release.wait(2):
            raise TimeoutError("test release was not signaled")
        return self.result


def request(
    *,
    idempotency_key: str | None = None,
    dicom_bytes: bytes = b"dicom",
) -> PredictionRequest:
    return PredictionRequest(
        case=CaseInput(BytesIO(dicom_bytes), "history"),
        mode=PredictionMode.FULL,
        idempotency_key=idempotency_key,
    )


def real_prediction_result() -> object:
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


class InMemorySubmissionTests(unittest.TestCase):
    def test_submit_returns_an_opaque_queued_handle(self) -> None:
        executor = CapturingExecutor()
        gateway = InMemoryGpuExecutionGateway(
            pipeline=PipelineStub(),
            capacity=2,
            executor=executor,
        )

        handle = gateway.submit(request(idempotency_key="browser-request-1"))
        status = gateway.status(handle.prediction_id)

        self.assertRegex(str(handle.prediction_id), re.compile(r"^[A-Za-z0-9_-]{32}$"))
        self.assertNotIn("browser-request", str(handle.prediction_id))
        self.assertEqual(handle.state, PredictionJobState.QUEUED)
        self.assertEqual(status.state, PredictionJobState.QUEUED)
        self.assertEqual(status.prediction_id, handle.prediction_id)
        self.assertEqual(len(executor.calls), 1)

    def test_duplicate_idempotency_key_returns_the_existing_job(self) -> None:
        executor = CapturingExecutor()
        gateway = InMemoryGpuExecutionGateway(
            pipeline=PipelineStub(),
            capacity=2,
            executor=executor,
        )

        first = gateway.submit(request(idempotency_key="same-upload"))
        replay = gateway.submit(request(idempotency_key="same-upload"))

        self.assertEqual(replay.prediction_id, first.prediction_id)
        self.assertTrue(replay.idempotent_replay)
        self.assertEqual(len(executor.calls), 1)

    def test_reusing_an_idempotency_key_for_other_input_is_rejected(self) -> None:
        executor = CapturingExecutor()
        gateway = InMemoryGpuExecutionGateway(
            pipeline=PipelineStub(),
            capacity=2,
            executor=executor,
        )
        gateway.submit(request(idempotency_key="same-key"))

        with self.assertRaises(IdempotencyConflict) as raised:
            gateway.submit(
                request(idempotency_key="same-key", dicom_bytes=b"other-dicom")
            )

        self.assertEqual(raised.exception.code, "prediction_idempotency_conflict")
        self.assertEqual(len(executor.calls), 1)

    def test_capacity_admission_is_atomic_for_concurrent_submitters(self) -> None:
        executor = CapturingExecutor()
        gateway = InMemoryGpuExecutionGateway(
            pipeline=PipelineStub(),
            capacity=1,
            executor=executor,
        )
        barrier = Barrier(3)
        handles: list[object] = []
        errors: list[BaseException] = []

        def submit(index: int) -> None:
            barrier.wait()
            try:
                handles.append(gateway.submit(request(dicom_bytes=f"d{index}".encode())))
            except BaseException as error:  # test thread must surface failures
                errors.append(error)

        threads = [Thread(target=submit, args=(index,)) for index in range(2)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(2)

        self.assertEqual(len(handles), 1)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], QueueSaturated)
        self.assertEqual(getattr(errors[0], "code", None), "prediction_queue_full")
        self.assertEqual(len(executor.calls), 1)


class InMemoryLifecycleTests(unittest.TestCase):
    def test_queued_job_runs_once_and_exposes_its_result(self) -> None:
        expected = object()
        executor = CapturingExecutor()
        pipeline = PipelineStub(expected)
        gateway = InMemoryGpuExecutionGateway(
            pipeline=pipeline,
            capacity=1,
            executor=executor,
        )
        handle = gateway.submit(request())

        with self.assertRaises(ResultNotReady):
            gateway.result(handle.prediction_id)
        executor.run_next()

        status = gateway.status(handle.prediction_id)
        self.assertEqual(status.state, PredictionJobState.SUCCEEDED)
        self.assertIsNotNone(status.started_at)
        self.assertIsNotNone(status.completed_at)
        self.assertIs(gateway.result(handle.prediction_id), expected)
        self.assertEqual(
            pipeline.calls,
            [(b"dicom", "history", PredictionMode.FULL)],
        )

    def test_execution_failure_is_terminal_sanitized_and_releases_capacity(self) -> None:
        executor = CapturingExecutor()
        gateway = InMemoryGpuExecutionGateway(
            pipeline=FailingPipelineStub(),
            capacity=1,
            executor=executor,
        )
        handle = gateway.submit(request())

        executor.run_next()

        status = gateway.status(handle.prediction_id)
        self.assertEqual(status.state, PredictionJobState.FAILED)
        self.assertEqual(status.failure.code, "prediction_execution_failed")
        self.assertNotIn("Alice", status.failure.detail)
        with self.assertRaises(PredictionFailed) as raised:
            gateway.result(handle.prediction_id)
        self.assertNotIn("Alice", str(raised.exception))
        gateway.submit(request(dicom_bytes=b"second"))

    def test_completed_result_expires_without_reusing_its_idempotency_binding(self) -> None:
        clock = FakeClock()
        executor = CapturingExecutor()
        gateway = InMemoryGpuExecutionGateway(
            pipeline=PipelineStub(object()),
            capacity=1,
            executor=executor,
            result_ttl_seconds=5,
            clock=clock,
        )
        first = gateway.submit(request(idempotency_key="expiring"))
        executor.run_next()
        clock.advance(6)

        status = gateway.status(first.prediction_id)

        self.assertEqual(status.state, PredictionJobState.EXPIRED)
        with self.assertRaises(ResultExpired):
            gateway.result(first.prediction_id)
        second = gateway.submit(request(idempotency_key="expiring"))
        self.assertNotEqual(second.prediction_id, first.prediction_id)

    def test_abandoned_queue_reservation_fails_and_releases_admission(self) -> None:
        clock = FakeClock()
        executor = CapturingExecutor()
        pipeline = PipelineStub(object())
        gateway = InMemoryGpuExecutionGateway(
            pipeline=pipeline,
            capacity=1,
            executor=executor,
            reservation_ttl_seconds=5,
            clock=clock,
        )
        abandoned = gateway.submit(request())
        clock.advance(6)

        status = gateway.status(abandoned.prediction_id)

        self.assertEqual(status.state, PredictionJobState.FAILED)
        self.assertEqual(status.failure.code, "prediction_reservation_expired")
        executor.run_next()
        self.assertEqual(pipeline.calls, [])
        gateway.submit(request(dicom_bytes=b"replacement"))

    def test_expired_running_lease_is_worker_loss_and_late_success_is_ignored(self) -> None:
        clock = FakeClock()
        executor = CapturingExecutor()
        entered = Event()
        release = Event()
        gateway = InMemoryGpuExecutionGateway(
            pipeline=BlockingPipelineStub(entered, release, object()),
            capacity=1,
            executor=executor,
            worker_loss_ttl_seconds=5,
            clock=clock,
        )
        handle = gateway.submit(request())
        worker = Thread(target=executor.run_next)
        worker.start()
        self.assertTrue(entered.wait(1))
        clock.advance(6)

        lost = gateway.status(handle.prediction_id)
        release.set()
        worker.join(2)

        self.assertEqual(lost.state, PredictionJobState.FAILED)
        self.assertEqual(lost.failure.code, "prediction_worker_lost")
        self.assertEqual(
            gateway.status(handle.prediction_id).state,
            PredictionJobState.FAILED,
        )
        gateway.submit(request(dicom_bytes=b"replacement"))

    def test_bounded_wait_returns_a_handle_without_cancelling_then_returns_result(self) -> None:
        expected = object()
        executor = CapturingExecutor()
        gateway = InMemoryGpuExecutionGateway(
            pipeline=PipelineStub(expected),
            capacity=1,
            executor=executor,
        )
        submitted = gateway.submit(request())

        timed_out = gateway.wait(submitted.prediction_id, timeout_seconds=0)
        executor.run_next()
        completed = gateway.wait(submitted.prediction_id, timeout_seconds=0)

        self.assertEqual(timed_out.prediction_id, submitted.prediction_id)
        self.assertEqual(timed_out.state, PredictionJobState.QUEUED)
        self.assertIs(completed, expected)

    def test_default_adapter_owns_one_background_worker_and_closes_cleanly(self) -> None:
        expected = object()

        with InMemoryGpuExecutionGateway(
            pipeline=PipelineStub(expected),
            capacity=1,
        ) as gateway:
            handle = gateway.submit(request())
            completed = gateway.wait(handle.prediction_id, timeout_seconds=2)

        self.assertIs(completed, expected)

    def test_unknown_prediction_has_a_stable_error(self) -> None:
        gateway = InMemoryGpuExecutionGateway(
            pipeline=PipelineStub(),
            capacity=1,
            executor=CapturingExecutor(),
        )

        with self.assertRaises(PredictionNotFound):
            gateway.status(PredictionId("unknown"))
        with self.assertRaises(PredictionNotFound):
            gateway.result(PredictionId("unknown"))


class EphemeralJobStoreTests(unittest.TestCase):
    def test_request_payload_round_trips_under_an_opaque_locator_then_purges(self) -> None:
        clock = FakeClock()
        with TemporaryDirectory() as directory:
            store = EphemeralJobStore(Path(directory), clock=clock)
            prediction_id = PredictionId("prediction-id")

            stored = store.store_request(
                prediction_id,
                request(dicom_bytes=b"private-dicom"),
                ttl_seconds=30,
            )
            loaded = store.load_request(prediction_id, stored.locator)

            self.assertRegex(stored.locator, re.compile(r"^[A-Za-z0-9_-]{32}$"))
            self.assertNotIn(str(prediction_id), stored.locator)
            self.assertEqual(loaded.mode, PredictionMode.FULL)
            self.assertEqual(loaded.case.clinical_history, "history")
            self.assertEqual(loaded.case.dicom_stream.read(), b"private-dicom")
            self.assertRegex(stored.request_fingerprint, re.compile(r"^[0-9a-f]{64}$"))

            store.purge_request(stored.locator)
            with self.assertRaises(JobPayloadNotFound):
                store.load_request(prediction_id, stored.locator)

    def test_result_write_is_idempotent_private_and_ttl_cleaned(self) -> None:
        result = real_prediction_result()
        clock = FakeClock()
        with TemporaryDirectory() as directory:
            store = EphemeralJobStore(Path(directory), clock=clock)
            prediction_id = PredictionId("prediction-id")
            stored = store.store_request(
                prediction_id,
                request(dicom_bytes=b"private-dicom"),
                ttl_seconds=30,
            )

            store.store_result(stored.locator, result, ttl_seconds=5)
            store.store_result(stored.locator, result, ttl_seconds=5)

            self.assertEqual(store.load_result(stored.locator), result)
            with self.assertRaises(JobPayloadNotFound):
                store.load_request(prediction_id, stored.locator)
            clock.advance(6)
            self.assertEqual(store.cleanup_expired(), 1)
            with self.assertRaises(JobResultNotFound):
                store.load_result(stored.locator)


class StoredGatewaySubmissionTests(unittest.TestCase):
    def test_broker_message_contains_only_opaque_job_tokens(self) -> None:
        clock = FakeClock()
        dispatcher = CapturingDispatcher()
        with TemporaryDirectory() as directory:
            store = EphemeralJobStore(Path(directory), clock=clock)
            state = InMemoryPredictionStateRepository(
                capacity=2,
                result_ttl_seconds=30,
                clock=clock,
            )
            gateway = StoredGpuExecutionGateway(
                state=state,
                store=store,
                dispatcher=dispatcher,
                reservation_ttl_seconds=10,
                clock=clock,
            )

            handle = gateway.submit(
                request(
                    idempotency_key="private-browser-key",
                    dicom_bytes=b"private-dicom-payload",
                )
            )

            self.assertEqual(len(dispatcher.messages), 1)
            broker_payload = json.dumps(dispatcher.messages[0])
            self.assertEqual(dispatcher.messages[0][0], str(handle.prediction_id))
            self.assertNotIn("private-dicom", broker_payload)
            self.assertNotIn("history", broker_payload)
            self.assertNotIn("browser-key", broker_payload)
            stored_request = store.load_request(
                handle.prediction_id,
                dispatcher.messages[0][1],
            )
            self.assertEqual(
                stored_request.case.dicom_stream.read(),
                b"private-dicom-payload",
            )

    def test_duplicate_submission_reuses_one_dispatch_and_one_payload(self) -> None:
        clock = FakeClock()
        dispatcher = CapturingDispatcher()
        with TemporaryDirectory() as directory:
            store = EphemeralJobStore(Path(directory), clock=clock)
            state = InMemoryPredictionStateRepository(
                capacity=1,
                result_ttl_seconds=30,
                clock=clock,
            )
            gateway = StoredGpuExecutionGateway(
                state=state,
                store=store,
                dispatcher=dispatcher,
                reservation_ttl_seconds=10,
                clock=clock,
            )

            first = gateway.submit(request(idempotency_key="same"))
            replay = gateway.submit(request(idempotency_key="same"))

            self.assertEqual(replay.prediction_id, first.prediction_id)
            self.assertTrue(replay.idempotent_replay)
            self.assertEqual(len(dispatcher.messages), 1)
            self.assertEqual(len(list(Path(directory).iterdir())), 1)

    def test_dispatch_failure_is_sanitized_cleaned_and_releases_admission(self) -> None:
        clock = FakeClock()
        dispatcher = FailingDispatcher()
        with TemporaryDirectory() as directory:
            store = EphemeralJobStore(Path(directory), clock=clock)
            state = InMemoryPredictionStateRepository(
                capacity=1,
                result_ttl_seconds=30,
                clock=clock,
            )
            gateway = StoredGpuExecutionGateway(
                state=state,
                store=store,
                dispatcher=dispatcher,
                reservation_ttl_seconds=10,
                clock=clock,
            )

            with self.assertRaises(GatewayUnavailable) as raised:
                gateway.submit(request(dicom_bytes=b"private"))

            self.assertNotIn("secret", str(raised.exception))
            self.assertEqual(list(Path(directory).iterdir()), [])
            dispatcher.fail = False
            gateway.submit(request(dicom_bytes=b"replacement"))


class StoredGatewayLifecycleTests(unittest.TestCase):
    def test_worker_transitions_job_writes_result_and_is_idempotent(self) -> None:
        clock = FakeClock()
        dispatcher = CapturingDispatcher()
        result = real_prediction_result()
        pipeline = PipelineStub(result)
        with TemporaryDirectory() as directory:
            store = EphemeralJobStore(Path(directory), clock=clock)
            state = InMemoryPredictionStateRepository(
                capacity=1,
                result_ttl_seconds=30,
                clock=clock,
            )
            gateway = StoredGpuExecutionGateway(
                state=state,
                store=store,
                dispatcher=dispatcher,
                reservation_ttl_seconds=10,
                clock=clock,
            )
            worker = PredictionJobWorker(
                state=state,
                store=store,
                pipeline=pipeline,
                worker_loss_ttl_seconds=60,
                result_ttl_seconds=30,
                clock=clock,
            )
            handle = gateway.submit(request())
            prediction_id, locator = dispatcher.messages[0]

            worker.execute(PredictionId(prediction_id), locator)
            worker.execute(PredictionId(prediction_id), locator)

            self.assertEqual(
                gateway.status(handle.prediction_id).state,
                PredictionJobState.SUCCEEDED,
            )
            self.assertEqual(gateway.result(handle.prediction_id), result)
            self.assertEqual(len(pipeline.calls), 1)
            with self.assertRaises(JobPayloadNotFound):
                store.load_request(handle.prediction_id, locator)

    def test_abandoned_stored_job_expires_and_releases_atomic_admission(self) -> None:
        clock = FakeClock()
        dispatcher = CapturingDispatcher()
        with TemporaryDirectory() as directory:
            store = EphemeralJobStore(Path(directory), clock=clock)
            state = InMemoryPredictionStateRepository(
                capacity=1,
                result_ttl_seconds=30,
                clock=clock,
            )
            gateway = StoredGpuExecutionGateway(
                state=state,
                store=store,
                dispatcher=dispatcher,
                reservation_ttl_seconds=10,
                clock=clock,
            )
            abandoned = gateway.submit(request())
            clock.advance(11)

            status = gateway.status(abandoned.prediction_id)

            self.assertEqual(status.state, PredictionJobState.FAILED)
            self.assertEqual(status.failure.code, "prediction_reservation_expired")
            self.assertEqual(store.cleanup_expired(), 1)
            gateway.submit(request(dicom_bytes=b"replacement"))

    def test_stored_running_lease_expiry_is_worker_loss(self) -> None:
        clock = FakeClock()
        dispatcher = CapturingDispatcher()
        entered = Event()
        release = Event()
        with TemporaryDirectory() as directory:
            store = EphemeralJobStore(Path(directory), clock=clock)
            state = InMemoryPredictionStateRepository(
                capacity=1,
                result_ttl_seconds=30,
                clock=clock,
            )
            gateway = StoredGpuExecutionGateway(
                state=state,
                store=store,
                dispatcher=dispatcher,
                reservation_ttl_seconds=10,
                clock=clock,
            )
            worker = PredictionJobWorker(
                state=state,
                store=store,
                pipeline=BlockingPipelineStub(
                    entered,
                    release,
                    real_prediction_result(),
                ),
                worker_loss_ttl_seconds=60,
                result_ttl_seconds=30,
                clock=clock,
            )
            handle = gateway.submit(request())
            prediction_id, locator = dispatcher.messages[0]
            thread = Thread(
                target=worker.execute,
                args=(PredictionId(prediction_id), locator),
            )
            thread.start()
            self.assertTrue(entered.wait(1))
            clock.advance(61)

            lost = gateway.status(handle.prediction_id)
            release.set()
            thread.join(2)

            self.assertEqual(lost.state, PredictionJobState.FAILED)
            self.assertEqual(lost.failure.code, "prediction_worker_lost")
            self.assertEqual(
                gateway.status(handle.prediction_id).state,
                PredictionJobState.FAILED,
            )
            gateway.submit(request(dicom_bytes=b"replacement"))

    def test_stored_gateway_wait_times_out_to_handle_without_cancelling(self) -> None:
        clock = FakeClock()
        dispatcher = CapturingDispatcher()
        result = real_prediction_result()
        with TemporaryDirectory() as directory:
            store = EphemeralJobStore(Path(directory), clock=clock)
            state = InMemoryPredictionStateRepository(
                capacity=1,
                result_ttl_seconds=30,
                clock=clock,
            )
            gateway = StoredGpuExecutionGateway(
                state=state,
                store=store,
                dispatcher=dispatcher,
                reservation_ttl_seconds=10,
                clock=clock,
            )
            worker = PredictionJobWorker(
                state=state,
                store=store,
                pipeline=PipelineStub(result),
                worker_loss_ttl_seconds=60,
                result_ttl_seconds=30,
                clock=clock,
            )
            submitted = gateway.submit(request())

            timed_out = gateway.wait(submitted.prediction_id, timeout_seconds=0)
            prediction_id, locator = dispatcher.messages[0]
            worker.execute(PredictionId(prediction_id), locator)
            completed = gateway.wait(submitted.prediction_id, timeout_seconds=0)

            self.assertEqual(timed_out.state, PredictionJobState.QUEUED)
            self.assertEqual(timed_out.prediction_id, submitted.prediction_id)
            self.assertEqual(completed, result)

    def test_gateway_observations_report_depth_wait_rejection_and_lifecycle(self) -> None:
        clock = FakeClock()
        dispatcher = CapturingDispatcher()
        with TemporaryDirectory() as directory:
            store = EphemeralJobStore(Path(directory), clock=clock)
            state = InMemoryPredictionStateRepository(
                capacity=1,
                result_ttl_seconds=30,
                clock=clock,
            )
            gateway = StoredGpuExecutionGateway(
                state=state,
                store=store,
                dispatcher=dispatcher,
                reservation_ttl_seconds=10,
                clock=clock,
            )
            worker = PredictionJobWorker(
                state=state,
                store=store,
                pipeline=PipelineStub(real_prediction_result()),
                worker_loss_ttl_seconds=60,
                result_ttl_seconds=30,
                clock=clock,
            )
            gateway.submit(request())
            with self.assertRaises(QueueSaturated):
                gateway.submit(request(dicom_bytes=b"rejected"))
            clock.advance(2)
            prediction_id, locator = dispatcher.messages[0]
            worker.execute(PredictionId(prediction_id), locator)

            observations = gateway.observations()

            self.assertEqual(observations.active_jobs, 0)
            self.assertEqual(observations.queued_jobs, 0)
            self.assertEqual(observations.running_jobs, 0)
            self.assertEqual(observations.admitted_total, 1)
            self.assertEqual(observations.rejected_total, 1)
            self.assertEqual(observations.succeeded_total, 1)
            self.assertEqual(observations.failed_total, 0)
            self.assertEqual(observations.queue_wait_ms_total, 2_000)

    def test_post_execution_state_failure_never_repeats_gpu_work(self) -> None:
        clock = FakeClock()
        dispatcher = CapturingDispatcher()
        result = real_prediction_result()
        pipeline = PipelineStub(result)
        with TemporaryDirectory() as directory:
            store = EphemeralJobStore(Path(directory), clock=clock)
            state = PostExecutionStateFailure(
                capacity=1,
                result_ttl_seconds=30,
                clock=clock,
            )
            gateway = StoredGpuExecutionGateway(
                state=state,
                store=store,
                dispatcher=dispatcher,
                reservation_ttl_seconds=10,
                clock=clock,
            )
            worker = PredictionJobWorker(
                state=state,
                store=store,
                pipeline=pipeline,
                worker_loss_ttl_seconds=60,
                result_ttl_seconds=30,
                clock=clock,
            )
            handle = gateway.submit(request())
            prediction_id, locator = dispatcher.messages[0]

            worker.execute(PredictionId(prediction_id), locator)

            self.assertEqual(len(pipeline.calls), 1)
            self.assertEqual(
                gateway.status(handle.prediction_id).state,
                PredictionJobState.FAILED,
            )
            self.assertEqual(store.load_result(locator), result)

    def test_stored_completed_result_becomes_expired_and_is_cleaned(self) -> None:
        clock = FakeClock()
        dispatcher = CapturingDispatcher()
        with TemporaryDirectory() as directory:
            store = EphemeralJobStore(Path(directory), clock=clock)
            state = InMemoryPredictionStateRepository(
                capacity=1,
                result_ttl_seconds=5,
                clock=clock,
            )
            gateway = StoredGpuExecutionGateway(
                state=state,
                store=store,
                dispatcher=dispatcher,
                reservation_ttl_seconds=10,
                clock=clock,
            )
            worker = PredictionJobWorker(
                state=state,
                store=store,
                pipeline=PipelineStub(real_prediction_result()),
                worker_loss_ttl_seconds=60,
                result_ttl_seconds=5,
                clock=clock,
            )
            handle = gateway.submit(request(idempotency_key="expiring"))
            prediction_id, locator = dispatcher.messages[0]
            worker.execute(PredictionId(prediction_id), locator)
            clock.advance(6)

            self.assertEqual(
                gateway.status(handle.prediction_id).state,
                PredictionJobState.EXPIRED,
            )
            with self.assertRaises(ResultExpired):
                gateway.result(handle.prediction_id)
            self.assertEqual(store.cleanup_expired(), 1)
            replay = gateway.submit(request(idempotency_key="expiring"))
            self.assertNotEqual(replay.prediction_id, handle.prediction_id)


class CeleryAdapterTests(unittest.TestCase):
    def test_web_side_gateway_import_cannot_initialize_cuda_or_model_factory(self) -> None:
        program = (
            "import sys; "
            f"sys.path.insert(0, {str(REPOSITORY_ROOT / 'src')!r}); "
            "import vision_model_serving.execution; "
            "forbidden={'torch','vision_model_serving.pipeline.factory'}; "
            "loaded=forbidden.intersection(sys.modules); "
            "assert not loaded, loaded"
        )

        completed = subprocess.run(
            [sys.executable, "-c", program],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_dispatch_and_worker_configuration_enforce_the_gpu_contract(self) -> None:
        app = CeleryAppFake()
        configure_gpu_worker(app, visibility_timeout_seconds=1_200)
        dispatcher = CeleryTaskDispatcher(app, queue_name="gpu-inference")

        dispatcher.dispatch(PredictionId("opaque-prediction"), "opaque-locator")

        self.assertEqual(
            app.sent,
            [
                {
                    "name": GPU_TASK_NAME,
                    "args": ["opaque-prediction", "opaque-locator"],
                    "kwargs": {},
                    "queue": "gpu-inference",
                }
            ],
        )
        self.assertTrue(app.conf["task_acks_late"])
        self.assertFalse(app.conf["task_reject_on_worker_lost"])
        self.assertEqual(app.conf["worker_concurrency"], 1)
        self.assertEqual(app.conf["worker_prefetch_multiplier"], 1)
        self.assertEqual(
            app.conf["broker_transport_options"],
            {"visibility_timeout": 1_200},
        )

    def test_registered_task_builds_the_pipeline_only_inside_execution(self) -> None:
        app = CeleryAppFake()
        executions: list[tuple[PredictionId, str]] = []

        class Worker:
            def execute(self, prediction_id: PredictionId, locator: str) -> None:
                executions.append((prediction_id, locator))

        factories: list[str] = []

        def build_worker() -> Worker:
            factories.append("built")
            return Worker()

        task = register_prediction_task(app, build_worker)
        self.assertEqual(factories, [])

        task("prediction", "locator")

        self.assertEqual(factories, ["built"])
        self.assertEqual(executions, [(PredictionId("prediction"), "locator")])
        self.assertTrue(app.task_options["acks_late"])
        self.assertFalse(app.task_options["reject_on_worker_lost"])
        self.assertEqual(app.task_options["max_retries"], 0)

    def test_production_gateway_composes_redis_storage_and_celery(self) -> None:
        app = CeleryAppFake()
        redis = RedisClientFake(lambda values: [1, values[5]])
        with TemporaryDirectory() as directory:
            gateway = CeleryRedisGpuExecutionGateway(
                redis_client=redis,
                celery_app=app,
                job_root=Path(directory),
                config=GpuExecutionConfig(
                    capacity=2,
                    reservation_ttl_seconds=30,
                    worker_loss_ttl_seconds=600,
                    result_ttl_seconds=60,
                    tombstone_ttl_seconds=30,
                    visibility_timeout_seconds=900,
                    queue_name="gpu-inference",
                ),
                clock=lambda: 1_000.0,
            )

            handle = gateway.submit(request(dicom_bytes=b"private"))

            self.assertEqual(handle.state, PredictionJobState.QUEUED)
            self.assertEqual(len(redis.calls), 1)
            self.assertEqual(len(app.sent), 1)
            self.assertEqual(app.sent[0]["name"], GPU_TASK_NAME)
            self.assertEqual(app.conf["worker_concurrency"], 1)


class RedisStateAdapterTests(unittest.TestCase):
    def test_admission_is_one_atomic_server_side_operation(self) -> None:
        redis = RedisClientFake([1, "opaque-prediction"])
        state = RedisPredictionStateRepository(
            redis,
            capacity=2,
            result_ttl_seconds=30,
            tombstone_ttl_seconds=10,
            clock=lambda: 1_000.0,
        )

        admission = state.admit(
            prediction_id=PredictionId("opaque-prediction"),
            locator="opaque-locator",
            request_fingerprint="a" * 64,
            idempotency_digest="b" * 64,
            submitted_at=1_000.0,
            reservation_expires_at=1_010.0,
        )

        self.assertEqual(admission.handle.state, PredictionJobState.QUEUED)
        self.assertEqual(len(redis.calls), 1)
        script, key_count, values = redis.calls[0]
        self.assertEqual(key_count, 4)
        self.assertIn("ZREMRANGEBYSCORE", script)
        self.assertIn("ZCARD", script)
        self.assertIn("HSET", script)
        self.assertIn("ZADD", script)
        self.assertIn("SET", script)
        serialized = json.dumps(values)
        self.assertNotIn("history", serialized)
        self.assertNotIn("dicom", serialized)

    def test_atomic_admission_outcomes_map_to_stable_gateway_errors(self) -> None:
        full = RedisPredictionStateRepository(
            RedisClientFake([0, ""]),
            capacity=1,
            result_ttl_seconds=30,
            tombstone_ttl_seconds=10,
            clock=lambda: 1_000.0,
        )
        conflict = RedisPredictionStateRepository(
            RedisClientFake([-1, "existing"]),
            capacity=1,
            result_ttl_seconds=30,
            tombstone_ttl_seconds=10,
            clock=lambda: 1_000.0,
        )
        arguments = {
            "prediction_id": PredictionId("opaque-prediction"),
            "locator": "opaque-locator",
            "request_fingerprint": "a" * 64,
            "idempotency_digest": "b" * 64,
            "submitted_at": 1_000.0,
            "reservation_expires_at": 1_010.0,
        }

        with self.assertRaises(QueueSaturated):
            full.admit(**arguments)
        with self.assertRaises(IdempotencyConflict):
            conflict.admit(**arguments)

    def test_status_refresh_exposes_worker_loss_without_private_payloads(self) -> None:
        redis = RedisClientFake(
            [
                b"prediction_id",
                b"opaque-prediction",
                b"locator",
                b"opaque-locator",
                b"request_fingerprint",
                b"a" * 64,
                b"idempotency_digest",
                b"",
                b"state",
                b"failed",
                b"submitted_at",
                b"900",
                b"lease_expires_at",
                b"999",
                b"result_expires_at",
                b"1030",
                b"started_at",
                b"910",
                b"completed_at",
                b"1000",
                b"failure_code",
                b"prediction_worker_lost",
                b"failure_detail",
                b"prediction worker was lost",
                b"failure_retryable",
                b"0",
            ]
        )
        state = RedisPredictionStateRepository(
            redis,
            capacity=1,
            result_ttl_seconds=30,
            tombstone_ttl_seconds=10,
            clock=lambda: 1_000.0,
        )

        status = state.status(PredictionId("opaque-prediction"))

        self.assertEqual(status.state, PredictionJobState.FAILED)
        self.assertEqual(status.failure.code, "prediction_worker_lost")
        self.assertNotIn("locator", repr(status))
        script = redis.calls[0][0]
        self.assertIn("prediction_worker_lost", script)
        self.assertIn("prediction_reservation_expired", script)
        self.assertIn("expired", script)

    def test_redis_loss_is_sanitized_as_gateway_unavailable(self) -> None:
        def fail(_values: tuple[object, ...]) -> object:
            raise RuntimeError("redis://secret@private-host")

        state = RedisPredictionStateRepository(
            RedisClientFake(fail),
            capacity=1,
            result_ttl_seconds=30,
            tombstone_ttl_seconds=10,
            clock=lambda: 1_000.0,
        )

        with self.assertRaises(GatewayUnavailable) as raised:
            state.admit(
                prediction_id=PredictionId("opaque-prediction"),
                locator="opaque-locator",
                request_fingerprint="a" * 64,
                idempotency_digest=None,
                submitted_at=1_000.0,
                reservation_expires_at=1_010.0,
            )

        self.assertNotIn("secret", str(raised.exception))

if __name__ == "__main__":
    unittest.main()
