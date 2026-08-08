from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import sys
from threading import Event, Lock, Thread
from types import ModuleType
import unittest
from unittest.mock import patch
import weakref


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from vision_model_serving.residency import (  # noqa: E402
    MemorySnapshot,
    ModelBinding,
    RuntimeInferenceError,
    RuntimeLoadError,
    RuntimeState,
    RuntimeUnavailableError,
    RuntimeUnloadError,
    SingleResidencyRuntime,
    TorchCudaLifecycle,
)


@dataclass(frozen=True)
class ArtifactStub:
    id: str
    sha256: str = "a" * 64
    repository_revision: str = "b" * 40


class ResidentStub:
    def __init__(self, model_id: str, *, fail_execute: bool = False):
        self.artifact = ArtifactStub(model_id)
        self.model_id = model_id
        self.fail_execute = fail_execute
        self.warmup_calls = 0
        self.execute_calls: list[object] = []

    def warmup(self) -> None:
        self.warmup_calls += 1

    def execute(self, inputs: object) -> object:
        self.execute_calls.append(inputs)
        if self.fail_execute:
            raise MemoryError("D:/patients/private/cuda-oom-secret")
        return f"{self.model_id}:{inputs}"


class LoaderStub:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.loads = 0
        self.generation = 1
        self.fail_load = False
        self.fail_execute = False
        self.last_resident: weakref.ReferenceType[ResidentStub] | None = None

    def failure_token(self) -> str:
        return f"{self.model_id}:generation-{self.generation}"

    def load(self) -> ResidentStub:
        self.loads += 1
        if self.fail_load:
            raise ValueError("D:/patients/private/checkpoint-load-secret")
        resident = ResidentStub(
            self.model_id,
            fail_execute=self.fail_execute,
        )
        self.last_resident = weakref.ref(resident)
        return resident


class BlockingResidentStub(ResidentStub):
    def __init__(self, model_id: str, entered: Event, release: Event):
        super().__init__(model_id)
        self._entered = entered
        self._release = release
        self._counter_lock = Lock()
        self.active = 0
        self.max_active = 0

    def execute(self, inputs: object) -> object:
        with self._counter_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        self._entered.set()
        if not self._release.wait(2):
            raise TimeoutError("test release was not signaled")
        with self._counter_lock:
            self.active -= 1
        return super().execute(inputs)


class BlockingLoaderStub(LoaderStub):
    def __init__(self, model_id: str, entered: Event, release: Event):
        super().__init__(model_id)
        self.entered = entered
        self.release = release

    def load(self) -> BlockingResidentStub:
        self.loads += 1
        resident = BlockingResidentStub(
            self.model_id,
            self.entered,
            self.release,
        )
        self.last_resident = weakref.ref(resident)
        return resident


class WarmupFailureResident(ResidentStub):
    def __init__(self, model_id: str, trace: list[str]):
        super().__init__(model_id)
        self._trace = trace

    def warmup(self) -> None:
        raise MemoryError("D:/patients/private/warmup-oom-secret")

    def __del__(self) -> None:
        self._trace.append("resident_deleted")


class WarmupFailureLoader(LoaderStub):
    def __init__(self, model_id: str, trace: list[str]):
        super().__init__(model_id)
        self.trace = trace

    def load(self) -> WarmupFailureResident:
        self.loads += 1
        resident = WarmupFailureResident(self.model_id, self.trace)
        self.last_resident = weakref.ref(resident)
        return resident


class BlockingLoadLoaderStub(LoaderStub):
    def __init__(self, model_id: str, entered: Event, release: Event):
        super().__init__(model_id)
        self.entered = entered
        self.release = release

    def load(self) -> ResidentStub:
        self.loads += 1
        self.entered.set()
        if not self.release.wait(2):
            raise TimeoutError("test load release was not signaled")
        resident = ResidentStub(self.model_id)
        self.last_resident = weakref.ref(resident)
        return resident


class AcceleratorStub:
    def __init__(self):
        self.events: list[str] = []

    def synchronize(self) -> None:
        self.events.append("synchronize")

    def reset_peak_memory_stats(self) -> None:
        self.events.append("reset_peak_memory_stats")

    def empty_cache(self) -> None:
        self.events.append("empty_cache")

    @contextmanager
    def inference_mode(self):
        self.events.append("inference_mode_enter")
        yield
        self.events.append("inference_mode_exit")

    def memory_snapshot(self) -> MemorySnapshot:
        self.events.append("memory_snapshot")
        return MemorySnapshot(
            allocated_bytes=100,
            reserved_bytes=200,
            peak_allocated_bytes=300,
            peak_reserved_bytes=400,
        )


class TracingAcceleratorStub(AcceleratorStub):
    def __init__(self, trace: list[str]):
        super().__init__()
        self.trace = trace

    def empty_cache(self) -> None:
        self.trace.append("empty_cache")
        super().empty_cache()


class BlockingCleanupAcceleratorStub(AcceleratorStub):
    def __init__(self, entered: Event, release: Event):
        super().__init__()
        self.entered = entered
        self.release = release
        self.block_next_synchronize = False

    def synchronize(self) -> None:
        if self.block_next_synchronize:
            self.block_next_synchronize = False
            self.entered.set()
            if not self.release.wait(2):
                raise TimeoutError("test cleanup release was not signaled")
        super().synchronize()


class RuntimeLoadTests(unittest.TestCase):
    def test_first_execute_loads_warms_and_reports_ready(self) -> None:
        loader = LoaderStub("focalnet-dino-detector")
        accelerator = AcceleratorStub()
        runtime = SingleResidencyRuntime(
            bindings=(
                ModelBinding(
                    model_id="focalnet-dino-detector",
                    load=loader.load,
                    failure_token=loader.failure_token,
                ),
            ),
            accelerator=accelerator,
        )

        before = runtime.status()
        output = runtime.execute("focalnet-dino-detector", "scan-1")
        after = runtime.status()

        self.assertEqual(before.state, RuntimeState.UNLOADED)
        self.assertEqual(output.model_id, "focalnet-dino-detector")
        self.assertEqual(output.value, "focalnet-dino-detector:scan-1")
        self.assertFalse(output.reused)
        self.assertEqual(loader.loads, 1)
        resident = loader.last_resident()
        self.assertIsNotNone(resident)
        self.assertEqual(resident.warmup_calls, 1)
        self.assertEqual(resident.execute_calls, ["scan-1"])
        self.assertEqual(after.state, RuntimeState.READY)
        self.assertEqual(after.active_model, "focalnet-dino-detector")
        self.assertEqual(after.artifact.id, "focalnet-dino-detector")
        self.assertEqual(after.memory.allocated_bytes, 100)
        self.assertEqual(after.memory.reserved_bytes, 200)
        self.assertEqual(after.memory.peak_allocated_bytes, 300)
        self.assertEqual(after.memory.peak_reserved_bytes, 400)
        self.assertEqual(after.metrics.load_count, 1)
        self.assertEqual(after.metrics.reuse_count, 0)
        self.assertEqual(after.active_inferences, 0)
        self.assertGreaterEqual(output.timings.load_ms, 0.0)
        self.assertGreaterEqual(output.timings.inference_ms, 0.0)
        self.assertIn("inference_mode_enter", accelerator.events)
        self.assertIn("inference_mode_exit", accelerator.events)
        self.assertFalse(hasattr(runtime, "model"))
        self.assertFalse(hasattr(output, "accelerator"))

    def test_cross_model_switch_proves_old_resident_is_unreachable(self) -> None:
        detector = LoaderStub("focalnet-dino-detector")
        classifier = LoaderStub("mmbcd-classifier")
        accelerator = AcceleratorStub()
        runtime = SingleResidencyRuntime(
            bindings=(
                ModelBinding(
                    model_id="focalnet-dino-detector",
                    load=detector.load,
                    failure_token=detector.failure_token,
                ),
                ModelBinding(
                    model_id="mmbcd-classifier",
                    load=classifier.load,
                    failure_token=classifier.failure_token,
                ),
            ),
            accelerator=accelerator,
        )

        runtime.execute("focalnet-dino-detector", "scan-1")
        old_resident = detector.last_resident
        switched = runtime.execute("mmbcd-classifier", "eight-rois")
        status = runtime.status()

        self.assertIsNone(old_resident())
        self.assertEqual(switched.model_id, "mmbcd-classifier")
        self.assertEqual(switched.value, "mmbcd-classifier:eight-rois")
        self.assertFalse(switched.reused)
        self.assertGreaterEqual(switched.timings.switch_ms, 0.0)
        self.assertEqual(status.state, RuntimeState.READY)
        self.assertEqual(status.active_model, "mmbcd-classifier")
        self.assertEqual(status.metrics.load_count, 2)
        self.assertEqual(status.metrics.switch_count, 1)
        self.assertEqual(status.metrics.unload_count, 1)
        self.assertGreaterEqual(status.timings.last_unload_ms, 0.0)
        self.assertGreaterEqual(status.timings.last_switch_ms, 0.0)
        self.assertIn("empty_cache", accelerator.events)

    def test_same_model_execute_reuses_the_ready_resident(self) -> None:
        loader = LoaderStub("mmbcd-classifier")
        runtime = SingleResidencyRuntime(
            bindings=(
                ModelBinding(
                    model_id="mmbcd-classifier",
                    load=loader.load,
                    failure_token=loader.failure_token,
                ),
            ),
            accelerator=AcceleratorStub(),
        )

        first = runtime.execute("mmbcd-classifier", "rois-1")
        second = runtime.execute("mmbcd-classifier", "rois-2")
        status = runtime.status()

        self.assertFalse(first.reused)
        self.assertTrue(second.reused)
        self.assertEqual(second.timings.load_ms, 0.0)
        self.assertEqual(loader.loads, 1)
        resident = loader.last_resident()
        self.assertEqual(resident.warmup_calls, 1)
        self.assertEqual(resident.execute_calls, ["rois-1", "rois-2"])
        self.assertEqual(status.metrics.load_count, 1)
        self.assertEqual(status.metrics.reuse_count, 1)
        self.assertEqual(status.metrics.switch_count, 0)
        self.assertEqual(status.metrics.unload_count, 0)

    def test_load_failure_retries_only_after_the_cause_token_changes(self) -> None:
        loader = LoaderStub("mmbcd-classifier")
        loader.fail_load = True
        runtime = SingleResidencyRuntime(
            bindings=(
                ModelBinding(
                    model_id="mmbcd-classifier",
                    load=loader.load,
                    failure_token=loader.failure_token,
                ),
            ),
            accelerator=AcceleratorStub(),
        )

        with self.assertRaises(RuntimeLoadError) as first_error:
            runtime.execute("mmbcd-classifier", "rois")
        failed = runtime.status()
        with self.assertRaises(RuntimeUnavailableError):
            runtime.execute("mmbcd-classifier", "rois")

        self.assertEqual(first_error.exception.code, "runtime_model_load_failed")
        self.assertNotIn("patients", str(first_error.exception))
        self.assertIsNone(first_error.exception.__cause__)
        self.assertEqual(loader.loads, 1)
        self.assertEqual(failed.state, RuntimeState.FAILED)
        self.assertIsNone(failed.active_model)
        self.assertEqual(failed.last_error.phase, "load")
        self.assertEqual(failed.last_error.code, "runtime_model_load_failed")
        self.assertEqual(failed.metrics.failure_count, 1)

        loader.generation = 2
        loader.fail_load = False
        recovered = runtime.execute("mmbcd-classifier", "rois")

        self.assertEqual(recovered.value, "mmbcd-classifier:rois")
        self.assertEqual(loader.loads, 2)
        self.assertEqual(runtime.status().state, RuntimeState.READY)
        self.assertIsNone(runtime.status().last_error)

    def test_warmup_failure_releases_resident_before_cache_cleanup(self) -> None:
        trace: list[str] = []
        loader = WarmupFailureLoader("mmbcd-classifier", trace)
        runtime = SingleResidencyRuntime(
            bindings=(
                ModelBinding(
                    model_id="mmbcd-classifier",
                    load=loader.load,
                    failure_token=loader.failure_token,
                ),
            ),
            accelerator=TracingAcceleratorStub(trace),
        )

        with self.assertRaises(RuntimeLoadError):
            runtime.execute("mmbcd-classifier", "rois")

        self.assertIsNone(loader.last_resident())
        self.assertEqual(trace, ["resident_deleted", "empty_cache"])
        self.assertEqual(runtime.status().state, RuntimeState.FAILED)

    def test_inference_failure_unloads_and_enters_sanitized_failed_state(self) -> None:
        loader = LoaderStub("focalnet-dino-detector")
        loader.fail_execute = True
        accelerator = AcceleratorStub()
        runtime = SingleResidencyRuntime(
            bindings=(
                ModelBinding(
                    model_id="focalnet-dino-detector",
                    load=loader.load,
                    failure_token=loader.failure_token,
                ),
            ),
            accelerator=accelerator,
        )

        with self.assertRaises(RuntimeInferenceError) as raised:
            runtime.execute("focalnet-dino-detector", "scan")
        failed = runtime.status()

        self.assertEqual(raised.exception.code, "runtime_model_inference_failed")
        self.assertNotIn("patients", str(raised.exception))
        self.assertIsNone(raised.exception.__cause__)
        self.assertIsNone(loader.last_resident())
        self.assertEqual(failed.state, RuntimeState.FAILED)
        self.assertIsNone(failed.active_model)
        self.assertEqual(failed.active_inferences, 0)
        self.assertEqual(failed.last_error.phase, "inference")
        self.assertEqual(
            failed.last_error.code,
            "runtime_model_inference_failed",
        )
        self.assertEqual(failed.metrics.failure_count, 1)
        self.assertEqual(failed.metrics.unload_count, 1)
        self.assertIn("empty_cache", accelerator.events)

        with self.assertRaises(RuntimeUnavailableError):
            runtime.execute("focalnet-dino-detector", "scan")

    def test_different_model_waits_in_draining_state_without_overlap(self) -> None:
        entered = Event()
        release = Event()
        detector = BlockingLoaderStub(
            "focalnet-dino-detector",
            entered,
            release,
        )
        classifier = LoaderStub("mmbcd-classifier")
        runtime = SingleResidencyRuntime(
            bindings=(
                ModelBinding(
                    model_id="focalnet-dino-detector",
                    load=detector.load,
                    failure_token=detector.failure_token,
                ),
                ModelBinding(
                    model_id="mmbcd-classifier",
                    load=classifier.load,
                    failure_token=classifier.failure_token,
                ),
            ),
            accelerator=AcceleratorStub(),
        )
        outputs: list[object] = []
        errors: list[BaseException] = []

        def run(model_id: str, inputs: str) -> None:
            try:
                outputs.append(runtime.execute(model_id, inputs))
            except BaseException as error:  # test thread must report failures
                errors.append(error)

        detector_thread = Thread(
            target=run,
            args=("focalnet-dino-detector", "scan"),
        )
        detector_thread.start()
        self.assertTrue(entered.wait(1))
        detector_resident = detector.last_resident()

        classifier_thread = Thread(
            target=run,
            args=("mmbcd-classifier", "rois"),
        )
        classifier_thread.start()
        classifier_thread.join(0.05)
        self.assertTrue(classifier_thread.is_alive())
        draining = runtime.status()

        self.assertEqual(draining.state, RuntimeState.DRAINING)
        self.assertEqual(draining.active_model, "focalnet-dino-detector")
        self.assertEqual(draining.active_inferences, 1)
        self.assertEqual(detector_resident.max_active, 1)
        del detector_resident

        release.set()
        detector_thread.join(2)
        classifier_thread.join(2)

        self.assertFalse(detector_thread.is_alive())
        self.assertFalse(classifier_thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(outputs), 2)
        self.assertIsNone(detector.last_resident())
        self.assertEqual(runtime.status().active_model, "mmbcd-classifier")

    def test_reachable_old_resident_fails_closed_after_allocator_cleanup(self) -> None:
        detector = LoaderStub("focalnet-dino-detector")
        classifier = LoaderStub("mmbcd-classifier")
        accelerator = AcceleratorStub()
        runtime = SingleResidencyRuntime(
            bindings=(
                ModelBinding(
                    model_id="focalnet-dino-detector",
                    load=detector.load,
                    failure_token=detector.failure_token,
                ),
                ModelBinding(
                    model_id="mmbcd-classifier",
                    load=classifier.load,
                    failure_token=classifier.failure_token,
                ),
            ),
            accelerator=accelerator,
        )
        runtime.execute("focalnet-dino-detector", "scan")
        leaked_resident = detector.last_resident()

        with self.assertRaises(RuntimeUnloadError) as raised:
            runtime.execute("mmbcd-classifier", "rois")
        failed = runtime.status()

        self.assertEqual(raised.exception.code, "runtime_model_unload_failed")
        self.assertIsNone(raised.exception.__cause__)
        self.assertEqual(failed.state, RuntimeState.FAILED)
        self.assertIsNone(failed.active_model)
        self.assertEqual(failed.last_error.phase, "unload")
        self.assertEqual(failed.last_error.code, "runtime_model_unload_failed")
        self.assertEqual(failed.metrics.failure_count, 1)
        self.assertIn("empty_cache", accelerator.events)
        self.assertIsNotNone(leaked_resident)

        del leaked_resident
        classifier.generation = 2
        recovered = runtime.execute("mmbcd-classifier", "rois")
        self.assertEqual(recovered.value, "mmbcd-classifier:rois")
        self.assertEqual(runtime.status().state, RuntimeState.READY)

    def test_status_observes_loading_and_unloading_transitions(self) -> None:
        load_entered = Event()
        load_release = Event()
        cleanup_entered = Event()
        cleanup_release = Event()
        detector = BlockingLoadLoaderStub(
            "focalnet-dino-detector",
            load_entered,
            load_release,
        )
        classifier = LoaderStub("mmbcd-classifier")
        accelerator = BlockingCleanupAcceleratorStub(
            cleanup_entered,
            cleanup_release,
        )
        runtime = SingleResidencyRuntime(
            bindings=(
                ModelBinding(
                    model_id="focalnet-dino-detector",
                    load=detector.load,
                    failure_token=detector.failure_token,
                ),
                ModelBinding(
                    model_id="mmbcd-classifier",
                    load=classifier.load,
                    failure_token=classifier.failure_token,
                ),
            ),
            accelerator=accelerator,
        )
        errors: list[BaseException] = []

        def execute(model_id: str, inputs: str) -> None:
            try:
                runtime.execute(model_id, inputs)
            except BaseException as error:  # test thread must report failures
                errors.append(error)

        loading_thread = Thread(
            target=execute,
            args=("focalnet-dino-detector", "scan"),
        )
        loading_thread.start()
        self.assertTrue(load_entered.wait(1))
        self.assertEqual(runtime.status().state, RuntimeState.LOADING)
        load_release.set()
        loading_thread.join(2)
        self.assertFalse(loading_thread.is_alive())

        accelerator.block_next_synchronize = True
        unloading_thread = Thread(
            target=execute,
            args=("mmbcd-classifier", "rois"),
        )
        unloading_thread.start()
        self.assertTrue(cleanup_entered.wait(1))
        self.assertEqual(runtime.status().state, RuntimeState.UNLOADING)
        cleanup_release.set()
        unloading_thread.join(2)

        self.assertFalse(unloading_thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(runtime.status().state, RuntimeState.READY)


class TorchCudaLifecycleTests(unittest.TestCase):
    def test_cuda_initialization_is_lazy_with_distinct_memory_metrics(self) -> None:
        events: list[tuple[str, object]] = []
        fake_torch = ModuleType("torch")
        fake_torch.cuda = type(
            "CudaStub",
            (),
            {
                "is_available": staticmethod(
                    lambda: events.append(("is_available", None)) or True
                ),
                "synchronize": staticmethod(
                    lambda device: events.append(("synchronize", device))
                ),
                "reset_peak_memory_stats": staticmethod(
                    lambda device: events.append(("reset_peak", device))
                ),
                "empty_cache": staticmethod(
                    lambda: events.append(("empty_cache", None))
                ),
                "memory_allocated": staticmethod(lambda device: 101),
                "memory_reserved": staticmethod(lambda device: 202),
                "max_memory_allocated": staticmethod(lambda device: 303),
                "max_memory_reserved": staticmethod(lambda device: 404),
            },
        )

        with patch.dict(sys.modules, {"torch": fake_torch}):
            accelerator = TorchCudaLifecycle(device="cuda:0")
            self.assertEqual(events, [])
            accelerator.reset_peak_memory_stats()
            accelerator.synchronize()
            snapshot = accelerator.memory_snapshot()
            accelerator.empty_cache()

        self.assertEqual(snapshot.allocated_bytes, 101)
        self.assertEqual(snapshot.reserved_bytes, 202)
        self.assertEqual(snapshot.peak_allocated_bytes, 303)
        self.assertEqual(snapshot.peak_reserved_bytes, 404)
        self.assertEqual(
            events,
            [
                ("is_available", None),
                ("reset_peak", "cuda:0"),
                ("synchronize", "cuda:0"),
                ("empty_cache", None),
            ],
        )


if __name__ == "__main__":
    unittest.main()
