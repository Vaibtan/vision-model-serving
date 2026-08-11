"""Serialized model lifecycle with exactly one private resident adapter."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
import gc
import hashlib
import math
import re
from threading import Lock
from time import perf_counter
from typing import ContextManager, Protocol
import weakref

from ._torch_policy import (
    TorchProcessConfigurationError,
    configure_deterministic_torch,
)


_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
_CUDA_DEVICE = re.compile(r"cuda(?::[0-9]+)?")


class ResidencyRuntimeError(RuntimeError):
    code = "runtime_failed"

    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(f"{self.code}: {detail}")


class RuntimeLoadError(ResidencyRuntimeError):
    code = "runtime_model_load_failed"


class RuntimeInferenceError(ResidencyRuntimeError):
    code = "runtime_model_inference_failed"


class RuntimeUnloadError(ResidencyRuntimeError):
    code = "runtime_model_unload_failed"


class RuntimeUnavailableError(ResidencyRuntimeError):
    code = "runtime_cause_unchanged"


class RuntimeCaseInputError(ResidencyRuntimeError):
    """The resident model rejected this case; the runtime itself is healthy."""

    code = "runtime_case_input_invalid"
    case_input_error = True


class UnknownModelError(ResidencyRuntimeError):
    code = "runtime_model_unknown"


class _ResidentReachabilityError(RuntimeError):
    pass


class RuntimeState(str, Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    DRAINING = "draining"
    UNLOADING = "unloading"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    allocated_bytes: int
    reserved_bytes: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (
                self.allocated_bytes,
                self.reserved_bytes,
                self.peak_allocated_bytes,
                self.peak_reserved_bytes,
            )
        ):
            raise ValueError("accelerator memory values must be non-negative integers")


@dataclass(frozen=True, slots=True)
class RuntimeArtifactIdentity:
    id: str
    sha256: str
    repository_revision: str

    def __post_init__(self) -> None:
        if not self.id or _SHA256.fullmatch(self.sha256) is None:
            raise ValueError("runtime artifact identity is invalid")
        if _GIT_COMMIT.fullmatch(self.repository_revision) is None:
            raise ValueError("runtime artifact revision is invalid")


@dataclass(frozen=True, slots=True)
class ExecutionTimings:
    load_ms: float
    inference_ms: float
    switch_ms: float

    def __post_init__(self) -> None:
        _validate_timings(self.load_ms, self.inference_ms, self.switch_ms)


@dataclass(frozen=True, slots=True)
class LifecycleTimings:
    last_load_ms: float
    last_inference_ms: float
    last_switch_ms: float
    last_unload_ms: float

    def __post_init__(self) -> None:
        _validate_timings(
            self.last_load_ms,
            self.last_inference_ms,
            self.last_switch_ms,
            self.last_unload_ms,
        )


@dataclass(frozen=True, slots=True)
class LifecycleMetrics:
    load_count: int
    reuse_count: int
    switch_count: int
    unload_count: int
    failure_count: int


@dataclass(frozen=True, slots=True)
class RuntimeFailure:
    model_id: str
    phase: str
    code: str
    cause_token_sha256: str


@dataclass(frozen=True, slots=True)
class ModelOutputs:
    model_id: str
    value: object
    artifact: RuntimeArtifactIdentity
    reused: bool
    timings: ExecutionTimings
    memory: MemorySnapshot


@dataclass(frozen=True, slots=True)
class RuntimeStatus:
    state: RuntimeState
    active_model: str | None
    active_inferences: int
    last_error: RuntimeFailure | None
    artifact: RuntimeArtifactIdentity | None
    memory: MemorySnapshot
    metrics: LifecycleMetrics
    timings: LifecycleTimings
    resident_models: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.active_inferences, bool)
            or not isinstance(self.active_inferences, int)
            or self.active_inferences not in {0, 1}
        ):
            raise ValueError("active inference count must be zero or one")
        if (
            not isinstance(self.resident_models, tuple)
            or len(self.resident_models) > 1
            or any(not isinstance(model, str) or not model for model in self.resident_models)
        ):
            raise ValueError("runtime may report at most one resident model")
        expected = () if self.active_model is None else (self.active_model,)
        if self.resident_models != expected:
            raise ValueError("active model and resident model must be identical")
        if (self.artifact is None) != (self.active_model is None):
            raise ValueError("runtime artifact and active model must change together")
        if self.state is RuntimeState.READY and not self.resident_models:
            raise ValueError("ready runtime must have one resident model")
        if (
            self.state
            in {
                RuntimeState.UNLOADED,
                RuntimeState.LOADING,
                RuntimeState.FAILED,
            }
            and self.resident_models
        ):
            raise ValueError(f"{self.state.value} runtime cannot retain a model")


class ResidentModel(Protocol):
    artifact: object

    def warmup(self, inputs: object) -> None: ...

    def execute(self, inputs: object) -> object: ...


class AcceleratorLifecycle(Protocol):
    def prepare(self) -> None: ...

    def synchronize(self) -> None: ...

    def reset_peak_memory_stats(self) -> None: ...

    def empty_cache(self) -> None: ...

    def inference_mode(self) -> ContextManager[None]: ...

    def memory_snapshot(self) -> MemorySnapshot: ...


class TorchCudaLifecycle:
    """Lazily access documented PyTorch CUDA lifecycle and allocator hooks."""

    def __init__(self, *, device: str = "cuda:0"):
        if not isinstance(device, str) or _CUDA_DEVICE.fullmatch(device) is None:
            raise ValueError("CUDA lifecycle device is invalid")
        self._device = device
        self._torch: object | None = None
        self._prepared = False

    def prepare(self) -> None:
        if self._prepared:
            return
        torch = self._import_torch()
        try:
            configure_deterministic_torch(torch, device=self._device)
        except TorchProcessConfigurationError as error:
            raise RuntimeUnavailableError(f"accelerator {error}") from None
        if not torch.cuda.is_available():
            raise RuntimeUnavailableError("CUDA is unavailable in the execution process")
        self._prepared = True

    def synchronize(self) -> None:
        self._cuda().synchronize(self._device)

    def reset_peak_memory_stats(self) -> None:
        self._cuda().reset_peak_memory_stats(self._device)

    def empty_cache(self) -> None:
        self._cuda().empty_cache()

    def memory_snapshot(self) -> MemorySnapshot:
        cuda = self._cuda()
        return MemorySnapshot(
            allocated_bytes=int(cuda.memory_allocated(self._device)),
            reserved_bytes=int(cuda.memory_reserved(self._device)),
            peak_allocated_bytes=int(cuda.max_memory_allocated(self._device)),
            peak_reserved_bytes=int(cuda.max_memory_reserved(self._device)),
        )

    def inference_mode(self) -> ContextManager[None]:
        return self._pytorch().inference_mode()

    def _cuda(self) -> object:
        return self._pytorch().cuda

    def _pytorch(self) -> object:
        self.prepare()
        return self._import_torch()

    def _import_torch(self) -> object:
        if self._torch is None:
            try:
                import torch
            except ImportError:
                raise RuntimeUnavailableError(
                    "PyTorch is unavailable in the execution process"
                ) from None
            self._torch = torch
        return self._torch


@dataclass(frozen=True, slots=True)
class ModelBinding:
    model_id: str
    load: Callable[[], ResidentModel]
    failure_token: Callable[[], str]

    def __post_init__(self) -> None:
        if not self.model_id or not callable(self.load) or not callable(self.failure_token):
            raise ValueError("model binding is invalid")


class SingleResidencyRuntime:
    """Serialize load, warmup, inference, and future model switches."""

    def __init__(
        self,
        *,
        bindings: Sequence[ModelBinding],
        accelerator: AcceleratorLifecycle,
    ):
        binding_map = {binding.model_id: binding for binding in bindings}
        if not binding_map or len(binding_map) != len(bindings):
            raise ValueError("runtime model bindings must be non-empty and unique")
        self._bindings = binding_map
        self._accelerator = accelerator
        self._accelerator.prepare()
        self._execution_lock = Lock()
        self._status_lock = Lock()
        self._resident: ResidentModel | None = None
        self._active_model: str | None = None
        self._artifact: RuntimeArtifactIdentity | None = None
        self._state = RuntimeState.UNLOADED
        self._active_inferences = 0
        self._last_error: RuntimeFailure | None = None
        self._failed_key: tuple[str, str] | None = None
        self._failed_resident: weakref.ReferenceType[ResidentModel] | None = None
        self._process_failure_latched = False
        self._closed = False
        self._memory = MemorySnapshot(0, 0, 0, 0)
        self._load_count = 0
        self._reuse_count = 0
        self._switch_count = 0
        self._unload_count = 0
        self._failure_count = 0
        self._last_load_ms = 0.0
        self._last_inference_ms = 0.0
        self._last_switch_ms = 0.0
        self._last_unload_ms = 0.0

    def execute(self, model_id: str, inputs: object) -> ModelOutputs:
        binding = self._bindings.get(model_id)
        if binding is None:
            raise UnknownModelError("model id is not registered")
        failure_key = _failure_key(binding)
        with self._status_lock:
            if self._closed:
                raise RuntimeUnavailableError("runtime is closed")
            if (
                self._active_model is not None
                and self._active_model != model_id
                and self._active_inferences > 0
            ):
                self._state = RuntimeState.DRAINING
        with self._execution_lock:
            with self._status_lock:
                if self._closed:
                    # A caller that queued behind close() must not reload a
                    # model into a process that is shutting down.
                    raise RuntimeUnavailableError("runtime is closed")
                if self._failure_cause_is_unchanged(model_id, failure_key):
                    raise RuntimeUnavailableError("runtime failure cause has not changed")
            reused = self._active_model == model_id and self._resident is not None
            load_ms = 0.0
            switch_ms = 0.0
            if not reused:
                switch_started: float | None = None
                if self._resident is not None:
                    switch_started = perf_counter()
                    self._set_state(RuntimeState.DRAINING)
                    try:
                        self._unload()
                    except Exception as error:
                        cleanup_error = type(error).__name__
                        process_failure = not isinstance(
                            error,
                            _ResidentReachabilityError,
                        )
                        del error
                        self._record_failure(
                            model_id=model_id,
                            phase="unload",
                            code=RuntimeUnloadError.code,
                            failure_key=failure_key,
                            process_global=process_failure,
                        )
                        raise RuntimeUnloadError(
                            f"failed resident cleanup ({cleanup_error})"
                        ) from None
                self._set_state(RuntimeState.LOADING)
                started = perf_counter()
                resident: ResidentModel | None = None
                load_error: str | None = None
                try:
                    resident = binding.load()
                    resident.warmup(inputs)
                    artifact = _artifact_identity(resident.artifact, model_id)
                    weakref.ref(resident)
                except Exception as error:
                    load_error = type(error).__name__
                    del error
                if load_error is not None:
                    load_ms = (perf_counter() - started) * 1000.0
                    try:
                        resident_reference = weakref.ref(resident) if resident is not None else None
                    except TypeError:
                        resident_reference = None
                    resident = None
                    try:
                        if self._resident is not None:
                            self._unload()
                        else:
                            self._cleanup_accelerator()
                    except Exception as error:
                        cleanup_error = type(error).__name__
                        del error
                        self._record_failure(
                            model_id=model_id,
                            phase="unload",
                            code=RuntimeUnloadError.code,
                            failure_key=failure_key,
                            process_global=True,
                        )
                        with self._status_lock:
                            self._last_load_ms = load_ms
                        raise RuntimeUnloadError(
                            f"failed partial-load cleanup ({cleanup_error})"
                        ) from None
                    if resident_reference is not None and resident_reference() is not None:
                        load_error = "ResidentReachabilityError"
                        with self._status_lock:
                            self._failed_resident = resident_reference
                    self._record_failure(
                        model_id=model_id,
                        phase="load",
                        code=RuntimeLoadError.code,
                        failure_key=failure_key,
                    )
                    with self._status_lock:
                        self._last_load_ms = load_ms
                    raise RuntimeLoadError(f"model preparation failed ({load_error})") from None
                if resident is None:
                    raise RuntimeLoadError("model loader returned no resident adapter")
                load_ms = (perf_counter() - started) * 1000.0
                with self._status_lock:
                    self._resident = resident
                    self._active_model = model_id
                    self._artifact = artifact
                    self._load_count += 1
                    self._last_load_ms = load_ms
                    self._state = RuntimeState.READY
                    self._last_error = None
                    self._failed_key = None
                    self._failed_resident = None
                    self._process_failure_latched = False
                    if switch_started is not None:
                        switch_ms = (perf_counter() - switch_started) * 1000.0
                        self._switch_count += 1
                        self._last_switch_ms = switch_ms
                del resident
            else:
                with self._status_lock:
                    self._reuse_count += 1

            inference_started = perf_counter()
            inference_error: str | None = None
            case_input_rejected = False
            try:
                self._accelerator.reset_peak_memory_stats()
                self._accelerator.synchronize()
                with self._status_lock:
                    self._active_inferences = 1
                with self._accelerator.inference_mode():
                    value = self._resident.execute(inputs)
                self._accelerator.synchronize()
                memory = self._accelerator.memory_snapshot()
            except Exception as error:
                inference_error = type(error).__name__
                case_input_rejected = getattr(error, "case_input_error", False) is True
                del error
            inference_ms = (perf_counter() - inference_started) * 1000.0
            if case_input_rejected:
                # A data-dependent rejection says nothing about the artifact
                # or the accelerator: the model stays resident and the
                # runtime stays READY for the next case.
                with self._status_lock:
                    self._active_inferences = 0
                    self._last_inference_ms = inference_ms
                raise RuntimeCaseInputError(
                    f"model rejected the case input ({inference_error})"
                ) from None
            if inference_error is not None:
                with self._status_lock:
                    self._active_inferences = 0
                    self._last_inference_ms = inference_ms
                try:
                    self._unload()
                except Exception as error:
                    cleanup_error = type(error).__name__
                    process_failure = not isinstance(
                        error,
                        _ResidentReachabilityError,
                    )
                    del error
                    self._record_failure(
                        model_id=model_id,
                        phase="unload",
                        code=RuntimeUnloadError.code,
                        failure_key=failure_key,
                        process_global=process_failure,
                    )
                    raise RuntimeUnloadError(f"failed resident cleanup ({cleanup_error})") from None
                self._record_failure(
                    model_id=model_id,
                    phase="inference",
                    code=RuntimeInferenceError.code,
                    failure_key=failure_key,
                )
                raise RuntimeInferenceError(f"model execution failed ({inference_error})") from None
            with self._status_lock:
                self._active_inferences = 0
                self._last_inference_ms = inference_ms
                self._memory = memory
                artifact = self._artifact
            if artifact is None:
                raise RuntimeError("runtime artifact identity is absent")
            return ModelOutputs(
                model_id=model_id,
                value=value,
                artifact=artifact,
                reused=reused,
                timings=ExecutionTimings(
                    load_ms=load_ms,
                    inference_ms=inference_ms,
                    switch_ms=switch_ms,
                ),
                memory=memory,
            )

    def status(self) -> RuntimeStatus:
        with self._status_lock:
            return RuntimeStatus(
                state=self._state,
                active_model=self._active_model,
                active_inferences=self._active_inferences,
                last_error=self._last_error,
                artifact=self._artifact,
                memory=self._memory,
                metrics=LifecycleMetrics(
                    load_count=self._load_count,
                    reuse_count=self._reuse_count,
                    switch_count=self._switch_count,
                    unload_count=self._unload_count,
                    failure_count=self._failure_count,
                ),
                timings=LifecycleTimings(
                    last_load_ms=self._last_load_ms,
                    last_inference_ms=self._last_inference_ms,
                    last_switch_ms=self._last_switch_ms,
                    last_unload_ms=self._last_unload_ms,
                ),
                resident_models=((self._active_model,) if self._active_model is not None else ()),
            )

    def close(self) -> None:
        with self._status_lock:
            self._closed = True
        with self._execution_lock:
            if self._resident is not None:
                self._unload()

    def _set_state(self, state: RuntimeState) -> None:
        with self._status_lock:
            self._state = state

    def _unload(self) -> None:
        self._set_state(RuntimeState.UNLOADING)
        resident_reference = weakref.ref(self._resident) if self._resident is not None else None
        started = perf_counter()
        with self._status_lock:
            self._resident = None
            self._active_model = None
            self._artifact = None
        memory = self._cleanup_accelerator()
        unload_ms = (perf_counter() - started) * 1000.0
        with self._status_lock:
            self._memory = memory
            self._last_unload_ms = unload_ms
        if resident_reference is not None and resident_reference() is not None:
            with self._status_lock:
                self._failed_resident = resident_reference
            raise _ResidentReachabilityError("resident adapter remains reachable after unload")
        with self._status_lock:
            self._failed_resident = None
            if resident_reference is not None:
                self._unload_count += 1
            self._state = RuntimeState.UNLOADED

    def _cleanup_accelerator(self) -> MemorySnapshot:
        self._accelerator.synchronize()
        gc.collect()
        self._accelerator.empty_cache()
        self._accelerator.synchronize()
        return self._accelerator.memory_snapshot()

    def _record_failure(
        self,
        *,
        model_id: str,
        phase: str,
        code: str,
        failure_key: str,
        process_global: bool = False,
    ) -> None:
        with self._status_lock:
            self._resident = None
            self._active_model = None
            self._artifact = None
            self._active_inferences = 0
            self._state = RuntimeState.FAILED
            self._failure_count += 1
            self._failed_key = (model_id, failure_key)
            self._process_failure_latched = self._process_failure_latched or process_global
            self._last_error = RuntimeFailure(
                model_id=model_id,
                phase=phase,
                code=code,
                cause_token_sha256=failure_key,
            )

    def _failure_cause_is_unchanged(
        self,
        model_id: str,
        failure_key: str,
    ) -> bool:
        if self._state is not RuntimeState.FAILED:
            return False
        if self._failed_resident is not None:
            if self._failed_resident() is not None:
                return True
            self._failed_resident = None
            self._failed_key = None
            self._last_error = None
            self._state = RuntimeState.UNLOADED
            return False
        if self._process_failure_latched:
            return True
        return self._failed_key == (model_id, failure_key)


def _artifact_identity(value: object, model_id: str) -> RuntimeArtifactIdentity:
    identity = RuntimeArtifactIdentity(
        id=str(getattr(value, "id", "")),
        sha256=str(getattr(value, "sha256", "")),
        repository_revision=str(getattr(value, "repository_revision", "")),
    )
    if identity.id != model_id:
        raise ValueError("resident artifact id differs from requested model")
    return identity


def _failure_key(binding: ModelBinding) -> str:
    try:
        token = binding.failure_token()
    except Exception as error:
        raise RuntimeLoadError(
            f"failure token resolution failed ({type(error).__name__})"
        ) from None
    if not isinstance(token, str) or not token:
        raise RuntimeLoadError("failure token is invalid")
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _validate_timings(*values: float) -> None:
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("runtime timings must be finite and non-negative")
