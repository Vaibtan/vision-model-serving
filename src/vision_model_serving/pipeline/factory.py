"""Composition root for the local, offline CUDA prediction pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock

from vision_model_serving.artifacts import ArtifactRegistry
from vision_model_serving.classifier import MmbcdClassifierAdapter
from vision_model_serving.detector import (
    FocalNetDinoAdapter,
    probe_focalnet_native_operator,
)
from vision_model_serving.dicom import DicomCanonicalizer
from vision_model_serving.model_ids import CLASSIFIER_MODEL_ID, DETECTOR_MODEL_ID
from vision_model_serving.residency import (
    ModelBinding,
    ModelOutputs,
    PersistentResidencyRuntime,
    SingleResidencyRuntime,
    TorchCudaLifecycle,
)

from .pipeline import PredictionPipeline


@dataclass(frozen=True, slots=True)
class LocalCudaPipelineConfig:
    project_root: Path
    artifact_root: Path
    tokenizer_root: Path
    focalnet_root: Path
    mmbcd_root: Path
    dino_root: Path
    device: str = "cuda:0"
    require_history_for_full: bool = True
    retain_models: bool = False

    def __post_init__(self) -> None:
        for name in (
            "project_root",
            "artifact_root",
            "tokenizer_root",
            "focalnet_root",
            "mmbcd_root",
            "dino_root",
        ):
            value = getattr(self, name)
            if not isinstance(value, Path):
                raise TypeError(f"{name} must be a pathlib.Path")
            object.__setattr__(self, name, value.expanduser().resolve())
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a non-empty string")
        if not isinstance(self.require_history_for_full, bool):
            raise TypeError("require_history_for_full must be boolean")
        if not isinstance(self.retain_models, bool):
            raise TypeError("retain_models must be boolean")


class _WarmupInputs:
    def __init__(self) -> None:
        self._current: dict[str, object] = {}

    def set(self, model_id: str, inputs: object) -> None:
        self._current[model_id] = inputs

    def get(self, model_id: str) -> object:
        try:
            return self._current[model_id]
        except KeyError:
            raise RuntimeError("model warmup input is unavailable") from None

    def clear(self, model_id: str) -> None:
        self._current.pop(model_id, None)


class _InputAwareRuntime:
    """Keep warmup inputs reachable only for one serialized execution."""

    def __init__(
        self,
        runtime: SingleResidencyRuntime,
        warmup_inputs: _WarmupInputs,
    ):
        self._runtime = runtime
        self._warmup_inputs = warmup_inputs
        self._lock = RLock()

    def execute(self, model_id: str, inputs: object) -> ModelOutputs:
        with self._lock:
            self._warmup_inputs.set(model_id, inputs)
            try:
                return self._runtime.execute(model_id, inputs)
            finally:
                self._warmup_inputs.clear(model_id)

    def status(self) -> object:
        return self._runtime.status()

    def close(self) -> None:
        close = getattr(self._runtime, "close", None)
        if callable(close):
            close()


def build_local_cuda_pipeline(config: LocalCudaPipelineConfig) -> PredictionPipeline:
    """Verify local assets and compose the real adapters without network access."""

    if not isinstance(config, LocalCudaPipelineConfig):
        raise TypeError("config must be a LocalCudaPipelineConfig")
    registry = ArtifactRegistry(
        config.project_root / "config" / "model-artifacts.json",
        artifact_root=config.artifact_root,
        tokenizer_root=config.tokenizer_root,
        repository_root=config.project_root,
        native_operator_probe=lambda: probe_focalnet_native_operator(
            config.focalnet_root,
            device=config.device,
        ),
    )
    report = registry.verify_all()
    if not report.ready:
        raise report.errors[0]
    verified = {artifact.id: artifact for artifact in report.verified_artifacts}
    detector_artifact = verified[DETECTOR_MODEL_ID]
    classifier_artifact = verified[CLASSIFIER_MODEL_ID]
    warmup_inputs = _WarmupInputs()

    class DetectorResident:
        def __init__(self) -> None:
            self.artifact = detector_artifact
            self._adapter = FocalNetDinoAdapter.from_local_source(
                detector_artifact,
                repository_root=config.focalnet_root,
                project_root=config.project_root,
                device=config.device,
            )

        def warmup(self) -> None:
            self._adapter.predict(warmup_inputs.get(DETECTOR_MODEL_ID))

        def execute(self, inputs: object) -> object:
            return self._adapter.predict(inputs)

    class ClassifierResident:
        def __init__(self) -> None:
            self.artifact = classifier_artifact
            self._adapter = MmbcdClassifierAdapter.from_local_assets(
                classifier_artifact,
                tokenizer_root=config.tokenizer_root,
                dino_root=config.dino_root,
                mmbcd_root=config.mmbcd_root,
                project_root=config.project_root,
                device=config.device,
            )

        def warmup(self) -> None:
            self.execute(warmup_inputs.get(CLASSIFIER_MODEL_ID))

        def execute(self, inputs: object) -> object:
            mammogram, rois, history = inputs
            return self._adapter.predict(mammogram, rois, history)

    runtime_type = (
        PersistentResidencyRuntime if config.retain_models else SingleResidencyRuntime
    )
    runtime = runtime_type(
        bindings=(
            ModelBinding(
                model_id=DETECTOR_MODEL_ID,
                load=DetectorResident,
                failure_token=lambda: (
                    f"{detector_artifact.sha256}:"
                    f"{detector_artifact.repository_revision}"
                ),
            ),
            ModelBinding(
                model_id=CLASSIFIER_MODEL_ID,
                load=ClassifierResident,
                failure_token=lambda: (
                    f"{classifier_artifact.sha256}:"
                    f"{classifier_artifact.repository_revision}"
                ),
            ),
        ),
        accelerator=TorchCudaLifecycle(device=config.device),
    )
    return PredictionPipeline(
        decoder=DicomCanonicalizer(),
        runtime=_InputAwareRuntime(runtime, warmup_inputs),
        require_history_for_full=config.require_history_for_full,
    )
