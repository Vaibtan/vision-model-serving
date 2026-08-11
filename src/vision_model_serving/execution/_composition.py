"""Private composition root for the persistent GPU executor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from vision_model_serving.artifacts import ArtifactRegistry
from vision_model_serving.classifier import MmbcdClassifierAdapter
from vision_model_serving.detector import (
    FocalNetDinoAdapter,
    probe_focalnet_native_operator,
)
from vision_model_serving.dicom import DicomCanonicalizer
from vision_model_serving.model_ids import CLASSIFIER_MODEL_ID, DETECTOR_MODEL_ID
from vision_model_serving.pipeline import PredictionPipeline
from vision_model_serving.residency import (
    ModelBinding,
    SingleResidencyRuntime,
    TorchCudaLifecycle,
)


@dataclass(frozen=True, slots=True)
class ExecutorPipelineConfig:
    project_root: Path
    artifact_root: Path
    tokenizer_root: Path
    focalnet_root: Path
    mmbcd_root: Path
    dino_root: Path
    device: str = "cuda:0"

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


@dataclass(frozen=True, slots=True)
class ExecutorComposition:
    pipeline: PredictionPipeline
    artifact_verification_ms: float
    runtime_initialization_ms: float


def build_executor_pipeline(config: ExecutorPipelineConfig) -> ExecutorComposition:
    """Verify local assets and compose the executor's fixed CUDA pipeline."""

    if not isinstance(config, ExecutorPipelineConfig):
        raise TypeError("config must be an ExecutorPipelineConfig")
    verification_started = perf_counter()
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
    artifact_verification_ms = (perf_counter() - verification_started) * 1_000.0
    runtime_started = perf_counter()
    verified = {artifact.id: artifact for artifact in report.verified_artifacts}
    detector_artifact = verified[DETECTOR_MODEL_ID]
    classifier_artifact = verified[CLASSIFIER_MODEL_ID]

    class DetectorResident:
        def __init__(self) -> None:
            self.artifact = detector_artifact
            self._adapter = FocalNetDinoAdapter.from_local_source(
                detector_artifact,
                repository_root=config.focalnet_root,
                project_root=config.project_root,
                device=config.device,
            )

        def warmup(self, inputs: object) -> None:
            # The cold execute() that follows immediately is the warm pass;
            # a real forward here would run the same patient case twice.
            del inputs

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

        def warmup(self, inputs: object) -> None:
            # See DetectorResident.warmup: the first execute() is the warm pass.
            del inputs

        def execute(self, inputs: object) -> object:
            mammogram, rois, history = inputs
            return self._adapter.predict(mammogram, rois, history)

    runtime = SingleResidencyRuntime(
        bindings=(
            ModelBinding(
                model_id=DETECTOR_MODEL_ID,
                load=DetectorResident,
                failure_token=lambda: (
                    f"{detector_artifact.sha256}:{detector_artifact.repository_revision}"
                ),
            ),
            ModelBinding(
                model_id=CLASSIFIER_MODEL_ID,
                load=ClassifierResident,
                failure_token=lambda: (
                    f"{classifier_artifact.sha256}:{classifier_artifact.repository_revision}"
                ),
            ),
        ),
        accelerator=TorchCudaLifecycle(device=config.device),
    )
    pipeline = PredictionPipeline(
        decoder=DicomCanonicalizer(),
        runtime=runtime,
    )
    return ExecutorComposition(
        pipeline=pipeline,
        artifact_verification_ms=artifact_verification_ms,
        runtime_initialization_ms=(perf_counter() - runtime_started) * 1_000.0,
    )
