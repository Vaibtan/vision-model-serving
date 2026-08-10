"""Private PyTorch execution boundary for the FocalNet-DINO detector."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import runpy
import sys
from time import perf_counter
from typing import Any, BinaryIO, ContextManager, Iterator, Protocol

import numpy as np

from vision_model_serving.model_ids import DETECTOR_MODEL_ID
from numpy.typing import NDArray

from vision_model_serving.artifacts import ArtifactRegistryError, load_manifest
from vision_model_serving.compatibility.environment import load_environment_spec
from vision_model_serving.compatibility.focalnet import prepare_focalnet_patches
from vision_model_serving.dicom import CanonicalMammogram
from vision_model_serving.residency._torch_policy import (
    TorchProcessConfigurationError,
    configure_deterministic_torch,
)

from .postprocessing import (
    DetectorAdapterError,
    DetectorInput,
    DetectorPostprocessor,
    DetectorPreprocessor,
    ProposalSelection,
)


_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
_DEVICE = re.compile(r"(?:cpu|cuda(?::[0-9]+)?)")


class DetectorLoadError(DetectorAdapterError):
    code = "detector_load_failed"


class DetectorInferenceError(DetectorAdapterError):
    code = "detector_inference_failed"


class DetectorArtifact(Protocol):
    id: str
    role: str
    sha256: str
    repository_revision: str
    strict_load_verified: bool

    def open_checkpoint(self) -> ContextManager[BinaryIO]: ...


DetectorModelFactory = Callable[[], object]


@dataclass(frozen=True, slots=True)
class DetectorArtifactIdentity:
    id: str
    sha256: str
    repository_revision: str


@dataclass(frozen=True, slots=True)
class DetectorTimings:
    load_ms: float
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float

    def __post_init__(self) -> None:
        for value in (
            self.load_ms,
            self.preprocess_ms,
            self.inference_ms,
            self.postprocess_ms,
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("detector timings must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class DetectorResult:
    artifact: DetectorArtifactIdentity
    detector_input: DetectorInput
    proposals: ProposalSelection
    timings: DetectorTimings


@dataclass(frozen=True, slots=True)
class _RuntimePrediction:
    logits: NDArray[np.float32]
    boxes: NDArray[np.float32]
    inference_ms: float


class _TorchDetectorRuntime:
    """Own the only model and device tensors used by one loaded adapter."""

    def __init__(
        self,
        artifact: DetectorArtifact,
        *,
        model_factory: DetectorModelFactory,
        device: str,
    ):
        loaded = _load_verified_detector_model(
            artifact,
            model_factory=model_factory,
            device=device,
        )
        self.identity = loaded.identity
        self._device = loaded.device
        self._torch = loaded.torch
        self._model = loaded.model
        self.load_ms = loaded.load_ms

    def predict(self, detector_input: DetectorInput) -> _RuntimePrediction:
        host_values = np.array(
            detector_input.tensor,
            dtype=np.float32,
            order="C",
            copy=True,
        )
        try:
            device_tensor = self._torch.from_numpy(host_values).to(
                device=self._device,
                dtype=self._torch.float32,
                non_blocking=False,
            )
            self._synchronize()
            started = perf_counter()
            with self._torch.inference_mode():
                output = self._model([device_tensor])
            self._synchronize()
            inference_ms = (perf_counter() - started) * 1000.0
        except Exception as error:
            raise DetectorInferenceError(
                f"detector execution failed ({type(error).__name__})"
            ) from None
        if not isinstance(output, Mapping) or not {
            "pred_logits",
            "pred_boxes",
        }.issubset(output):
            raise DetectorInferenceError("detector output keys are invalid")
        try:
            logits = _tensor_to_numpy(output["pred_logits"])
            boxes = _tensor_to_numpy(output["pred_boxes"])
        except Exception as error:
            raise DetectorInferenceError(
                f"detector output transfer failed ({type(error).__name__})"
            ) from None
        if logits.shape != (1, 900, 1) or boxes.shape != (1, 900, 4):
            raise DetectorInferenceError(
                "detector output shapes differ from the verified artifact contract"
            )
        return _RuntimePrediction(logits, boxes, inference_ms)

    def _synchronize(self) -> None:
        if self._device.startswith("cuda"):
            self._torch.cuda.synchronize(self._device)


@dataclass(frozen=True, slots=True)
class _LoadedDetectorModel:
    identity: DetectorArtifactIdentity
    device: str
    torch: object
    model: object
    load_ms: float


def _load_verified_detector_model(
    artifact: DetectorArtifact,
    *,
    model_factory: DetectorModelFactory,
    device: str,
) -> _LoadedDetectorModel:
    """Build and strict-load the one detector used by eager validation."""

    identity = _validate_artifact(artifact)
    validated_device = _validate_device(device)
    try:
        import torch
    except ImportError:
        raise DetectorLoadError("PyTorch is unavailable in the pinned detector runtime") from None
    try:
        configure_deterministic_torch(torch, device=validated_device)
    except TorchProcessConfigurationError as error:
        raise DetectorLoadError(f"detector {error}") from None

    started = perf_counter()
    try:
        model = model_factory()
    except DetectorLoadError:
        raise
    except Exception as error:
        raise DetectorLoadError(f"detector construction failed ({type(error).__name__})") from None

    try:
        with artifact.open_checkpoint() as stream:
            with torch.serialization.safe_globals([argparse.Namespace]):
                checkpoint = torch.load(
                    stream,
                    map_location="cpu",
                    weights_only=True,
                )
    except ArtifactRegistryError:
        raise
    except Exception as error:
        raise DetectorLoadError(
            f"restricted checkpoint loading failed ({type(error).__name__})"
        ) from None
    if not isinstance(checkpoint, Mapping) or "model" not in checkpoint:
        raise DetectorLoadError("verified checkpoint has no model state root")
    state_dict = checkpoint["model"]
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise DetectorLoadError("verified model state root is invalid")

    try:
        load_result = model.load_state_dict(state_dict, strict=True)
    except Exception as error:
        raise DetectorLoadError(
            f"strict detector state load failed ({type(error).__name__})"
        ) from None
    if load_result.missing_keys or load_result.unexpected_keys:
        raise DetectorLoadError("strict detector state load reported key differences")
    try:
        model = model.to(device=validated_device, dtype=torch.float32)
        model.eval()
    except Exception as error:
        raise DetectorLoadError(
            f"detector device initialization failed ({type(error).__name__})"
        ) from None
    return _LoadedDetectorModel(
        identity=identity,
        device=validated_device,
        torch=torch,
        model=model,
        load_ms=(perf_counter() - started) * 1_000.0,
    )


class FocalNetDinoAdapter:
    """Expose canonical inputs and immutable detector results, never the model."""

    def __init__(
        self,
        runtime: _TorchDetectorRuntime,
        *,
        preprocessor: DetectorPreprocessor | None = None,
        postprocessor: DetectorPostprocessor | None = None,
    ):
        self._runtime = runtime
        self._preprocessor = preprocessor or DetectorPreprocessor()
        self._postprocessor = postprocessor or DetectorPostprocessor()

    @classmethod
    def from_artifact(
        cls,
        artifact: DetectorArtifact,
        *,
        model_factory: DetectorModelFactory,
        device: str = "cuda:0",
        preprocessor: DetectorPreprocessor | None = None,
        postprocessor: DetectorPostprocessor | None = None,
    ) -> FocalNetDinoAdapter:
        runtime = _TorchDetectorRuntime(
            artifact,
            model_factory=model_factory,
            device=device,
        )
        return cls(
            runtime,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
        )

    @classmethod
    def from_local_source(
        cls,
        artifact: DetectorArtifact,
        *,
        repository_root: str | Path,
        project_root: str | Path,
        device: str = "cuda:0",
    ) -> FocalNetDinoAdapter:
        """Construct only from the patched, pinned local FocalNet source tree."""

        factory = _LocalFocalNetDinoFactory(
            repository_root=repository_root,
            project_root=project_root,
            expected_revision=getattr(artifact, "repository_revision", ""),
            device=device,
        )
        return cls.from_artifact(
            artifact,
            model_factory=factory.build,
            device=device,
        )

    def predict(self, mammogram: CanonicalMammogram) -> DetectorResult:
        preprocess_started = perf_counter()
        try:
            detector_input = self._preprocessor.prepare(mammogram.pixels)
        except (AttributeError, TypeError, ValueError) as error:
            raise DetectorInferenceError(
                f"canonical detector input is invalid ({type(error).__name__})"
            ) from None
        preprocess_ms = (perf_counter() - preprocess_started) * 1000.0
        prediction = self._runtime.predict(detector_input)
        postprocess_started = perf_counter()
        try:
            proposals = self._postprocessor.process(
                prediction.logits,
                prediction.boxes,
                mammogram.geometry,
            )
        except DetectorAdapterError:
            raise
        except (AttributeError, TypeError, ValueError) as error:
            raise DetectorInferenceError(
                f"detector postprocessing failed ({type(error).__name__})"
            ) from None
        postprocess_ms = (perf_counter() - postprocess_started) * 1000.0
        return DetectorResult(
            artifact=self._runtime.identity,
            detector_input=detector_input,
            proposals=proposals,
            timings=DetectorTimings(
                load_ms=self._runtime.load_ms,
                preprocess_ms=preprocess_ms,
                inference_ms=prediction.inference_ms,
                postprocess_ms=postprocess_ms,
            ),
        )


def _validate_artifact(artifact: DetectorArtifact) -> DetectorArtifactIdentity:
    if (
        getattr(artifact, "id", None) != DETECTOR_MODEL_ID
        or getattr(artifact, "role", None) != "detector"
    ):
        raise DetectorLoadError("artifact is not the manifest-owned detector")
    if getattr(artifact, "strict_load_verified", None) is not True:
        raise DetectorLoadError("artifact lacks strict-load evidence")
    sha256 = getattr(artifact, "sha256", "")
    revision = getattr(artifact, "repository_revision", "")
    if _SHA256.fullmatch(sha256) is None or _GIT_COMMIT.fullmatch(revision) is None:
        raise DetectorLoadError("artifact identity is malformed")
    return DetectorArtifactIdentity(artifact.id, sha256, revision)


def _validate_device(device: str) -> str:
    if not isinstance(device, str) or _DEVICE.fullmatch(device) is None:
        raise DetectorLoadError("detector device is invalid")
    return device


def _tensor_to_numpy(tensor: Any) -> NDArray[np.float32]:
    values = tensor.detach().cpu().contiguous().numpy()
    return np.ascontiguousarray(values, dtype=np.float32)


class _LocalFocalNetDinoFactory:
    """Validate local source/config identity before invoking the pinned builder."""

    def __init__(
        self,
        *,
        repository_root: str | Path,
        project_root: str | Path,
        expected_revision: str,
        device: str,
    ):
        self._repository_root = Path(repository_root).expanduser().resolve()
        self._project_root = Path(project_root).expanduser().resolve()
        self._expected_revision = expected_revision
        self._device = _validate_device(device)

    def build(self) -> object:
        try:
            environment_spec = load_environment_spec(
                self._project_root / "config" / "l4-fp32-environment.json"
            )
            if environment_spec.upstream_commits.get("focalnet_dino") != (self._expected_revision):
                raise DetectorLoadError("artifact and source revision contracts do not match")
            patch_results = prepare_focalnet_patches(
                self._repository_root,
                self._project_root,
                environment_spec,
                apply=False,
            )
        except DetectorLoadError:
            raise
        except Exception as error:
            raise DetectorLoadError(
                f"pinned detector source validation failed ({type(error).__name__})"
            ) from None
        if not patch_results or any(result.state != "already_applied" for result in patch_results):
            raise DetectorLoadError("required serving source patches are not applied")

        config_path = self._verified_config_path()
        try:
            config_values = {
                key: value
                for key, value in runpy.run_path(str(config_path)).items()
                if not key.startswith("__")
            }
        except Exception as error:
            raise DetectorLoadError(
                f"pinned detector configuration failed ({type(error).__name__})"
            ) from None
        arguments = argparse.Namespace(**config_values)
        arguments.device = "cuda" if self._device.startswith("cuda") else self._device
        arguments.use_checkpoint = False
        arguments.pretrain_model_path = ""
        arguments.nms_iou_threshold = -1
        if (
            arguments.num_classes,
            arguments.num_queries,
            arguments.num_select,
        ) != (1, 900, 300):
            raise DetectorLoadError("pinned detector output contract differs")

        ops_root = self._repository_root / "models" / "dino" / "ops"
        try:
            with _prepend_import_paths(self._repository_root, ops_root):
                extension = importlib.import_module("MultiScaleDeformableAttention")
                dino_module = importlib.import_module("models.dino")
            _require_module_inside(extension, ops_root)
            _require_module_inside(dino_module, self._repository_root)
            model, _, _ = dino_module.build_dino(arguments)
        except DetectorLoadError:
            raise
        except Exception as error:
            raise DetectorLoadError(
                f"pinned detector construction failed ({type(error).__name__})"
            ) from None
        if not all(
            callable(getattr(model, name, None)) for name in ("load_state_dict", "to", "eval")
        ):
            raise DetectorLoadError("pinned detector builder returned an invalid model")
        return model

    def _verified_config_path(self) -> Path:
        manifest_path = self._project_root / "config" / "model-artifacts.json"
        try:
            load_manifest(manifest_path)
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            record = next(
                item
                for item in payload["repository_assets"]
                if item["id"] == "focalnet-dino-config"
            )
            relative = PurePosixPath(record["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("unsafe configuration path")
            config_path = self._project_root.joinpath(*relative.parts).resolve()
            if not config_path.is_relative_to(self._project_root):
                raise ValueError("configuration escapes project root")
            if _sha256_file(config_path) != record["sha256"]:
                raise ValueError("configuration checksum mismatch")
        except Exception as error:
            raise DetectorLoadError(
                f"pinned detector configuration is invalid ({type(error).__name__})"
            ) from None
        return config_path


def probe_focalnet_native_operator(
    repository_root: str | Path,
    *,
    device: str = "cuda:0",
) -> bool:
    """Verify the pinned CUDA device and native detector operator are usable."""

    root = Path(repository_root).expanduser().resolve()
    selected_device = _validate_device(device)
    if not selected_device.startswith("cuda"):
        return False
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        probe = torch.ones(1, device=selected_device)
        torch.cuda.synchronize(selected_device)
        if not bool(torch.isfinite(probe).all().item()):
            return False
        ops_root = root / "models" / "dino" / "ops"
        with _prepend_import_paths(root, ops_root):
            extension = importlib.import_module("MultiScaleDeformableAttention")
        _require_module_inside(extension, ops_root)
        return True
    except Exception:
        return False


@contextmanager
def _prepend_import_paths(*paths: Path) -> Iterator[None]:
    values = [str(path) for path in paths]
    sys.path[:0] = values
    try:
        yield
    finally:
        del sys.path[: len(values)]


def _require_module_inside(module: object, root: Path) -> None:
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str):
        raise DetectorLoadError("pinned detector module has no source identity")
    try:
        valid = Path(module_file).resolve().is_relative_to(root.resolve())
    except OSError:
        valid = False
    if not valid:
        raise DetectorLoadError("detector module resolved outside the pinned source")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
