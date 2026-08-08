"""Private, offline PyTorch runtime for the pinned MMBCD classifier."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import sys
from time import perf_counter
from typing import Any, BinaryIO, ContextManager, Protocol

import numpy as np
from numpy.typing import NDArray

from vision_model_serving.artifacts import ArtifactRegistryError
from vision_model_serving.compatibility.environment import load_environment_spec

from .adapter import (
    ClassifierAdapterError,
    ClassifierArtifactIdentity,
    ClassifierRuntimeOutput,
    TokenBatch,
)


_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
_DEVICE = re.compile(r"(?:cpu|cuda(?::[0-9]+)?)")
_ALIASES = (
    ("img_fc1.weight", "img_fc_layer.1.weight"),
    ("img_fc1.bias", "img_fc_layer.1.bias"),
    ("txt_fc1.weight", "txt_fc_layer.1.weight"),
    ("txt_fc1.bias", "txt_fc_layer.1.bias"),
)


class ClassifierLoadError(ClassifierAdapterError):
    code = "classifier_load_failed"


class ClassifierInferenceError(ClassifierAdapterError):
    code = "classifier_inference_failed"


class ClassifierArtifact(Protocol):
    id: str
    role: str
    sha256: str
    repository_revision: str
    strict_load_verified: bool

    def open_checkpoint(self) -> ContextManager[BinaryIO]: ...


ClassifierModelFactory = Callable[[], object]


class TorchMmbcdRuntime:
    """Own the only live MMBCD model and device tensors for an adapter."""

    def __init__(
        self,
        artifact: ClassifierArtifact,
        *,
        model_factory: ClassifierModelFactory,
        device: str,
    ):
        self.identity = _validate_artifact(artifact)
        self._device = _validate_device(device)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        prior_workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        workspace_config = os.environ.setdefault(
            "CUBLAS_WORKSPACE_CONFIG",
            ":4096:8",
        )
        if workspace_config != ":4096:8":
            raise ClassifierLoadError(
                "classifier CUBLAS determinism configuration differs"
            )
        try:
            import torch
        except ImportError:
            raise ClassifierLoadError(
                "PyTorch is unavailable in the pinned classifier runtime"
            ) from None
        self._torch = torch
        if (
            self._device.startswith("cuda")
            and torch.cuda.is_initialized()
            and prior_workspace_config != ":4096:8"
        ):
            raise ClassifierLoadError(
                "CUBLAS determinism was configured after CUDA initialization"
            )
        _configure_determinism(torch)

        started = perf_counter()
        try:
            model = model_factory()
        except ClassifierLoadError:
            raise
        except Exception as error:
            raise ClassifierLoadError(
                f"classifier construction failed ({type(error).__name__})"
            ) from None

        try:
            with artifact.open_checkpoint() as stream:
                raw_state = torch.load(
                    stream,
                    map_location="cpu",
                    weights_only=True,
                )
        except ArtifactRegistryError:
            raise
        except Exception as error:
            raise ClassifierLoadError(
                f"restricted checkpoint loading failed ({type(error).__name__})"
            ) from None
        state = _canonical_state_dict(raw_state)
        _verify_aliases(state, torch)
        try:
            load_result = model.load_state_dict(state, strict=True)
        except Exception as error:
            raise ClassifierLoadError(
                f"strict classifier state load failed ({type(error).__name__})"
            ) from None
        if load_result.missing_keys or load_result.unexpected_keys:
            raise ClassifierLoadError(
                "strict classifier state load reported key differences"
            )
        try:
            self._model = model.to(device=self._device, dtype=torch.float32)
            self._model.eval()
        except Exception as error:
            raise ClassifierLoadError(
                f"classifier device initialization failed ({type(error).__name__})"
            ) from None
        self.load_ms = (perf_counter() - started) * 1000.0

    def execute(
        self,
        crops: NDArray[np.float32],
        tokens: TokenBatch,
    ) -> ClassifierRuntimeOutput:
        host_crops = np.array(crops, dtype=np.float32, order="C", copy=True)
        host_ids = np.array(tokens.input_ids, dtype=np.int64, order="C", copy=True)
        host_mask = np.array(
            tokens.attention_mask,
            dtype=np.int64,
            order="C",
            copy=True,
        )
        try:
            device_crops = self._torch.from_numpy(host_crops).to(
                device=self._device,
                dtype=self._torch.float32,
            )
            input_ids = self._torch.from_numpy(host_ids).to(
                device=self._device,
                dtype=self._torch.int64,
            )
            attention_mask = self._torch.from_numpy(host_mask).to(
                device=self._device,
                dtype=self._torch.int64,
            )
            self._synchronize()
            started = perf_counter()
            with self._torch.inference_mode():
                output = self._model(device_crops, input_ids, attention_mask)
            self._synchronize()
            inference_ms = (perf_counter() - started) * 1000.0
        except Exception as error:
            raise ClassifierInferenceError(
                f"classifier execution failed ({type(error).__name__})"
            ) from None
        if not isinstance(output, tuple) or len(output) != 3:
            raise ClassifierInferenceError("classifier output contract is invalid")
        try:
            logits, embeddings, attention = (
                _tensor_to_numpy(value) for value in output
            )
        except Exception as error:
            raise ClassifierInferenceError(
                f"classifier output transfer failed ({type(error).__name__})"
            ) from None
        return ClassifierRuntimeOutput(
            logits=logits,
            fused_embeddings=embeddings,
            roi_attention=attention,
            inference_ms=inference_ms,
        )

    def _synchronize(self) -> None:
        if self._device.startswith("cuda"):
            self._torch.cuda.synchronize(self._device)


@dataclass(frozen=True, slots=True)
class LocalMmbcdModelFactory:
    """Reconstruct MMBCD only after both local source identities are pinned."""

    dino_root: Path
    mmbcd_root: Path
    project_root: Path
    expected_mmbcd_revision: str

    def build(self) -> object:
        try:
            environment = load_environment_spec(
                self.project_root / "config" / "l4-fp32-environment.json"
            )
            if environment.upstream_commits["mmbcd"] != self.expected_mmbcd_revision:
                raise ClassifierLoadError(
                    "artifact and MMBCD source revision contracts do not match"
                )
            _verify_git_head(
                self.dino_root,
                environment.upstream_commits["dino"],
                "DINO",
            )
            _verify_git_head(
                self.mmbcd_root,
                environment.upstream_commits["mmbcd"],
                "MMBCD",
            )
            model = _build_model(self.dino_root)
        except ClassifierLoadError:
            raise
        except Exception as error:
            raise ClassifierLoadError(
                f"pinned classifier source validation failed ({type(error).__name__})"
            ) from None
        return model


def _load_pinned_dino_architecture(dino_root: Path) -> object:
    root = dino_root.resolve()
    module_path = root / "vision_transformer.py"
    utils_path = root / "utils.py"
    if any(
        not path.is_file() or path.is_symlink()
        for path in (module_path, utils_path)
    ):
        raise ClassifierLoadError("pinned DINO architecture source is absent")

    missing = object()
    prior_utils = sys.modules.get("utils", missing)
    try:
        utils_spec = importlib.util.spec_from_file_location("utils", utils_path)
        if utils_spec is None or utils_spec.loader is None:
            raise ClassifierLoadError("pinned DINO utilities cannot be loaded")
        utils_module = importlib.util.module_from_spec(utils_spec)
        sys.modules["utils"] = utils_module
        utils_spec.loader.exec_module(utils_module)

        architecture_spec = importlib.util.spec_from_file_location(
            "vision_model_serving_pinned_dino",
            module_path,
        )
        if architecture_spec is None or architecture_spec.loader is None:
            raise ClassifierLoadError("pinned DINO architecture cannot be loaded")
        architecture = importlib.util.module_from_spec(architecture_spec)
        architecture_spec.loader.exec_module(architecture)
        return architecture
    finally:
        if prior_utils is missing:
            sys.modules.pop("utils", None)
        else:
            sys.modules["utils"] = prior_utils


def _build_model(dino_root: Path) -> object:
    try:
        import torch
        import torch.nn as nn
        from transformers import RobertaConfig, RobertaForSequenceClassification

        dino_vit = _load_pinned_dino_architecture(dino_root)
    except ClassifierLoadError:
        raise
    except Exception as error:
        raise ClassifierLoadError(
            f"pinned architecture import failed ({type(error).__name__})"
        ) from None

    class MmbcdInference(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.img_size = 224
            self.image_encoder = dino_vit.vit_small(patch_size=8, num_classes=0)
            self.img_fc1 = nn.Linear(384, 256)
            self.img_fc_layer = nn.Sequential(
                nn.BatchNorm1d(384),
                self.img_fc1,
                nn.GELU(),
            )
            config = RobertaConfig(
                vocab_size=50265,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_act="gelu",
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                max_position_embeddings=514,
                type_vocab_size=1,
                initializer_range=0.02,
                layer_norm_eps=1e-5,
                pad_token_id=1,
                bos_token_id=0,
                eos_token_id=2,
                position_embedding_type="absolute",
                use_cache=True,
                classifier_dropout=None,
                num_labels=2,
                output_hidden_states=True,
            )
            self.text_encoder = RobertaForSequenceClassification(config)
            self.txt_fc1 = nn.Linear(768, 256)
            self.txt_fc_layer = nn.Sequential(
                nn.BatchNorm1d(768),
                self.txt_fc1,
                nn.GELU(),
            )
            self.attention = nn.MultiheadAttention(
                embed_dim=256,
                num_heads=1,
                batch_first=True,
                dropout=0.3,
            )
            self.model_fc2 = nn.Linear(768, 2)

        def forward(
            self,
            image_tensor: Any,
            input_ids: Any,
            attention_mask: Any,
        ) -> tuple[Any, Any, Any]:
            images = image_tensor.view(-1, 3, self.img_size, self.img_size)
            features = self.image_encoder(images).squeeze(-1).squeeze(-1)
            image_embeddings = self.img_fc_layer(features).view(
                image_tensor.shape[0],
                image_tensor.shape[1],
                -1,
            )
            maxpooled, _ = torch.max(image_embeddings, dim=1)
            text_output = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            sentence = text_output.hidden_states[-1][:, 0, :]
            text_embeddings = self.txt_fc_layer(sentence)
            attended, attention_weights = self.attention(
                text_embeddings.unsqueeze(1),
                image_embeddings,
                image_embeddings,
            )
            fused = torch.cat(
                (attended.squeeze(1), text_embeddings, maxpooled),
                dim=1,
            )
            return self.model_fc2(fused), fused, attention_weights

    return MmbcdInference()


def _validate_artifact(
    artifact: ClassifierArtifact,
) -> ClassifierArtifactIdentity:
    if (
        getattr(artifact, "id", None) != "mmbcd-classifier"
        or getattr(artifact, "role", None) != "classifier"
    ):
        raise ClassifierLoadError("artifact is not the manifest-owned classifier")
    if getattr(artifact, "strict_load_verified", None) is not True:
        raise ClassifierLoadError("artifact lacks strict-load evidence")
    sha256 = getattr(artifact, "sha256", "")
    revision = getattr(artifact, "repository_revision", "")
    if _SHA256.fullmatch(sha256) is None or _GIT_COMMIT.fullmatch(revision) is None:
        raise ClassifierLoadError("artifact identity is malformed")
    return ClassifierArtifactIdentity(artifact.id, sha256, revision)


def _validate_device(device: str) -> str:
    if not isinstance(device, str) or _DEVICE.fullmatch(device) is None:
        raise ClassifierLoadError("classifier device is invalid")
    return device


def _canonical_state_dict(raw_state: object) -> dict[str, object]:
    if not isinstance(raw_state, Mapping) or not raw_state:
        raise ClassifierLoadError("verified checkpoint is not a raw state dictionary")
    canonical: dict[str, object] = {}
    for key, value in raw_state.items():
        if not isinstance(key, str):
            raise ClassifierLoadError("verified checkpoint has a non-text state key")
        normalized = key[7:] if key.startswith("module.") else key
        if normalized in canonical:
            raise ClassifierLoadError(
                "state key collision after approved prefix normalization"
            )
        canonical[normalized] = value
    return canonical


def _verify_aliases(state: Mapping[str, object], torch: object) -> None:
    if not all(
        left in state
        and right in state
        and torch.equal(state[left], state[right])
        for left, right in _ALIASES
    ):
        raise ClassifierLoadError("classifier checkpoint aliases differ")


def _verify_git_head(root: Path, expected: str, label: str) -> None:
    path = root.expanduser().resolve()
    if _GIT_COMMIT.fullmatch(expected) is None or not path.is_dir():
        raise ClassifierLoadError(f"pinned {label} source identity is invalid")
    git_prefix = [
        "git",
        "-c",
        f"safe.directory={path.as_posix()}",
        "-C",
        str(path),
    ]
    try:
        result = subprocess.run(
            [*git_prefix, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
        status = subprocess.run(
            [*git_prefix, "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except (OSError, subprocess.SubprocessError):
        raise ClassifierLoadError(f"pinned {label} source cannot be verified") from None
    if result.stdout.strip() != expected:
        raise ClassifierLoadError(f"pinned {label} source revision differs")
    if status.stdout.strip():
        raise ClassifierLoadError(f"pinned {label} source has tracked changes")


def _tensor_to_numpy(tensor: Any) -> NDArray[np.float32]:
    values = tensor.detach().cpu().contiguous().numpy()
    return np.ascontiguousarray(values, dtype=np.float32)


def _configure_determinism(torch: object) -> None:
    try:
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.use_deterministic_algorithms(True, warn_only=False)
    except Exception as error:
        raise ClassifierLoadError(
            f"classifier determinism setup failed ({type(error).__name__})"
        ) from None
