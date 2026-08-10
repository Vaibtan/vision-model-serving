"""Private process-wide PyTorch accelerator policy for every model adapter."""

from __future__ import annotations

import os


_CUBLAS_WORKSPACE_CONFIG = ":4096:8"


class TorchProcessConfigurationError(RuntimeError):
    pass


def configure_deterministic_torch(torch: object, *, device: str) -> None:
    """Apply the one deterministic FP32 policy before model construction."""

    prior_workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if prior_workspace_config not in {None, _CUBLAS_WORKSPACE_CONFIG}:
        raise TorchProcessConfigurationError("CUBLAS determinism configuration differs")
    if (
        device.startswith("cuda")
        and torch.cuda.is_initialized()
        and prior_workspace_config != _CUBLAS_WORKSPACE_CONFIG
    ):
        raise TorchProcessConfigurationError(
            "CUBLAS determinism was configured after CUDA initialization"
        )
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", _CUBLAS_WORKSPACE_CONFIG)
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
        raise TorchProcessConfigurationError(
            f"determinism setup failed ({type(error).__name__})"
        ) from None
