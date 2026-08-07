#!/usr/bin/env python3
"""Fail-fast probe for the validated Lightning L4/PyTorch environment."""

from __future__ import annotations
import argparse
import json
import os
import platform
import sys
import numpy
import torch
import torchvision
import transformers
from torch.utils.cpp_extension import CUDA_HOME


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-torch", default = "2.8.0+cu128")
    parser.add_argument("--expected-cuda", default = "12.8")
    parser.add_argument("--expected-capability", default = "8.9")
    parser.add_argument("--expected-gpu", default = "NVIDIA L4")
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("PyTorch cannot access CUDA")
    capability = ".".join(map(str, torch.cuda.get_device_capability(0)))
    gpu = torch.cuda.get_device_name(0)
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "transformers": transformers.__version__,
        "numpy": numpy.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_home": CUDA_HOME,
        "gpu": gpu,
        "compute_capability": capability,
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
    }
    print(json.dumps(report, indent=2, sort_keys=True))

    assert torch.__version__ == args.expected_torch
    assert torch.version.cuda == args.expected_cuda
    assert capability == args.expected_capability
    assert gpu == args.expected_gpu

    value = torch.randn(1024, 1024, device="cuda")
    result = value @ value.T
    assert torch.isfinite(result).all()
    print("LIGHTNING L4 ENVIRONMENT PASSED")


if __name__ == "__main__":
    main()
