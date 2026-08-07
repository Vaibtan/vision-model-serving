#!/usr/bin/env python3
"""Import and functionally validate the FocalNet-DINO CUDA extension."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
from pathlib import Path
import sys

from _common import default_paths, require_directory


def main() -> None:
    defaults = default_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ops-dir",
        type=Path,
        default=defaults["focalnet_repo"] / "models" / "dino" / "ops",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Report a skipped GPU gate instead of failing when CUDA is unavailable",
    )
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        if args.allow_cpu:
            print("FOCALNET CUDA EXTENSION SKIPPED: CUDA unavailable")
            return
        raise RuntimeError("CUDA is unavailable")

    ops_dir = require_directory(args.ops_dir)
    sys.path.insert(0, str(ops_dir))
    extension = importlib.import_module("MultiScaleDeformableAttention")
    extension_path = Path(extension.__file__).resolve()

    test_path = ops_dir / "test.py"
    test_spec = importlib.util.spec_from_file_location(
        "focalnet_deformable_attention_test", test_path
    )
    if test_spec is None or test_spec.loader is None:
        raise RuntimeError(f"Cannot load extension parity checks from {test_path}")
    test_module = importlib.util.module_from_spec(test_spec)
    test_spec.loader.exec_module(test_module)

    print("PyTorch:", torch.__version__)
    print("Extension:", extension_path)
    print("Extension SHA256:", _sha256(extension_path))
    test_module.check_forward_equal_with_pytorch_double()
    test_module.check_forward_equal_with_pytorch_float()
    print("FOCALNET CUDA EXTENSION PASSED")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
