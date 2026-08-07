#!/usr/bin/env python3
"""Import and run serving-relevant FocalNet-DINO CUDA extension checks."""

from __future__ import annotations

import argparse
import sys
import torch
from pathlib import Path
from _common import default_paths, require_directory
import MultiScaleDeformableAttention as extension

def main() -> None:
    defaults = default_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--ops-dir", type = Path, default = defaults["focalnet_repo"] / "models" / "dino" / "ops")
    args = parser.parse_args()
    ops_dir = require_directory(args.ops_dir)
    sys.path.insert(0, str(ops_dir))
    from test import (check_forward_equal_with_pytorch_double, check_forward_equal_with_pytorch_float)
    print("PyTorch:", torch.__version__)
    print("Extension:", extension.__file__)
    check_forward_equal_with_pytorch_double()
    check_forward_equal_with_pytorch_float()
    print("FOCALNET CUDA EXTENSION PASSED")

if __name__ == "__main__": main()
