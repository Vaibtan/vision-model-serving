#!/usr/bin/env python3
"""Evaluate the current host against the checked-in Lightning L4 FP32 lane."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vision_model_serving.compatibility.cli import environment_main  # noqa: E402


if __name__ == "__main__":
    default_spec = PROJECT_ROOT / "config" / "l4-fp32-environment.json"
    raise SystemExit(environment_main([str(default_spec), *sys.argv[1:]]))
