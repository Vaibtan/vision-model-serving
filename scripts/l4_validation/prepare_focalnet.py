#!/usr/bin/env python3
"""Check, apply, or build the pinned FocalNet-DINO compatibility patches."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vision_model_serving.compatibility.cli import focalnet_main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(focalnet_main())
