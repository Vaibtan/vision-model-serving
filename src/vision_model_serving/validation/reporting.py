"""Atomic JSON report writing shared by validation workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Write one complete JSON report or leave the prior report intact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
