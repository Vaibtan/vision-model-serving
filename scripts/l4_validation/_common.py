"""Shared private helpers and immutable references for the L4 validation scripts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


STUDIO_ROOT = Path("/teamspace/studios/this_studio")
SERIES_UID = "1.3.6.1.4.1.9590.100.1.2.100131208110604806117271735422083351547"

FOCALNET_COMMIT = "23901e021dc6ec8f66bad47983f45a25574452cc"
MMBCD_COMMIT = "14ac5e099c79253b01e0885d2ebefa6f86cfd8f0"
DINO_COMMIT = "7c446df5b9f45747937fb0d72314eb9f7b66930a"
ROBERTA_REVISION = "e2da8e2f811d1448a5b465c236feacd80ffbac7b"

FOCALNET_TRAINING_SHA256 = (
    "67a7b0cd787a3aaba199cf1ff82ed2934c33ffe37544473379d7a837ab1637b4"
)
FOCALNET_INFERENCE_SHA256 = (
    "a7fed981c7309d12c19532624e148b45087462c56568797cb62b60055bb62f04"
)
MMBCD_SHA256 = "2264351216f9fb4945af35e300459ff4ce2e7f5445519348024f3bf1eec721a4"
DICOM_SHA256 = "9f70081672a460f29231bb471e8a9e26dd3ed26a2ebbd91c064e575e7842a19c"

PREPROCESSED_ARRAY_SHA256 = (
    "97fa0f80a696ce7f822c1681a8c3f7c072da9262b2bd91239c9f1637eaf68552"
)
DETECTOR_PREDICTION_SHA256 = (
    "4cdd09d986702e8839acff8d7517a63f263ca2a01b0607d78d6b2086c886a9a5"
)
MMBCD_INPUT_TENSOR_SHA256 = (
    "89cda9694e3696f63eb70706a6ae4cc2dd16e9be214f27ba6f106479eac90155"
)
MMBCD_PREDICTION_SHA256 = (
    "43ec1c4593c0549510098ea082ea7092c7fd5631c95d8b912ecf31633185899b"
)


def default_paths() -> dict[str, Path]:
    fixture_dir = STUDIO_ROOT / "fixtures" / "cbis-ddsm" / SERIES_UID
    return {
        "studio_root": STUDIO_ROOT,
        "project_repo": STUDIO_ROOT / "vision-model-serving",
        "focalnet_repo": STUDIO_ROOT / "src" / "FocalNet-DINO",
        "mmbcd_repo": STUDIO_ROOT / "src" / "MMBCD",
        "dino_repo": STUDIO_ROOT / "src" / "dino",
        "artifact_dir": STUDIO_ROOT / "vision-model-serving-artifacts",
        "fixture_dir": fixture_dir,
        "preprocess_dir": fixture_dir / "preprocessed",
        "detector_dir": fixture_dir / "detector",
        "mmbcd_input_dir": fixture_dir / "mmbcd" / "input",
        "mmbcd_output_dir": fixture_dir / "mmbcd" / "output",
        "tokenizer_dir": (
            STUDIO_ROOT / "assets" / f"roberta-base-tokenizer-{ROBERTA_REVISION}"
        ),
    }


def path_argument(
    parser: argparse.ArgumentParser,
    name: str,
    default: Path,
    help_text: str,
) -> None:
    parser.add_argument(name, type=Path, default=default, help=help_text)


def require_file(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def require_directory(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise NotADirectoryError(path)
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: Any) -> str:
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def verify_sha256(path: Path, expected: str) -> str:
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise RuntimeError(
            f"SHA-256 mismatch for {path}: expected {expected}, observed {actual}"
        )
    return actual


def git_commit(repository: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def verify_git_commit(repository: Path, expected: str) -> str:
    actual = git_commit(require_directory(repository))
    if actual != expected:
        raise RuntimeError(
            f"Commit mismatch for {repository}: expected {expected}, observed {actual}"
        )
    return actual


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(require_file(path).read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    canonical: dict[str, Any] = {}
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith("module.") else key
        if new_key in canonical:
            raise RuntimeError(f"State key collision after prefix stripping: {new_key}")
        canonical[new_key] = value
    return canonical
