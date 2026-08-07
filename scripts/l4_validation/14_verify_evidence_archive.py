#!/usr/bin/env python3
"""Verify the downloaded L4 evidence archive using only the Python standard library."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath


EXPECTED_ARCHIVE_SHA256 = (
    "5a1311b121edd6c03feb279e5aa7170be5a545d859a1e1fc2f70fc32252f097f"
)
REQUIRED_MEMBERS = {
    "environment/dino-commit.txt",
    "environment/focalnet-commit.txt",
    "environment/mmbcd-commit.txt",
    "environment/model-artifacts.sha256",
    "manifests/dicom-manifest.json",
    "manifests/preprocess-manifest.json",
    "manifests/detector-inference-manifest.json",
    "manifests/mmbcd-input-manifest.json",
    "manifests/mmbcd-inference-manifest.json",
    "patches/focalnet-pytorch-2.8-compat.patch",
    "bundles/detections-top8.txt",
    "bundles/mmbcd-inputs.npz",
    "bundles/mmbcd-outputs.npz",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_member(name: str) -> str:
    while name.startswith("./"):
        name = name[2:]
    return name.rstrip("/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("sidecar", type=Path)
    parser.add_argument("--expected-sha256", default=EXPECTED_ARCHIVE_SHA256)
    args = parser.parse_args()

    archive = args.archive.expanduser().resolve()
    sidecar = args.sidecar.expanduser().resolve()
    if not archive.is_file() or not sidecar.is_file():
        raise FileNotFoundError((archive, sidecar))
    sidecar_hash = sidecar.read_text(encoding="utf-8").split()[0].lower()
    observed_hash = sha256_file(archive)
    assert sidecar_hash == observed_hash
    assert observed_hash == args.expected_sha256.lower()

    with tarfile.open(archive, mode="r:gz") as tar:
        members = {normalize_member(member.name): member for member in tar.getmembers()}
        for member_name, member in members.items():
            pure = PurePosixPath(member_name)
            if pure.is_absolute() or ".." in pure.parts:
                raise RuntimeError(f"Unsafe archive member: {member.name}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"Links are forbidden in evidence archive: {member.name}")
        missing = REQUIRED_MEMBERS - members.keys()
        if missing:
            raise RuntimeError(f"Evidence archive is incomplete: {sorted(missing)}")

        def read_json(name: str):
            extracted = tar.extractfile(members[name])
            assert extracted is not None
            return json.load(extracted)

        dicom = read_json("manifests/dicom-manifest.json")
        preprocess = read_json("manifests/preprocess-manifest.json")
        detector = read_json("manifests/detector-inference-manifest.json")
        mmbcd_input = read_json("manifests/mmbcd-input-manifest.json")
        mmbcd = read_json("manifests/mmbcd-inference-manifest.json")

    assert dicom["sha256"] == (
        "9f70081672a460f29231bb471e8a9e26dd3ed26a2ebbd91c064e575e7842a19c"
    )
    assert preprocess["artifacts"]["resized"]["array_sha256"] == (
        "97fa0f80a696ce7f822c1681a8c3f7c072da9262b2bd91239c9f1637eaf68552"
    )
    assert detector["outputs"]["prediction_sha256"] == (
        "4cdd09d986702e8839acff8d7517a63f263ca2a01b0607d78d6b2086c886a9a5"
    )
    assert detector["outputs"]["determinism_max_abs_diff"] == 0.0
    assert detector["outputs"]["mmbcd_topk_count"] == 8
    assert mmbcd_input["image_transform"]["tensor_sha256"] == (
        "89cda9694e3696f63eb70706a6ae4cc2dd16e9be214f27ba6f106479eac90155"
    )
    assert mmbcd_input["text"]["label_information_used"] is False
    assert mmbcd["outputs"]["prediction_sha256"] == (
        "43ec1c4593c0549510098ea082ea7092c7fd5631c95d8b912ecf31633185899b"
    )
    assert mmbcd["determinism"]["bitwise_equal_logits"] is True
    assert mmbcd["determinism"]["bitwise_equal_embeddings"] is True

    print("Archive SHA256:", observed_hash)
    print("Members:", len(members))
    print("Detector prediction SHA256:", detector["outputs"]["prediction_sha256"])
    print("MMBCD prediction SHA256:", mmbcd["outputs"]["prediction_sha256"])
    print("LIGHTNING L4 EVIDENCE ARCHIVE PASSED")


if __name__ == "__main__":
    main()
