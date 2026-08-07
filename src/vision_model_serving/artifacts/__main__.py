"""Command-line entry point for artifact manifest verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from .manifest import ManifestValidationError, load_manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate model-artifact metadata without loading checkpoints."
    )
    parser.add_argument("manifest", type=Path, help="Path to the JSON manifest")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the validated inventory as JSON",
    )
    args = parser.parse_args(argv)

    try:
        manifest = load_manifest(args.manifest)
    except ManifestValidationError as error:
        print(error, file=sys.stderr)
        return 2

    inventory = manifest.inventory()
    if args.json:
        print(json.dumps(inventory, indent=2, sort_keys=True))
    else:
        print(f"VALID {manifest.manifest_id}")
        print("Pipeline: " + " -> ".join(manifest.pipeline_stages))
        for artifact in manifest.artifacts:
            print(
                f"- {artifact.role}: {artifact.filename} "
                f"({artifact.size_bytes} bytes, sha256:{artifact.sha256})"
            )
        print("Clinical semantics: unverified; labels and thresholds disabled")
        print("Public checkpoint uploads: disabled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
