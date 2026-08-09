"""Command-line interface for the L4 compatibility modules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence

from vision_model_serving.validation.reporting import write_json_atomic

from .environment import (
    EnvironmentSnapshot,
    EnvironmentSpecError,
    evaluate_environment,
    load_environment_spec,
)
from .focalnet import (
    PatchCheckError,
    build_focalnet_extension,
    prepare_focalnet_patches,
)


def environment_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a host against the pinned Lightning L4 FP32 lane."
    )
    parser.add_argument("spec", type=Path)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Skip only CUDA/device gates; dependency mismatches still fail",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    try:
        spec = load_environment_spec(args.spec)
        result = evaluate_environment(
            spec,
            EnvironmentSnapshot.collect(spec),
            allow_cpu=args.allow_cpu,
        )
    except EnvironmentSpecError as error:
        print(f"L4 ENVIRONMENT FAILED: {error}", file=sys.stderr)
        return 2

    payload = {"lane_id": spec.lane_id, **result.as_dict()}
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        write_json_atomic(args.output.expanduser().resolve(), payload)
    if args.json:
        print(serialized, end="")
    else:
        for issue in result.issues:
            print(f"- {issue}")
        print(result.marker)
    return 0 if result.succeeded else 1


def focalnet_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check, apply, or build the pinned FocalNet-DINO patches."
    )
    parser.add_argument("action", choices=("check", "apply", "build"))
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--max-jobs", type=int, default=4)
    args = parser.parse_args(argv)

    try:
        spec = load_environment_spec(args.spec)
        results = prepare_focalnet_patches(
            args.repo,
            args.project_root,
            spec,
            apply=args.action in {"apply", "build"},
        )
        for result in results:
            print(f"{result.state}: {result.path} sha256:{result.sha256}")
        print("FOCALNET PATCH CHECK PASSED")
        if args.action == "build":
            extension = build_focalnet_extension(
                args.repo, spec, max_jobs=args.max_jobs
            )
            print("Extension:", extension)
            print("FOCALNET CUDA EXTENSION BUILD PASSED")
    except (
        EnvironmentSpecError,
        PatchCheckError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"FOCALNET PREPARATION FAILED: {error}", file=sys.stderr)
        return 1
    return 0
