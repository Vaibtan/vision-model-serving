"""Reference command-line adapter for the public prediction pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Callable, Sequence, TextIO

from .contracts import CaseInput, PredictionMode
from .factory import LocalCudaPipelineConfig, build_local_cuda_pipeline
from .pipeline import PredictionPipeline
from .serialization import prediction_to_dict


PipelineBuilder = Callable[[LocalCudaPipelineConfig], PredictionPipeline]


def main(
    argv: Sequence[str] | None = None,
    *,
    pipeline_builder: PipelineBuilder = build_local_cuda_pipeline,
    stdout: TextIO | None = None,
) -> int:
    args = _parser().parse_args(argv)
    output_stream = stdout or sys.stdout
    config = LocalCudaPipelineConfig(
        project_root=args.project_root,
        artifact_root=args.artifact_root,
        tokenizer_root=args.tokenizer_root,
        focalnet_root=args.focalnet_root,
        mmbcd_root=args.mmbcd_root,
        dino_root=args.dino_root,
        device=args.device,
        require_history_for_full=not args.allow_empty_history,
    )
    pipeline = pipeline_builder(config)
    history = _clinical_history(args.clinical_history_file)
    try:
        with args.dicom.open("rb") as dicom_stream:
            result = pipeline.infer(
                CaseInput(
                    dicom_stream=dicom_stream,
                    clinical_history=history,
                ),
                PredictionMode(args.mode),
            )
        payload = prediction_to_dict(result)
        if args.output is None:
            json.dump(payload, output_stream, allow_nan=False, indent=2)
            output_stream.write("\n")
        else:
            with args.output.open("w", encoding="utf-8", newline="\n") as stream:
                json.dump(payload, stream, allow_nan=False, indent=2)
                stream.write("\n")
    finally:
        history = None
    return 0


def _clinical_history(path: Path | None) -> str | None:
    if path is None:
        return None
    return path.read_text(encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description="Run the offline detector or detector-to-MMBCD pipeline.",
    )
    parser.add_argument("dicom", type=Path)
    parser.add_argument(
        "--mode",
        choices=tuple(mode.value for mode in PredictionMode),
        default=PredictionMode.DETECTION.value,
    )
    parser.add_argument("--clinical-history-file", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--focalnet-root", type=Path, required=True)
    parser.add_argument("--mmbcd-root", type=Path, required=True)
    parser.add_argument("--dino-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--allow-empty-history",
        action="store_true",
        help="Allow the provisional full mode to run with an empty history.",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
