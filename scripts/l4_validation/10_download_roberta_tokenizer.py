#!/usr/bin/env python3
"""Download the exact small RoBERTa tokenizer snapshot used by the validation."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import ROBERTA_REVISION, default_paths


def main() -> None:
    defaults = default_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="FacebookAI/roberta-base")
    parser.add_argument("--revision", default=ROBERTA_REVISION)
    parser.add_argument("--output-dir", type=Path, default=defaults["tokenizer_dir"])
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=output_dir,
        allow_patterns=[
            "vocab.json",
            "merges.txt",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "config.json",
        ],
    )
    print("Tokenizer snapshot:", output)
    print("ROBERTA TOKENIZER SNAPSHOT PASSED")


if __name__ == "__main__":
    main()
