"""Checksum-pinned, network-free RoBERTa tokenizer adapter."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath

import numpy as np

from vision_model_serving.artifacts import load_manifest

from .adapter import (
    ClassifierAdapterError,
    ClassifierInputError,
    LocalTokenizerIdentity,
    TokenBatch,
)


class TokenizerVerificationError(ClassifierAdapterError):
    code = "classifier_tokenizer_invalid"


class LocalRobertaTokenizer:
    """Load only the five files owned by the artifact manifest."""

    def __init__(self, tokenizer: object, identity: LocalTokenizerIdentity):
        self._tokenizer = tokenizer
        self.identity = identity

    @classmethod
    def from_manifest(
        cls,
        tokenizer_root: str | Path,
        manifest_path: str | Path,
    ) -> LocalRobertaTokenizer:
        root = Path(tokenizer_root).expanduser().resolve()
        manifest = Path(manifest_path).expanduser().resolve()
        try:
            load_manifest(manifest)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            spec = payload["tokenizer"]
            records = spec["files"]
            verified: list[tuple[str, str]] = []
            for record in records:
                filename = str(record["filename"])
                pure = PurePosixPath(filename)
                if pure.is_absolute() or len(pure.parts) != 1:
                    raise ValueError("unsafe tokenizer filename")
                path = root / filename
                if path.is_symlink() or not path.is_file():
                    raise ValueError("tokenizer file is absent or symbolic")
                if path.stat().st_size != int(record["size_bytes"]):
                    raise ValueError("tokenizer file size differs")
                observed = _sha256_file(path)
                if observed != record["sha256"]:
                    raise ValueError("tokenizer file checksum differs")
                verified.append((filename, observed))
            identity = LocalTokenizerIdentity(
                id=str(spec["id"]),
                revision=str(spec["revision"]),
                file_sha256=tuple(sorted(verified)),
            )
        except Exception as error:
            raise TokenizerVerificationError(
                f"tokenizer snapshot verification failed ({type(error).__name__})"
            ) from None

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            from transformers import RobertaTokenizer

            tokenizer = RobertaTokenizer.from_pretrained(
                str(root),
                local_files_only=True,
                trust_remote_code=False,
            )
        except Exception as error:
            raise TokenizerVerificationError(
                f"offline tokenizer loading failed ({type(error).__name__})"
            ) from None
        return cls(tokenizer, identity)

    def encode(self, prompt: str, *, max_length: int) -> TokenBatch:
        if max_length != 90:
            raise ClassifierInputError("MMBCD tokenizer length must be exactly 90")
        try:
            encoded = self._tokenizer(
                [prompt],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="np",
            )
            return TokenBatch(
                input_ids=np.ascontiguousarray(encoded["input_ids"], dtype=np.int64),
                attention_mask=np.ascontiguousarray(
                    encoded["attention_mask"],
                    dtype=np.int64,
                ),
            )
        except ClassifierInputError:
            raise
        except Exception as error:
            raise ClassifierInputError(
                f"offline tokenization failed ({type(error).__name__})"
            ) from None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
