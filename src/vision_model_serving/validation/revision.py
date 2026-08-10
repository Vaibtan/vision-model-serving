"""Fail-closed Git revision identity for publishable validation evidence."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess


_COMMIT = re.compile(r"[0-9a-f]{40}")


class RevisionEvidenceError(RuntimeError):
    """Raised when evidence cannot be bound to one exact clean commit."""


def require_clean_revision(root: str | Path, expected: str | None = None) -> str:
    """Return the exact HEAD, rejecting malformed, dirty, or differing state."""

    repository = Path(root).expanduser().resolve()
    if expected is not None and _COMMIT.fullmatch(expected) is None:
        raise RevisionEvidenceError("expected revision must be a full Git commit")
    prefix = [
        "git",
        "-c",
        f"safe.directory={repository.as_posix()}",
        "-C",
        str(repository),
    ]
    try:
        observed = subprocess.check_output(
            [*prefix, "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            [*prefix, "status", "--porcelain"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RevisionEvidenceError(
            f"Git revision inspection failed ({type(error).__name__})"
        ) from None
    if _COMMIT.fullmatch(observed) is None:
        raise RevisionEvidenceError("observed revision is not a full Git commit")
    if expected is not None and observed != expected:
        raise RevisionEvidenceError("observed revision differs from the requested commit")
    if dirty:
        raise RevisionEvidenceError("validation evidence requires a clean worktree")
    return observed
