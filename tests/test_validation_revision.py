from __future__ import annotations

from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest

from vision_model_serving.validation.revision import (
    RevisionEvidenceError,
    require_clean_revision,
)


class RevisionEvidenceTests(unittest.TestCase):
    def test_requires_exact_clean_commit(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._git(root, "init")
            (root / "evidence.txt").write_text("clean\n", encoding="utf-8")
            self._git(root, "add", "evidence.txt")
            self._git(
                root,
                "-c",
                "user.name=Codex Test",
                "-c",
                "user.email=codex@example.invalid",
                "commit",
                "-m",
                "fixture",
            )
            revision = self._git(root, "rev-parse", "HEAD").stdout.strip()

            self.assertEqual(require_clean_revision(root, revision), revision)
            with self.assertRaises(RevisionEvidenceError):
                require_clean_revision(root, "0" * 40)

            (root / "evidence.txt").write_text("dirty\n", encoding="utf-8")
            with self.assertRaises(RevisionEvidenceError):
                require_clean_revision(root, revision)

    @staticmethod
    def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )


if __name__ == "__main__":
    unittest.main()
