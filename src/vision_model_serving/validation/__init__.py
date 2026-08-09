"""Golden-evidence verification and packaged-validation interfaces."""

from .golden import (
    BoxComparison,
    EvidenceSummary,
    EvidenceVerificationError,
    compare_box_records,
    verify_evidence_archive,
)

__all__ = [
    "BoxComparison",
    "EvidenceSummary",
    "EvidenceVerificationError",
    "compare_box_records",
    "verify_evidence_archive",
]
