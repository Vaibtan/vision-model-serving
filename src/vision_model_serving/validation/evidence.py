from __future__ import annotations

from pathlib import Path
from typing import Iterable


def sanitize_error_detail(
    error: BaseException | str,
    roots: Iterable[Path],
    *,
    limit: int = 20_000,
) -> str:
    """Remove host-specific roots and bound generated validation evidence."""

    value = str(error)
    replacements: dict[str, str] = {}
    for root in roots:
        expanded = root.expanduser().resolve()
        label = f"<{expanded.name or 'root'}>"
        native = str(expanded)
        replacements[native] = label
        replacements[native.replace("\\", "/")] = label
    for source in sorted(replacements, key=len, reverse=True):
        value = value.replace(source, replacements[source])
    return value[:limit]
