"""Resolve simulation paths once, relative to the case root.

This module contains no solver policy beyond path ownership.  In particular,
it never creates directories and never decides whether an existing run may be
overwritten; those decisions belong to the owning lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def resolve_case_path(case_dir: str | Path, value: str | Path | None, default: str) -> Path:
    """Resolve an absolute or case-relative path without consulting CWD."""
    root = Path(case_dir).expanduser().resolve()
    candidate = Path(default if value is None else value).expanduser()
    return (candidate if candidate.is_absolute() else root / candidate).resolve()


@dataclass(frozen=True, slots=True)
class CasePaths:
    """Canonical artifact roots for one simulation case."""

    case_dir: Path
    solution_dir: Path
    samples_dir: Path

    @classmethod
    def resolve(
        cls,
        case_dir: str | Path,
        *,
        solution_dir: str | Path | None = None,
        samples_dir: str | Path | None = None,
        solution_default: str = "solutions",
    ) -> CasePaths:
        root = Path(case_dir).expanduser().resolve()
        return cls(
            case_dir=root,
            solution_dir=resolve_case_path(root, solution_dir, solution_default),
            samples_dir=resolve_case_path(root, samples_dir, "samples"),
        )
