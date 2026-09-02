# SPDX-License-Identifier: GPL-3.0-or-later
"""Typed quality-optimisation diagnostics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OptimisationDiagnostics:
    """Quality values measured after a mesh stage."""

    max_non_orthogonality_deg: float | None = None
    max_skewness: float | None = None
    max_aspect_ratio: float | None = None

    @classmethod
    def from_quality(cls, quality: dict) -> OptimisationDiagnostics:
        """Adapt authoritative validation metrics into a typed stage result."""
        return cls(
            max_non_orthogonality_deg=_optional_float(quality, "max_non_orthogonality_deg"),
            max_skewness=_optional_float(quality, "max_skewness"),
            max_aspect_ratio=_optional_float(quality, "max_aspect_ratio"),
        )

    def as_dict(self) -> dict[str, float | None]:
        """Return a serialisable diagnostics snapshot."""
        return {
            "max_non_orthogonality_deg": self.max_non_orthogonality_deg,
            "max_skewness": self.max_skewness,
            "max_aspect_ratio": self.max_aspect_ratio,
        }


def _optional_float(values: dict, key: str) -> float | None:
    value = values.get(key)
    return None if value is None else float(value)


__all__ = ["OptimisationDiagnostics"]
