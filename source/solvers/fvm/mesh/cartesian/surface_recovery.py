# SPDX-License-Identifier: GPL-3.0-or-later
"""Surface-recovery diagnostics for the staged mesher."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RecoveryDiagnostics:
    """Summary of a surface-recovery transaction."""

    attempted: int
    accepted: int
    rejected: int
    partial_accepted: int = 0

    @classmethod
    def from_mesh(cls, mesh_data: dict) -> RecoveryDiagnostics:
        """Extract recovery counts emitted by the native recovery stage."""
        projection = mesh_data.get("mesh_generation", {}).get("surface_projection", {})
        attempted = int(projection.get("attempted_points", 0))
        accepted = int(projection.get("accepted_points", 0))
        partial = int(projection.get("partial_accepted_points", 0))
        rejected = max(0, attempted - accepted)
        return cls(attempted, accepted, rejected, partial)

    def as_dict(self) -> dict[str, int]:
        """Return a serialisable diagnostics snapshot."""
        return {
            "attempted": self.attempted,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "partial_accepted": self.partial_accepted,
        }


__all__ = ["RecoveryDiagnostics"]
