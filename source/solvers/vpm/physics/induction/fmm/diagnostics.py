"""FMM runtime diagnostics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class FMMDiagnostics:
    """Counters recorded for one evaluator instance."""

    hierarchy_builds: int = 0
    p2p_interactions: int = 0
    m2l_interactions: int = 0
    last_uncorrected_rate_defect: float = 0.0


__all__ = ["FMMDiagnostics"]
