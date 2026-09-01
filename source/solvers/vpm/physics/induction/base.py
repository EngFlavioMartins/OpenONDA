"""Contracts shared by VPM induction backends.

The contract deliberately describes rates rather than a particular numerical
algorithm.  A caller supplies the complete temporary RK stage state and owns
the output fields.  Implementations must read only that supplied state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class StageState:
    """The particle source state used for one physical stage evaluation."""

    position: Any
    vortex_strength: Any
    core_radius: Any
    count: int
    time: float = 0.0


@dataclass(frozen=True, slots=True)
class StageRates:
    """Output fields produced by one induction evaluation."""

    velocity: Any
    vortex_strength_rate: Any
    velocity_gradient: Any | None = None


class InductionMethod(Protocol):
    """Evaluate self-induced particle rates for one supplied RK stage."""

    def evaluate_stage(
        self,
        *,
        position: Any,
        vortex_strength: Any,
        core_radius: Any,
        count: int,
        velocity_out: Any,
        vortex_strength_rate_out: Any,
        velocity_gradient_out: Any | None = None,
        stage_time: float = 0.0,
    ) -> None:
        """Write velocity, strength rate, and optionally gradient to outputs."""
