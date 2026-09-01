"""Central stage right-hand side for coupled VPM evolution."""

from __future__ import annotations

from typing import Protocol

from .induction.base import InductionMethod, StageRates, StageState


class ExternalStageContribution(Protocol):
    """Add explicitly modelled external rates at one RK stage."""

    def add_stage_rates(
        self, stage_state: StageState, stage_time: float, stage_rates: StageRates
    ) -> None:
        """Accumulate external velocity/rate contributions into stage outputs."""


class StageRHS:
    """Combine selected self-induced induction and external stage providers.

    The object has no accepted-state access.  Every provider receives the exact
    temporary position/strength fields and physical time of the current RK
    stage.  Providers that only model velocity leave the strength-rate field
    untouched; providers that model external stretching or forcing explicitly
    add the corresponding rate.
    """

    def __init__(
        self,
        induction: InductionMethod,
        providers: tuple[ExternalStageContribution, ...] = (),
    ) -> None:
        self.induction = induction
        self.providers = tuple(providers)

    def evaluate(
        self, stage_state: StageState, stage_time: float, stage_rates: StageRates
    ) -> None:
        """Evaluate self-induced and external rates for one common stage state."""
        self.induction.evaluate_stage(
            position=stage_state.position,
            vortex_strength=stage_state.vortex_strength,
            core_radius=stage_state.core_radius,
            count=stage_state.count,
            velocity_out=stage_rates.velocity,
            vortex_strength_rate_out=stage_rates.vortex_strength_rate,
            velocity_gradient_out=stage_rates.velocity_gradient,
            stage_time=stage_time,
        )
        for provider in self.providers:
            provider.add_stage_rates(stage_state, stage_time, stage_rates)


__all__ = ["ExternalStageContribution", "StageRHS"]
