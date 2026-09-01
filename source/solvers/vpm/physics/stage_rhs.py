"""Central stage right-hand side for coupled VPM evolution."""

from typing import Protocol

import numpy as np
import taichi as ti

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
        *,
        strength_enabled: bool = True,
    ) -> None:
        self.induction = induction
        self.providers = tuple(providers)
        self.strength_enabled = bool(strength_enabled)

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
        if not self.strength_enabled:
            _zero_stage_field(stage_rates.vortex_strength_rate, stage_state.count)


@ti.kernel
def _zero_stage_field(field: ti.template(), count: ti.i32):
    for i in range(count):
        field[i] = ti.Vector([0.0, 0.0, 0.0])


@ti.data_oriented
class ParticleExternalStageContribution:
    """Apply stage-time external velocities to a particle RHS.

    The provider is intentionally small: self-induced induction belongs to the
    selected induction object, while freestream, body and overlap corrections
    are explicit stage contributions.  Python callbacks are given the supplied
    stage positions, never the accepted particle field.
    """

    def __init__(self, particles, physics) -> None:
        self.particles = particles
        self.physics = physics

    @ti.kernel
    def _add_background(self, velocity: ti.template(), count: ti.i32):
        for i in range(count):
            velocity[i] += self.particles.velocity_background[None]

    def add_stage_rates(
        self, stage_state: StageState, stage_time: float, stage_rates: StageRates
    ) -> None:
        del stage_time
        count = stage_state.count
        if count == 0:
            return
        self._add_background(stage_rates.velocity, count)
        body_field = getattr(self.physics, "body_velocity_field", None)
        if body_field is not None:
            body_field(stage_state.position, stage_rates.velocity, count)

        body = getattr(self.physics, "body_velocity", None)
        override = getattr(self.physics, "velocity_override", None)
        if body is None and override is None:
            return
        position = stage_state.position.to_numpy()[:count]
        velocity = stage_rates.velocity.to_numpy()[:count]
        if body is not None:
            velocity += np.asarray(body(position), dtype=velocity.dtype).reshape(count, 3)
        if override is not None:
            if hasattr(override, "blend_into"):
                override.blend_into(position, velocity, velocity)
            else:
                velocity[...] = np.asarray(override(position, velocity), dtype=velocity.dtype)
        uploaded = np.zeros((self.physics.max_n_particles, 3), dtype=velocity.dtype)
        uploaded[:count] = velocity
        stage_rates.velocity.from_numpy(uploaded)


__all__ = ["ExternalStageContribution", "ParticleExternalStageContribution", "StageRHS"]
