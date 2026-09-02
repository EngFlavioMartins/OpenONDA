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

    def evaluate(self, stage_state: StageState, stage_time: float, stage_rates: StageRates) -> None:
        """Evaluate self-induced and external rates for one common stage state."""
        self.induction.evaluate_stage(
            position=stage_state.position,
            vortex_strength=stage_state.vortex_strength,
            core_radius=stage_state.core_radius,
            count=stage_state.count,
            velocity_out=stage_rates.velocity,
            vortex_strength_rate_out=stage_rates.vortex_strength_rate,
            velocity_gradient_out=stage_rates.velocity_gradient,
            strength_rate_enabled=stage_rates.strength_rate_enabled,
            stage_time=stage_time,
        )
        for provider in self.providers:
            provider.add_stage_rates(stage_state, stage_time, stage_rates)
        if not self.strength_enabled or not stage_rates.strength_rate_enabled:
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

    def __init__(self, particles, physics, source_owner=None) -> None:
        self.particles = particles
        self.physics = physics
        self.source_owner = source_owner

    @ti.kernel
    def _add_background(self, velocity: ti.template(), count: ti.i32):
        for i in range(count):
            velocity[i] += self.particles.velocity_background[None]

    def add_stage_rates(
        self, stage_state: StageState, stage_time: float, stage_rates: StageRates
    ) -> None:
        count = stage_state.count
        if count == 0:
            return
        self._add_background(stage_rates.velocity, count)

        # Surface-source/blockage particles are an external velocity term, not
        # a diagnostic-only target correction.  Evaluate them against the exact
        # temporary RK positions so every coupled stage sees the same source
        # contribution as arbitrary target queries.
        owner = self.source_owner
        source_count = int(getattr(owner, "n_sources", 0)) if owner is not None else 0
        if source_count:
            self.physics.kernels["compute_target_source_velocity_kernel"](
                stage_state.position,
                owner.source_position,
                owner.source_strength,
                owner.source_core_radius,
                stage_rates.velocity,
                count,
                source_count,
            )

        body_field = getattr(self.physics, "body_velocity_field", None)
        if body_field is not None:
            _call_stage_velocity_field(body_field, stage_state, stage_rates.velocity, count)

        body = getattr(self.physics, "body_velocity", None)
        override = getattr(self.physics, "velocity_override", None)
        if body is None and override is None:
            return
        position = stage_state.position.to_numpy()[:count]
        velocity = stage_rates.velocity.to_numpy()[:count]
        if body is not None:
            velocity += np.asarray(
                _call_stage_velocity_callback(body, position, stage_time), dtype=velocity.dtype
            ).reshape(count, 3)
        if override is not None:
            if hasattr(override, "blend_into"):
                override.blend_into(position, velocity, velocity)
            else:
                velocity[...] = np.asarray(
                    _call_stage_velocity_callback(override, position, stage_time, velocity),
                    dtype=velocity.dtype,
                )
        uploaded = np.zeros((self.physics.max_n_particles, 3), dtype=velocity.dtype)
        uploaded[:count] = velocity
        stage_rates.velocity.from_numpy(uploaded)


@ti.data_oriented
class AxisymmetricNoSwirlStageProjection:
    """Project stage velocity and strength rates onto the declared symmetry."""

    def __init__(self, physics, orbit_id, axis: int) -> None:
        if axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1, or 2")
        self.physics = physics
        self.orbit_id = orbit_id
        self.axis = int(axis)

    def add_stage_rates(
        self, stage_state: StageState, stage_time: float, stage_rates: StageRates
    ) -> None:
        del stage_time
        self.physics.average_axisymmetric_no_swirl_rhs(
            stage_state.position,
            stage_rates.velocity,
            stage_rates.vortex_strength_rate,
            self.orbit_id,
            self.axis,
            stage_state.count,
        )


def _call_stage_velocity_callback(callback, position, stage_time, velocity=None):
    """Call a legacy or stage-aware host velocity callback.

    Existing body callbacks accept ``(position)`` while new coupled providers
    may accept ``(position, stage_time)`` or ``(position, stage_time, velocity)``.
    Signature inspection would reject callable objects with dynamic signatures;
    the small arity ladder keeps both forms explicit and backwards-compatible.
    """
    if velocity is not None:
        try:
            return callback(position, stage_time, velocity)
        except TypeError as exc:
            try:
                return callback(position, velocity)
            except TypeError:
                try:
                    return callback(position, stage_time)
                except TypeError:
                    if exc.__traceback__ is not None:
                        raise
                    raise
    try:
        return callback(position, stage_time)
    except TypeError as exc:
        try:
            return callback(position)
        except TypeError:
            if exc.__traceback__ is not None:
                raise
            raise


def _call_stage_velocity_field(callback, stage_state, velocity, count):
    """Invoke a device callback with the stage time when it supports it."""
    try:
        callback(stage_state.position, velocity, count, stage_state.time)
    except TypeError:
        callback(stage_state.position, velocity, count)


__all__ = [
    "AxisymmetricNoSwirlStageProjection",
    "ExternalStageContribution",
    "ParticleExternalStageContribution",
    "StageRHS",
]
