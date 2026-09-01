"""One generic coupled explicit Runge--Kutta engine for VPM particles."""

from typing import Any, Protocol

import taichi as ti

from ..physics.induction.base import StageRates, StageState
from .rk_tableaux import SSPRK3, RKTableau


def _make_construct_stage_kernel(scalar_dtype):
    """Create the stage-construction kernel at the particle precision."""

    @ti.kernel
    def construct(
        position: ti.template(),
        vortex_strength: ti.template(),
        stage_position: ti.template(),
        stage_vortex_strength: ti.template(),
        velocity_0: ti.template(),
        velocity_1: ti.template(),
        velocity_2: ti.template(),
        velocity_3: ti.template(),
        strength_rate_0: ti.template(),
        strength_rate_1: ti.template(),
        strength_rate_2: ti.template(),
        strength_rate_3: ti.template(),
        time_step_size: scalar_dtype,
        a0: scalar_dtype,
        a1: scalar_dtype,
        a2: scalar_dtype,
        a3: scalar_dtype,
        count: ti.i32,
    ):
        for i in range(count):
            stage_position[i] = position[i] + time_step_size * (
                a0 * velocity_0[i]
                + a1 * velocity_1[i]
                + a2 * velocity_2[i]
                + a3 * velocity_3[i]
            )
            stage_vortex_strength[i] = vortex_strength[i] + time_step_size * (
                a0 * strength_rate_0[i]
                + a1 * strength_rate_1[i]
                + a2 * strength_rate_2[i]
                + a3 * strength_rate_3[i]
            )

    return construct


def _make_combine_kernel(scalar_dtype):
    """Create the final RK combination kernel at the particle precision."""

    @ti.kernel
    def combine(
        position: ti.template(),
        vortex_strength: ti.template(),
        velocity_0: ti.template(),
        velocity_1: ti.template(),
        velocity_2: ti.template(),
        velocity_3: ti.template(),
        strength_rate_0: ti.template(),
        strength_rate_1: ti.template(),
        strength_rate_2: ti.template(),
        strength_rate_3: ti.template(),
        time_step_size: scalar_dtype,
        b0: scalar_dtype,
        b1: scalar_dtype,
        b2: scalar_dtype,
        b3: scalar_dtype,
        count: ti.i32,
    ):
        for i in range(count):
            position[i] += time_step_size * (
                b0 * velocity_0[i]
                + b1 * velocity_1[i]
                + b2 * velocity_2[i]
                + b3 * velocity_3[i]
            )
            vortex_strength[i] += time_step_size * (
                b0 * strength_rate_0[i]
                + b1 * strength_rate_1[i]
                + b2 * strength_rate_2[i]
                + b3 * strength_rate_3[i]
            )

    return combine


class CoupledStageRHS(Protocol):
    """Evaluate both rates for one supplied temporary stage state."""

    def evaluate(
        self, stage_state: StageState, stage_time: float, stage_rates: StageRates
    ) -> None:
        """Write velocity and vortex-strength rate into ``stage_rates``."""


@ti.data_oriented
class RungeKutta:
    """Advance position and vortex strength with one shared RK tableau.

    All stage fields are allocated once.  The engine has no knowledge of the
    induction backend, diffusion scheme, or physical forcing; it calls the
    supplied stage right-hand side exactly once per tableau stage.
    """

    def __init__(
        self,
        tableau: RKTableau | None = None,
        *,
        max_n_particles: int,
        dtype=ti.f32,
    ) -> None:
        self.tableau = SSPRK3() if tableau is None else tableau
        self.max_n_particles = int(max_n_particles)
        if self.max_n_particles < 1:
            raise ValueError("max_n_particles must be positive")
        if self.tableau.stages > 4:
            raise ValueError("the VPM RK workspace supports at most four stages")

        self.stage_position = ti.Vector.field(3, dtype=dtype, shape=(self.max_n_particles,))
        self.stage_vortex_strength = ti.Vector.field(
            3, dtype=dtype, shape=(self.max_n_particles,)
        )
        self.stage_velocity = [
            ti.Vector.field(3, dtype=dtype, shape=(self.max_n_particles,))
            for _ in range(4)
        ]
        self.stage_strength_rate = [
            ti.Vector.field(3, dtype=dtype, shape=(self.max_n_particles,))
            for _ in range(4)
        ]
        scalar_dtype = ti.f64 if dtype == ti.f64 else ti.f32
        self._construct_stage_kernel = _make_construct_stage_kernel(scalar_dtype)
        self._combine_kernel = _make_combine_kernel(scalar_dtype)

    @property
    def name(self) -> str:
        """Name of the active tableau for configuration reporting."""
        return self.tableau.name

    def advance(
        self,
        *,
        position: Any,
        vortex_strength: Any,
        core_radius: Any,
        count: int,
        time: float,
        time_step_size: float,
        right_hand_side: CoupledStageRHS,
        velocity_gradient_out: Any | None = None,
    ) -> None:
        """Advance one coupled particle state over ``time_step_size``."""
        count = int(count)
        if count < 0 or count > self.max_n_particles:
            raise ValueError(f"stage count {count} exceeds RK capacity {self.max_n_particles}")
        if count == 0 or time_step_size == 0.0:
            return

        def padded(values: tuple[float, ...]) -> tuple[float, float, float, float]:
            return tuple(values) + (0.0,) * (4 - len(values))

        zero_field = self.stage_velocity[0]
        for stage in range(self.tableau.stages):
            coefficients = self.tableau.a[stage]
            if stage == 0:
                self._construct_stage_kernel(
                    position,
                    vortex_strength,
                    self.stage_position,
                    self.stage_vortex_strength,
                    zero_field,
                    zero_field,
                    zero_field,
                    zero_field,
                    self.stage_strength_rate[0],
                    self.stage_strength_rate[1],
                    self.stage_strength_rate[2],
                    self.stage_strength_rate[3],
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    count,
                )
            else:
                self._construct_stage_kernel(
                    position,
                    vortex_strength,
                    self.stage_position,
                    self.stage_vortex_strength,
                    self.stage_velocity[0],
                    self.stage_velocity[1],
                    self.stage_velocity[2],
                    self.stage_velocity[3],
                    self.stage_strength_rate[0],
                    self.stage_strength_rate[1],
                    self.stage_strength_rate[2],
                    self.stage_strength_rate[3],
                    float(time_step_size),
                    *padded(coefficients),
                    count,
                )

            stage_state = StageState(
                position=self.stage_position,
                vortex_strength=self.stage_vortex_strength,
                core_radius=core_radius,
                count=count,
                time=float(time + self.tableau.c[stage] * time_step_size),
            )
            stage_rates = StageRates(
                velocity=self.stage_velocity[stage],
                vortex_strength_rate=self.stage_strength_rate[stage],
                velocity_gradient=velocity_gradient_out,
            )
            right_hand_side.evaluate(stage_state, stage_state.time, stage_rates)

        coefficients = self.tableau.b
        self._combine_kernel(
            position,
            vortex_strength,
            self.stage_velocity[0],
            self.stage_velocity[1],
            self.stage_velocity[2],
            self.stage_velocity[3],
            self.stage_strength_rate[0],
            self.stage_strength_rate[1],
            self.stage_strength_rate[2],
            self.stage_strength_rate[3],
            float(time_step_size),
            *padded(coefficients),
            count,
        )


__all__ = ["CoupledStageRHS", "RungeKutta"]
