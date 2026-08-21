"""Local stabilization operators for the vortex-particle field.

Both operators here act on one particle at a time from quantities the solver
has already evaluated, so they cost O(N) and need no grid, no neighbour search,
and no linear solve.  They are the two cheapest entries in the stabilization
hierarchy; the moment-constrained mechanisms (filament refinement, Winckelmans
projection, conservative regularization) live beside them in this package.

``apply_stretching_viscosity``
    Adds a stretching-aware residual viscosity to ``nu_eff``, so the energy it
    removes leaves through the configured viscous operator and is auditable as
    viscous dissipation rather than as an unreported clip.

``apply_pedrizzetti_relaxation``
    Rotates each strength toward the vorticity direction it induces, which is
    the quantity the divergence-driven three-dimensional instability grows.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: August 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import numpy as np
import taichi as ti


@ti.data_oriented
class StabilizationOperators:
    """Per-particle stabilization kernels owned by the stabilization subsystem.

    The operators are standalone: they hold the Taichi fields they write and act
    on particle-array templates handed in by the caller, so neither ``physics``
    nor the particle container needs a dependency on this module.  One instance
    is constructed by :class:`StabilizationManager`, which decides when the
    operators may act.
    """

    def __init__(self, compute_dtype, max_particles: int) -> None:
        self.accumulator_dtype = compute_dtype
        self.stabilization_viscosity = ti.field(dtype=compute_dtype, shape=(max_particles,))
        self._stabilization_viscosity_sum = ti.field(dtype=compute_dtype, shape=())
        self._stabilization_viscosity_max = ti.field(dtype=compute_dtype, shape=())
        self._stabilization_viscosity_active = ti.field(dtype=ti.i32, shape=())
        self._pedrizzetti_misalignment_sum = ti.field(dtype=compute_dtype, shape=())
        self._pedrizzetti_misalignment_max = ti.field(dtype=compute_dtype, shape=())
        self._pedrizzetti_strength_before = ti.field(dtype=compute_dtype, shape=())
        self._pedrizzetti_strength_after = ti.field(dtype=compute_dtype, shape=())
        self._pedrizzetti_relaxed_count = ti.field(dtype=ti.i32, shape=())

    @ti.kernel
    def _apply_stretching_viscosity_kernel(
        self,
        strength: ti.template(),
        strain_rate: ti.template(),
        volume: ti.template(),
        viscosity: ti.template(),
        viscosity_turbulent: ti.template(),
        viscosity_effective: ti.template(),
        coefficient: ti.f32,
        count: ti.i32,
    ):
        for i in range(count):
            gamma = strength[i]
            gamma_sq = gamma.dot(gamma)
            production = ti.cast(0.0, self.accumulator_dtype)
            if gamma_sq > ti.cast(1.0e-30, self.accumulator_dtype):
                production = ti.max(gamma.dot(strain_rate[i] @ gamma) / gamma_sq, 0.0)
            delta_sq = ti.pow(ti.max(volume[i], 0.0), 2.0 / 3.0)
            nu_stabilization = coefficient * delta_sq * production
            self.stabilization_viscosity[i] = nu_stabilization
            viscosity_effective[i] = viscosity[i] + viscosity_turbulent[i] + nu_stabilization
            ti.atomic_add(self._stabilization_viscosity_sum[None], nu_stabilization)
            ti.atomic_max(self._stabilization_viscosity_max[None], nu_stabilization)
            if nu_stabilization > 0.0:
                ti.atomic_add(self._stabilization_viscosity_active[None], 1)

    def apply_stretching_viscosity(self, particles, coefficient: float) -> dict[str, float]:
        """Add positive-production residual viscosity to ``nu_eff``.

        ``particles.strain_rate`` must describe the same state as the current
        particle strengths.  The returned statistics are used only for audit
        output and do not take part in the numerical update.
        """
        count = len(particles)
        self._stabilization_viscosity_sum.fill(0.0)
        self._stabilization_viscosity_max.fill(0.0)
        self._stabilization_viscosity_active.fill(0)
        if count <= 0 or coefficient <= 0.0:
            return {
                "stabilization_viscosity_mean": 0.0,
                "stabilization_viscosity_max": 0.0,
                "stabilization_viscosity_active_fraction": 0.0,
            }
        self._apply_stretching_viscosity_kernel(
            particles.vortex_strength,
            particles.strain_rate,
            particles.volume,
            particles.kinematic_viscosity,
            particles.eddy_viscosity,
            particles.effective_viscosity,
            float(coefficient),
            count,
        )
        ti.sync()
        return {
            "stabilization_viscosity_mean": float(self._stabilization_viscosity_sum[None]) / count,
            "stabilization_viscosity_max": float(self._stabilization_viscosity_max[None]),
            "stabilization_viscosity_active_fraction": float(
                self._stabilization_viscosity_active[None]
            )
            / count,
        }

    @ti.kernel
    def _apply_pedrizzetti_relaxation_kernel(
        self,
        strength: ti.template(),
        velocity_gradient: ti.template(),
        factor: ti.f32,
        conserve_strength: ti.i32,
        count: ti.i32,
    ):
        for i in range(count):
            gamma = strength[i]
            gradient = velocity_gradient[i]
            vorticity = ti.Vector(
                [
                    gradient[2, 1] - gradient[1, 2],
                    gradient[0, 2] - gradient[2, 0],
                    gradient[1, 0] - gradient[0, 1],
                ]
            )
            gamma_norm = gamma.norm()
            vorticity_norm = vorticity.norm()
            tiny = ti.cast(1.0e-30, self.accumulator_dtype)
            if gamma_norm > tiny and vorticity_norm > tiny:
                cosine = ti.min(
                    ti.max(gamma.dot(vorticity) / (gamma_norm * vorticity_norm), -1.0),
                    1.0,
                )
                relaxed = (1.0 - factor) * gamma + (
                    factor * gamma_norm / vorticity_norm
                ) * vorticity
                relaxed_norm = relaxed.norm()
                if conserve_strength == 1 and relaxed_norm > tiny:
                    relaxed *= gamma_norm / relaxed_norm
                    relaxed_norm = gamma_norm
                strength[i] = relaxed
                misalignment = ti.acos(cosine)
                ti.atomic_add(self._pedrizzetti_misalignment_sum[None], misalignment * gamma_norm)
                ti.atomic_max(self._pedrizzetti_misalignment_max[None], misalignment)
                ti.atomic_add(self._pedrizzetti_strength_before[None], gamma_norm)
                ti.atomic_add(self._pedrizzetti_strength_after[None], relaxed_norm)
                ti.atomic_add(self._pedrizzetti_relaxed_count[None], 1)

    def apply_pedrizzetti_relaxation(
        self,
        particles,
        factor: float,
        *,
        conserve_strength: bool = True,
    ) -> dict[str, float]:
        """Rotate every strength toward the vorticity direction it induces.

        The vorticity is taken as the curl of ``particles.velocity_gradient``,
        so this must be called while that gradient still describes the state
        being relaxed.  Each strength moves along the short arc between
        ``alpha_p`` and ``omega(x_p)``, which bounds the correction by the
        misalignment itself; with ``conserve_strength`` the rotation is exact
        and no particle strength is created or destroyed.

        Vector circulation, linear impulse, and angular impulse are *not*
        preserved by the rotation.  The returned statistics report the angle
        that was removed and the strength that the uncorrected form would have
        dissipated; they are audit output and take no part in the update.
        """
        count = len(particles)
        self._pedrizzetti_misalignment_sum.fill(0.0)
        self._pedrizzetti_misalignment_max.fill(0.0)
        self._pedrizzetti_strength_before.fill(0.0)
        self._pedrizzetti_strength_after.fill(0.0)
        self._pedrizzetti_relaxed_count.fill(0)
        if count <= 0 or factor <= 0.0:
            return {
                "pedrizzetti_misalignment_deg": 0.0,
                "pedrizzetti_misalignment_max_deg": 0.0,
                "pedrizzetti_strength_change_relative": 0.0,
                "pedrizzetti_relaxed_fraction": 0.0,
            }
        self._apply_pedrizzetti_relaxation_kernel(
            particles.vortex_strength,
            particles.velocity_gradient,
            float(factor),
            1 if conserve_strength else 0,
            count,
        )
        ti.sync()
        strength_before = float(self._pedrizzetti_strength_before[None])
        strength_after = float(self._pedrizzetti_strength_after[None])
        return {
            "pedrizzetti_misalignment_deg": np.degrees(
                float(self._pedrizzetti_misalignment_sum[None])
                / max(strength_before, np.finfo(float).tiny)
            ),
            "pedrizzetti_misalignment_max_deg": np.degrees(
                float(self._pedrizzetti_misalignment_max[None])
            ),
            "pedrizzetti_strength_change_relative": (strength_after - strength_before)
            / max(strength_before, np.finfo(float).tiny),
            "pedrizzetti_relaxed_fraction": float(self._pedrizzetti_relaxed_count[None]) / count,
        }
