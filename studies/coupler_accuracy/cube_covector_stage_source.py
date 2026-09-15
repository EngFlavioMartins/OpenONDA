"""Frozen-source RK experiment for a nodal covector-rate correction.

This source is a Gaussian-filtered, finite-support approximation to
``-2 J.T (omega_G - curl(u))``. It is not an exact Helmholtz correction.
It changes only vortex-strength rates on the existing stage particles.
No accepted-state velocity, vorticity or gradient is used.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import time

from cube_wake_measured_longitudinal import density_at
from cube_wake_operator_audit import curl
from cube_wake_particle_probe import rms
from cube_wake_vorticity_consistency import gaussian_vorticity_and_divergence
import numpy as np


class NodalCovectorSource:
    """Add a measured covector source on a common-volume, Gaussian RK cloud.

    The explicit volume is fixed during RK. This study accepts the cube's
    common volume only; it makes no claim about variable-volume evolution.
    Host neighbour sums are deliberately transparent and are not a production
    performance implementation. Every supplied temporary state is reconstructed
    independently, including the second stage at the same particle count.
    """

    requires_velocity_gradient = True

    def __init__(self, solver):
        self.physics = solver.physics
        self.solver = solver
        self.volume = 0.06**3
        np.testing.assert_allclose(solver.particle_volume, self.volume, rtol=1e-6, atol=0)
        self.calls = 0
        self.wall_seconds = 0.0
        self.first_stages = []
        self.last_rate_budget = {}
        self.step_budgets = []
        self.diffusion_budgets = []
        self._stage_budgets = []

    def invariants(self):
        """Read raw owned fields for scalar bookkeeping, bypassing host caches."""
        particles = self.solver.particles
        count = len(particles)
        position = self.physics._download_vector_field(particles.position, count).astype(float)
        strength = self.physics._download_vector_field(particles.vortex_strength, count).astype(
            float
        )
        return {
            "particles": count,
            "net_strength": strength.sum(axis=0).tolist(),
            "linear_impulse_per_density": (0.5 * np.cross(position, strength).sum(axis=0)).tolist(),
            "strength_l1": float(np.linalg.norm(strength, axis=1).sum()),
        }

    @contextmanager
    def integration_step(self, tableau, time_step_size, strength_enabled):
        """Bookkeep the actual RK tableau and state change without a moment repair."""
        if not strength_enabled:
            raise ValueError("This experiment requires active vortex-strength evolution")
        before = self.invariants()
        self._stage_budgets = []
        yield
        if len(self._stage_budgets) != tableau.stages:
            raise ValueError("Source audit did not observe every RK stage exactly once")
        after = self.invariants()

        def integrate(key):
            return time_step_size * sum(
                weight * np.asarray(stage[key])
                for weight, stage in zip(tableau.b, self._stage_budgets, strict=True)
            )

        net_increment = integrate("applied_total_net_strength_rate")
        source_increment = integrate("applied_correction_net_strength_rate")
        impulse_increment = integrate("applied_total_impulse_rate_per_density")
        source_impulse = integrate("applied_correction_impulse_rate_per_density")
        self.step_budgets.append(
            {
                "before_rk": before,
                "after_rk": after,
                "tableau_weights": list(tableau.b),
                "time_step_size": time_step_size,
                "source_strength_increment": source_increment.tolist(),
                "source_impulse_increment_per_density": source_impulse.tolist(),
                "source_l1_increment": float(integrate("applied_correction_strength_rate_l1")),
                "net_strength_rk_closure": (
                    np.asarray(after["net_strength"]) - before["net_strength"] - net_increment
                ).tolist(),
                # Impulse is bilinear in the RK variables. Its exact state
                # difference need not equal RK quadrature of its derivative.
                "impulse_state_minus_rate_quadrature": (
                    np.asarray(after["linear_impulse_per_density"])
                    - before["linear_impulse_per_density"]
                    - impulse_increment
                ).tolist(),
                "stages": self._stage_budgets,
            }
        )

    def observe_diffusion(self, operation):
        """Wrap a study's diffusion call with read-only before/after measurements."""

        def measured(time_step_size):
            before = self.invariants()
            result = operation(time_step_size)
            self.diffusion_budgets.append(
                {
                    "start_time": self.solver.time,
                    "time_step_size": time_step_size,
                    "completed_rk_updates": len(self.step_budgets),
                    "before": before,
                    "after": self.invariants(),
                }
            )
            return result

        return measured

    def add_stage_rates(self, state, stage_time, rates):
        """Reconstruct this stage and add ``-2 V J.T ell`` in m³/s²."""
        # Accepted-field refreshes consume velocity/gradient only. Their
        # unused strength rates must not pay for this reconstruction.
        if state.stage_index is None or not rates.strength_rate_enabled:
            return
        started = time.perf_counter()
        count = state.count
        if rates.velocity_gradient is None:
            raise ValueError("Covector control requires the complete temporary-stage Jacobian")
        position = self.physics._download_vector_field(state.position, count).astype(float)
        strength = self.physics._download_vector_field(state.vortex_strength, count).astype(float)
        radius = self.physics._download_scalar_field(state.core_radius, count).astype(float)
        jacobian = self.physics._download_matrix_field(rates.velocity_gradient, count).astype(float)
        jacobian = jacobian.reshape(count, 3, 3)
        native_rate = self.physics._download_vector_field(rates.vortex_strength_rate, count)
        velocity = self.physics._download_vector_field(rates.velocity, count).astype(float)
        omega = density_at(position, position, strength, radius)
        longitudinal = omega - curl(jacobian)
        correction = -2 * self.volume * np.einsum("nji,nj->ni", jacobian, longitudinal)
        applied_total = np.asarray(native_rate + correction, dtype=self.physics.np_dtype)
        applied_correction = applied_total.astype(float) - native_rate.astype(float)
        self.physics._upload_vector_array(
            applied_total,
            rates.vortex_strength_rate,
            count,
        )
        if self.calls < 2:
            selected = np.linspace(0, count - 1, 24, dtype=int)
            independent, _ = gaussian_vorticity_and_divergence(
                position[selected], position, strength, radius
            )
            error = rms(omega[selected] - independent)
            if error > 1e-6:
                raise ValueError(f"Truncated Gaussian stage density disagrees by {error}")
            self.first_stages.append(
                {
                    "stage_index": state.stage_index,
                    "time": stage_time,
                    "position_sha256": hashlib.sha256(position.tobytes()).hexdigest(),
                    "strength_sha256": hashlib.sha256(strength.tobytes()).hexdigest(),
                    "density_absolute_rms_at_24_particles": error,
                    "longitudinal_density_rms": rms(longitudinal),
                }
            )
        self.last_rate_budget = {
            "time": stage_time,
            "stage_index": state.stage_index,
            "native_net_strength_rate": native_rate.astype(float).sum(axis=0).tolist(),
            "correction_net_strength_rate": correction.sum(axis=0).tolist(),
            "correction_linear_impulse_rate_per_density": (
                0.5 * np.cross(position, correction).sum(axis=0)
            ).tolist(),
            "correction_strength_rate_l1": float(np.linalg.norm(correction, axis=1).sum()),
            "applied_correction_net_strength_rate": applied_correction.sum(axis=0).tolist(),
            "applied_correction_impulse_rate_per_density": (
                0.5 * np.cross(position, applied_correction).sum(axis=0)
            ).tolist(),
            "applied_correction_strength_rate_l1": float(
                np.linalg.norm(applied_correction, axis=1).sum()
            ),
            "applied_total_net_strength_rate": applied_total.astype(float).sum(axis=0).tolist(),
            "applied_total_impulse_rate_per_density": (
                0.5
                * (
                    np.cross(velocity, strength) + np.cross(position, applied_total.astype(float))
                ).sum(axis=0)
            ).tolist(),
        }
        self._stage_budgets.append(self.last_rate_budget)
        self.calls += 1
        self.wall_seconds += time.perf_counter() - started

    def measurements(self):
        """Return bounded stage verification and cumulative diagnostic cost."""
        return {
            "stage_calls": self.calls,
            "wall_seconds_including_density_validation": self.wall_seconds,
            "first_stages": self.first_stages,
            "last_rate_budget": self.last_rate_budget,
            "step_budgets": self.step_budgets,
            "diffusion_budgets": self.diffusion_budgets,
        }
