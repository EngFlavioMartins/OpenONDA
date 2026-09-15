"""Integrate the compact conservative source on remeshed Cartesian particles.

This research operator holds positions, occupied cells and the 0.18 m support
taper fixed during each source substep. Explicit midpoint reevaluates both the
free-space potential and body-complete velocity Jacobian from trial strengths.
It adds no particles and performs no moment repair. It is not a production
operator or an assertion of second-order accuracy for GBD plus renewal.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import time

from cube_covector_flux_control import direct_potential, potential_flux_source, potential_on_lattice
from cube_wake_operator_audit import curl
from cube_wake_particle_probe import rms
import numpy as np
from scipy.ndimage import distance_transform_edt


def midpoint_increment(
    strength: np.ndarray, duration: float, rate: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """Return the explicit-midpoint strength increment [m³/s] for one source ODE."""
    first_rate = rate(strength)
    return duration * rate(strength + 0.5 * duration * first_rate)


def invariants(solver) -> dict:
    """Read active device fields and compute circulation/impulse without cached fields."""
    particles = solver.particles
    position = particles.position_cpu(use_cache=False).astype(float)
    strength = particles.vortex_strength_cpu(use_cache=False).astype(float)
    return {
        "particles": len(position),
        "net_strength": strength.sum(axis=0).tolist(),
        "linear_impulse_per_density": (0.5 * np.cross(position, strength).sum(axis=0)).tolist(),
        "strength_l1": float(np.linalg.norm(strength, axis=1).sum()),
    }


@dataclass(frozen=True)
class SourceCells:
    """Occupied cells, ghost identifiers and a compact scalar taper on a fixed lattice."""

    position: np.ndarray
    index: np.ndarray
    identifiers: np.ndarray
    taper: np.ndarray
    core_radius: float
    spacing: float = 0.06

    @classmethod
    def from_particles(cls, position: np.ndarray, radius: np.ndarray) -> SourceCells:
        """Require unique common-core centres on the cube's actual 0.06 m lattice."""
        spacing = 0.06
        np.testing.assert_array_equal(radius, np.full(len(radius), radius[0]))
        origin = 0.03 + spacing * (np.floor((position.min(axis=0) - 0.03) / spacing) - 1)
        index = np.rint((position - origin) / spacing).astype(int)
        if np.max(np.abs(position - origin - index * spacing)) > 3e-7:
            raise ValueError("Compact face flux requires a remeshed Cartesian particle state")
        shape = tuple((index.max(axis=0) + 2).tolist())
        identifiers = np.full(shape, -1, dtype=int)
        identifiers[tuple(index.T)] = np.arange(len(position))
        if np.count_nonzero(identifiers >= 0) != len(position):
            raise ValueError("Multiple particles occupy the same source cell")
        distance = np.maximum(
            distance_transform_edt(identifiers >= 0, sampling=spacing) - spacing / 2, 0
        )
        taper = 0.5 * (1 - np.cos(np.pi * np.clip(distance / 0.18, 0, 1)))
        return cls(position, index, identifiers, taper[tuple(index.T)], float(radius[0]))

    def rate(self, strength: np.ndarray, jacobian: np.ndarray) -> tuple[np.ndarray, dict]:
        """Compute Γdot=-2∮χψ Jᵀn dA [m³/s²] using shared internal faces."""
        coefficient = np.zeros((*self.identifiers.shape, 3))
        coefficient[tuple(self.index.T)] = strength
        potential = potential_on_lattice(coefficient, self.spacing, self.core_radius)
        psi = potential[tuple(self.index.T)]
        phi = self.taper * psi
        source, _, _, links = potential_flux_source(
            phi, jacobian, self.index, self.identifiers, self.spacing
        )
        net = source.sum(axis=0)
        strength_l1 = float(np.linalg.norm(source, axis=1).sum())
        if np.linalg.norm(net) > 1e-12 * max(1, strength_l1):
            raise AssertionError("Shared-face source failed algebraic circulation conservation")
        return source, {
            "internal_faces": links,
            "source_net_strength_rate": net.tolist(),
            "source_strength_rate_l1": strength_l1,
            "source_impulse_rate_per_density": (
                0.5 * np.cross(self.position, source).sum(axis=0)
            ).tolist(),
            "midpoint_impulse_target": (
                -(self.spacing**3) * np.sum(phi[:, None] * curl(jacobian), axis=0)
            ).tolist(),
            "strength_sha256": hashlib.sha256(strength.tobytes()).hexdigest(),
            "potential": psi,
        }


class CompactFluxStep:
    """Apply source substeps through the solver's strength and panel-state owners.

    Trial strengths are uploaded for consistent tree and panel evaluations.
    A failed substep restores its input strengths. The physical solver clock
    and particle geometry do not advance during this autonomous source solve.
    """

    def __init__(self, solver):
        self.solver = solver
        self.calls: list[dict] = []
        self.wall_seconds = 0.0
        self.evaluations = 0

    def _strength(self) -> np.ndarray:
        return self.solver.particles.vortex_strength_cpu(use_cache=False).astype(float)

    def _set_strength(self, strength: np.ndarray) -> None:
        current = self._strength()
        self.solver.update_particle_vortex_strength(
            np.ones(len(current), dtype=bool), strength - current
        )

    def advance(self, duration: float, phase: str) -> None:
        """Advance Γ at fixed x by explicit midpoint over ``duration`` seconds."""
        started = time.perf_counter()
        before = invariants(self.solver)
        position = self.solver.particles.position_cpu(use_cache=False).astype(float)
        radius = self.solver.particles.core_radius_cpu(use_cache=False).astype(float)
        cells = SourceCells.from_particles(position, radius)
        initial = self._strength()
        stages = []

        def evaluate(strength):
            evaluation_started = time.perf_counter()
            self._set_strength(strength)
            actual_strength = self._strength()
            self.solver.refresh_boundary_element_solution()
            jacobian = self.solver.compute_velocity_gradient_at_points(
                position, particle_spacing=cells.spacing
            )
            gradient_done = time.perf_counter()
            source, budget = cells.rate(actual_strength, jacobian)
            psi = budget.pop("potential")
            if self.evaluations == 0:
                selected = np.linspace(0, len(position) - 1, 48, dtype=int)
                independent = direct_potential(
                    position[selected], position, actual_strength, radius
                )
                error = rms((psi[selected] - independent)[:, None])
                if error > 2e-6:
                    raise AssertionError(f"Free-space potential check failed: {error}")
                budget["potential_direct_error_rms"] = error
            budget["gradient_wall_seconds"] = gradient_done - evaluation_started
            budget["potential_and_flux_wall_seconds"] = time.perf_counter() - gradient_done
            stages.append(budget)
            self.evaluations += 1
            return source

        try:
            increment = midpoint_increment(initial, duration, evaluate)
            self._set_strength(initial + increment)
        except Exception:
            self._set_strength(initial)
            self.solver.refresh_boundary_element_solution()
            raise
        after = invariants(self.solver)
        elapsed = time.perf_counter() - started
        self.wall_seconds += elapsed
        self.calls.append(
            {
                "time": self.solver.time,
                "phase": phase,
                "duration": duration,
                "before": before,
                "after": after,
                "integrated_source_strength": increment.sum(axis=0).tolist(),
                "integrated_source_impulse_per_density": (
                    0.5 * np.cross(position, increment).sum(axis=0)
                ).tolist(),
                "storage_net_strength_closure": (
                    np.asarray(after["net_strength"])
                    - before["net_strength"]
                    - increment.sum(axis=0)
                ).tolist(),
                "wall_seconds": elapsed,
                "stages": stages,
            }
        )

    def measurements(self) -> dict:
        """Return scalar provenance and actual source-stage conservation budgets."""
        return {
            "integration": "S(dt/2) A(dt) S(dt/2); each S uses explicit midpoint",
            "total_order_claim": "None: native A includes GBD splitting and algebraic renewal",
            "taper_width_m": 0.18,
            "wall_seconds": self.wall_seconds,
            "evaluations": self.evaluations,
            "substeps": self.calls,
        }
