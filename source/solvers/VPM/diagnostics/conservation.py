"""
Conservation Diagnostics for VLM-VPM Coupling.
==============================================

This module implements vector-strength and integral-invariant tracking for
hybrid VLM-VPM simulations.

Key concept: bound/wake vortex-strength closure
------------------------------------------------
For a discretized inviscid vortex system, the oriented bound and wake
filament strengths close:

    d(alpha_total) / dt = 0

where alpha = Gamma dl has units L^3/T. At shedding, scalar VLM circulation
Gamma [L^2/T] is converted to the VPM vector strength alpha_p [L^3/T] by the
oriented filament length; the two quantities are never added directly.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: February 2026
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..core.solver import VPMSolver


@dataclass
class ConservationState:
    """Snapshot of conservation quantities at a single time step."""

    time: float
    """Physical time [s]."""

    bound_vortex_strength: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Oriented bound-vortex strength sum from VLM panels [m³/s]."""

    wake_vortex_strength: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Wake particle vortex-strength sum [m³/s]."""

    total_vortex_strength: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Bound plus wake vector strength [m³/s]."""

    vortex_strength_error: float = 0.0
    """Relative drift in bound/wake vector-strength closure [%]."""

    impulse_wake: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Wake linear impulse [kg m/s]."""

    impulse_total: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Tracked total linear impulse [kg m/s]."""

    force_kutta_joukowski: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Force from Kutta-Joukowski on panels [N]."""

    kinetic_energy: float = 0.0
    """Total kinetic energy [J]."""

    energy_dissipation_rate: float = 0.0
    """Viscous energy dissipation [W]."""

    n_particles_total: int = 0
    """Total number of VPM particles."""

    n_particles_shed: int = 0
    """Number of particles shed this step."""

    n_particles_removed: int = 0
    """Number of particles removed."""

    vortex_strength_removed: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Vector strength removed with discarded particles [m³/s]."""


class ConservationTracker:
    """Track conservation properties during VLM-VPM coupled simulations."""

    def __init__(self, density: float = 1.225):
        self.density = density
        self.history: list[ConservationState] = []
        self._initial_vortex_strength: float | None = None

    def record_state(self, solver: "VPMSolver") -> ConservationState:
        """Record conservation quantities at the current time step."""
        state = ConservationState(time=solver.time)

        state.wake_vortex_strength = solver.total_vortex_strength
        state.impulse_wake = solver.total_linear_impulse * self.density
        state.kinetic_energy = solver.total_kinetic_energy * self.density
        state.energy_dissipation_rate = solver.kinetic_energy_dissipation_rate * self.density
        state.n_particles_total = solver.particles.n_particles

        if hasattr(solver, "_particles_removed_this_step"):
            state.n_particles_removed = solver._particles_removed_this_step
            state.vortex_strength_removed = solver._vortex_strength_removed_this_step

        if solver.vlm_solver is not None and solver.vlm_solver._solved:
            state.bound_vortex_strength = solver.vlm_solver.compute_total_bound_vortex_strength()
            try:
                forces = solver.vlm_solver.compute_forces(
                    density=self.density,
                    reference_speed=float(np.linalg.norm(solver.freestream_velocity)),
                )
                state.force_kutta_joukowski = np.array([forces["Fx"], forces["Fy"], forces["Fz"]])
            except Exception:
                pass

        state.total_vortex_strength = state.bound_vortex_strength + state.wake_vortex_strength
        state.impulse_total = state.impulse_wake

        if self._initial_vortex_strength is None:
            self._initial_vortex_strength = np.linalg.norm(state.total_vortex_strength)

        if self._initial_vortex_strength > 1e-10:
            state.vortex_strength_error = (
                100.0
                * abs(np.linalg.norm(state.total_vortex_strength) - self._initial_vortex_strength)
                / self._initial_vortex_strength
            )

        self.history.append(state)
        return state

    def export_csv(
        self,
        case_dir: str | Path,
        file_name: str = "vpm_conservation.csv",
    ) -> Path | None:
        """Export history to ``<case_dir>/samples/<file_name>``."""
        if len(self.history) == 0:
            print("[WARNING] No conservation data to export")
            return None

        filename = Path(case_dir) / "samples" / file_name
        filename.parent.mkdir(parents=True, exist_ok=True)

        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "time",
                    "bound_vortex_strength_mag",
                    "wake_vortex_strength_mag",
                    "total_vortex_strength_mag",
                    "vortex_strength_error_pct",
                    "impulse_wake_x",
                    "impulse_wake_y",
                    "impulse_wake_z",
                    "force_kj_x",
                    "force_kj_y",
                    "force_kj_z",
                    "kinetic_energy",
                    "energy_dissipation_rate",
                    "n_particles",
                ]
            )

            for state in self.history:
                writer.writerow(
                    [
                        state.time,
                        np.linalg.norm(state.bound_vortex_strength),
                        np.linalg.norm(state.wake_vortex_strength),
                        np.linalg.norm(state.total_vortex_strength),
                        state.vortex_strength_error,
                        *state.impulse_wake,
                        *state.force_kutta_joukowski,
                        state.kinetic_energy,
                        state.energy_dissipation_rate,
                        state.n_particles_total,
                    ]
                )

        print(f"[INFO] Conservation diagnostics exported to: {filename}")
        return filename

    def print_summary(self) -> None:
        """Print a short conservation-quality summary."""
        if len(self.history) == 0:
            print("[WARNING] No conservation data recorded")
            return

        final = self.history[-1]

        print("\n" + "=" * 70)
        print("CONSERVATION DIAGNOSTICS SUMMARY")
        print("=" * 70)
        print(f"\nSimulation time: {final.time:.4f} s")
        print(f"Time steps recorded: {len(self.history)}")

        print("\n--- Bound/Wake Vortex-Strength Closure ---")
        print(f"Initial total strength: {self._initial_vortex_strength:.6e} m^3/s")
        print(f"Final total strength:   {np.linalg.norm(final.total_vortex_strength):.6e} m^3/s")
        print(f"Closure error:          {final.vortex_strength_error:.3f}%")

        if final.vortex_strength_error < 0.1:
            status = "EXCELLENT"
        elif final.vortex_strength_error < 1.0:
            status = "GOOD"
        elif final.vortex_strength_error < 5.0:
            status = "ACCEPTABLE"
        else:
            status = "POOR"
        print(f"Status: {status}")

        print("\n--- Surface Force ---")
        print(
            f"Kutta-Joukowski force:     [{final.force_kutta_joukowski[0]:+.4e}, "
            f"{final.force_kutta_joukowski[1]:+.4e}, {final.force_kutta_joukowski[2]:+.4e}] N"
        )

        print("\n--- Energy Budget ---")
        print(f"Kinetic energy:            {final.kinetic_energy:.6e} J")
        print(f"Dissipation rate:          {final.energy_dissipation_rate:.6e} W")

        print("\n--- Particle Statistics ---")
        print(f"Total particles:           {final.n_particles_total}")

        print("=" * 70 + "\n")
