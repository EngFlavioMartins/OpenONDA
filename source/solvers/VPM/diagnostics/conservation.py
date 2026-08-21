"""
Conservation Diagnostics for VLM-VPM Coupling.
==============================================

This module implements circulation and integral-invariant tracking for hybrid
VLM-VPM simulations.

Key concept: circulation conservation (Kelvin's theorem)
--------------------------------------------------------
For inviscid flow, total circulation is conserved:

    d(Gamma_total) / dt = 0

where Gamma_total = Gamma_bound + Gamma_wake. At shedding, circulation must
transfer exactly between the VLM bound system and the VPM wake.

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

    circulation_bound: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Bound circulation vector from VLM [m^2/s]."""

    circulation_wake: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Wake circulation vector from VPM particles [m^2/s]."""

    circulation_total: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Total circulation (bound + wake) [m^2/s]."""

    vortex_strength_error: float = 0.0
    """Relative error in circulation conservation [%]."""

    impulse_wake: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Wake linear impulse from VPM [kg m^2/s]."""

    impulse_total: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Tracked total linear impulse [kg m^2/s]."""

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

    vortex_strength_removed: float = 0.0
    """Magnitude of circulation lost to particle removal [m^2/s]."""


class ConservationTracker:
    """Track conservation properties during VLM-VPM coupled simulations."""

    def __init__(self, density: float = 1.225):
        self.density = density
        self.history: list[ConservationState] = []
        self._initial_circulation: float | None = None

    def record_state(self, solver: "VPMSolver") -> ConservationState:
        """Record conservation quantities at the current time step."""
        state = ConservationState(time=solver.time)

        state.circulation_wake = solver.total_strength
        state.impulse_wake = solver.total_linear_impulse * self.density
        state.kinetic_energy = solver.total_kinetic_energy
        state.energy_dissipation_rate = solver.kinetic_energy_dissipation_rate
        state.n_particles_total = solver.particles.n_particles

        if hasattr(solver, "_particles_removed_this_step"):
            state.n_particles_removed = solver._particles_removed_this_step
            state.vortex_strength_removed = solver._vortex_strength_removed_this_step

        if solver.vlm_solver is not None and solver.vlm_solver._solved:
            state.circulation_bound = solver.vlm_solver.compute_total_circulation()
            try:
                forces = solver.vlm_solver.compute_forces(
                    density=self.density,
                    reference_speed=float(np.linalg.norm(solver.freestream_velocity)),
                )
                state.force_kutta_joukowski = np.array([forces["Fx"], forces["Fy"], forces["Fz"]])
            except Exception:
                pass

        state.circulation_total = state.circulation_bound + state.circulation_wake
        state.impulse_total = state.impulse_wake

        if self._initial_circulation is None:
            self._initial_circulation = np.linalg.norm(state.circulation_total)

        if self._initial_circulation > 1e-10:
            state.vortex_strength_error = (
                100.0
                * abs(np.linalg.norm(state.circulation_total) - self._initial_circulation)
                / self._initial_circulation
            )

        self.history.append(state)
        return state

    def export_csv(self, filename: str = "solution/conservation.csv") -> None:
        """Export conservation history to a CSV file."""
        if len(self.history) == 0:
            print("[WARNING] No conservation data to export")
            return

        Path(filename).parent.mkdir(parents=True, exist_ok=True)

        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "time",
                    "circulation_bound_mag",
                    "circulation_wake_mag",
                    "circulation_total_mag",
                    "circulation_error_pct",
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
                        np.linalg.norm(state.circulation_bound),
                        np.linalg.norm(state.circulation_wake),
                        np.linalg.norm(state.circulation_total),
                        state.vortex_strength_error,
                        *state.impulse_wake,
                        *state.force_kutta_joukowski,
                        state.kinetic_energy,
                        state.energy_dissipation_rate,
                        state.n_particles_total,
                    ]
                )

        print(f"[INFO] Conservation diagnostics exported to: {filename}")

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

        print("\n--- Circulation Conservation (Kelvin's Theorem) ---")
        print(f"Initial total circulation: {self._initial_circulation:.6e} m^2/s")
        print(f"Final total circulation:   {np.linalg.norm(final.circulation_total):.6e} m^2/s")
        print(f"Conservation error:        {final.circulation_error:.3f}%")

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
