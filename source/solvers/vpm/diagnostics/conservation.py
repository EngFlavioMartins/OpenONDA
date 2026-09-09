"""
Conservation Diagnostics for VLM-VPM Coupling.
==============================================

This module implements vector-strength and integral-invariant tracking for
hybrid VLM-VPM simulations.

Key concept: bound/wake vortex-strength closure
------------------------------------------------
For a discretized inviscid vortex system, the oriented bound and wake
filament vortex_strength close:

    d(Gamma_total) / dt = 0

where Gamma = scalar circulation times the oriented filament segment and has
units L^3/T.
At shedding, scalar VLM circulation [L^2/T] is converted to VPM vector strength
Gamma_p [L^3/T] by the oriented filament length; the two quantities are never
added directly.

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

    net_vortex_strength: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Bound plus wake vector strength [m³/s]."""

    vortex_strength_closure_error_percent: float = 0.0
    """Relative drift in bound/wake vector-strength closure [%]."""

    impulse_wake: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Wake linear impulse [kg m/s]."""

    impulse_bound: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Finite bound-field linear impulse [kg m/s]."""

    impulse_total: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Tracked total linear impulse [kg m/s]."""

    kutta_joukowski_force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Force from Kutta-Joukowski on panels [N]."""

    unsteady_pressure_force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Surface potential-jump time derivative contribution [N]."""

    total_kinetic_energy: float = 0.0
    """Total kinetic energy [J]."""

    viscous_kinetic_energy_rate: float = 0.0
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
        """Create an in-memory conservation history recorder.

        Parameters
        ----------
        density : float, default=1.225
            Fluid density in kg/m³. It scales impulse and kinetic-energy
            quantities reconstructed from the VPM fields and is also passed to
            the VLM force calculation.

        Notes
        -----
        The tracker owns the append-only :attr:`history` list but does not
        copy or mutate the solver's particle arrays. Call :meth:`record_state`
        after an accepted, fully refreshed solver step.
        """
        self.density = density
        self.history: list[ConservationState] = []

    def record_state(self, solver: "VPMSolver") -> ConservationState:
        """Record conservation quantities at the current time step."""
        state = ConservationState(time=solver.time)

        state.wake_vortex_strength = solver.net_vortex_strength
        state.impulse_wake = solver.total_linear_impulse * self.density
        state.total_kinetic_energy = solver.total_kinetic_energy * self.density
        state.viscous_kinetic_energy_rate = solver.viscous_kinetic_energy_rate * self.density
        state.n_particles_total = solver.particles.n_particles_total

        if hasattr(solver, "_particles_removed_this_step"):
            state.n_particles_removed = solver._particles_removed_this_step
            state.vortex_strength_removed = solver._vortex_strength_removed_this_step

        if solver.vlm_solver is not None and solver.vlm_solver._solved:
            state.bound_vortex_strength = solver.vlm_solver.compute_total_bound_vortex_strength()
            state.impulse_bound = solver.vlm_solver.compute_bound_linear_impulse() * self.density
            forces = solver.vlm_solver.compute_forces(density=self.density)
            state.unsteady_pressure_force = np.array(
                [forces[f"unsteady_force_{axis}"] for axis in "xyz"]
            )
            state.kutta_joukowski_force = (
                np.array([forces["force_x"], forces["force_y"], forces["force_z"]])
                - state.unsteady_pressure_force
            )

        state.net_vortex_strength = state.bound_vortex_strength + state.wake_vortex_strength
        state.impulse_total = state.impulse_wake + state.impulse_bound

        closure_scale = max(
            np.linalg.norm(state.bound_vortex_strength),
            np.linalg.norm(state.wake_vortex_strength),
        )
        if closure_scale > np.finfo(float).tiny:
            state.vortex_strength_closure_error_percent = (
                100.0 * np.linalg.norm(state.net_vortex_strength) / closure_scale
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
                    "bound_vortex_strength_magnitude",
                    "wake_vortex_strength_magnitude",
                    "net_vortex_strength_magnitude",
                    "vortex_strength_closure_error_percent",
                    "linear_impulse_x",
                    "linear_impulse_y",
                    "linear_impulse_z",
                    "kutta_joukowski_force_x",
                    "kutta_joukowski_force_y",
                    "kutta_joukowski_force_z",
                    "total_kinetic_energy",
                    "viscous_kinetic_energy_rate",
                    "n_particles_total",
                    "bound_linear_impulse_x",
                    "bound_linear_impulse_y",
                    "bound_linear_impulse_z",
                    "total_linear_impulse_x",
                    "total_linear_impulse_y",
                    "total_linear_impulse_z",
                    "unsteady_pressure_force_x",
                    "unsteady_pressure_force_y",
                    "unsteady_pressure_force_z",
                ]
            )

            for state in self.history:
                writer.writerow(
                    [
                        state.time,
                        np.linalg.norm(state.bound_vortex_strength),
                        np.linalg.norm(state.wake_vortex_strength),
                        np.linalg.norm(state.net_vortex_strength),
                        state.vortex_strength_closure_error_percent,
                        *state.impulse_wake,
                        *state.kutta_joukowski_force,
                        state.total_kinetic_energy,
                        state.viscous_kinetic_energy_rate,
                        state.n_particles_total,
                        *state.impulse_bound,
                        *state.impulse_total,
                        *state.unsteady_pressure_force,
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
        initial = self.history[0]

        print("\n" + "=" * 70)
        print("CONSERVATION DIAGNOSTICS SUMMARY")
        print("=" * 70)
        print(f"\nSimulation time: {final.time:.4f} s")
        print(f"Time steps recorded: {len(self.history)}")

        print("\n--- Bound/Wake Vortex-Strength Closure ---")
        print(f"Initial total strength: {np.linalg.norm(initial.net_vortex_strength):.6e} m^3/s")
        print(f"Final total strength:   {np.linalg.norm(final.net_vortex_strength):.6e} m^3/s")
        print(f"Closure error:          {final.vortex_strength_closure_error_percent:.3f}%")

        print("\n--- Surface Force ---")
        print(
            f"Kutta-Joukowski force:     [{final.kutta_joukowski_force[0]:+.4e}, "
            f"{final.kutta_joukowski_force[1]:+.4e}, {final.kutta_joukowski_force[2]:+.4e}] N"
        )

        print("\n--- Energy Budget ---")
        print(f"Kinetic energy:            {final.total_kinetic_energy:.6e} J")
        print(f"Dissipation rate:          {final.viscous_kinetic_energy_rate:.6e} W")

        print("\n--- Particle Statistics ---")
        print(f"Total particles:           {final.n_particles_total}")

        print("=" * 70 + "\n")
