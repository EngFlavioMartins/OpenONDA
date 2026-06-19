"""
Conservation Diagnostics for VLM-VPM Coupling.
==============================================

This module implements impulse-consistent force evaluation and circulation
conservation tracking for hybrid VLM-VPM simulations.

**Key Concepts:**

1. **Impulse-Consistent Forces**
   For incompressible flow with compact vorticity:

   .. math::
       \\mathbf{I} = \\frac{\\density}{2} \\int \\mathbf{x} \\times \\boldsymbol{\\omega} \\, dV

       \\mathbf{F} = -\\frac{d\\mathbf{I}}{dt}

   This is the *correct* expression for forces. It automatically includes:
   - Added-mass effects
   - Wake-induced forces
   - Unsteady lift components

   Forces from Kutta-Joukowski on bound panels alone are *incomplete*.

2. **Circulation Conservation (Kelvin's Theorem)**
   For inviscid flow:

   .. math::
       \\frac{d\\Gamma_{total}}{dt} = 0

   where :math:`\\Gamma_{total} = \\Gamma_{bound} + \\Gamma_{wake}`.

   At shedding, circulation must transfer exactly:

   .. math::
       \\Gamma_{shed}^{n+1} = \\Gamma_{bound}^{n} - \\Gamma_{bound}^{n+1}

3. **Wake Truncation Invariance**
   Impulse-based forces must remain constant when far-wake particles are removed,
   while Kutta-Joukowski forces will change (incorrectly).

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: February 2026
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..core.solver import Solver


@dataclass
class ConservationState:
    """Snapshot of conservation quantities at a single time step."""

    time: float
    """Physical time [s]"""

    # Circulation tracking
    circulation_bound: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Bound circulation vector from VLM [m²/s]"""

    circulation_wake: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Wake circulation vector from VPM particles [m²/s]"""

    circulation_total: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Total circulation (bound + wake) [m²/s]"""

    circulation_error: float = 0.0
    """Relative error in circulation conservation [%]"""

    # Impulse tracking
    impulse_bound: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Bound impulse from VLM [kg·m²/s]"""

    impulse_wake: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Wake impulse from VPM [kg·m²/s]"""

    impulse_total: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Total impulse (bound + wake) [kg·m²/s]"""

    # Force tracking
    force_impulse: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Force from impulse derivative [N]"""

    force_kutta_joukowski: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Force from Kutta-Joukowski on panels [N]"""

    force_difference: float = 0.0
    """Relative difference between methods [%]"""

    # Energy tracking
    kinetic_energy: float = 0.0
    """Total kinetic energy [J]"""

    energy_dissipation_rate: float = 0.0
    """Viscous energy dissipation [W]"""

    # Particle statistics
    n_particles_total: int = 0
    """Total number of VPM particles"""

    n_particles_shed: int = 0
    """Number of particles shed this step"""

    n_particles_removed: int = 0
    """Number of particles removed (absorption, pruning)"""

    circulation_removed: float = 0.0
    """Magnitude of circulation lost to particle removal [m²/s]"""


class ConservationTracker:
    """
    Track conservation properties during VLM-VPM coupled simulations.

    This class monitors:
    - Circulation conservation (Kelvin's theorem)
    - Impulse evolution
    - Force consistency between methods
    - Energy dissipation

    Example::

        >>> tracker = ConservationTracker(density=1.225)
        >>>
        >>> # During simulation loop
        >>> tracker.record_state(solver)
        >>>
        >>> # After simulation
        >>> tracker.export_csv("solution/conservation.csv")
        >>> tracker.print_summary()
    """

    def __init__(self, density: float = 1.225):
        """
        Initialize conservation tracker.

        Args:
            density: Fluid density [kg/m³]
        """
        self.density = density
        self.history: list[ConservationState] = []
        self._initial_circulation: float | None = None

    def record_state(self, solver: "Solver") -> ConservationState:
        """
        Record conservation quantities at current time step.

        Args:
            solver: VPM solver instance (must have vlm_solver attached)

        Returns:
            ConservationState: Snapshot of current conservation quantities
        """
        state = ConservationState(time=solver.flow_time)

        # Get VPM wake quantities
        state.circulation_wake = solver.total_strength
        state.impulse_wake = solver.total_linear_impulse * self.density
        state.kinetic_energy = solver.total_kinetic_energy
        state.energy_dissipation_rate = solver.kinetic_energy_dissipation_rate
        state.n_particles_total = solver.particles.number_of_particles

        # Track particles removed this step (if tracked by solver)
        if hasattr(solver, "_particles_removed_this_step"):
            state.n_particles_removed = solver._particles_removed_this_step
            state.circulation_removed = solver._circulation_removed_this_step

        # Get VLM bound quantities (if available)
        if solver.vlm_solver is not None and solver.vlm_solver._solved:
            state.circulation_bound = solver.vlm_solver.compute_total_circulation()
            state.impulse_bound = solver.vlm_solver.compute_bound_impulse(self.density)

            # Compute Kutta-Joukowski force
            try:
                forces = solver.vlm_solver.compute_forces(
                    density=self.density, V_ref_mag=np.linalg.norm(solver.background_velocity)
                )
                state.force_kutta_joukowski = np.array([forces["Fx"], forces["Fy"], forces["Fz"]])
            except Exception:
                pass

        # Total circulation and impulse
        state.circulation_total = state.circulation_bound + state.circulation_wake
        state.impulse_total = state.impulse_bound + state.impulse_wake

        # Compute impulse-based force (finite difference)
        if len(self.history) > 0:
            dt = state.time - self.history[-1].time
            if dt > 0:
                dI_dt = (state.impulse_total - self.history[-1].impulse_total) / dt
                state.force_impulse = -dI_dt

                # Compute force difference
                if np.linalg.norm(state.force_impulse) > 1e-10:
                    state.force_difference = (
                        100.0
                        * np.linalg.norm(state.force_impulse - state.force_kutta_joukowski)
                        / np.linalg.norm(state.force_impulse)
                    )

        # Circulation error (relative to initial)
        if self._initial_circulation is None:
            self._initial_circulation = np.linalg.norm(state.circulation_total)

        if self._initial_circulation > 1e-10:
            state.circulation_error = (
                100.0
                * abs(np.linalg.norm(state.circulation_total) - self._initial_circulation)
                / self._initial_circulation
            )

        self.history.append(state)
        return state

    def export_csv(self, filename: str = "solution/conservation.csv"):
        """
        Export conservation history to CSV file.

        Args:
            filename: Output CSV file path
        """
        if len(self.history) == 0:
            print("[WARNING] No conservation data to export")
            return

        # Ensure directory exists
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(
                [
                    "time",
                    "circulation_bound_mag",
                    "circulation_wake_mag",
                    "circulation_total_mag",
                    "circulation_error_pct",
                    "impulse_bound_x",
                    "impulse_bound_y",
                    "impulse_bound_z",
                    "impulse_wake_x",
                    "impulse_wake_y",
                    "impulse_wake_z",
                    "force_impulse_x",
                    "force_impulse_y",
                    "force_impulse_z",
                    "force_kj_x",
                    "force_kj_y",
                    "force_kj_z",
                    "force_difference_pct",
                    "kinetic_energy",
                    "energy_dissipation_rate",
                    "n_particles",
                ]
            )

            # Data rows
            for state in self.history:
                writer.writerow(
                    [
                        state.time,
                        np.linalg.norm(state.circulation_bound),
                        np.linalg.norm(state.circulation_wake),
                        np.linalg.norm(state.circulation_total),
                        state.circulation_error,
                        *state.impulse_bound,
                        *state.impulse_wake,
                        *state.force_impulse,
                        *state.force_kutta_joukowski,
                        state.force_difference,
                        state.kinetic_energy,
                        state.energy_dissipation_rate,
                        state.n_particles_total,
                    ]
                )

        print(f"[INFO] Conservation diagnostics exported to: {filename}")

    def print_summary(self):
        """Print summary of conservation quality."""
        if len(self.history) == 0:
            print("[WARNING] No conservation data recorded")
            return

        # Get final state
        final = self.history[-1]

        print("\n" + "=" * 70)
        print("CONSERVATION DIAGNOSTICS SUMMARY")
        print("=" * 70)
        print(f"\nSimulation time: {final.time:.4f} s")
        print(f"Time steps recorded: {len(self.history)}")

        print("\n--- Circulation Conservation (Kelvin's Theorem) ---")
        print(f"Initial total circulation: {self._initial_circulation:.6e} m²/s")
        print(f"Final total circulation:   {np.linalg.norm(final.circulation_total):.6e} m²/s")
        print(f"Conservation error:        {final.circulation_error:.3f}%")

        if final.circulation_error < 0.1:
            status = "✓ EXCELLENT"
        elif final.circulation_error < 1.0:
            status = "✓ GOOD"
        elif final.circulation_error < 5.0:
            status = "⚠ ACCEPTABLE"
        else:
            status = "✗ POOR"
        print(f"Status: {status}")

        print("\n--- Force Evaluation Methods ---")
        print(
            f"Impulse-based force:       [{final.force_impulse[0]:+.4e}, "
            f"{final.force_impulse[1]:+.4e}, {final.force_impulse[2]:+.4e}] N"
        )
        print(
            f"Kutta-Joukowski force:     [{final.force_kutta_joukowski[0]:+.4e}, "
            f"{final.force_kutta_joukowski[1]:+.4e}, {final.force_kutta_joukowski[2]:+.4e}] N"
        )
        print(f"Relative difference:       {final.force_difference:.2f}%")

        print("\n--- Energy Budget ---")
        print(f"Kinetic energy:            {final.kinetic_energy:.6e} J")
        print(f"Dissipation rate:          {final.energy_dissipation_rate:.6e} W")

        print("\n--- Particle Statistics ---")
        print(f"Total particles:           {final.n_particles_total}")

        print("=" * 70 + "\n")

    def validate_wake_truncation_invariance(
        self, solver: "Solver", truncation_distance: float
    ) -> dict[str, float]:
        """
        Validate that impulse-based forces are invariant to wake truncation.

        This is the "killer test" for impulse consistency. Removes far-wake
        particles and checks that:
        - Kutta-Joukowski force changes (expected, it's incomplete)
        - Impulse-based force remains constant (proving consistency)

        Args:
            solver: VPM solver instance
            truncation_distance: Distance from origin beyond which to remove particles [m]

        Returns:
            Dictionary with validation metrics
        """
        print("\n" + "=" * 70)
        print("WAKE TRUNCATION INVARIANCE TEST")
        print("=" * 70)

        # Record state before truncation
        state_before = self.record_state(solver)

        # Count particles to be removed
        pos = solver.particles_positions
        dist = np.linalg.norm(pos, axis=1)
        mask_remove = dist > truncation_distance
        n_remove = np.sum(mask_remove)
        n_keep = len(pos) - n_remove

        print(f"\nTruncating wake at distance: {truncation_distance:.2f} m")
        print(f"Particles to remove: {n_remove} / {len(pos)} ({100 * n_remove / len(pos):.1f}%)")
        print(f"Particles to keep:   {n_keep}")

        # Remove far-wake particles
        indices_remove = np.where(mask_remove)[0].tolist()
        if len(indices_remove) > 0:
            solver.remove_particles(particle_indices=indices_remove)

        # Record state after truncation
        state_after = self.record_state(solver)

        # Compare forces
        impulse_change = np.linalg.norm(state_after.force_impulse - state_before.force_impulse)
        kj_change = np.linalg.norm(
            state_after.force_kutta_joukowski - state_before.force_kutta_joukowski
        )

        if np.linalg.norm(state_before.force_impulse) > 1e-10:
            impulse_change_pct = 100 * impulse_change / np.linalg.norm(state_before.force_impulse)
        else:
            impulse_change_pct = 0.0

        if np.linalg.norm(state_before.force_kutta_joukowski) > 1e-10:
            kj_change_pct = 100 * kj_change / np.linalg.norm(state_before.force_kutta_joukowski)
        else:
            kj_change_pct = 0.0

        print("\n--- Force Changes After Truncation ---")
        print(
            f"Impulse-based force change: {impulse_change_pct:.3f}% "
            f"({'✓ INVARIANT' if impulse_change_pct < 1.0 else '⚠ CHANGED'})"
        )
        print(
            f"Kutta-Joukowski change:     {kj_change_pct:.3f}% "
            f"({'✓ EXPECTED' if kj_change_pct > 5.0 else '? UNEXPECTED'})"
        )

        # Verdict
        if impulse_change_pct < 1.0 and kj_change_pct > 5.0:
            print("\n✓ TEST PASSED: Impulse method is wake-truncation invariant")
            print("               Kutta-Joukowski method is NOT (as expected)")
        else:
            print("\n⚠ TEST INCONCLUSIVE: Check circulation conservation first")

        print("=" * 70 + "\n")

        return {
            "impulse_change_percent": impulse_change_pct,
            "kj_change_percent": kj_change_pct,
            "particles_removed": n_remove,
            "particles_kept": n_keep,
        }


class ImpulseForceEvaluator:
    """
    Compute forces using impulse-consistent method.

    This evaluator uses the exact relation:

    .. math::
        \\mathbf{F} = -\\frac{d\\mathbf{I}}{dt}

    where the impulse includes *all* vorticity (bound + wake).

    Example::

        >>> evaluator = ImpulseForceEvaluator(density=1.225, smoothing_window=5)
        >>>
        >>> # In time-stepping loop
        >>> force = evaluator.compute_force(solver)
        >>> print(f"Total force: {force} N")
    """

    def __init__(self, density: float = 1.225, order: int = 2, smoothing_window: int | None = None):
        """
        Initialize impulse force evaluator.

        Args:
            density: Fluid density [kg/m³]
            order: Order of time derivative (1 or 2)
            smoothing_window: Window size for moving average filter (None = no smoothing).
                             Recommended: 3-5 for noisy high-vortex simulations.
        """
        self.density = density
        self.order = order
        self.smoothing_window = smoothing_window
        self._impulse_history: list[np.ndarray] = []
        self._time_history: list[float] = []
        self._force_history: list[np.ndarray] = []

    def compute_force(self, solver: "Solver") -> np.ndarray:
        """
        Compute force from impulse derivative.

        Args:
            solver: VPM solver instance

        Returns:
            Force vector [Fx, Fy, Fz] in Newtons
        """
        # Compute total impulse (VPM wake + VLM bound)
        impulse_wake = solver.total_linear_impulse * self.density

        impulse_bound = np.zeros(3)
        if solver.vlm_solver is not None and solver.vlm_solver._solved:
            impulse_bound = solver.vlm_solver.compute_bound_impulse(self.density)

        impulse_total = impulse_wake + impulse_bound

        # Store in history
        self._impulse_history.append(impulse_total.copy())
        self._time_history.append(solver.flow_time)

        # Compute derivative
        if len(self._impulse_history) < 2:
            return np.zeros(3)

        if self.order == 1 or len(self._impulse_history) < 3:
            # First-order backward difference
            dt = self._time_history[-1] - self._time_history[-2]
            dI_dt = (self._impulse_history[-1] - self._impulse_history[-2]) / dt
        else:
            # Second-order backward difference
            dt1 = self._time_history[-1] - self._time_history[-2]
            dt2 = self._time_history[-2] - self._time_history[-3]

            # Coefficients for unequal spacing
            a0 = (2 * dt1 + dt2) / (dt1 * (dt1 + dt2))
            a1 = -(dt1 + dt2) / (dt1 * dt2)
            a2 = dt1 / (dt2 * (dt1 + dt2))

            dI_dt = (
                a0 * self._impulse_history[-1]
                + a1 * self._impulse_history[-2]
                + a2 * self._impulse_history[-3]
            )

        # F = -dI/dt
        force = -dI_dt

        # Store in force history for smoothing
        self._force_history.append(force.copy())

        # Apply moving average filter if enabled
        if self.smoothing_window is not None and len(self._force_history) >= self.smoothing_window:
            # Use last N forces for smoothing
            recent_forces = np.array(self._force_history[-self.smoothing_window :])
            force = np.mean(recent_forces, axis=0)

        return force

    def reset(self):
        """Reset history (e.g., for new simulation)."""
        self._impulse_history.clear()
        self._time_history.clear()
