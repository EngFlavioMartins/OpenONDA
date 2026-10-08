"""VLM coupling within a VPM time step.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..core.solver import VPMSolver


class CouplingStepper:
    """Advance the coupled VLM solver and append shed particles."""

    def __init__(self, solver: VPMSolver) -> None:
        """Keep the VPM solver that owns the particle cloud and clock."""
        self.solver = solver

    def advance_vlm(self, time_step_size: float) -> None:
        """Advance VLM–VPM coupling and append shed wake particles."""
        solver = self.solver
        if solver.vlm_solver is None:
            return

        particles_before = getattr(solver.particles, "n_particles_total", None)
        wake_particles = solver.vlm_solver.advance_coupled(
            particles=solver.particles,
            physics=solver.physics,
            config=solver.setup,
            time_step_size=getattr(solver, "_release_interval", time_step_size),
            step=solver.stepper.step,
            time=solver.stepper.time,
            release_wake=getattr(solver, "_release_wake_particles", True),
        )

        if wake_particles is not None:
            solver.add_vortex_particles(**wake_particles)
        elif particles_before is not None:
            particles_added = solver.particles.n_particles_total - particles_before
            if particles_added > 0 and solver.stabilization.reference_vortex_strength is not None:
                # VLM inserts on the device; read only its shed batch for refinement reference.
                lattice = solver.vlm_solver.lattice
                strength = lattice.wake_vortex_strength.to_numpy()[:particles_added]
                volume = lattice.wake_volume.to_numpy()[:particles_added]
                solver.stabilization.on_add(
                    np.linalg.norm(strength, axis=1), volume, particles_before
                )
