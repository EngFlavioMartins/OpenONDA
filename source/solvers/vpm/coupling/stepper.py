"""Advance the coupled VLM solver during a VPM step.

:class:`CouplingStepper` calls the VLM solver's coupled advance method with the
current VPM state and appends any wake particles it sheds. The implementation
lives in ``boundary_elements.vlm``; the coupling stepper never re-implements it.

The stepper holds a back-reference to the solver but names each capability it
uses.  It is intentionally not a forwarding facade for the full solver API.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.solver import VPMSolver


class CouplingStepper:
    """Advance the coupled VLM solver and append shed particles."""

    def __init__(self, solver: VPMSolver) -> None:
        """Attach the boundary-element coupling orchestrator to a solver.

        Parameters
        ----------
        solver : VPMSolver
            VPM solver that owns the particle arrays, time-step clock, and
            optional VLM solver. The reference is retained; no arrays
            are copied.

        Notes
        -----
        :meth:`advance_vlm` may append shed wake particles to the owning solver.
        This helper is normally constructed by ``VPMSolver`` and is not an
        independent time integrator.
        """
        self.solver = solver

    def advance_vlm(self, time_step_size: float) -> None:
        """Advance VLM–VPM coupling and append shed wake particles."""
        solver = self.solver
        if solver.vlm_solver is None:
            return

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
