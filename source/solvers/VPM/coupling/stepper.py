"""Advance the coupled panel/VLM boundary-element solvers during a VPM step.

:class:`CouplingStepper` owns the *orchestration* of VLM and panel coupling:
it calls the coupled solver's advance method with the current VPM state and
appends any wake particles the solver sheds.  The solver implementations live
in ``boundary_elements``; the coupling stepper never re-implements them.

The stepper holds a back-reference to the solver and routes attribute reads
through it (``__getattr__``), matching the delegation pattern of
:class:`~source.solvers.VPM.core.evolution.EvolutionStepper`.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..core.solver import Solver


class CouplingStepper:
    """Advance the coupled boundary-element solvers and append shed particles."""

    def __init__(self, solver: Solver) -> None:
        self.solver = solver

    def __getattr__(self, name: str):
        return getattr(self.solver, name)

    def advance_panel(self):
        """Advance panel–VPM coupling and append any shed particles."""
        new_particles = self.panel_solver.advance(
            particles=self.particles,
            physics=self.physics,
            V_inf=self.freestream_velocity,
            dt=self.time_step_size,
            time=self.flow_time,
            step=self.time_step,
            logging_frequency=self.logging_frequency,
            density=getattr(self.config, "density", 1.0),
        )
        if new_particles is not None:
            n = len(new_particles["points"])
            if n > 0:
                visc_cfg = getattr(self.config, "viscous", None)
                nu = getattr(visc_cfg, "viscosity", None) if visc_cfg is not None else None
                if nu is None or nu <= 0:
                    nu = 1e-2
                viscosity = np.full(n, nu, dtype=self.np_dtype)

                pos = new_particles["points"].astype(self.np_dtype)
                strength = new_particles["strengths"].astype(self.np_dtype)
                rad = new_particles["radii"].astype(self.np_dtype)
                vol = new_particles["volumes"].astype(self.np_dtype)

                self.add_vortex_particles(
                    position=pos,
                    velocity=np.zeros((n, 3), dtype=self.np_dtype),
                    circulation=strength,
                    radius=rad,
                    volume=vol,
                    viscosity=viscosity,
                )

    def advance_vlm(self, dt: float) -> None:
        """Advance VLM–VPM coupling and append shed wake particles."""
        if self.vlm_solver is None:
            return

        wake_particles = self.vlm_solver.advance_coupled(
            particles=self.particles,
            physics=self.physics,
            config=self.config,
            dt=dt,
            time_step=self.time_step,
            time=self.flow_time,
        )

        if wake_particles is not None:
            self.add_vortex_particles(**wake_particles)
