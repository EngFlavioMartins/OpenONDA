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
    from ..core.solver import VPMSolver


class CouplingStepper:
    """Advance the coupled boundary-element solvers and append shed particles."""

    def __init__(self, solver: VPMSolver) -> None:
        self.solver = solver

    def __getattr__(self, name: str):
        return getattr(self.solver, name)

    def advance_panel(self):
        """Advance panel–VPM coupling and append any shed particles."""
        new_particles = self.panel_solver.advance(
            particles=self.particles,
            physics=self.physics,
            freestream_velocity=self.freestream_velocity,
            time_step_size=self.time_step_size,
            time=self.time,
            step=self.step,
            logging_interval_steps=self.logging_interval_steps,
        )
        if new_particles is not None:
            n = len(new_particles["points"])
            if n > 0:
                visc_cfg = getattr(self.setup, "viscous", None)
                if visc_cfg is None or visc_cfg.scheme == "NONE":
                    nu_shed = 0.0
                else:
                    configured_nu = visc_cfg.kinematic_viscosity
                    if configured_nu is None:
                        raise ValueError(
                            "ViscousConfig scheme "
                            f"{visc_cfg.scheme!r} requires kinematic_viscosity so shed "
                            "wake particles receive the molecular value; no fallback "
                            "is applied"
                        )
                    nu_shed = float(configured_nu)
                viscosity = np.full(n, nu_shed, dtype=self.np_dtype)

                pos = new_particles["points"].astype(self.np_dtype)
                strength = new_particles["strengths"].astype(self.np_dtype)
                rad = new_particles["radii"].astype(self.np_dtype)
                vol = new_particles["volumes"].astype(self.np_dtype)

                self.add_vortex_particles(
                    position=pos,
                    velocity=np.zeros((n, 3), dtype=self.np_dtype),
                    vortex_strength=strength,
                    core_radius=rad,
                    volume=vol,
                    kinematic_viscosity=viscosity,
                )

    def advance_vlm(self, time_step_size: float) -> None:
        """Advance VLM–VPM coupling and append shed wake particles."""
        if self.vlm_solver is None:
            return

        wake_particles = self.vlm_solver.advance_coupled(
            particles=self.particles,
            physics=self.physics,
            config=self.setup,
            time_step_size=time_step_size,
            step=self.step,
            time=self.time,
        )

        if wake_particles is not None:
            self.add_vortex_particles(**wake_particles)
