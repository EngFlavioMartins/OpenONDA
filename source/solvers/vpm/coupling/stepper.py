"""Advance the coupled panel/VLM boundary-element solvers during a VPM step.

:class:`CouplingStepper` owns the *orchestration* of VLM and panel coupling:
it calls the coupled solver's advance method with the current VPM state and
appends any wake particles the solver sheds.  The solver implementations live
in ``boundary_elements``; the coupling stepper never re-implements them.

The stepper holds a back-reference to the solver and routes attribute reads
through it (``__getattr__``), matching the delegation pattern of
:class:`~source.solvers.vpm.core.evolution.EvolutionStepper`.

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
        if getattr(self.panel_solver, "coupling_scope", "full") == "vpm_boundary_condition":
            return
        new_particles = self.panel_solver.advance(
            particles=self.particles,
            physics=self.physics,
            freestream_velocity=self.freestream_velocity,
            time_step_size=self.time_step_size,
            time=self.time,
            step=self.step,
        )
        if new_particles is not None:
            n = len(new_particles["vertex_position"])
            if n > 0:
                visc_cfg = getattr(self.setup, "viscous", None)
                if visc_cfg is None or visc_cfg.scheme == "NONE":
                    shed_kinematic_viscosity = 0.0
                else:
                    configured_kinematic_viscosity = visc_cfg.kinematic_viscosity
                    if configured_kinematic_viscosity is None:
                        raise ValueError(
                            "ViscousConfig scheme "
                            f"{visc_cfg.scheme!r} requires kinematic_viscosity so shed "
                            "wake particles receive the molecular value; no fallback "
                            "is applied"
                        )
                    shed_kinematic_viscosity = float(configured_kinematic_viscosity)
                kinematic_viscosity = np.full(n, shed_kinematic_viscosity, dtype=self.np_dtype)

                position = new_particles["vertex_position"].astype(self.np_dtype)
                vortex_strength = new_particles["vortex_strength"].astype(self.np_dtype)
                core_radius = new_particles["core_radius"].astype(self.np_dtype)
                particle_volume = new_particles["particle_volume"].astype(self.np_dtype)

                self.add_vortex_particles(
                    position=position,
                    velocity=np.zeros((n, 3), dtype=self.np_dtype),
                    vortex_strength=vortex_strength,
                    core_radius=core_radius,
                    particle_volume=particle_volume,
                    kinematic_viscosity=kinematic_viscosity,
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
