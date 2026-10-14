"""A small CPU vortex-particle case for restart and backend checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from openonda.vpm import (
    SSPRK3,
    DirectInduction,
    LoggingConfig,
    Numerics,
    Output,
    StabilizationConfig,
    ViscousConfig,
    VPMCase,
    VPMSolver,
)


def small_case(
    case_directory: Path,
    *,
    time_step_size: float = 0.01,
    max_n_particles: int = 64,
    integrator=None,
    viscous: ViscousConfig | None = None,
    induction=None,
    random_seed: int = 42,
    stabilization: StabilizationConfig | None = None,
) -> VPMCase:
    return VPMCase(
        name="small_case",
        directory=case_directory,
        output=Output(precision="f16"),
        logging=LoggingConfig(console=False),
        numerics=Numerics(
            time_step_size=time_step_size,
            compute_device="cpu",
            max_n_particles=max_n_particles,
            domain_bounds=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
            integrator=SSPRK3() if integrator is None else integrator,
            viscous=(
                ViscousConfig.cs(kinematic_viscosity=0.01, particle_spacing=0.2)
                if viscous is None
                else viscous
            ),
            induction=DirectInduction() if induction is None else induction,
            random_seed=random_seed,
            stabilization=(
                StabilizationConfig.disabled() if stabilization is None else stabilization
            ),
        ),
    )


def add_counter_rotating_pair(solver: VPMSolver) -> None:
    solver.add_vortex_particles(
        position=np.array([[-0.25, 0.0, 0.0], [0.25, 0.0, 0.0]]),
        velocity=np.zeros((2, 3)),
        vortex_strength=np.array([[0.0, 0.1, 0.0], [0.0, -0.1, 0.0]]),
        core_radius=np.full(2, 0.2),
        particle_volume=np.full(2, 0.2**3),
        kinematic_viscosity=np.full(2, 0.01),
    )
