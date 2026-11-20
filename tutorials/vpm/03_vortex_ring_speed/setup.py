#!/usr/bin/env python3
"""One Gaussian vortex ring in still fluid: translation speed against Saffman."""

from pathlib import Path

import numpy as np
import openonda.vpm as vpm

from assets.initial_conditions import ToroidalDistribution, VortexRing
from assets.ring_diagnostics import RingDiagnosticsSampler


RING_RADIUS = 1.0  # [m]
RING_CIRCULATION = np.pi  # [m²/s]
REYNOLDS_NUMBER = 3000.0  # Gamma_0 / nu
KINEMATIC_VISCOSITY = RING_CIRCULATION / REYNOLDS_NUMBER  # [m²/s]
CORE_RADIUS = 0.1  # [m], Gaussian e-folding radius a_0
PARTICLE_SPACING = 0.05  # [m]
PARTICLE_CORE_RADIUS = 1.5 * PARTICLE_SPACING  # sigma/h = 1.5: blobs overlap
REMESH_CORE_RADIUS = 2.2 * PARTICLE_SPACING  # core spreading grows sigma; remesh above this
TOROIDAL_TAIL_FRACTION = 1.0e-4
TIME_STEP_SIZE = 0.00375  # [s]
END_TIME = 3.6  # [s], t Gamma_0 / R_0^2 = 11.3
TUTORIAL_DIR = Path(__file__).parent


def build_case():
    tube = np.sqrt(CORE_RADIUS**2 - PARTICLE_CORE_RADIUS**2) * np.sqrt(
        -np.log(TOROIDAL_TAIL_FRACTION)
    )
    ring = VortexRing(
        radius=RING_RADIUS,
        circulation=RING_CIRCULATION,
        vortex_core_radius=CORE_RADIUS,
        kinematic_viscosity=KINEMATIC_VISCOSITY,
        distribution=ToroidalDistribution(
            ring_radius=RING_RADIUS,
            tube_radius=tube,
            spacing=PARTICLE_SPACING,
            core_radius_ratio=PARTICLE_CORE_RADIUS / PARTICLE_SPACING,
        ),
    )
    # Gaussian redistribution restores the lattice once core spreading has grown
    # the cores to REMESH_CORE_RADIUS; it conserves strength and impulse.
    remesh = vpm.StabilizationConfig(
        regularization_interval_steps=20,
        regularization_grid_spacing=PARTICLE_SPACING,
        regularization_core_radius=PARTICLE_CORE_RADIUS,
        regularization_core_radius_trigger=REMESH_CORE_RADIUS,
        regularization_tail_budget=0.003,
        regularization_transfer_only=True,
        regularization_divergence_trigger=None,
        regularization_misalignment_trigger=None,
        regularization_total_kinetic_energy_dissipation_limit=0.01,
        regularization_total_enstrophy_dissipation_limit=0.01,
    )
    return vpm.VPMCase(
        name="single_ring",
        directory=TUTORIAL_DIR,
        variant="single_ring",
        initial_conditions=(ring,),
        numerics=vpm.Numerics(
            time_step_size=TIME_STEP_SIZE,
            induction=vpm.DirectInduction(),
            viscous=vpm.ViscousConfig.cs(
                particle_spacing=PARTICLE_SPACING,
                kinematic_viscosity=KINEMATIC_VISCOSITY,
                core_radius_ratio=PARTICLE_CORE_RADIUS / PARTICLE_SPACING,
            ),
            turbulence=vpm.TurbulenceConfig.dns(),
            stabilization=remesh,
            random_seed=42,
            state_limits=vpm.ParticleStateLimits(
                lagrangian_cfl=vpm.LagrangianCFLLimit(maximum=1.0),
                divergence=vpm.DivergenceLimit(maximum=0.12),
            ),
        ),
        run=vpm.RunPlan(end_time=END_TIME, state_limit_action="stop"),
        samplers=vpm.Samplers(
            samples=(
                vpm.FlowIntegralsSampler(schedule=vpm.EverySteps(10)),
                RingDiagnosticsSampler(schedule=vpm.EverySteps(10)),
            ),
        ),
        backup=vpm.Backup(schedule=vpm.EverySteps(200)),
    )


if __name__ == "__main__":
    vpm.VPMSolver(build_case()).run()
