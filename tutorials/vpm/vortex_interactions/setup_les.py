#!/usr/bin/env python3
"""Compare the GBD/LES leapfrogging baseline with three stabilizers.

All experiments share the same unperturbed rings and numerical resolution.
Edit the physical and numerical inputs below; baseline is run first.
"""

import argparse
from pathlib import Path

import numpy as np

import openonda.vpm as vpm
from openonda.tutorial_runner import case_package

__package__ = case_package(Path(__file__).parent)
from .setup import core_section_samplers

# Ring properties
RING_RADIUS = 1.0  # [m]
RING_CIRCULATION = np.pi  # [m^2/s]
REYNOLDS_NUMBER = 3000.0
CORE_RADIUS = 0.1  # Gaussian core radius [m]
RING_SEPARATION = 1.0  # axial distance [m]
KINEMATIC_VISCOSITY = RING_CIRCULATION / REYNOLDS_NUMBER  # [m^2/s]

# Particle resolution and numerical model
PARTICLE_SPACING = 0.04  # [m]
CORE_RADIUS_RATIO = 1.0
TAIL_FRACTION = 1.0e-4
SMAGORINSKY_COEFFICIENT = 0.20
REALIGNMENT_FREQUENCY = 0.384684814725  # [1/s]
STRETCHING_VISCOSITY_COEFFICIENT = 0.5
SPLITTING_STRENGTH_FACTOR = 2.0  # relative to the initial peak particle strength
MAX_N_PARTICLES = 1_000_000

# Time integration and output
TIME_STEP_SIZE = 0.0075  # [s]
N_STEPS = 1200  # t=9 s; allows both cores to reach x/R0=7 at the estimated slow speed
FIELD_INTERVAL_TIME = 0.15  # [s]
BACKUP_INTERVAL_STEPS = 100
WALL_MINUTES = 150
VARIANTS = ("baseline", "stretching_viscosity", "p_moments", "splitting", "halfdt")


def stabilization(variant, rings, time_step_size):
    if variant == "splitting":
        peak = max(np.linalg.norm(ring.build().vortex_strength, axis=1).max() for ring in rings)
        return vpm.StabilizationConfig(
            filament_refinement=vpm.FilamentRefinementConfig.adaptive(
                interval_steps=5,
                max_vortex_strength_factor=2.0,
                max_absolute_vortex_strength=SPLITTING_STRENGTH_FACTOR * peak,
                offset_fraction=0.25,
                max_n_particles=MAX_N_PARTICLES,
            )
        )
    return {
        "baseline": vpm.StabilizationConfig.disabled(),
        "halfdt": vpm.StabilizationConfig.disabled(),
        "stretching_viscosity": vpm.StabilizationConfig.stretching_viscosity(
            coefficient=STRETCHING_VISCOSITY_COEFFICIENT
        ),
        "p_moments": vpm.StabilizationConfig(
            pedrizzetti_relaxation_factor=REALIGNMENT_FREQUENCY * time_step_size,
            pedrizzetti_relaxation_preserve_vortex_strength=False,
            pedrizzetti_relaxation_preserve_moments=True,
        ),
    }[variant]


def build_case(variant):
    refinement = 2 if variant == "halfdt" else 1
    time_step_size = TIME_STEP_SIZE / refinement
    name = f"les_{variant}"
    particle_core_radius = CORE_RADIUS_RATIO * PARTICLE_SPACING
    tube_radius = np.sqrt(CORE_RADIUS**2 - particle_core_radius**2) * np.sqrt(
        -np.log(TAIL_FRACTION)
    )
    rings = tuple(
        vpm.VortexRing(
            centre=(centre_x, 0.0, 0.0),
            radius=RING_RADIUS,
            circulation=RING_CIRCULATION,
            vortex_core_radius=CORE_RADIUS,
            kinematic_viscosity=KINEMATIC_VISCOSITY,
            disturbance=vpm.WidnallDisturbance.single_mode(
                amplitude=0.0, mode=8, direction="radial"
            ),
            core_compensation=vpm.ParticleCoreCompensation(),
            distribution=vpm.ToroidalDistribution(
                centre=(centre_x, 0.0, 0.0),
                ring_radius=RING_RADIUS,
                tube_radius=tube_radius,
                spacing=PARTICLE_SPACING,
                core_radius_ratio=CORE_RADIUS_RATIO,
            ),
            group_id=group,
        )
        for group, centre_x in enumerate((-RING_SEPARATION / 2, RING_SEPARATION / 2))
    )
    return vpm.VPMCase(
        name=name,
        directory=Path(__file__).resolve().parent,
        initial_conditions=rings,
        numerics=vpm.Numerics(
            time_step_size=time_step_size,
            integrator=vpm.SSPRK3(),
            induction=vpm.TreecodeInduction(
                theta=0.5, multipole_order=3, stretching_scheme="TRANSPOSED"
            ),
            viscous=vpm.ViscousConfig.gbd(
                particle_spacing=PARTICLE_SPACING,
                kinematic_viscosity=KINEMATIC_VISCOSITY,
                core_radius_ratio=CORE_RADIUS_RATIO,
                padding=5.0,
                threshold=TAIL_FRACTION,
                max_nodes=MAX_N_PARTICLES,
                remeshing_kernel="LAGRANGE6",
            ),
            turbulence=vpm.TurbulenceConfig.les_smagorinsky(
                smagorinsky_coefficient=SMAGORINSKY_COEFFICIENT
            ),
            stabilization=stabilization(variant, rings, time_step_size),
            domain_bounds=(-2.0, 12.0, -3.0, 3.0, -3.0, 3.0),
            max_n_particles=MAX_N_PARTICLES,
            write_precision="f32",
            random_seed=42,
            verbose=False,
            health_limits=vpm.HealthLimits(
                lagrangian_cfl=vpm.LagrangianCFLLimit(maximum=1.0),
                divergence=vpm.DivergenceLimit(maximum=0.12),
                misalignment=vpm.MisalignmentLimit(maximum_degrees=25.0),
            ),
        ),
        backup=vpm.Backup(
            interval_steps=BACKUP_INTERVAL_STEPS,
            directory=f"solution/{name}",
            log_directory=f"solution/{name}",
        ),
        samplers=vpm.Samplers(
            directory=name,
            samples=(
                vpm.FlowIntegralsSampler(schedule=vpm.EverySteps(10), initial=True),
                vpm.RingDiagnosticsSampler(schedule=vpm.EverySteps(10), initial=True),
                *core_section_samplers(interval=FIELD_INTERVAL_TIME),
                *(
                    vpm.SurfaceSampler(
                        point=[0.0, 0.0, 0.0],
                        normal=[0.0, 1.0, 0.0],
                        bounds=[-2.0, 14.0, -1.8, 1.8],
                        spacing=0.04,
                        file_name="cross_section",
                        include_derivatives=False,
                        schedule=schedule,
                        initial=initial,
                    )
                    for schedule, initial in ((vpm.EveryTime(0.30), True), (vpm.FinalOnly(), False))
                ),
            ),
        ),
        run=vpm.RunPlan(
            steps=N_STEPS * refinement,
            final_backup=True,
            health_limit_action="STOP",
            wall_time_limit_seconds=60.0 * WALL_MINUTES,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    args = parser.parse_args()
    vpm.VPMSolver(build_case(args.variant)).run()
