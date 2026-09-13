#!/usr/bin/env python3
"""Compare the CS/LES leapfrogging baseline with three stabilizers.

The default scenario is the unperturbed Re_Gamma=3000 kinematic control.
The separate ``seeded_breakdown`` scenario follows the distinct instability
case in Cheng, Lou and Lim (2015): Re_Gamma=3415, epsilon/R0=.05 and an
axial mode-eight displacement.  Its outputs are deliberately named apart
from the kinematic control.  Edit the physical and numerical inputs below;
baseline is run first.
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
BREAKDOWN_REYNOLDS_NUMBER = 3415.0
CORE_RADIUS = 0.1  # Gaussian core radius [m]
RING_SEPARATION = 1.0  # axial distance [m]

# Particle resolution and numerical model
# Selected by bounded h=.08/h=.06 qualification before the official battery.
PARTICLE_SPACING = 0.06  # [m]
CORE_RADIUS_RATIO = 1.0
TAIL_FRACTION = 1.0e-4
SMAGORINSKY_COEFFICIENT = 0.20
REALIGNMENT_FREQUENCY = 0.384684814725  # [1/s]
STRETCHING_VISCOSITY_COEFFICIENT = 0.5
SPLITTING_STRENGTH_FACTOR = 2.0  # relative to the initial peak particle strength
MAX_N_PARTICLES = 1_000_000
# Trial resource guards: 600k active particles leaves margin below the fixed
# 1M device capacity and is well above the 27,518 particles observed in the
# prior split control.  These checks run at accepted-step boundaries.
RESOURCE_MAX_PARTICLES = 600_000
PROCESS_RSS_LIMIT_BYTES = 12 * 1024**3
AVAILABLE_MEMORY_FLOOR_BYTES = 2 * 1024**3
BREAKDOWN_DISTURBANCE_AMPLITUDE = 0.05
BREAKDOWN_DISTURBANCE_MODE = 8
BREAKDOWN_DISTURBANCE_DIRECTION = "axial"

# Time integration and output
TIME_STEP_SIZE = 0.0075  # [s]
N_STEPS = 1200  # t=9 s; allows both cores to reach x/R0=7 at the estimated slow speed
FIELD_INTERVAL_TIME = 0.15  # [s]
BACKUP_INTERVAL_STEPS = 100
WALL_MINUTES = 24
QUALIFICATION_STEPS = 80
QUALIFICATION_WALL_MINUTES = 15
VARIANTS = ("baseline", "stretching_viscosity", "p_moments", "splitting", "halfdt")
SCENARIOS = ("kinematic", "seeded_breakdown")


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
        "p_moments": vpm.StabilizationConfig.pedrizzetti_relaxation(
            factor=REALIGNMENT_FREQUENCY * time_step_size,
            preserve_vortex_strength=False,
            preserve_moments=True,
        ),
    }[variant]


def build_case(
    variant,
    *,
    scenario: str = "kinematic",
    compute_device: str = "AUTO",
    steps: int | None = None,
    wall_minutes: float | None = None,
    qualification: bool = False,
    particle_spacing: float | None = None,
    particle_core_radius: float | None = None,
    smagorinsky_coefficient: float | None = None,
    case_name: str | None = None,
    particle_capacity: int | None = None,
):
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; choose from {SCENARIOS}")
    refinement = 2 if variant == "halfdt" else 1
    time_step_size = TIME_STEP_SIZE / refinement
    spacing = PARTICLE_SPACING if particle_spacing is None else particle_spacing
    if spacing <= 0.0:
        raise ValueError("particle spacing must be positive")
    numerical_core = (
        CORE_RADIUS_RATIO * spacing if particle_core_radius is None else particle_core_radius
    )
    if not 0.0 < numerical_core < CORE_RADIUS:
        raise ValueError(
            f"particle core radius must be between zero and the physical core radius {CORE_RADIUS}"
        )
    core_radius_ratio = numerical_core / spacing
    les_coefficient = (
        SMAGORINSKY_COEFFICIENT if smagorinsky_coefficient is None else smagorinsky_coefficient
    )
    if les_coefficient < 0.0:
        raise ValueError("Smagorinsky coefficient must be non-negative")
    seeded = scenario == "seeded_breakdown"
    reynolds_number = BREAKDOWN_REYNOLDS_NUMBER if seeded else REYNOLDS_NUMBER
    kinematic_viscosity = RING_CIRCULATION / reynolds_number
    disturbance_amplitude = BREAKDOWN_DISTURBANCE_AMPLITUDE if seeded else 0.0
    disturbance_direction = BREAKDOWN_DISTURBANCE_DIRECTION if seeded else "radial"
    disturbance_mode = BREAKDOWN_DISTURBANCE_MODE if seeded else 8
    disturbance = vpm.WidnallDisturbance.single_mode(
        amplitude=disturbance_amplitude,
        mode=disturbance_mode,
        direction=disturbance_direction,
    )
    if case_name is not None:
        name = case_name
    elif qualification:
        name = "cs_breakdown_qualification" if seeded else "cs_qualification"
    else:
        name = f"cs_breakdown_{variant}" if seeded else f"cs_{variant}"
    tube_radius = np.sqrt(CORE_RADIUS**2 - numerical_core**2) * np.sqrt(-np.log(TAIL_FRACTION))
    rings = tuple(
        vpm.VortexRing(
            centre=(centre_x, 0.0, 0.0),
            radius=RING_RADIUS,
            circulation=RING_CIRCULATION,
            vortex_core_radius=CORE_RADIUS,
            kinematic_viscosity=kinematic_viscosity,
            disturbance=disturbance,
            core_compensation=vpm.ParticleCoreCompensation(),
            distribution=vpm.ToroidalDistribution(
                centre=(centre_x, 0.0, 0.0),
                ring_radius=RING_RADIUS,
                tube_radius=tube_radius,
                spacing=spacing,
                core_radius_ratio=core_radius_ratio,
                disturbance=disturbance if seeded else None,
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
            viscous=vpm.ViscousConfig.cs(
                particle_spacing=spacing,
                kinematic_viscosity=kinematic_viscosity,
                core_radius_ratio=core_radius_ratio,
            ),
            turbulence=vpm.TurbulenceConfig.les_smagorinsky(
                smagorinsky_coefficient=les_coefficient
            ),
            stabilization=stabilization(variant, rings, time_step_size),
            domain_bounds=(-2.0, 12.0, -3.0, 3.0, -3.0, 3.0),
            max_n_particles=MAX_N_PARTICLES if particle_capacity is None else particle_capacity,
            compute_device=compute_device,
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
            steps=N_STEPS * refinement if steps is None else steps,
            final_backup=True,
            health_limit_action="STOP",
            wall_time_limit_seconds=60.0 * (WALL_MINUTES if wall_minutes is None else wall_minutes),
            resource_limits=vpm.ResourceLimits(
                max_particles=RESOURCE_MAX_PARTICLES,
                max_rss_bytes=PROCESS_RSS_LIMIT_BYTES,
                min_available_memory_bytes=AVAILABLE_MEMORY_FLOOR_BYTES,
            ),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, default="kinematic")
    parser.add_argument("--compute-device", choices=("AUTO", "CPU", "METAL"), default="AUTO")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--wall-minutes", type=float)
    parser.add_argument("--qualification", action="store_true")
    parser.add_argument("--particle-spacing", type=float)
    parser.add_argument("--particle-core-radius", type=float)
    parser.add_argument("--smagorinsky", type=float)
    parser.add_argument("--particle-capacity", type=int,
                        help="Allocated particle capacity; leaves the initial population unchanged")
    parser.add_argument("--case-name", "--qualification-name", dest="case_name")
    args = parser.parse_args()
    if args.qualification and args.variant != "baseline":
        parser.error("--qualification is only valid with --variant baseline")
    steps = args.steps
    wall_minutes = args.wall_minutes
    if args.qualification:
        steps = QUALIFICATION_STEPS if steps is None else steps
        wall_minutes = QUALIFICATION_WALL_MINUTES if wall_minutes is None else wall_minutes
    vpm.VPMSolver(
        build_case(
            args.variant,
            scenario=args.scenario,
            compute_device=args.compute_device,
            steps=steps,
            wall_minutes=wall_minutes,
            qualification=args.qualification,
            particle_spacing=args.particle_spacing,
            particle_core_radius=args.particle_core_radius,
            smagorinsky_coefficient=args.smagorinsky,
            case_name=args.case_name,
            particle_capacity=args.particle_capacity,
        )
    ).run()
