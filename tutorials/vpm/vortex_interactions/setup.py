#!/usr/bin/env python3
"""Compare VPM stabilization methods for two leapfrogging vortex rings.

Every case uses the same transposed LES formulation and SSPRK3 update. The
only changed quantity is the stabilization method selected by the case name.
Crossing a resolution limit ends that case normally so the comparison can
continue.
"""

from __future__ import annotations

import argparse
from functools import cache
from pathlib import Path

import numpy as np

import openonda.vpm as vpm
from openonda.vpm import Backup, Samplers


# Physics
RING_RADIUS = 1.0  # ring major radius [m]
RING_CIRCULATION = np.pi  # circulation of each ring [m²/s]
REYNOLDS_NUMBER = 3000.0  # Re = Gamma/nu
CORE_RADIUS = 0.1 * RING_RADIUS  # physical Gaussian core radius [m]
RING_SEPARATION = 1.0 * RING_RADIUS  # initial axial separation [m]
KINEMATIC_VISCOSITY = RING_CIRCULATION / REYNOLDS_NUMBER  # [m^2/s]
DISTURBANCE_AMPLITUDE = 0.05  # fraction of ring radius
DISTURBANCE_MODE = 8

# Numerics
PARTICLE_SPACING = 0.035 * RING_RADIUS  # [m]
PARTICLE_CORE_RADIUS = 2.0 * PARTICLE_SPACING
# Circulation normalization amplifies a truncated Gaussian. A 5% tail cut
# produced a 6--7% peak excess on the study lattice; retain the physical tail.
TOROIDAL_TAIL_FRACTION = 1.0e-4
TIME_STEP_SIZE = 20.0 * PARTICLE_SPACING**2 / RING_CIRCULATION  # [s]
N_STEPS = 1200
SAMPLE_INTERVAL_STEPS = 5
CORE_SECTION_INTERVAL = 1.5  # physical seconds, independent of the integration timestep
CORE_SECTION_SPACING = 0.02 * RING_RADIUS
BACKUP_INTERVAL_STEPS = 50
MAX_N_PARTICLES = 120_000
SMAGORINSKY_COEFFICIENT = 0.20
RANDOM_SEED = 42

MAX_LAGRANGIAN_CFL = 1.0
MAX_VORTICITY_DIVERGENCE = 0.12
MAX_VORTEX_MISALIGNMENT = 25.0

# Stabilization
STRETCHING_VISCOSITY_COEFFICIENT = 0.5
PEDRIZZETTI_FACTOR = 0.3
PEDRIZZETTI_INTERVAL_STEPS = 1
SPLITTING_STRENGTH_FACTOR = 2.0
SPLITTING_INTERVAL_STEPS = 1
SPLITTING_OFFSET_FRACTION = 0.25
DIVERGENCE_RELAXATION_INTERVAL_STEPS = 25
REMESH_CORE_RADIUS_FACTOR = 2.0
REMESH_CORE_RADIUS_TRIGGER = REMESH_CORE_RADIUS_FACTOR * PARTICLE_CORE_RADIUS
REMESH_INTERVAL_STEPS = round(
    3.0 * PARTICLE_CORE_RADIUS**2 / (4.0 * KINEMATIC_VISCOSITY * TIME_STEP_SIZE)
)
REMESH_TAIL_BUDGET = 1.0e-3

CASES = (
    "baseline",
    "stretching_viscosity",
    "pedrizzetti",
    "splitting",
    "divergence_relaxation",
    "remeshing",
)
CASE_LABELS = {
    "baseline": "Baseline",
    "stretching_viscosity": "Stretching viscosity",
    "pedrizzetti": "Pedrizzetti relaxation",
    "splitting": "Filament refinement",
    "divergence_relaxation": "Divergence relaxation",
    "remeshing": "Conservative regularization",
}


TUTORIAL_DIR = Path(__file__).resolve().parent


def core_section_samplers(
    *, interval: float = CORE_SECTION_INTERVAL
) -> tuple[vpm.SurfaceSampler, ...]:
    """Sample curl(u) on z=0, y>=0; here omega_theta equals omega_z.

    The fixed grid covers both rings' travel. Postprocessing crops the saved
    plane around the cores without recomputing or azimuthally averaging fields.
    A final sampler also records runs that end between cadence boundaries.
    """
    options = dict(
        point=[0.0, 0.0, 0.0],
        normal=[0.0, 0.0, 1.0],
        bounds=[-2.0 * RING_RADIUS, 14.0 * RING_RADIUS, 0.0, 1.8 * RING_RADIUS],
        spacing=CORE_SECTION_SPACING,
        file_name="core_section",
        include_derivatives=False,
    )
    return (
        vpm.SurfaceSampler(**options, schedule=vpm.EveryTime(interval), initial=True),
        vpm.SurfaceSampler(**options, schedule=vpm.FinalOnly()),
    )


def create_ring(centre_x: float, group_id: int) -> vpm.VortexRing:
    """Build one disturbed Gaussian vortex ring."""
    represented_core_sq = CORE_RADIUS**2 - PARTICLE_CORE_RADIUS**2
    tube_radius = np.sqrt(represented_core_sq) * np.sqrt(-np.log(TOROIDAL_TAIL_FRACTION))
    centre = (centre_x, 0.0, 0.0)
    distribution = vpm.ToroidalDistribution(
        ring_radius=RING_RADIUS,
        tube_radius=tube_radius,
        spacing=PARTICLE_SPACING,
        core_radius_ratio=PARTICLE_CORE_RADIUS / PARTICLE_SPACING,
        centre=centre,
    )
    return vpm.VortexRing(
        kinematic_viscosity=KINEMATIC_VISCOSITY,
        centre=centre,
        radius=RING_RADIUS,
        circulation=RING_CIRCULATION,
        vortex_core_radius=CORE_RADIUS,
        disturbance=vpm.WidnallDisturbance.single_mode(
            amplitude=DISTURBANCE_AMPLITUDE,
            mode=DISTURBANCE_MODE,
        ),
        core_compensation=vpm.ParticleCoreCompensation(),
        distribution=distribution,
        group_id=group_id,
    )


@cache
def initial_peak_strength() -> float:
    """Return the largest initial particle strength."""
    particles = create_ring(-0.5 * RING_SEPARATION, 0).build()
    return float(np.linalg.norm(particles.vortex_strength, axis=1).max())


def stabilization(case_name: str) -> vpm.StabilizationConfig:
    return {
        "baseline": vpm.StabilizationConfig.disabled(),
        "stretching_viscosity": vpm.StabilizationConfig.stretching_viscosity(
            coefficient=STRETCHING_VISCOSITY_COEFFICIENT
        ),
        "pedrizzetti": vpm.StabilizationConfig.pedrizzetti_relaxation(
            factor=PEDRIZZETTI_FACTOR,
            interval_steps=PEDRIZZETTI_INTERVAL_STEPS,
        ),
        "splitting": vpm.StabilizationConfig(
            filament_refinement=vpm.FilamentRefinementConfig.adaptive(
                interval_steps=SPLITTING_INTERVAL_STEPS,
                max_vortex_strength_factor=np.inf,
                max_absolute_vortex_strength=(SPLITTING_STRENGTH_FACTOR * initial_peak_strength()),
                offset_fraction=SPLITTING_OFFSET_FRACTION,
                max_n_particles=MAX_N_PARTICLES,
            )
        ),
        "divergence_relaxation": vpm.StabilizationConfig(
            divergence_relaxation=vpm.DivergenceRelaxationConfig.constrained(
                interval_steps=DIVERGENCE_RELAXATION_INTERVAL_STEPS,
                start_step=DIVERGENCE_RELAXATION_INTERVAL_STEPS,
                grid_spacing=PARTICLE_SPACING,
            )
        ),
        "remeshing": vpm.StabilizationConfig(
            regularization_interval_steps=REMESH_INTERVAL_STEPS,
            regularization_start_step=REMESH_INTERVAL_STEPS,
            regularization_grid_spacing=PARTICLE_SPACING,
            regularization_tail_budget=REMESH_TAIL_BUDGET,
            regularization_max_particles=MAX_N_PARTICLES,
            regularization_divergence_trigger=None,
            regularization_misalignment_trigger=None,
            regularization_core_radius_trigger=REMESH_CORE_RADIUS_TRIGGER,
            regularization_core_radius=PARTICLE_CORE_RADIUS,
        ),
    }[case_name]


def build_case(
    case_name: str,
    *,
    n_steps: int = N_STEPS,
    compute_device: str = "AUTO",
) -> vpm.VPMCase:
    """Build one LES and transposed-stretching comparison case."""

    initial_conditions = tuple(
        create_ring(centre_x, group_id)
        for group_id, centre_x in enumerate((-0.5 * RING_SEPARATION, 0.5 * RING_SEPARATION))
    )
    return vpm.VPMCase(
        name=case_name,
        numerics=vpm.Numerics(
            time_step_size=TIME_STEP_SIZE,
            compute_device=compute_device,
            integrator=vpm.SSPRK3(),
            induction=vpm.TreecodeInduction(stretching_scheme="TRANSPOSED"),
            viscous=vpm.ViscousConfig.cs(),
            turbulence=vpm.TurbulenceConfig.les_smagorinsky(
                smagorinsky_coefficient=SMAGORINSKY_COEFFICIENT
            ),
            stabilization=stabilization(case_name),
            particle_kernel="GAUSSIAN",
            write_precision="f32",
            max_n_particles=MAX_N_PARTICLES,
            random_seed=RANDOM_SEED,
            health_limits=vpm.HealthLimits(
                lagrangian_cfl=vpm.LagrangianCFLLimit(maximum=MAX_LAGRANGIAN_CFL),
                divergence=vpm.DivergenceLimit(maximum=MAX_VORTICITY_DIVERGENCE),
                misalignment=vpm.MisalignmentLimit(maximum_degrees=MAX_VORTEX_MISALIGNMENT),
            ),
        ),
        initial_conditions=initial_conditions,
        backup=Backup(
            interval_steps=BACKUP_INTERVAL_STEPS,
            directory=str(Path("solution") / case_name),
            log_directory=str(Path("solution") / case_name),
        ),
        samplers=Samplers(
            samples=(
                vpm.FlowIntegralsSampler(
                    schedule=vpm.EverySteps(SAMPLE_INTERVAL_STEPS), initial=True
                ),
                vpm.RingDiagnosticsSampler(
                    schedule=vpm.EverySteps(SAMPLE_INTERVAL_STEPS), initial=True
                ),
                *core_section_samplers(),
            ),
            directory=case_name,
        ),
        run=vpm.RunPlan(
            steps=n_steps,
            final_backup=False,
            health_limit_action="STOP",
        ),
        directory=TUTORIAL_DIR,
    )


def run_case(
    case_name: str,
    *,
    n_steps: int = N_STEPS,
    compute_device: str = "AUTO",
) -> None:
    """Run one stabilization case and retain all available diagnostics."""

    case = build_case(case_name, n_steps=n_steps, compute_device=compute_device)
    vpm.VPMSolver(case).run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=CASES)
    args = parser.parse_args()
    run_case(args.case)
