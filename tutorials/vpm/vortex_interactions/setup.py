#!/usr/bin/env python3
"""Particle-stabilization comparison for two leapfrogging vortex rings.

Usage:
    python setup.py --case leapfrog_les_splitting_remeshing
"""

from __future__ import annotations

import argparse
from functools import cache
from pathlib import Path

import numpy as np

import openonda.vpm as vpm
from openonda.vpm import Backup, Samplers

TUTORIAL_DIR = Path(__file__).resolve().parent

# ---- Cheng, Lou & Lim (2015) leapfrogging reference -----------------------
RING_RADIUS = 1.0  # R0 [m]
RING_CIRCULATION = np.pi  # Gamma0 [m^2/s]
REYNOLDS_NUMBER = 3000.0  # Re_Gamma = Gamma0/nu
CORE_RADIUS = 0.1 * RING_RADIUS  # a0/R0 = 0.1 [m]
RING_SEPARATION = 1.0 * RING_RADIUS  # h0/R0 = 1
KINEMATIC_VISCOSITY = RING_CIRCULATION / REYNOLDS_NUMBER
DISTURBANCE_AMPLITUDE = 0.05 * RING_RADIUS
DISTURBANCE_MODE = 8
DISTURBANCE_PHASE = 0.0

# ---- VPM discretization ----------------------------------------------------
PARTICLE_SPACING = 0.035 * RING_RADIUS
PARTICLE_CORE_RADIUS = 2.0 * PARTICLE_SPACING
WEAK_PARTICLE_PERCENT = 5.0
TIME_STEP_SIZE = 20.0 * PARTICLE_SPACING**2 / RING_CIRCULATION
NUMBER_OF_STEPS = 1200
DIAGNOSTIC_INTERVAL_STEPS = 5
BACKUP_INTERVAL_STEPS = 50
TREECODE_THETA = 0.30
MAXIMUM_PARTICLES = 120_000
# The historical 0.15 value limited internal coupled substeps; it is not a
# calibrated rejection threshold for the complete accepted macro-step.
MAX_LAGRANGIAN_CFL: float | None = None

# ---- Common LES closure and independently varied stabilization ------------
SMAGORINSKY_COEFFICIENT = 0.20

# Split when |alpha_p| exceeds twice the largest initial |alpha_p|.
SPLITTING_STRENGTH_FACTOR = 2.0
SPLITTING_INTERVAL_STEPS = 1
SPLITTING_OFFSET_FRACTION = 0.25

# Gaussian CS gives sigma^2(t) = sigma_0^2 + 4 nu t. The first time at which
# sigma = 2 sigma_0 is 3 sigma_0^2/(4 nu)
REMESH_CORE_RADIUS_FACTOR = 2.0
REMESH_CORE_RADIUS_TRIGGER = REMESH_CORE_RADIUS_FACTOR * PARTICLE_CORE_RADIUS
REMESH_INTERVAL_STEPS = round(
    3.0 * PARTICLE_CORE_RADIUS**2 / (4.0 * KINEMATIC_VISCOSITY * TIME_STEP_SIZE)
)
REMESH_GRID_SPACING = PARTICLE_SPACING
REMESH_TAIL_BUDGET = 1.0e-3

CASES = {
    "leapfrog_les": (False, False),
    "leapfrog_les_splitting": (True, False),
    "leapfrog_les_remeshing": (False, True),
    "leapfrog_les_splitting_remeshing": (True, True),
}


def create_ring(centre_x: float, group_id: int | None = None) -> vpm.VortexRing:
    """Build one declarative disturbed Gaussian-ring condition."""
    centre = np.array([centre_x, 0.0, 0.0])
    axial_extent = CORE_RADIUS + PARTICLE_CORE_RADIUS
    radial_extent = RING_RADIUS + 2.0 * CORE_RADIUS
    distribution = vpm.RectangularDistribution(
        bounds=(
            (centre_x - axial_extent, centre_x + axial_extent),
            (-radial_extent, radial_extent),
            (-radial_extent, radial_extent),
        ),
        spacing=PARTICLE_SPACING,
        core_radius_ratio=PARTICLE_CORE_RADIUS / PARTICLE_SPACING,
    )
    return vpm.VortexRing(
        kinematic_viscosity=KINEMATIC_VISCOSITY,
        centre=centre,
        radius=RING_RADIUS,
        circulation=RING_CIRCULATION,
        vortex_core_radius=CORE_RADIUS,
        disturbance=vpm.WidnallDisturbance.single_mode(
            amplitude=DISTURBANCE_AMPLITUDE / RING_RADIUS,
            mode=DISTURBANCE_MODE,
            phase=DISTURBANCE_PHASE,
        ),
        core_compensation=vpm.ParticleCoreCompensation(),
        distribution=distribution,
        group_id=group_id,
    )


@cache
def initial_peak_strength() -> float:
    """Return the maximum strength magnitude in the original particle cloud."""
    particles = create_ring(-0.5 * RING_SEPARATION).build()
    return float(np.linalg.norm(particles.vortex_strength, axis=1).max())


def stabilization(case_name: str) -> vpm.StabilizationConfig:
    """Arm the requested splitting and core-growth remeshing mechanisms."""
    has_splitting, has_remeshing = CASES[case_name]
    filament_refinement = (
        vpm.FilamentRefinementConfig.adaptive(
            interval_steps=SPLITTING_INTERVAL_STEPS,
            max_vortex_strength_factor=np.inf,
            max_absolute_vortex_strength=(SPLITTING_STRENGTH_FACTOR * initial_peak_strength()),
            offset_fraction=SPLITTING_OFFSET_FRACTION,
            max_n_particles=MAXIMUM_PARTICLES,
        )
        if has_splitting
        else vpm.FilamentRefinementConfig.disabled()
    )
    return vpm.StabilizationConfig(
        filament_refinement=filament_refinement,
        regularization_interval_steps=(REMESH_INTERVAL_STEPS if has_remeshing else 0),
        regularization_start_step=(REMESH_INTERVAL_STEPS if has_remeshing else 0),
        regularization_grid_spacing=(REMESH_GRID_SPACING if has_remeshing else None),
        regularization_tail_budget=REMESH_TAIL_BUDGET,
        regularization_max_particles=(MAXIMUM_PARTICLES if has_remeshing else None),
        regularization_divergence_trigger=None,
        regularization_misalignment_trigger=None,
        regularization_core_radius_trigger=(REMESH_CORE_RADIUS_TRIGGER if has_remeshing else None),
        regularization_core_radius=(PARTICLE_CORE_RADIUS if has_remeshing else None),
    )


def build_case(case_name: str) -> vpm.VPMCase:
    """Build one of the four declarative stabilization comparisons."""
    initial_conditions = tuple(
        create_ring(centre_x, group_id=group_id)
        for group_id, centre_x in enumerate((-0.5 * RING_SEPARATION, 0.5 * RING_SEPARATION))
    )
    return vpm.VPMCase(
        numerics=vpm.Numerics(
            time_step_size=TIME_STEP_SIZE,
            integrator=vpm.SSPRK3(),
            viscous=vpm.ViscousConfig.cs(
                kinematic_viscosity=KINEMATIC_VISCOSITY,
                particle_spacing=PARTICLE_SPACING,
            ),
            turbulence=vpm.TurbulenceConfig.les_smagorinsky(
                smagorinsky_coefficient=SMAGORINSKY_COEFFICIENT,
            ),
            stabilization=stabilization(case_name),
            health_limits=vpm.HealthLimits(
                lagrangian_cfl=vpm.LagrangianCFLLimit(maximum=MAX_LAGRANGIAN_CFL)
            ),
            induction=vpm.TreecodeInduction(
                theta=TREECODE_THETA,
                sort_particle_targets=True,
                traversal_block_dim=128,
            ),
            particle_kernel="GAUSSIAN",
            write_precision="f32",
            max_n_particles=MAXIMUM_PARTICLES,
        ),
        initial_conditions=initial_conditions,
        backup=Backup(
            interval_steps=BACKUP_INTERVAL_STEPS,
            directory=str(Path("solution") / case_name),
            log_directory=str(Path("solution") / case_name),
        ),
        samplers=Samplers(
            samples=(
                vpm.FlowIntegralsSampler(schedule=vpm.EverySteps(DIAGNOSTIC_INTERVAL_STEPS)),
                vpm.RingDiagnosticsSampler(schedule=vpm.EverySteps(DIAGNOSTIC_INTERVAL_STEPS)),
            ),
            directory=case_name,
        ),
        run=vpm.RunPlan(steps=NUMBER_OF_STEPS),
        initial_weak_particle_percent=WEAK_PARTICLE_PERCENT,
        directory=TUTORIAL_DIR,
    )


def run_case(case_name: str) -> None:
    """Run one case and retain its terminal manifest beside its backups."""
    root_manifest = TUTORIAL_DIR / "run_manifest.json"
    case_manifest = TUTORIAL_DIR / "solution" / case_name / "run_manifest.json"
    root_manifest.unlink(missing_ok=True)
    try:
        vpm.VPMSolver(build_case(case_name)).run()
    finally:
        if root_manifest.is_file():
            case_manifest.parent.mkdir(parents=True, exist_ok=True)
            root_manifest.replace(case_manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=tuple(CASES))
    arguments = parser.parse_args()
    run_case(arguments.case)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
