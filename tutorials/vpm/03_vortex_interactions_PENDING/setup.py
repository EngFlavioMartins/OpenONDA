#!/usr/bin/env python3
"""Unperturbed Fig. 5 rings: an LES baseline and two stabilization methods."""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import openonda.vpm as vpm

RING_RADIUS = 1.0
RING_CIRCULATION = np.pi
REYNOLDS_NUMBER = 3000.0
KINEMATIC_VISCOSITY = RING_CIRCULATION / REYNOLDS_NUMBER
CORE_RADIUS = 0.1
RING_SEPARATION = 1.0
DISTURBANCE_AMPLITUDE = 0.0
DISTURBANCE_MODE = 8
PARTICLE_SPACING = 0.05
PARTICLE_CORE_RADIUS = 0.05
TOROIDAL_TAIL_FRACTION = 1.0e-4
TIME_STEP_SIZE = 0.00375
N_STEPS = 2400
SMAGORINSKY_COEFFICIENT = 0.20
LES_FILTER_WIDTH = 0.053
MAX_N_PARTICLES = 120_000
TUTORIAL_DIR = Path(__file__).parent

CASES = ("baseline", "stretching_viscosity", "p_moments")
CASE_LABELS = {
    "fig5_baseline": "Baseline",
    "fig5_stretching_viscosity": "Stretching viscosity",
    "fig5_p_moments": "Moment-preserving relaxation",
}


def create_ring(x, group):
    centre = (x, 0.0, 0.0)
    tube = np.sqrt(CORE_RADIUS**2 - PARTICLE_CORE_RADIUS**2) * np.sqrt(
        -np.log(TOROIDAL_TAIL_FRACTION)
    )
    return vpm.VortexRing(
        centre=centre,
        radius=RING_RADIUS,
        circulation=RING_CIRCULATION,
        vortex_core_radius=CORE_RADIUS,
        kinematic_viscosity=KINEMATIC_VISCOSITY,
        disturbance=vpm.WidnallDisturbance.single_mode(
            amplitude=DISTURBANCE_AMPLITUDE, mode=DISTURBANCE_MODE
        ),
        core_compensation=vpm.ParticleCoreCompensation(),
        distribution=vpm.ToroidalDistribution(
            centre=centre,
            ring_radius=RING_RADIUS,
            tube_radius=tube,
            spacing=PARTICLE_SPACING,
            core_radius_ratio=1.0,
        ),
        group_id=group,
    )


def core_section_samplers(*, interval=0.15):
    plane = dict(
        point=[0, 0, 0],
        normal=[0, 0, 1],
        bounds=[-2, 14, 0, 1.8],
        spacing=0.02,
        file_name="core_section",
        include_derivatives=False,
    )
    return (
        vpm.SurfaceSampler(**plane, schedule=vpm.EveryTime(interval), initial=True),
        vpm.SurfaceSampler(**plane, schedule=vpm.FinalOnly()),
    )


def baseline_case(name, *, n_steps=N_STEPS, compute_device="AUTO"):
    # Conservative transfer maintains particle resolution; no damping or projection.
    transfer = vpm.StabilizationConfig(
        regularization_interval_steps=20,
        regularization_grid_spacing=0.05,
        regularization_core_radius=0.05,
        regularization_core_radius_trigger=0.10,
        regularization_tail_budget=0.003,
        regularization_max_particles=MAX_N_PARTICLES,
        regularization_transfer_only=True,
        regularization_divergence_trigger=None,
        regularization_misalignment_trigger=None,
        regularization_total_kinetic_energy_dissipation_limit=0.01,
        regularization_total_enstrophy_dissipation_limit=0.01,
    )
    return vpm.VPMCase(
        name=name,
        directory=TUTORIAL_DIR,
        initial_conditions=(
            create_ring(-RING_SEPARATION / 2, 0),
            create_ring(RING_SEPARATION / 2, 1),
        ),
        numerics=vpm.Numerics(
            time_step_size=TIME_STEP_SIZE,
            compute_device=compute_device,
            integrator=vpm.SSPRK3(),
            induction=vpm.TreecodeInduction(
                theta=0.5, multipole_order=3, stretching_scheme="TRANSPOSED"
            ),
            viscous=vpm.ViscousConfig.cs(
                particle_spacing=PARTICLE_SPACING,
                kinematic_viscosity=KINEMATIC_VISCOSITY,
                core_radius_ratio=1.0,
            ),
            turbulence=vpm.TurbulenceConfig.les_smagorinsky(
                smagorinsky_coefficient=SMAGORINSKY_COEFFICIENT,
                filter_width=LES_FILTER_WIDTH,
            ),
            stabilization=transfer,
            domain_bounds=(-2, 12, -3, 3, -3, 3),
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
            interval_steps=100, directory=f"solution/{name}", log_directory=f"solution/{name}"
        ),
        samplers=vpm.Samplers(
            directory=name,
            samples=(
                vpm.FlowIntegralsSampler(schedule=vpm.EverySteps(10), initial=True),
                vpm.RingDiagnosticsSampler(schedule=vpm.EverySteps(10), initial=True),
                *core_section_samplers(),
            ),
        ),
        run=vpm.RunPlan(
            steps=n_steps,
            final_backup=True,
            health_limit_action="STOP",
            resource_limits=vpm.ResourceLimits(
                max_particles=600_000,
                max_rss_bytes=12 * 1024**3,
                min_available_memory_bytes=2 * 1024**3,
            ),
        ),
    )


def build_case(method, *, n_steps=N_STEPS, compute_device="AUTO"):
    case = baseline_case(f"fig5_{method}", n_steps=n_steps, compute_device=compute_device)
    base = case.numerics.stabilization
    if method == "baseline":
        return case
    if method == "stretching_viscosity":
        added = replace(base, stretching_viscosity_coefficient=0.5)
    else:  # p_moments
        added = replace(
            base,
            pedrizzetti_relaxation_factor=0.384684814725 * TIME_STEP_SIZE,
            pedrizzetti_relaxation_preserve_vortex_strength=False,
            pedrizzetti_relaxation_preserve_moments=True,
        )
    return replace(case, numerics=replace(case.numerics, stabilization=added))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=CASES)
    case = build_case(parser.parse_args().method)
    vpm.VPMSolver(case).run()
