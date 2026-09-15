#!/usr/bin/env python3
"""Cheng et al. Fig. 5 rings: a DNS baseline and three stabilization methods."""

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
# Fig. 5 at Re=3000 excludes the instability perturbation. The paper's
# epsilon/R0=0.05, n=8 axial perturbation belongs to Fig. 3 at Re=3415.
DISTURBANCE_AMPLITUDE = 0.0
DISTURBANCE_MODE = 8
PARTICLE_SPACING = 0.05
PARTICLE_CORE_RADIUS = 0.05
TOROIDAL_TAIL_FRACTION = 1.0e-4
TIME_STEP_SIZE = 0.00375
N_STEPS = 2400
MAX_N_PARTICLES = 600_000
TUTORIAL_DIR = Path(__file__).parent

CASES = (
    "baseline",
    "selective_eddy_viscosity",
    "pedrizzetti_relaxation",
    "particle_splitting",
)


def create_ring(x, group):
    centre = (x, 0.0, 0.0)
    disturbance = (
        vpm.WidnallDisturbance.single_mode(
            amplitude=DISTURBANCE_AMPLITUDE,
            mode=DISTURBANCE_MODE,
            direction="axial",
        )
        if DISTURBANCE_AMPLITUDE > 0.0
        else None
    )
    tube = np.sqrt(CORE_RADIUS**2 - PARTICLE_CORE_RADIUS**2) * np.sqrt(
        -np.log(TOROIDAL_TAIL_FRACTION)
    )
    return vpm.VortexRing(
        centre=centre,
        radius=RING_RADIUS,
        circulation=RING_CIRCULATION,
        vortex_core_radius=CORE_RADIUS,
        kinematic_viscosity=KINEMATIC_VISCOSITY,
        disturbance=disturbance,
        core_compensation=vpm.ParticleCoreCompensation(),
        distribution=vpm.ToroidalDistribution(
            centre=centre,
            ring_radius=RING_RADIUS,
            tube_radius=tube,
            spacing=PARTICLE_SPACING,
            core_radius_ratio=1.0,
            disturbance=disturbance,
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
        regularization_preserve_groups=True,
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
            turbulence=vpm.TurbulenceConfig.dns(),
            stabilization=transfer,
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
                vpm.RingDiagnosticsSampler(schedule=vpm.FinalOnly()),
                *core_section_samplers(),
            ),
        ),
        run=vpm.RunPlan(
            steps=n_steps,
            final_backup=True,
            health_limit_action="STOP",
            resource_limits=vpm.ResourceLimits(max_particles=MAX_N_PARTICLES),
        ),
    )


def build_case(method, *, n_steps=N_STEPS, compute_device="AUTO"):
    case = baseline_case(method, n_steps=n_steps, compute_device=compute_device)
    base = case.numerics.stabilization
    if method == "baseline":
        return case
    if method == "selective_eddy_viscosity":
        added = replace(base, selective_eddy_viscosity_coefficient=0.5)
    elif method == "particle_splitting":
        added = replace(
            base,
            filament_refinement=vpm.FilamentRefinementConfig.adaptive(
                interval_steps=5,
                max_vortex_strength_factor=2.0,
                offset_fraction=0.25,
                max_n_particles=MAX_N_PARTICLES,
            ),
        )
    else:  # pedrizzetti_relaxation
        added = replace(
            base,
            pedrizzetti_relaxation_factor=0.385 * TIME_STEP_SIZE,
            pedrizzetti_relaxation_preserve_vortex_strength=False,
            pedrizzetti_relaxation_preserve_moments=True,
        )
    return replace(case, numerics=replace(case.numerics, stabilization=added))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", nargs="?", default="baseline", choices=CASES)
    case = build_case(parser.parse_args().method)
    vpm.VPMSolver(case).run()
