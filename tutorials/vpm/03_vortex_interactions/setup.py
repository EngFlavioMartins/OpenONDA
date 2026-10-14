#!/usr/bin/env python3
"""Leapfrogging rings of Cheng, Lou & Lim (2015) Fig. 5: DNS, LES and stabilized LES."""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import openonda.vpm as vpm

from assets.initial_conditions import ToroidalDistribution, VortexRing
from assets.ring_diagnostics import AzimuthalModeSampler, RingDiagnosticsSampler

START_FROM = "latest"  # Resume the latest backup; ./allrun.sh cleans first.

RING_RADIUS = 1.0
RING_CIRCULATION = np.pi
REYNOLDS_NUMBER = 3000.0
HIGH_REYNOLDS_NUMBER = 30000.0
KINEMATIC_VISCOSITY = RING_CIRCULATION / REYNOLDS_NUMBER
CORE_RADIUS = 0.1
RING_SEPARATION = 1.0
PARTICLE_SPACING = 0.05
PARTICLE_CORE_RADIUS = 1.5 * PARTICLE_SPACING  # sigma/h = 1.5: blobs overlap
REMESH_CORE_RADIUS = 2.2 * PARTICLE_SPACING  # core spreading grows sigma; remesh above this
TOROIDAL_TAIL_FRACTION = 1.0e-4
TIME_STEP_SIZE = 0.00375
N_STEPS = 3200  # 12 s
HIGH_RE_STEPS = 2400  # 9 s
SINGLE_RING_STEPS = 300
MAX_N_PARTICLES = 600_000
RELAXATION_FACTOR = 0.02  # per step: a relaxation rate of 5.3 1/s
TUTORIAL_DIR = Path(__file__).parent

CASES = ("single_ring", "dns", "les", "les_stabilized", "high_re_dns", "high_re_les")


def create_ring(x, group, kinematic_viscosity):
    centre = (x, 0.0, 0.0)
    tube = np.sqrt(CORE_RADIUS**2 - PARTICLE_CORE_RADIUS**2) * np.sqrt(
        -np.log(TOROIDAL_TAIL_FRACTION)
    )
    return VortexRing(
        centre=centre,
        radius=RING_RADIUS,
        circulation=RING_CIRCULATION,
        vortex_core_radius=CORE_RADIUS,
        kinematic_viscosity=kinematic_viscosity,
        distribution=ToroidalDistribution(
            centre=centre,
            ring_radius=RING_RADIUS,
            tube_radius=tube,
            spacing=PARTICLE_SPACING,
            core_radius_ratio=PARTICLE_CORE_RADIUS / PARTICLE_SPACING,
        ),
        group_id=group,
    )


def core_section_samplers(*, interval=0.15):
    plane = dict(
        point=[0, 0, 0],
        normal=[0, 0, 1],
        bounds=[-2, 17, 0, 1.8],
        spacing=0.02,
        file_name="core_section",
        include_derivatives=False,
    )
    return (
        vpm.SurfaceSampler(**plane, schedule=vpm.EveryTime(interval), initial=True),
        vpm.SurfaceSampler(**plane, schedule=vpm.FinalOnly()),
    )


def dns_case(name, *, reynolds_number=REYNOLDS_NUMBER, n_steps=N_STEPS, direct=False):
    kinematic_viscosity = RING_CIRCULATION / reynolds_number
    # Gaussian redistribution restores the lattice once core spreading has grown
    # the cores to REMESH_CORE_RADIUS; it conserves strength, impulse and groups.
    remesh = vpm.StabilizationConfig(
        regularization_interval_steps=20,
        regularization_grid_spacing=PARTICLE_SPACING,
        regularization_core_radius=PARTICLE_CORE_RADIUS,
        regularization_core_radius_trigger=REMESH_CORE_RADIUS,
        regularization_tail_budget=0.003,
        regularization_transfer_only=True,
        regularization_preserve_groups=True,
        regularization_divergence_trigger=None,
        regularization_misalignment_trigger=None,
        regularization_total_kinetic_energy_dissipation_limit=0.01,
        regularization_total_enstrophy_dissipation_limit=0.01,
    )
    rings = tuple(
        create_ring(x, group, kinematic_viscosity)
        for group, x in enumerate((-RING_SEPARATION / 2, RING_SEPARATION / 2))
    )
    # Remeshing spreads the Re = 3000 rings over up to 3e5 particles, where the
    # fast multipole method is quicker; below about 5e4 particles (one ring, or
    # Re = 3e4 where the cores hardly grow) direct summation is faster and exact.
    induction = (
        vpm.DirectInduction(stretching_scheme="TRANSPOSED")
        if direct
        else vpm.FMMInduction(stretching_scheme="TRANSPOSED")
    )
    return vpm.VPMCase(
        name=name,
        directory=TUTORIAL_DIR,
        initial_conditions=rings[1:] if name == "single_ring" else rings,
        numerics=vpm.Numerics(
            time_step_size=TIME_STEP_SIZE,
            induction=induction,
            viscous=vpm.ViscousConfig.cs(
                particle_spacing=PARTICLE_SPACING,
                kinematic_viscosity=kinematic_viscosity,
                core_radius_ratio=PARTICLE_CORE_RADIUS / PARTICLE_SPACING,
            ),
            turbulence=vpm.TurbulenceConfig.dns(),
            stabilization=remesh,
            max_n_particles=MAX_N_PARTICLES,
            random_seed=42,
            verbose=False,
            # The run stops once the particles no longer represent a solenoidal
            # vorticity field: the lifetime of each case.
            state_limits=vpm.ParticleStateLimits(
                lagrangian_cfl=vpm.LagrangianCFLLimit(maximum=1.0),
                divergence=vpm.DivergenceLimit(maximum=0.12),
            ),
        ),
        backup=vpm.Backup(
            interval_steps=200, directory=f"solution/{name}", log_directory=f"solution/{name}"
        ),
        samplers=vpm.Samplers(
            directory=name,
            samples=(
                vpm.FlowIntegralsSampler(schedule=vpm.EverySteps(10), initial=True),
                RingDiagnosticsSampler(schedule=vpm.EverySteps(10), initial=True),
                AzimuthalModeSampler(schedule=vpm.EverySteps(10), initial=True),
                *core_section_samplers(),
            ),
        ),
        run=vpm.RunPlan(steps=n_steps, final_backup=True, state_limit_action="STOP"),
    )


def build_case(name):
    if name == "single_ring":
        return dns_case(name, n_steps=SINGLE_RING_STEPS, direct=True)
    if name.startswith("high_re"):
        case = dns_case(
            name, reynolds_number=HIGH_REYNOLDS_NUMBER, n_steps=HIGH_RE_STEPS, direct=True
        )
    else:
        case = dns_case(name)
    numerics = case.numerics
    if name.endswith(("les", "les_stabilized")):
        # Sigma model, active only where a particle's strength departs from the
        # resolved vorticity, applied by strength exchange so cores stay uniform.
        numerics = replace(numerics, turbulence=vpm.TurbulenceConfig.les_sigma())
    if name == "les_stabilized":
        # Pedrizzetti relaxation turns each strength toward the resolved vorticity;
        # net strength and impulse are restored after every application.
        stabilization = replace(
            numerics.stabilization,
            pedrizzetti_relaxation_factor=RELAXATION_FACTOR,
            pedrizzetti_relaxation_preserve_vortex_strength=False,
            pedrizzetti_relaxation_preserve_moments=True,
        )
        numerics = replace(numerics, stabilization=stabilization)
    return replace(case, numerics=numerics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", nargs="?", default="dns", choices=CASES)
    case = build_case(parser.parse_args().case)
    vpm.VPMSolver(case).run(start_from=START_FROM)
