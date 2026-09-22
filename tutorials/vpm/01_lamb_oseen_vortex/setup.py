#!/usr/bin/env python3
"""Run a Lamb--Oseen vortex, dipole, or merging pair.

Examples (from this case directory)::

    python setup.py vortex CS
    python setup.py dipole DVH
    python setup.py merging GBD
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

import openonda.vpm as vpm
from openonda.tutorial_runner import case_package
from openonda.vpm import Backup, Samplers

__package__ = case_package(Path(__file__).parent)

# Physics (Lamb--Oseen benchmark)
START_FROM = "latest"  # Resume the latest backup; ./allrun.sh cleans first.

CIRCULATION_REYNOLDS_NUMBER = 530.0  # Re_Γ = |Γ|/ν — sets the vortex Reynolds number
BETA_RMAX = 1.12  # r(u_θ,max)/a — velocity-peak radius / Gaussian core radius
CORE_RADIUS = 0.125  # a₀ — initial velocity-peak radius [m] (defines the analytic profile)
GAUSSIAN_CORE_RADIUS = CORE_RADIUS / BETA_RMAX  # Gaussian 1/e vorticity radius [m]
SEPARATION = 1.0  # distance between the two vortex centres [m]
COLUMN_LENGTH = 40.0 * CORE_RADIUS  # finite vortex column length along z [m]

# Numerical setup shared by every viscous scheme
SPACING = 0.60 * CORE_RADIUS  # 2k--4k initial particles
CORE_RADIUS_RATIO = 1.2  # DVH/GBD core radius ratio for regeneration
PARTICLE_RADIUS = CORE_RADIUS_RATIO * SPACING  # vortex particle core radius (1.2× spacing)
FIELD_SPACING = 0.15 * CORE_RADIUS  # sampling field resolution for surface output
TIME_STEP_SIZE = 0.291 / 9.0  # Δt [s]
TOTAL_TIME = 103.0 * 0.291  # total simulation time [s]
SAMPLE_INTERVAL_TIME = 2.0 * 0.291  # time between field samples [s]
MERGING_SAMPLE_INTERVAL_STEPS = 6  # resolve the rapid final collapse of the two vorticity peaks
BACKUP_INTERVAL_TIME = 10.0 * 0.291  # time between snapshots [s]
INITIAL_STRENGTH_CUTOFF = 1e-4  # discard particles with Γ < x% of peak
MAX_PARTICLES = 400_000  # particle-container capacity (largest DVH/GBD population)

VISCOUS_SCHEMES = ("CS", "DVH", "GBD")
ALL_VISCOUS_SCHEMES = ("CS", "RWM", "DVH", "GBD")
RWM_ENSEMBLE_SIZE = 10

# Backend selection is independent of the stretching formulation.
STRETCHING_SCHEME = "transposed"  # "direct", "mixed", or "transposed"

COMPUTE_METHOD = {
    "CS": "DIRECT",
    "RWM": "DIRECT",
    "DVH": "TREECODE",
    "GBD": "TREECODE",
}

# Physical case definitions
PHYSICS_CIRCULATIONS = {
    "vortex": (+1.0,),
    "dipole": (+1.0, -1.0),
    "merging": (+1.0, +1.0),
}


TUTORIAL_DIR = Path(__file__).resolve().parent


def viscous_config(scheme: str, kinematic_viscosity: float, spacing: float) -> vpm.ViscousConfig:
    return {
        "CS": vpm.ViscousConfig.cs(
            kinematic_viscosity=kinematic_viscosity,
            particle_spacing=spacing,
        ),
        "RWM": vpm.ViscousConfig.rwm(
            kinematic_viscosity=kinematic_viscosity,
            particle_spacing=spacing,
        ),
        "DVH": vpm.ViscousConfig.dvh(
            particle_spacing=spacing,
            padding=5,
            kinematic_viscosity=kinematic_viscosity,
            dvh_support_radius_ratio=4,
            threshold=1e-4,
            threshold_mode="budget",
            max_nodes=MAX_PARTICLES,
            core_radius_ratio=CORE_RADIUS_RATIO,
        ),
        "GBD": vpm.ViscousConfig.gbd(
            particle_spacing=spacing,
            padding=5,
            kinematic_viscosity=kinematic_viscosity,
            threshold=1e-4,
            threshold_mode="budget",
            max_nodes=MAX_PARTICLES,
            core_radius_ratio=CORE_RADIUS_RATIO,
        ),
    }[scheme.upper()]


def induction_config(scheme: str):
    """Select an induction backend while keeping stretching independently configurable."""
    methods = {
        "DIRECT": vpm.DirectInduction,
        "TREECODE": vpm.TreecodeInduction,
        "FMM": vpm.FMMInduction,
    }
    return methods[COMPUTE_METHOD[scheme.upper()]](stretching_scheme=STRETCHING_SCHEME)


def _initial_conditions(physics: str, kinematic_viscosity: float):
    """Build the shared initial-condition descriptions and their geometry."""
    circulations = PHYSICS_CIRCULATIONS[physics]
    y_positions = (0.0,) if physics == "vortex" else (SEPARATION / 2, -SEPARATION / 2)
    initial_half_width = max(abs(y) for y in y_positions) + 7.0 * CORE_RADIUS
    column_half_length = COLUMN_LENGTH / 2.0
    initial_bounds = [
        -initial_half_width,
        initial_half_width,
        -initial_half_width,
        initial_half_width,
        -column_half_length,
        column_half_length,
    ]
    distribution = vpm.TriangularPrismDistribution(
        bounds=(
            (initial_bounds[0], initial_bounds[1]),
            (initial_bounds[2], initial_bounds[3]),
            (initial_bounds[4], initial_bounds[5]),
        ),
        spacing=SPACING,
        core_radius_ratio=PARTICLE_RADIUS / SPACING,
    )
    conditions = tuple(
        vpm.VortexFilament(
            kinematic_viscosity=kinematic_viscosity,
            centre=(0.0, y_position, 0.0),
            direction=(0.0, 0.0, 1.0),
            circulation=circulation,
            vortex_core_radius=GAUSSIAN_CORE_RADIUS,
            distribution=distribution,
            group_id=group_id,
            core_compensation=vpm.ParticleCoreCompensation(),
            tail_minimum_relative_strength=INITIAL_STRENGTH_CUTOFF,
            tail_circulation_per_length=circulation,
            tail_represented_length=COLUMN_LENGTH,
        )
        for group_id, (circulation, y_position) in enumerate(
            zip(circulations, y_positions, strict=True)
        )
    )
    return conditions, circulations, y_positions, initial_half_width, column_half_length


def run_case(
    physics: str,
    scheme: str,
    *,
    name: str | None = None,
    random_seed: int = 42,
    surfaces: bool = True,
    backup_steps: int | None = None,
    compute_device: str = "AUTO",
) -> None:
    scheme = scheme.upper()
    case_name = name or f"{physics}_{scheme.lower()}"
    # Derived physical quantities
    spacing = SPACING
    field_spacing = FIELD_SPACING
    circulations = PHYSICS_CIRCULATIONS[physics]
    circulation = abs(circulations[0])
    kinematic_viscosity = circulation / CIRCULATION_REYNOLDS_NUMBER  # ν = |Γ|/Re_Γ
    viscous = viscous_config(scheme, kinematic_viscosity, spacing)

    # Time stepping
    sample_steps = round(SAMPLE_INTERVAL_TIME / TIME_STEP_SIZE)
    if backup_steps is None:
        backup_steps = round(BACKUP_INTERVAL_TIME / TIME_STEP_SIZE)
    field_interval_steps = MERGING_SAMPLE_INTERVAL_STEPS if physics == "merging" else sample_steps

    # Initial vortex geometry
    (
        initial_conditions,
        circulations,
        y_positions,
        initial_half_width,
        column_half_length,
    ) = _initial_conditions(physics, kinematic_viscosity)
    # Domain sizing (must contain vortex at t=end_time)
    final_core_radius = BETA_RMAX * np.sqrt(
        GAUSSIAN_CORE_RADIUS**2 + 4.0 * kinematic_viscosity * TOTAL_TIME
    )
    padding = (
        0.0 if physics == "vortex" else 4.0 * final_core_radius
    )  # extra room for dipole/merging spread
    lateral_half_width = (
        initial_half_width if physics == "vortex" else max(abs(y) for y in y_positions) + padding
    )
    diffusion_padding = 3.6 * np.sqrt(4.0 * kinematic_viscosity * TOTAL_TIME)
    lateral_half_width = max(lateral_half_width, initial_half_width + diffusion_padding)
    field_padding = 0.0 if physics == "vortex" else 3.0 * final_core_radius
    field_lateral_half_width = (
        initial_half_width
        if physics == "vortex"
        else max(abs(y) for y in y_positions) + field_padding
    )
    axial_half_length = column_half_length + max(padding, diffusion_padding)
    downstream_length = (
        8.0 * SEPARATION if physics == "dipole" else 0.0
    )  # dipole advects downstream

    domain_bounds = [
        -lateral_half_width,
        lateral_half_width + downstream_length,
        -lateral_half_width,
        lateral_half_width,
        -axial_half_length,
        axial_half_length,
    ]
    field_bounds = [
        -field_lateral_half_width,
        field_lateral_half_width + downstream_length,
        -field_lateral_half_width,
        field_lateral_half_width,
    ]
    # Field samplers
    sample_plane_fraction = 0.25  # sample at z = L/4
    integral_interval_steps = field_interval_steps
    if scheme == "DVH":
        diffusion_steps = math.ceil(viscous.dvh_required_time_step_size() / TIME_STEP_SIZE)
        # Each energy secant must include an actual heat transfer. Between
        # transfers DVH advances only advection, whose small discretization
        # drift is not a viscous energy-rate estimate. Keep field output dense.
        integral_interval_steps *= math.ceil(diffusion_steps / integral_interval_steps)
    samplers = [vpm.FlowIntegralsSampler(schedule=vpm.EverySteps(integral_interval_steps))]
    if surfaces:
        samplers.append(
            vpm.SurfaceSampler(
                point=[0, 0, sample_plane_fraction * COLUMN_LENGTH],
                normal=[0, 0, 1],
                bounds=field_bounds,
                spacing=field_spacing,
                file_name=f"{case_name}_zq",
                include_derivatives=False,
                schedule=vpm.EverySteps(field_interval_steps),
            )
        )

    sample_directory = case_name
    solution_directory = Path("solution") / case_name

    n_steps = round(TOTAL_TIME / TIME_STEP_SIZE)

    case = vpm.VPMCase(
        name=case_name,
        numerics=vpm.Numerics(
            time_step_size=TIME_STEP_SIZE,
            viscous=viscous,
            integrator=vpm.RK2(),
            induction=induction_config(scheme),
            particle_kernel="GAUSSIAN",
            write_precision="f32",
            precision="f32",
            max_n_particles=MAX_PARTICLES,
            domain_bounds=domain_bounds,
            compute_device=compute_device,
            random_seed=random_seed,
        ),
        initial_conditions=initial_conditions,
        backup=Backup(
            interval_steps=backup_steps,
            directory=str(solution_directory),
            log_directory=str(solution_directory),
        ),
        samplers=Samplers(samples=tuple(samplers), directory=sample_directory),
        run=vpm.RunPlan(steps=n_steps),
        directory=TUTORIAL_DIR,
    )

    solver = vpm.VPMSolver(case)
    solver.run(start_from=START_FROM)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", nargs="?", default="vortex", choices=tuple(PHYSICS_CIRCULATIONS))
    parser.add_argument("viscous_scheme", nargs="?", default="CS", choices=ALL_VISCOUS_SCHEMES)
    parser.add_argument("--ensemble", action="store_true", help="Run/resume independent RWM seeds")
    parser.add_argument("--number-of-realizations", type=int, default=RWM_ENSEMBLE_SIZE)
    parser.add_argument("--converge", action="store_true")
    parser.add_argument("--maximum-realizations", type=int, default=80)
    args = parser.parse_args()
    if args.ensemble:
        if args.viscous_scheme != "RWM":
            parser.error("--ensemble requires RWM")
        from .assets.rwm_ensemble import run_converged_ensemble, run_ensemble

        if args.converge:
            run_converged_ensemble(args.case, args.number_of_realizations, 42000, args.maximum_realizations)
        else:
            run_ensemble(args.case, args.number_of_realizations)
    else:
        run_case(args.case, args.viscous_scheme)
