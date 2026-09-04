#!/usr/bin/env python3
"""Lamb--Oseen vortex benchmark: single vortex, vortex dipole, and merging pair.

The Lamb--Oseen vortex is an exact solution of the two-dimensional
Navier--Stokes equations for an isolated vortex, so the computed flow can be
compared directly with the analytic profile.  This script runs the chosen
benchmark case (single vortex, vortex dipole, or merging vortex pair) with the
chosen diffusion scheme and stores the snapshots under ``solution`` while the
sampled z=L/4 velocity/vorticity fields and flow integrals are written under
``samples``.

The user-facing entry point accepts the physical case, viscous scheme, and an
optional induction method::

    python -m tutorials.vpm.lamb_oseen_vortex.setup vortex CS DIRECT
    python -m tutorials.vpm.lamb_oseen_vortex.setup dipole DVH TREECODE
    python -m tutorials.vpm.lamb_oseen_vortex.setup merging GBD TREECODE
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import openonda.vpm as vpm
from openonda.vpm import Backup, Samplers

TUTORIAL_DIR = Path(__file__).resolve().parent

# ---- Physics (Lamb--Oseen benchmark) ------------------------------------
CIRCULATION_REYNOLDS_NUMBER = 530.0  # Re_Γ = |Γ|/ν — sets the vortex Reynolds number
BETA_RMAX = 1.12  # r(u_θ,max)/a — velocity-peak radius / Gaussian core radius
CORE_RADIUS = 0.125  # a₀ — initial velocity-peak radius [m] (defines the analytic profile)
GAUSSIAN_CORE_RADIUS = CORE_RADIUS / BETA_RMAX  # Gaussian 1/e vorticity radius [m]
SEPARATION = 1.0  # distance between the two vortex centres [m]
COLUMN_LENGTH = 40.0 * CORE_RADIUS  # finite vortex column length along z [m]

# ---- Numerical setup shared by every viscous scheme ----------------------
SPACING = 0.45 * CORE_RADIUS  # particle spacing in every direction (controls resolution)
CORE_RADIUS_RATIO = 1.2  # DVH/GBD core radius ratio for regeneration
PARTICLE_RADIUS = CORE_RADIUS_RATIO * SPACING  # vortex particle core radius (1.5× spacing)
FIELD_SPACING = 0.15 * CORE_RADIUS  # sampling field resolution for surface output
TIME_STEP_SIZE = 0.291 / 9.0  # Δt [s]
TOTAL_TIME = 103.0 * 0.291  # total simulation time [s]
SAMPLE_INTERVAL_TIME = 2.0 * 0.291  # time between field samples [s]
MERGING_SAMPLE_INTERVAL_STEPS = 6  # resolve the rapid final collapse of the two vorticity peaks
BACKUP_INTERVAL_TIME = 10.0 * 0.291  # time between snapshots [s]
INITIAL_STRENGTH_CUTOFF = 1e-4  # discard particles with Γ < x% of peak
MAX_PARTICLES = 400_000  # particle-container capacity (largest DVH/GBD population)

VISCOUS_SCHEMES = ("CS", "DVH", "GBD")
INDUCTION_METHODS = ("DIRECT", "TREECODE", "FMM")

# ---- Physical case definitions -------------------------------------------
PHYSICS_CIRCULATIONS = {
    "vortex": (+1.0,),
    "dipole": (+1.0, -1.0),
    "merging": (+1.0, +1.0),
}


def viscous_config(scheme: str, kinematic_viscosity: float, spacing: float) -> vpm.ViscousConfig:
    if scheme == "CS":
        return vpm.ViscousConfig.cs(
            kinematic_viscosity=kinematic_viscosity,
            particle_spacing=spacing,
        )
    if scheme == "RWM":
        return vpm.ViscousConfig.rwm(
            kinematic_viscosity=kinematic_viscosity,
            particle_spacing=spacing,
        )
    if scheme == "DVH":
        return vpm.ViscousConfig.dvh(
            particle_spacing=spacing,
            padding=5,
            kinematic_viscosity=kinematic_viscosity,
            dvh_support_radius_ratio=4,
            threshold=1e-4,
            threshold_mode="budget",
            max_nodes=MAX_PARTICLES,
            core_radius_ratio=CORE_RADIUS_RATIO,
        )
    return vpm.ViscousConfig.gbd(
        particle_spacing=spacing,
        padding=5,
        kinematic_viscosity=kinematic_viscosity,
        threshold=1e-4,
        threshold_mode="budget",
        max_nodes=MAX_PARTICLES,
        core_radius_ratio=CORE_RADIUS_RATIO,
    )


def induction_config(method: str):
    """Construct the explicitly selected particle-induction backend."""
    method = method.upper()
    if method == "DIRECT":
        return vpm.DirectInduction()
    if method == "TREECODE":
        return vpm.TreecodeInduction()
    if method == "FMM":
        return vpm.FMMInduction()
    raise ValueError(f"Unknown induction method {method!r}; expected one of {INDUCTION_METHODS}")


def write_run_metadata(
    *,
    physics: str,
    scheme: str,
    sample_directory: str,
    circulations: tuple[float, ...],
    kinematic_viscosity: float,
    spacing: float,
    particle_core_radius: float,
    field_spacing: float,
    n_steps: int,
    random_seed: int,
    solver,
) -> None:
    """
    This is for post-processing
    """
    metadata = {
        "schema_version": 1,
        "status": "complete",
        "completed": True,
        "case": physics,
        "scheme": scheme.lower(),
        "circulations": [float(value) for value in circulations],
        "kinematic_viscosity": float(kinematic_viscosity),
        "circulation_reynolds_number": float(CIRCULATION_REYNOLDS_NUMBER),
        "core_radius": float(GAUSSIAN_CORE_RADIUS),
        "gaussian_core_radius": float(GAUSSIAN_CORE_RADIUS),
        "velocity_peak_radius": float(CORE_RADIUS),
        "velocity_peak_radius_factor": float(BETA_RMAX),
        "vortex_separation": float(SEPARATION),
        "vortex_column_length": float(COLUMN_LENGTH),
        "column_length": float(COLUMN_LENGTH),
        "column_half_length": float(COLUMN_LENGTH / 2.0),
        "particle_spacing": float(spacing),
        "particle_core_radius": float(particle_core_radius),
        "field_spacing": float(field_spacing),
        "sample_plane_fraction": 0.25,
        "sample_plane_z": 0.25 * float(COLUMN_LENGTH),
        "time_step_size": float(TIME_STEP_SIZE),
        "total_time": float(TOTAL_TIME),
        "end_time": float(TOTAL_TIME),
        "number_of_steps": int(n_steps),
        "integrator": "SSPRK3",
        "induction_backend": type(solver.induction).__name__,
        "strength_rate_formulation": solver.induction.strength_rate_mode,
        "particle_kernel": "GAUSSIAN",
        "diffusion_scheme": scheme,
        "compute_backend": getattr(solver, "compute_device", "AUTO"),
        "precision": "f32",
        "write_precision": "f32",
        "random_seed": int(random_seed),
        "final_time": float(solver.time),
        "initial_n_particles_total": int(getattr(solver.particles, "n_particles_total", 0)),
    }
    destination = TUTORIAL_DIR / "samples" / sample_directory / "run_metadata.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


def run_case(
    physics: str,
    scheme: str,
    *,
    name: str | None = None,
    random_seed: int = 42,
    surfaces: bool = True,
    backup_steps: int | None = None,
    compute_device: str = "AUTO",
    induction_method: str | None = None,
) -> None:
    scheme = scheme.upper()
    induction_method = (
        ("DIRECT" if scheme in {"CS", "RWM"} else "TREECODE")
        if induction_method is None
        else induction_method.upper()
    )
    case_name = name or f"{physics}_{scheme.lower()}"
    circulations = PHYSICS_CIRCULATIONS[physics]

    # ---- Derived physical quantities ----
    spacing = SPACING
    particle_core_radius = 1.5 * spacing
    field_spacing = FIELD_SPACING
    circulation = abs(circulations[0])
    kinematic_viscosity = circulation / CIRCULATION_REYNOLDS_NUMBER  # ν = |Γ|/Re_Γ
    viscous = viscous_config(scheme, kinematic_viscosity, spacing)

    # ---- Time stepping ----
    sample_steps = round(SAMPLE_INTERVAL_TIME / TIME_STEP_SIZE)
    if backup_steps is None:
        backup_steps = round(BACKUP_INTERVAL_TIME / TIME_STEP_SIZE)
    field_interval_steps = MERGING_SAMPLE_INTERVAL_STEPS if physics == "merging" else sample_steps

    # ---- Initial vortex geometry ----
    y_positions = (0.0,) if physics == "vortex" else (SEPARATION / 2, -SEPARATION / 2)
    initial_half_width = max(abs(y) for y in y_positions) + 7.0 * CORE_RADIUS
    column_half_length = COLUMN_LENGTH / 2.0

    # ---- Domain sizing (must contain vortex at t=end_time) ----
    final_core_radius = BETA_RMAX * np.sqrt(
        GAUSSIAN_CORE_RADIUS**2 + 4.0 * kinematic_viscosity * TOTAL_TIME
    )
    padding = (
        0.0 if physics == "vortex" else 4.0 * final_core_radius
    )  # extra room for dipole/merging spread
    lateral_half_width = (
        initial_half_width if physics == "vortex" else max(abs(y) for y in y_positions) + padding
    )
    field_padding = 0.0 if physics == "vortex" else 3.0 * final_core_radius
    field_lateral_half_width = (
        initial_half_width
        if physics == "vortex"
        else max(abs(y) for y in y_positions) + field_padding
    )
    axial_half_length = column_half_length + padding
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
    initial_bounds = [
        -initial_half_width,
        initial_half_width,
        -initial_half_width,
        initial_half_width,
        -column_half_length,
        column_half_length,
    ]
    # ---- Particle distribution and field sampler ----
    distribution = vpm.TriangularPrismDistribution(
        bounds=(
            (initial_bounds[0], initial_bounds[1]),
            (initial_bounds[2], initial_bounds[3]),
            (initial_bounds[4], initial_bounds[5]),
        ),
        spacing=spacing,
        core_radius_ratio=particle_core_radius / spacing,
    )

    sample_plane_fraction = 0.25  # sample at z = L/4
    samplers = [vpm.FlowIntegralsSampler(schedule=vpm.EverySteps(field_interval_steps))]
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

    initial_conditions = tuple(
        vpm.VortexFilament(
            kinematic_viscosity=kinematic_viscosity,
            centre=(0.0, y_position, 0.0),
            direction=(0.0, 0.0, 1.0),
            circulation=circ,
            vortex_core_radius=GAUSSIAN_CORE_RADIUS,
            distribution=distribution,
            group_id=group_id,
            core_compensation=vpm.ParticleCoreCompensation(),
            tail_minimum_relative_strength=INITIAL_STRENGTH_CUTOFF,
            tail_circulation_per_length=circ,
            tail_represented_length=COLUMN_LENGTH,
        )
        for group_id, (circ, y_position) in enumerate(zip(circulations, y_positions, strict=True))
    )
    n_steps = round(TOTAL_TIME / TIME_STEP_SIZE)

    case = vpm.VPMCase(
        numerics=vpm.Numerics(
            time_step_size=TIME_STEP_SIZE,
            viscous=viscous,
            integrator=vpm.SSPRK3(),
            induction=induction_config(induction_method),
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
    solver.run()

    write_run_metadata(
        physics=physics,
        scheme=scheme,
        sample_directory=sample_directory,
        circulations=circulations,
        kinematic_viscosity=kinematic_viscosity,
        spacing=spacing,
        particle_core_radius=particle_core_radius,
        field_spacing=field_spacing,
        n_steps=n_steps,
        random_seed=random_seed,
        solver=solver,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=tuple(PHYSICS_CIRCULATIONS))
    parser.add_argument("viscous_scheme", choices=VISCOUS_SCHEMES)
    parser.add_argument("induction_method", nargs="?", choices=INDUCTION_METHODS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_case(args.case, args.viscous_scheme, induction_method=args.induction_method)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
