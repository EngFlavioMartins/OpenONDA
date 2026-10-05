#!/usr/bin/env python3
"""Coupled FVM–VPM flow past a spanwise cylinder section at Re = 150.

The body-fitted FVM resolves the cylinder and a compact near-body box. The
VPM carries vorticity through the outer domain. The FVM force is normalized
by the resolved span.

Usage:
    ./allrun.sh
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import openonda.coupler as coupling
import openonda.fvm as fvm
import openonda.fvm.mesher as msh
import openonda.vpm as vpm
from openonda.vpm import Backup, Samplers
from openonda.tutorial_runner import case_package

if not __package__:
    __package__ = case_package(Path(__file__).resolve().parent)

from .assets.startup import run_coupled_cylinder
from .assets.configuration import (
    align_cylinder_sampling,
    positive_coupling_steps,
    steps,
    validate_inputs,
    validate_authority,
)

# Physical problem
START_FROM = "latest"  # allrun.sh preserves outputs; allclean.sh is explicit.

CASE_NAME = "coupled_cylinder_flow"
DIAMETER = 1.0
FREESTREAM_VELOCITY = (1.0, 0.0, 0.0)
STARTUP_FREESTREAM_VELOCITY = (1.0, 0.1, 0.0)
STARTUP_DURATION = 2.0
STARTUP_TRANSITION_DURATION = 1.0
INITIAL_PERTURBATION = {"amplitude": 1e-3, "radius": 0.5, "centre": 0.65}
DENSITY = 1.0
REYNOLDS_NUMBER = 150.0
KINEMATIC_VISCOSITY = np.linalg.norm(FREESTREAM_VELOCITY) * DIAMETER / REYNOLDS_NUMBER

# FVM domain and mesh
FVM_CORES = 4
CELL_SIZE = 0.04
FVM_RESOLVED_SPAN = 0.96
FVM_HALF_SPAN = 0.5 * FVM_RESOLVED_SPAN
FVM_BOX = (
    -1.60,
    2.40,
    -1.60,
    1.60,
    -FVM_HALF_SPAN,
    FVM_HALF_SPAN,
)
TRANSFER_REGION_BOX = (
    -1.25,
    2.05,
    -1.25,
    1.25,
    -FVM_HALF_SPAN,
    FVM_HALF_SPAN,
)

# VPM domain and resolution.
VPM_DOMAIN = (*(-5.0, 15.0, -5.0, 5.0), -FVM_HALF_SPAN, FVM_HALF_SPAN)
# Keep renewal and grid-based diffusion on one VPM lattice.
PARTICLE_LIMIT = 1_000_000
PARTICLE_SPACING_RATIO = 1.0
CORE_RADIUS_RATIO = 1.0
BLEND_WIDTH_RATIO = 6.0
RELEASE_WIDTH_RATIO = 2.0
COMPUTE_DEVICE = "AUTO"
INTERFACE_ITERATIONS = 3
INTERFACE_TOLERANCE = 1.0e-5

# Time and output
FVM_TIME_STEP_SIZE = 0.008
VPM_TIME_STEP_MULTIPLIER = 5
VPM_TIME_STEP_SIZE = VPM_TIME_STEP_MULTIPLIER * FVM_TIME_STEP_SIZE
END_TIME = 100.0
STATISTICS_START = 40.0
GBD_VORTICITY_FLOOR = 0.01

# Physical output intervals
FORCES = 0.04
PROFILES = 0.2
SLICES = 0.4
BACKUPS = 1.0
VOLUMES = 4.0

# Case files and derived sampling data
CASE_DIR = Path(__file__).resolve().parent
CYLINDER_STL = CASE_DIR / "assets" / "cylinder_long.stl"

FVM_PATCHES = msh.BoxPatches(
    xmin="numericalBoundary",
    xmax="numericalBoundary",
    ymin="numericalBoundary",
    ymax="numericalBoundary",
    zmin="zmin",
    zmax="zmax",
)


def phase_lines(outer=False):
    """Identical coordinates in the reference and corresponding coupled region."""
    lo, hi, count = (1.6, 8.0, 65) if outer else (0.6, 1.4, 9)
    region = "wake" if outer else "near"
    return [
        (f"phase_{region}_{side}", [lo, y, 0.0], [hi, y, 0.0], count)
        for side, y in (("upper", 0.6), ("lower", -0.6))
    ]


def field_lines(reference, *, span):
    lines = [("centreline", [0.6, 0.0, 0.0], [1.4, 0.0, 0.0], 11)]
    for label, z in (("lower", -span / 4), ("middle", 0.0), ("upper", span / 4)):
        lines.append((f"span_{label}", [1.0, -1.2, z], [1.0, 1.2, z], 31))
    if reference:
        for x in (2, 4):
            lines.append((f"transverse_x{x}", [x, -2.0, 0.0], [x, 2.0, 0.0], 51))
    return lines


def sampling_plan(period, *, end, exchange_dt, fvm_time_step):
    """Resolve physical periods onto the actual accepted exchange clock."""
    steps(exchange_dt, fvm_time_step)
    steps(end, exchange_dt)
    return align_cylinder_sampling(
        end_time=end,
        exchange_dt=exchange_dt,
        fvm_time_step=fvm_time_step,
        sample_period=period,
        slice_period=SLICES,
        output_period=VOLUMES,
    )


def fvm_samplers(reference, *, span, freestream_speed, diameter, end, exchange_dt, fvm_time_step):
    clocks = {"end": end, "exchange_dt": exchange_dt, "fvm_time_step": fvm_time_step}
    force, profile = sampling_plan(FORCES, **clocks), sampling_plan(PROFILES, **clocks)
    fast = fvm.RunSchedule(every_n_steps=force.fvm_sample_steps)
    slow = fvm.RunSchedule(every_n_steps=profile.fvm_sample_steps)
    samplers = [
        fvm.ForceSampler(
            patch_names=["cylinder"],
            reference_velocity=freestream_speed,
            reference_area=diameter * span,
            reference_length=diameter,
            file_name="forces_history",
            schedule=fast,
        )
    ]
    for name, start, end, count in phase_lines() + (phase_lines(True) if reference else []):
        samplers.append(
            fvm.LineSampler(
                start=start,
                end=end,
                n_points=count,
                k=12,
                reconstruction="affine",
                file_name=name,
                schedule=fast,
            )
        )
    for name, start, end, count in field_lines(reference, span=span):
        samplers.append(
            fvm.LineSampler(
                start=start,
                end=end,
                n_points=count,
                k=12,
                reconstruction="affine",
                file_name=name,
                schedule=slow,
            )
        )
    samplers.append(
        fvm.SurfaceSampler(
            point=[0.0, 0.0, 0.0],
            normal=[0.0, 0.0, 1.0],
            bounds=[
                -1.6,
                8.0 if reference else 1.6,
                -2.0 if reference else -1.6,
                2.0 if reference else 1.6,
            ],
            spacing=0.1,
            file_name="midspan",
            schedule=fvm.RunSchedule(every_n_steps=profile.fvm_slice_steps),
            body_bounds=[-0.5, 0.5, -0.5, 0.5, -6.0, 6.0],
            body_geometry="cylinder_z",
        )
    )
    return tuple(samplers)


def vpm_samplers(*, end, exchange_dt, fvm_time_step):
    """Particle-side observations shared by the ordinary tutorial entry point."""
    clocks = {"end": end, "exchange_dt": exchange_dt, "fvm_time_step": fvm_time_step}
    force, profile = sampling_plan(FORCES, **clocks), sampling_plan(PROFILES, **clocks)
    substeps = steps(exchange_dt, fvm_time_step)
    samples = []
    for name, start, finish, count in phase_lines() + phase_lines(True):
        samples.append(
            vpm.LineSampler(
                start=start,
                end=finish,
                spacing=math.dist(start, finish) / (count - 1) * (1 + 1e-12),
                file_name="vpm_" + name,
                schedule=vpm.EverySteps(force.sample_steps),
            )
        )
    for x in (2, 4):
        samples.append(
            vpm.LineSampler(
                start=[x, -2.0, 0.0],
                end=[x, 2.0, 0.0],
                spacing=0.08 * (1 + 1e-12),
                file_name=f"vpm_transverse_x{x}",
                schedule=vpm.EverySteps(profile.sample_steps),
            )
        )
    samples.append(
        vpm.SurfaceSampler(
            point=[0.0, 0.0, 0.0],
            normal=[0.0, 0.0, 1.0],
            bounds=[1.6, 8.0, -2.0, 2.0],
            spacing=0.1,
            file_name="vpm_midspan",
            include_derivatives=False,
            schedule=vpm.EverySteps(profile.fvm_slice_steps // substeps),
        )
    )
    return tuple(samples)


def build_case(
    *,
    end_time: float | None = None,
    overrides: dict[str, object] | None = None,
):
    """Construct the physical case, mesh and accepted-time schedules."""
    values = dict(overrides or {})
    hxy = float(values.pop("hxy", CELL_SIZE))
    span = float(values.pop("span", FVM_RESOLVED_SPAN))
    dz_target = float(values.pop("dz", hxy))
    hp_ratio = float(values.pop("particle_spacing_ratio", PARTICLE_SPACING_RATIO))
    core_ratio = float(values.pop("core_radius_ratio", CORE_RADIUS_RATIO))
    blend_ratio = float(values.pop("blend_width_ratio", BLEND_WIDTH_RATIO))
    release_ratio = float(values.pop("release_width_ratio", RELEASE_WIDTH_RATIO))
    exchange_dt = float(values.pop("exchange_dt", VPM_TIME_STEP_SIZE))
    cores = int(values.pop("cores", FVM_CORES))
    compute_device = str(values.pop("compute_device", COMPUTE_DEVICE))
    particle_limit = int(values.pop("particle_limit", PARTICLE_LIMIT))
    physical_end = END_TIME if end_time is None else float(end_time)
    validate_inputs(
        hxy,
        span,
        dz_target,
        hp_ratio,
        core_ratio,
        exchange_dt,
        physical_end,
        release_ratio,
        blend_ratio,
        cores,
        particle_limit,
    )
    steps(exchange_dt, FVM_TIME_STEP_SIZE)
    steps(physical_end, exchange_dt)
    half_span = span / 2.0
    fvm_box = (*FVM_BOX[:4], -half_span, half_span)
    transfer_box = (*TRANSFER_REGION_BOX[:4], -half_span, half_span)
    particle_spacing = span / max(6, math.ceil(span / (hxy * hp_ratio)))
    authority_edge = min(-transfer_box[0], transfer_box[1], -transfer_box[2], transfer_box[3])
    validate_authority(authority_edge, blend_ratio * particle_spacing, DIAMETER / 2, 1e-6)
    axial_layers = max(4, math.ceil(span / dz_target))
    realized_dz = span / axial_layers
    vpm_domain = (*VPM_DOMAIN[:4], -half_span, half_span)

    source_half_span = 16.0 * hxy
    source_mesh = msh.CartesianMesher(
        domain=msh.BoxDomain(
            bounds=(*fvm_box[:4], -source_half_span, source_half_span),
            patches=FVM_PATCHES,
        ),
        surfaces=(msh.STLSurface(CYLINDER_STL, patch="cylinder"),),
        max_cell_size=hxy,
        cell_size_anchor=hxy,
        patch_refinements=(msh.PatchRefinement("cylinder", hxy),),
        surface_may_cross_domain_boundary=True,
    )
    mesh = msh.ExtrudedCartesianMesher(
        source=source_mesh,
        domain=msh.BoxDomain(
            bounds=(*source_mesh.domain.bounds[:4], -half_span, half_span),
            patches=FVM_PATCHES,
        ),
        levels=tuple(-half_span + layer * realized_dz for layer in range(axial_layers + 1)),
    )

    clocks = dict(end=physical_end, exchange_dt=exchange_dt, fvm_time_step=FVM_TIME_STEP_SIZE)
    sampling = sampling_plan(PROFILES, **clocks)
    fvm_setup = fvm.FVMSetup(
        case_name=CASE_NAME,
        cores=cores,
        execution=fvm.ComputeConfig(operator_backend="numba"),
        output=fvm.OutputConfig(compression="lz4", ghost_layers=0),
        time=fvm.TimeConfig(
            time_step_size=FVM_TIME_STEP_SIZE,
            end_time=physical_end,
            output_schedule=fvm.RunSchedule(every_n_steps=sampling.fvm_output_steps),
        ),
        schemes=fvm.DiscretizationConfig(
            convection_scheme="limitedLinear", gradient_scheme="lsq", time_scheme="backward"
        ),
        linear=fvm.LinearSolverConfig(
            pressure_solver="amg",
            pressure_tolerance=1.0e-7,
            pressure_relative_tolerance=0.005,
            momentum_tolerance=1.0e-6,
            momentum_relative_tolerance=0.05,
        ),
        pimple=fvm.PimpleControl(
            n_outer_correctors=2, n_correctors=2, velocity_relaxation=0.7, pressure_relaxation=0.3
        ),
        backup=fvm.BackupConfig(schedule=None, write_at_end=False),
        samplers=fvm_samplers(
            False,
            span=span,
            freestream_speed=float(np.linalg.norm(FREESTREAM_VELOCITY)),
            diameter=DIAMETER,
            **clocks,
        ),
        transport=fvm.TransportConfig(density=DENSITY, kinematic_viscosity=KINEMATIC_VISCOSITY),
        turbulence=fvm.TurbulenceConfig.none(),
        boundaries=[
            fvm.BoundaryConfig(
                name="numericalBoundary",
                velocity_type="fixedValue",
                velocity_value=list(STARTUP_FREESTREAM_VELOCITY),
                pressure_type="fixedFluxPressure",
            ),
            fvm.BoundaryConfig.slip("zmin"),
            fvm.BoundaryConfig.slip("zmax"),
            fvm.BoundaryConfig.wall("cylinder"),
        ],
        initial_velocity=list(STARTUP_FREESTREAM_VELOCITY),
    )
    vpm_case = vpm.VPMCase(
        name=CASE_NAME,
        numerics=vpm.Numerics(
            time_step_size=exchange_dt,
            compute_device=compute_device,
            freestream_velocity=STARTUP_FREESTREAM_VELOCITY,
            viscous=vpm.ViscousConfig.gbd(
                particle_spacing=particle_spacing,
                padding=5.0,
                kinematic_viscosity=KINEMATIC_VISCOSITY,
                threshold_mode="absolute",
                threshold=GBD_VORTICITY_FLOOR * particle_spacing**3,
                core_radius_ratio=core_ratio,
            ),
            integrator=vpm.RK2(),
            turbulence=vpm.TurbulenceConfig.inviscid(),
            induction=vpm.SlipSlabInduction(
                vpm.FMMInduction(),
                z_min=-half_span,
                z_max=half_span,
                tail_tolerance=1.0e-4,
                max_shells=129,
            ),
            stabilization=vpm.StabilizationConfig.bounded_domain(vpm_domain),
            max_n_particles=particle_limit,
            domain_bounds=vpm_domain,
        ),
        backup=Backup(interval_steps=0, directory="solution", log_directory="solution"),
        samplers=Samplers(samples=vpm_samplers(**clocks)),
        run=vpm.RunPlan(steps=round(physical_end / exchange_dt)),
        directory=CASE_DIR,
    )
    if "transfer_region_scale" in values:
        scale = values.pop("transfer_region_scale")
        transfer_box = tuple(value * scale for value in transfer_box)
    coupling_values = dict(
        freestream_velocity=list(STARTUP_FREESTREAM_VELOCITY),
        transfer_region_bounds=transfer_box,
        eta_blend_width=blend_ratio * particle_spacing,
        vpm_only_width=release_ratio * particle_spacing,
        interface_iterations=INTERFACE_ITERATIONS,
        interface_normal_tolerance=INTERFACE_TOLERANCE,
        interface_gradient_tolerance=INTERFACE_TOLERANCE,
        backup_interval_steps=max(1, round(BACKUPS / exchange_dt)),
        transfer_diagnostic_interval_steps=max(1, round(2.4 / exchange_dt)),
    )
    coupler_setup = coupling.CouplerSetup(**{**coupling_values, **values})
    return fvm_setup, vpm_case, coupler_setup, mesh


def create_solver(
    *,
    output_root: Path | None = None,
    end_time: float | None = None,
    restart_from: Path | None = None,
    max_coupling_steps: int | None = None,
    overrides: dict[str, object] | None = None,
) -> int:
    """Run the coupled case in an optional isolated campaign directory."""

    return run_coupled_cylinder(
        build_case,
        start_from=START_FROM,
        output_root=output_root,
        end_time=end_time,
        restart_from=restart_from,
        max_coupling_steps=max_coupling_steps,
        overrides=overrides,
        startup_duration=STARTUP_DURATION,
        startup_transition_duration=STARTUP_TRANSITION_DURATION,
        steady_freestream_velocity=FREESTREAM_VELOCITY,
        perturbation=INITIAL_PERTURBATION,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-coupling-steps",
        type=positive_coupling_steps,
        help="Stop after this many accepted exchanges and save a native checkpoint; "
        "the configured 100 s physical horizon is unchanged.",
    )
    options = parser.parse_args(argv)
    create_solver(max_coupling_steps=options.max_coupling_steps)
    return 0


if __name__ == "__main__":
    main()
