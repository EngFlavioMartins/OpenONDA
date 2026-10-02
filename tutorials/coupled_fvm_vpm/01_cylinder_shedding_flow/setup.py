#!/usr/bin/env python3
"""Coupled FVM–VPM flow past a spanwise cylinder section at Re = 150.

The body-fitted FVM resolves the cylinder and a compact near-body box. The
VPM carries vorticity through the outer domain. The FVM force is normalized
by the resolved span.

Usage:
    ./allrun.sh
"""

import argparse
import math
from pathlib import Path

import numpy as np

import openonda.coupler as coupling
import openonda.fvm as fvm
import openonda.fvm.mesher as msh
import openonda.vpm as vpm
from openonda.cylinder_campaign import positive_coupling_steps, run_coupled_cylinder
from openonda.cylinder_case import (
    DEFAULT_CYLINDER_CASE,
    resolve_cylinder_particle_spacing,
    resolve_cylinder_variant,
    validate_cylinder_authority,
)
from openonda.tutorial_support import cylinder_sampling as observations
from openonda.vpm import Backup, Samplers

# Physical problem
START_FROM = "latest"  # allrun.sh preserves outputs; allclean.sh is explicit.

CASE_NAME = "coupled_cylinder_flow"
DIAMETER = DEFAULT_CYLINDER_CASE.diameter
FREESTREAM_VELOCITY = (1.0, 0.0, 0.0)
DENSITY = 1.0
REYNOLDS_NUMBER = DEFAULT_CYLINDER_CASE.reynolds_number
KINEMATIC_VISCOSITY = np.linalg.norm(FREESTREAM_VELOCITY) * DIAMETER / REYNOLDS_NUMBER

# FVM domain and mesh
FVM_CORES = 4
CELL_SIZE = DEFAULT_CYLINDER_CASE.coupled_wall_spacing
FVM_RESOLVED_SPAN = DEFAULT_CYLINDER_CASE.resolved_span
FVM_HALF_SPAN = 0.5 * FVM_RESOLVED_SPAN
FVM_BOX = (
    -1.60,
    1.60,
    -1.60,
    1.60,
    -FVM_HALF_SPAN,
    FVM_HALF_SPAN,
)
TRANSFER_REGION_BOX = (
    -1.25,
    1.25,
    -1.25,
    1.25,
    -FVM_HALF_SPAN,
    FVM_HALF_SPAN,
)

# VPM domain and resolution. Choose a spanwise lattice spacing that closes the
# physical slab exactly; the slab induction wrapper supplies the free-slip
# images at its two span boundaries.
VPM_DOMAIN = (*(-5.0, 15.0, -5.0, 5.0), -FVM_HALF_SPAN, FVM_HALF_SPAN)
# Keep renewal and grid-based diffusion on one VPM lattice.
PARTICLE_LIMIT = 1_000_000
INTERFACE_ITERATIONS = 3
INTERFACE_TOLERANCE = 1.0e-5

# Time and output
FVM_TIME_STEP_SIZE = 0.008
VPM_TIME_STEP_MULTIPLIER = 5
VPM_TIME_STEP_SIZE = VPM_TIME_STEP_MULTIPLIER * FVM_TIME_STEP_SIZE
END_TIME = DEFAULT_CYLINDER_CASE.coupled_end_time
GBD_VORTICITY_FLOOR = 0.01

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


def build_case(
    *,
    end_time: float | None = None,
    overrides: dict[str, object] | None = None,
):
    """Construct the physical case, mesh and accepted-time schedules."""
    overrides = {
        "hxy": CELL_SIZE,
        "dz": CELL_SIZE,
        "particle_spacing_ratio": 1.0,
        "compute_device": "AUTO",
        **(overrides or {}),
    }
    variant = resolve_cylinder_variant(
        overrides,
        hxy=CELL_SIZE,
        span=FVM_RESOLVED_SPAN,
        exchange_dt=VPM_TIME_STEP_SIZE,
        cores=FVM_CORES,
        particle_limit=PARTICLE_LIMIT,
        end_time=END_TIME if end_time is None else end_time,
        fvm_time_step=FVM_TIME_STEP_SIZE,
    )
    hxy, span, dz_target = variant.hxy, variant.span, variant.dz
    hp_ratio, core_ratio = variant.particle_spacing_ratio, variant.core_radius_ratio
    blend_ratio, release_ratio = variant.blend_width_ratio, variant.release_width_ratio
    exchange_dt, physical_end = variant.exchange_dt, variant.end_time
    cores, compute_device = variant.cores, variant.compute_device
    particle_limit = variant.particle_limit
    values = dict(variant.coupler_overrides)
    half_span = span / 2.0
    fvm_box = (*FVM_BOX[:4], -half_span, half_span)
    transfer_box = (*TRANSFER_REGION_BOX[:4], -half_span, half_span)
    particle_spacing = resolve_cylinder_particle_spacing(span=span, hxy=hxy, ratio=hp_ratio)
    validate_cylinder_authority(
        transfer_edge_x=transfer_box[1],
        radius=DIAMETER / 2.0,
        blend_width=blend_ratio * particle_spacing,
    )
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
    sampling = observations.sampling_plan(observations.PROFILES, **clocks)
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
        samplers=observations.fvm_samplers(
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
                velocity_value=list(FREESTREAM_VELOCITY),
                pressure_type="fixedFluxPressure",
            ),
            fvm.BoundaryConfig.slip("zmin"),
            fvm.BoundaryConfig.slip("zmax"),
            fvm.BoundaryConfig.wall("cylinder"),
        ],
        initial_velocity=list(FREESTREAM_VELOCITY),
    )
    vpm_case = vpm.VPMCase(
        name=CASE_NAME,
        numerics=vpm.Numerics(
            time_step_size=exchange_dt,
            compute_device=compute_device,
            freestream_velocity=FREESTREAM_VELOCITY,
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
        samplers=Samplers(samples=observations.vpm_samplers(**clocks)),
        run=vpm.RunPlan(steps=round(physical_end / exchange_dt)),
        directory=CASE_DIR,
    )
    if "transfer_region_scale" in values:
        scale = values.pop("transfer_region_scale")
        transfer_box = tuple(value * scale for value in transfer_box)
    coupling_values = dict(
        freestream_velocity=list(FREESTREAM_VELOCITY),
        transfer_region_bounds=transfer_box,
        eta_blend_width=blend_ratio * particle_spacing,
        vpm_only_width=release_ratio * particle_spacing,
        interface_iterations=INTERFACE_ITERATIONS,
        interface_normal_tolerance=INTERFACE_TOLERANCE,
        interface_gradient_tolerance=INTERFACE_TOLERANCE,
        backup_interval_steps=max(1, round(observations.BACKUPS / exchange_dt)),
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

    def resolved_case(**kwargs):
        setup, particles, coupling_setup, mesh = build_case(**kwargs)
        # This is the ordinary case mesh cache, not a separate run directory.
        cached_mesh = (
            CASE_DIR if output_root is None else Path(output_root)
        ) / "solution/fvm/mesh.npz"
        if cached_mesh.is_file() and not kwargs.get("overrides"):
            mesh = cached_mesh
        return setup, particles, coupling_setup, mesh

    return run_coupled_cylinder(
        resolved_case,
        start_from=START_FROM,
        output_root=output_root,
        end_time=end_time,
        restart_from=restart_from,
        max_coupling_steps=max_coupling_steps,
        overrides=overrides,
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
