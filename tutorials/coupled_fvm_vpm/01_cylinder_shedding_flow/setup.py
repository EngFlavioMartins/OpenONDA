#!/usr/bin/env python3
"""Coupled FVM--VPM flow past a circular cylinder at Re=150.

The body-fitted FVM resolves the cylinder and near wake. The VPM carries the
outer flow and receives vorticity from the FVM overlap region. No immersed-
boundary model is used.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import openonda.coupler as coupling
import openonda.fvm as fvm
import openonda.fvm.mesher as msh
import openonda.vpm as vpm
from openonda.vpm import Backup, Samplers

CASE_DIR = Path(__file__).resolve().parent
CYLINDER_STL = CASE_DIR / "assets" / "cylinder_long.stl"

# Physics
DIAMETER = 1.0
# This finite-cylinder setup includes the complete supplied z=-6..6 surface.
# Force coefficients must use its actual projected area D * L.
CYLINDER_LENGTH = 12.0
REYNOLDS_NUMBER = 150.0
DENSITY = 1.0
FREESTREAM_VELOCITY = [1.0, 0.0, 0.0]
KINEMATIC_VISCOSITY = 1.0 / REYNOLDS_NUMBER
REFERENCE_AREA = DIAMETER * CYLINDER_LENGTH

# Domains
# The supplied finite surface spans z=-6..6; the typed mesher requires the
# geometric authority to be strictly inside its outer domain.
FVM_DOMAIN = (-3.0, 4.5, -3.5, 3.5, -6.25, 6.25)
TRANSFER_REGION = (-2.75, 4.25, -3.25, 3.25, -1.5, 1.5)
VPM_DOMAIN = (-8.0, 20.0, -8.0, 8.0, -4.0, 4.0)

# Time, mesh, and output
FVM_TIME_STEP_SIZE = 0.002
VPM_TIME_STEP_SIZE = 0.05
TOTAL_TIME = 60.0
FORCE_INTERVAL_TIME = 0.02
LINE_INTERVAL_TIME = 0.1
SLICE_INTERVAL_TIME = 0.5
FIELD_INTERVAL_TIME = 2.5

# Medium adaptive FVM mesh. The built-in mesher supplies cfMesh's automatic
# wrapper; configurable user-designed inflation layers are intentionally not
# part of this tutorial contract.
FVM_BACKGROUND_CELL_SIZE = 0.5
FVM_SURFACE_CELL_SIZE = 1.0 / 32.0
VPM_PARTICLE_SPACING = 1.0 / 16.0
SAMPLE_SPACING = 1.0 / 16.0
VPM_PARTICLE_LIMIT = 750_000
GBD_VORTICITY_FLOOR = 0.01
TRANSFER_VORTICITY_CUTOFF = 0.05


def interval_steps(interval: float, time_step_size: float) -> int:
    steps = round(interval / time_step_size)
    return steps


FVM_MESH = msh.CartesianMesher(
    domain=msh.BoxDomain(
        bounds=FVM_DOMAIN,
        patches=msh.BoxPatches(
            xmin="numericalBoundary",
            xmax="numericalBoundary",
            ymin="numericalBoundary",
            ymax="numericalBoundary",
            zmin="zmin",
            zmax="zmax",
        ),
    ),
    surfaces=(msh.STLSurface(CYLINDER_STL, patch="cylinder"),),
    max_cell_size=FVM_BACKGROUND_CELL_SIZE,
    boundary_cell_size=FVM_SURFACE_CELL_SIZE,
    min_cell_size=FVM_SURFACE_CELL_SIZE,
    boundary_layers=(),
)

FVM_FORCE_SCHEDULE = fvm.RunSchedule(
    every_n_steps=interval_steps(FORCE_INTERVAL_TIME, FVM_TIME_STEP_SIZE)
)
FVM_LINE_SCHEDULE = fvm.RunSchedule(
    every_n_steps=interval_steps(LINE_INTERVAL_TIME, FVM_TIME_STEP_SIZE)
)
FVM_SLICE_SCHEDULE = fvm.RunSchedule(
    every_n_steps=interval_steps(SLICE_INTERVAL_TIME, FVM_TIME_STEP_SIZE)
)
VPM_LINE_SCHEDULE = vpm.EverySteps(interval_steps(LINE_INTERVAL_TIME, VPM_TIME_STEP_SIZE))
VPM_SLICE_SCHEDULE = vpm.EverySteps(interval_steps(SLICE_INTERVAL_TIME, VPM_TIME_STEP_SIZE))


def fvm_transverse_line(x: float) -> fvm.LineSampler:
    return fvm.LineSampler(
        start=[x, -3.0, 0.0],
        end=[x, 3.0, 0.0],
        spacing=SAMPLE_SPACING,
        k=12,
        reconstruction="affine",
        file_name=f"fvm_transverse_x{x:g}",
        schedule=FVM_LINE_SCHEDULE,
    )


def vpm_transverse_line(x: float) -> vpm.LineSampler:
    return vpm.LineSampler(
        start=[x, -3.0, 0.0],
        end=[x, 3.0, 0.0],
        spacing=SAMPLE_SPACING,
        file_name=f"vpm_transverse_x{x:g}",
        schedule=VPM_LINE_SCHEDULE,
    )


FVM_SETUP = fvm.FVMSetup(
    case_name="cylinder_shedding_flow",
    cores=1,
    execution=fvm.ComputeConfig(operator_backend="numba"),
    output=fvm.OutputConfig(
        format="vtk_xml",
        data_location="cell",
        encoding="appended",
        compression="lz4",
        precision="f32",
        asynchronous=False,
        ghost_layers=0,
    ),
    time=fvm.TimeConfig(
        time_step_size=FVM_TIME_STEP_SIZE,
        start_time=0.0,
        end_time=TOTAL_TIME,
        output_schedule=fvm.RunSchedule(
            every_n_steps=interval_steps(FIELD_INTERVAL_TIME, FVM_TIME_STEP_SIZE)
        ),
    ),
    schemes=fvm.DiscretizationConfig(
        convection_scheme="limitedLinear",
        gradient_scheme="lsq",
        time_scheme="backward",
    ),
    linear=fvm.LinearSolverConfig(
        linear_solver="bicgstab",
        pressure_solver="amg",
        pressure_tolerance=1.0e-7,
        pressure_relative_tolerance=0.005,
        momentum_tolerance=1.0e-6,
        momentum_relative_tolerance=0.05,
    ),
    pimple=fvm.PimpleControl(
        n_correctors=2,
        n_outer_correctors=2,
        n_nonorthogonal_correctors=1,
        velocity_relaxation=0.7,
        pressure_relaxation=0.3,
    ),
    samplers=(
        fvm.ForceSampler(
            patch_names=["cylinder"],
            reference_velocity=1.0,
            reference_area=REFERENCE_AREA,
            reference_length=DIAMETER,
            moment_centre=[0.0, 0.0, 0.0],
            file_name="forces_history",
            schedule=FVM_FORCE_SCHEDULE,
        ),
        fvm.LineSampler(
            start=[1.5, 0.0, 0.0],
            end=[1.5, 0.0, 0.0],
            n_points=1,
            k=12,
            reconstruction="affine",
            file_name="fvm_midspan_probe",
            schedule=FVM_FORCE_SCHEDULE,
        ),
        fvm.LineSampler(
            start=[-2.0, 0.0, 0.0],
            end=[12.0, 0.0, 0.0],
            spacing=SAMPLE_SPACING,
            k=12,
            reconstruction="affine",
            file_name="fvm_centreline",
            schedule=FVM_LINE_SCHEDULE,
        ),
        fvm_transverse_line(1.0),
        fvm_transverse_line(2.0),
        fvm_transverse_line(4.0),
        fvm.SurfaceSampler(
            point=[0.0, 0.0, 0.0],
            normal=[0.0, 0.0, 1.0],
            bounds=[FVM_DOMAIN[0], FVM_DOMAIN[1], FVM_DOMAIN[2], FVM_DOMAIN[3]],
            spacing=SAMPLE_SPACING,
            k=12,
            reconstruction="affine",
            file_name="fvm_midspan",
            schedule=FVM_SLICE_SCHEDULE,
            body_bounds=[-0.5, 0.5, -0.5, 0.5, -6.0, 6.0],
            body_geometry="cylinder_z",
        ),
    ),
    transport=fvm.TransportConfig(
        density=DENSITY,
        kinematic_viscosity=KINEMATIC_VISCOSITY,
    ),
    turbulence=fvm.TurbulenceConfig.none(),
    boundaries=[
        fvm.BoundaryConfig(
            name="numericalBoundary",
            velocity_type="fixedValue",
            velocity_value=FREESTREAM_VELOCITY,
            pressure_type="fixedFluxPressure",
        ),
        fvm.BoundaryConfig.slip("zmin"),
        fvm.BoundaryConfig.slip("zmax"),
        fvm.BoundaryConfig.wall("cylinder"),
    ],
    initial_velocity=FREESTREAM_VELOCITY,
    initial_kinematic_pressure=0.0,
)

VPM_PANEL_SOLVER = vpm.PanelSolver(
    max_n_panels=2048,
    float_dtype="f32",
    linear_solver="SCIPY",
    boundary_condition_type="NEUMANN",
    density=DENSITY,
    freestream_velocity=np.asarray(FREESTREAM_VELOCITY),
    coupling_scope="vpm_boundary_condition",
)

VPM_CASE = vpm.VPMCase(
    name="cylinder_shedding_flow",
    numerics=vpm.Numerics(
        time_step_size=VPM_TIME_STEP_SIZE,
        freestream_velocity=FREESTREAM_VELOCITY,
        viscous=vpm.ViscousConfig.gbd(
            particle_spacing=VPM_PARTICLE_SPACING,
            padding=5.0,
            kinematic_viscosity=KINEMATIC_VISCOSITY,
            threshold_mode="absolute",
            threshold=GBD_VORTICITY_FLOOR * VPM_PARTICLE_SPACING**3,
            max_nodes=VPM_PARTICLE_LIMIT,
            core_radius_ratio=1.0,
        ),
        integrator=vpm.RK2(),
        turbulence=vpm.TurbulenceConfig.inviscid(),
        induction=vpm.TreecodeInduction(),
        stabilization=vpm.StabilizationConfig.bounded_domain(VPM_DOMAIN),
        particle_kernel="GAUSSIAN",
        precision="f32",
        compute_device="AUTO",
        max_n_particles=VPM_PARTICLE_LIMIT,
        max_evaluation_points=VPM_PARTICLE_LIMIT,
        domain_bounds=list(VPM_DOMAIN),
        write_precision="f32",
        panel_solver=VPM_PANEL_SOLVER,
        bodies=(
            vpm.PanelBodySetup(
                stl=str(CYLINDER_STL),
                uid="cylinder",
                reference_area=REFERENCE_AREA,
            ),
        ),
    ),
    # Coupled restart saves own restart writes; VPM samplers remain independent.
    backup=Backup(interval_steps=0, directory="solution", log_directory="solution"),
    samplers=Samplers(
        samples=(
            vpm.LineSampler(
                start=[1.5, 0.0, 0.0],
                end=[1.5, 0.0, 0.0],
                spacing=SAMPLE_SPACING,
                file_name="vpm_midspan_probe",
                schedule=VPM_LINE_SCHEDULE,
            ),
            vpm.LineSampler(
                start=[-2.0, 0.0, 0.0],
                end=[12.0, 0.0, 0.0],
                spacing=SAMPLE_SPACING,
                file_name="vpm_centreline",
                schedule=VPM_LINE_SCHEDULE,
            ),
            vpm_transverse_line(1.0),
            vpm_transverse_line(2.0),
            vpm_transverse_line(4.0),
            vpm.SurfaceSampler(
                point=[0.0, 0.0, 0.0],
                normal=[0.0, 0.0, 1.0],
                bounds=[-2.0, 12.0, -3.0, 3.0],
                spacing=SAMPLE_SPACING,
                file_name="vpm_midspan",
                include_derivatives=False,
                schedule=VPM_SLICE_SCHEDULE,
            ),
        ),
    ),
    run=vpm.RunPlan(steps=round(TOTAL_TIME / VPM_TIME_STEP_SIZE)),
    directory=CASE_DIR,
)

COUPLER_SETUP = coupling.CouplerSetup(
    freestream_velocity=FREESTREAM_VELOCITY,
    transfer_method="buffered_m4_renewal",
    transfer_region_bounds=TRANSFER_REGION,
    backup_interval_steps=interval_steps(FIELD_INTERVAL_TIME, VPM_TIME_STEP_SIZE),
    boundary_condition_mode="vorticity_mixed",
    fvm_consistency_width=0.25,
    eta_blend_width=6.0 * VPM_PARTICLE_SPACING,
    vpm_only_width=0.0,
    transfer_vorticity_cutoff=TRANSFER_VORTICITY_CUTOFF,
    transfer_amplification_cap=1.8,
    transfer_diagnostic_interval_steps=interval_steps(LINE_INTERVAL_TIME, VPM_TIME_STEP_SIZE),
)


def main() -> None:
    with coupling.create_coupler(FVM_SETUP, VPM_CASE, COUPLER_SETUP, mesh=FVM_MESH) as solver:
        solver.run(backup_at_stop=True)


if __name__ == "__main__":
    main()
