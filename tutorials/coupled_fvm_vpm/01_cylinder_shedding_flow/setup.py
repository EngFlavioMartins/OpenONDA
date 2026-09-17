#!/usr/bin/env python3
"""Coupled FVM--VPM flow past a finite circular cylinder at Re = 150.

The body-fitted FVM resolves the cylinder and a compact near-body box. The
VPM carries vorticity through the outer domain. The supplied body is 12D long,
so this remains a three-dimensional calculation even though the midspan flow
is compared with the quasi-two-dimensional FVM reference.

Usage:
    python setup.py
"""

from pathlib import Path

import numpy as np

import openonda.coupler as coupling
import openonda.fvm as fvm
import openonda.fvm.mesher as msh
import openonda.vpm as vpm
from openonda.vpm import Backup, Samplers

# Physical problem
CASE_NAME = "coupled_cylinder_flow"
DIAMETER = 1.0
CYLINDER_LENGTH = 12.0
FREESTREAM_VELOCITY = (1.0, 0.0, 0.0)
DENSITY = 1.0
REYNOLDS_NUMBER = 150.0
KINEMATIC_VISCOSITY = np.linalg.norm(FREESTREAM_VELOCITY) * DIAMETER / REYNOLDS_NUMBER
REFERENCE_AREA = DIAMETER * CYLINDER_LENGTH

# Geometry and resolution. CELL_SIZE matches the reference-flow fine case.
CASE_DIR = Path(__file__).resolve().parent
CYLINDER_STL = CASE_DIR / "assets" / "cylinder_long.stl"
CELL_SIZE = 0.03
SPANWISE_CELL_SIZE = 1.0 / 9.0
FVM_HALF_SPAN = 6.0
FVM_BOX = (-1.44, 1.92, -1.44, 1.44, -FVM_HALF_SPAN, FVM_HALF_SPAN)
TRANSFER_REGION = (-1.20, 1.68, -1.20, 1.20, -5.76, 5.76)
VPM_DOMAIN = (-5.0, 15.0, -5.0, 5.0, -6.60, 6.60)
PARTICLE_SPACING = CELL_SIZE
SAMPLE_SPACING = 2.0 * CELL_SIZE
PARTICLE_LIMIT = 1_500_000

# Numerical controls and output schedules
FVM_CORES = 6
TIME_STEP_SIZE = 0.004
END_TIME = 60.0
SAMPLING_INTERVAL_TIME = 0.1
SLICE_INTERVAL_TIME = 0.5
BACKUP_INTERVAL_TIME = 2.5
GBD_VORTICITY_FLOOR = 0.01

SAMPLING_INTERVAL_STEPS = round(SAMPLING_INTERVAL_TIME / TIME_STEP_SIZE)
SLICE_INTERVAL_STEPS = round(SLICE_INTERVAL_TIME / TIME_STEP_SIZE)
BACKUP_INTERVAL_STEPS = round(BACKUP_INTERVAL_TIME / TIME_STEP_SIZE)

FVM_PATCHES = msh.BoxPatches(
    xmin="numericalBoundary",
    xmax="numericalBoundary",
    ymin="numericalBoundary",
    ymax="numericalBoundary",
    zmin="zmin",
    zmax="zmax",
)
FVM_SOURCE_BOX = (*FVM_BOX[:4], -16.0 * CELL_SIZE, 16.0 * CELL_SIZE)
FVM_SPAN_LEVELS = tuple(
    np.linspace(
        FVM_BOX[4],
        FVM_BOX[5],
        round((FVM_BOX[5] - FVM_BOX[4]) / SPANWISE_CELL_SIZE) + 1,
    )
)
FVM_MESH = msh.ExtrudedCartesianMesher(
    source=msh.CartesianMesher(
        domain=msh.BoxDomain(bounds=FVM_SOURCE_BOX, patches=FVM_PATCHES),
        surfaces=(msh.STLSurface(CYLINDER_STL, patch="cylinder"),),
        max_cell_size=8.0 * CELL_SIZE,
        cell_size_anchor=CELL_SIZE,
        refinements=(
            msh.BoxRefinement(
                name="nearBody",
                bounds=(-1.0, 1.92, -1.0, 1.0, FVM_SOURCE_BOX[4], FVM_SOURCE_BOX[5]),
                cell_size=2.0 * CELL_SIZE,
            ),
        ),
        patch_refinements=(msh.PatchRefinement("cylinder", CELL_SIZE),),
        surface_may_cross_domain_boundary=True,
    ),
    domain=msh.BoxDomain(bounds=FVM_BOX, patches=FVM_PATCHES),
    levels=FVM_SPAN_LEVELS,
)

FVM_SAMPLING_SCHEDULE = fvm.RunSchedule(every_n_steps=SAMPLING_INTERVAL_STEPS)
FVM_SLICE_SCHEDULE = fvm.RunSchedule(every_n_steps=SLICE_INTERVAL_STEPS)
VPM_SAMPLING_SCHEDULE = vpm.EverySteps(SAMPLING_INTERVAL_STEPS)
VPM_SLICE_SCHEDULE = vpm.EverySteps(SLICE_INTERVAL_STEPS)

FVM_SAMPLERS = (
    fvm.ForceSampler(
        patch_names=["cylinder"],
        reference_velocity=np.linalg.norm(FREESTREAM_VELOCITY),
        reference_area=REFERENCE_AREA,
        reference_length=DIAMETER,
        moment_centre=[0.0, 0.0, 0.0],
        file_name="forces_history",
        schedule=FVM_SAMPLING_SCHEDULE,
    ),
    fvm.LineSampler(
        start=[FVM_BOX[0], 0.0, 0.0],
        end=[FVM_BOX[1], 0.0, 0.0],
        spacing=SAMPLE_SPACING,
        file_name="fvm_centreline",
        schedule=FVM_SAMPLING_SCHEDULE,
    ),
    fvm.LineSampler(
        start=[1.0, -1.5, 0.0],
        end=[1.0, 1.5, 0.0],
        spacing=SAMPLE_SPACING,
        file_name="fvm_transverse_x1",
        schedule=FVM_SAMPLING_SCHEDULE,
    ),
    fvm.SurfaceSampler(
        point=[0.0, 0.0, 0.0],
        normal=[0.0, 0.0, 1.0],
        bounds=list(FVM_BOX[:4]),
        spacing=SAMPLE_SPACING,
        file_name="fvm_midspan",
        schedule=FVM_SLICE_SCHEDULE,
        body_bounds=[-0.5, 0.5, -0.5, 0.5, -6.0, 6.0],
        body_geometry="cylinder_z",
    ),
)

FVM_SETUP = fvm.FVMSetup(
    case_name=CASE_NAME,
    cores=FVM_CORES,
    execution=fvm.ComputeConfig(operator_backend="numba"),
    output=fvm.OutputConfig(compression="lz4", asynchronous=True, ghost_layers=0),
    time=fvm.TimeConfig(
        time_step_size=TIME_STEP_SIZE,
        end_time=END_TIME,
        output_schedule=fvm.RunSchedule(every_n_steps=BACKUP_INTERVAL_STEPS),
    ),
    schemes=fvm.DiscretizationConfig(
        convection_scheme="limitedLinear",
        gradient_scheme="lsq",
        time_scheme="euler_implicit",
    ),
    linear=fvm.LinearSolverConfig(
        pressure_solver="amg",
        pressure_tolerance=1.0e-7,
        pressure_relative_tolerance=0.005,
        momentum_tolerance=1.0e-6,
        momentum_relative_tolerance=0.05,
    ),
    pimple=fvm.PimpleControl(
        n_correctors=2,
        n_outer_correctors=2,
        velocity_relaxation=0.7,
        pressure_relaxation=0.3,
    ),
    samplers=FVM_SAMPLERS,
    transport=fvm.TransportConfig(
        density=DENSITY,
        kinematic_viscosity=KINEMATIC_VISCOSITY,
    ),
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
    initial_kinematic_pressure=0.0,
)

COUPLER_SETUP = coupling.CouplerSetup(
    freestream_velocity=list(FREESTREAM_VELOCITY),
    transfer_method="buffered_m4_renewal",
    transfer_region_bounds=TRANSFER_REGION,
    backup_interval_steps=BACKUP_INTERVAL_STEPS,
    boundary_condition_mode="vorticity_mixed",
    fvm_consistency_width=0.0,
    interface_iterations=12,
    eta_blend_width=6.0 * PARTICLE_SPACING,
    vpm_only_width=2.0 * PARTICLE_SPACING,
    transfer_vorticity_cutoff=0.05,
    transfer_amplification_cap=1.8,
    transfer_diagnostic_interval_steps=BACKUP_INTERVAL_STEPS,
)

VPM_SAMPLERS = (
    vpm.LineSampler(
        start=[VPM_DOMAIN[0], 0.0, 0.0],
        end=[VPM_DOMAIN[1], 0.0, 0.0],
        spacing=SAMPLE_SPACING,
        file_name="vpm_centreline",
        schedule=VPM_SAMPLING_SCHEDULE,
    ),
    vpm.LineSampler(
        start=[1.0, -3.0, 0.0],
        end=[1.0, 3.0, 0.0],
        spacing=SAMPLE_SPACING,
        file_name="vpm_transverse_x1",
        schedule=VPM_SAMPLING_SCHEDULE,
    ),
    vpm.LineSampler(
        start=[2.0, -3.0, 0.0],
        end=[2.0, 3.0, 0.0],
        spacing=SAMPLE_SPACING,
        file_name="vpm_transverse_x2",
        schedule=VPM_SAMPLING_SCHEDULE,
    ),
    vpm.LineSampler(
        start=[4.0, -3.0, 0.0],
        end=[4.0, 3.0, 0.0],
        spacing=SAMPLE_SPACING,
        file_name="vpm_transverse_x4",
        schedule=VPM_SAMPLING_SCHEDULE,
    ),
    vpm.SurfaceSampler(
        point=[0.0, 0.0, 0.0],
        normal=[0.0, 0.0, 1.0],
        bounds=[VPM_DOMAIN[0], VPM_DOMAIN[1], -3.0, 3.0],
        spacing=SAMPLE_SPACING,
        file_name="vpm_midspan",
        include_derivatives=False,
        schedule=VPM_SLICE_SCHEDULE,
    ),
)

VPM_PANEL_SOLVER = vpm.PanelSolver(
    max_n_panels=2048,
    float_dtype="f32",
    linear_solver="SCIPY",
    boundary_condition_type="NEUMANN",
    density=DENSITY,
    freestream_velocity=np.asarray(FREESTREAM_VELOCITY),
    coupling_scope="fvm_vpm",
)

VPM_CASE = vpm.VPMCase(
    name=CASE_NAME,
    numerics=vpm.Numerics(
        time_step_size=TIME_STEP_SIZE,
        freestream_velocity=FREESTREAM_VELOCITY,
        viscous=vpm.ViscousConfig.gbd(
            particle_spacing=PARTICLE_SPACING,
            padding=5.0,
            kinematic_viscosity=KINEMATIC_VISCOSITY,
            threshold_mode="absolute",
            threshold=GBD_VORTICITY_FLOOR * PARTICLE_SPACING**3,
            max_nodes=PARTICLE_LIMIT,
            core_radius_ratio=1.0,
        ),
        integrator=vpm.RK2(),
        turbulence=vpm.TurbulenceConfig.inviscid(),
        induction=vpm.FMMInduction(),
        stabilization=vpm.StabilizationConfig.bounded_domain(VPM_DOMAIN),
        particle_kernel="GAUSSIAN",
        precision="f32",
        compute_device="AUTO",
        max_n_particles=PARTICLE_LIMIT,
        max_evaluation_points=PARTICLE_LIMIT,
        domain_bounds=VPM_DOMAIN,
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
    backup=Backup(interval_steps=0, directory="solution", log_directory="solution"),
    samplers=Samplers(samples=VPM_SAMPLERS),
    run=vpm.RunPlan(steps=round(END_TIME / TIME_STEP_SIZE)),
    directory=CASE_DIR,
)


def main() -> int:
    with coupling.create_coupler(FVM_SETUP, VPM_CASE, COUPLER_SETUP, mesh=FVM_MESH) as solver:
        solver.run()
    return 0


if __name__ == "__main__":
    main()
