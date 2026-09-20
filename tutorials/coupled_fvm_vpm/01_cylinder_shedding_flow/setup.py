#!/usr/bin/env python3
"""Coupled FVM–VPM flow past a spanwise cylinder section at Re = 150.

The body-fitted FVM resolves the cylinder and a compact near-body box. The
VPM carries vorticity through the outer domain.  The FVM force is normalized
by its resolved one-diameter span; the longer panel body supplies the harmonic
body field without introducing panel wake shedding.

Usage:
    ./allrun.sh
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
FVM_RESOLVED_SPAN = 1.0
FREESTREAM_VELOCITY = (1.0, 0.0, 0.0)
DENSITY = 1.0
REYNOLDS_NUMBER = 150.0
KINEMATIC_VISCOSITY = np.linalg.norm(FREESTREAM_VELOCITY) * DIAMETER / REYNOLDS_NUMBER
REFERENCE_AREA = DIAMETER * FVM_RESOLVED_SPAN
PANEL_REFERENCE_AREA = DIAMETER * CYLINDER_LENGTH

# FVM domain and mesh
FVM_CORES = 4
PIMPLE_CORRECTORS = 2

# The reference-flow fine mesh realizes its --dx 0.050 request as 0.0375 m
# near the cylinder. The compact coupled FVM uses that spacing as the finest
# level and retains four realized dyadic levels through its background region.
CELL_SIZE = 0.0375
SPANWISE_CELL_SIZE = CELL_SIZE
FVM_HALF_SPAN = 0.5 * FVM_RESOLVED_SPAN
FVM_BOX = (
    -1.5,
    1.5,
    -1.5,
    1.5,
    -FVM_HALF_SPAN,
    FVM_HALF_SPAN,
)
TRANSFER_REGION_BOX = (
    -1.25,
    1.25,
    -1.25,
    1.25,
    -FVM_HALF_SPAN + 0.5 * SPANWISE_CELL_SIZE,
    FVM_HALF_SPAN - 0.5 * SPANWISE_CELL_SIZE,
)

# VPM domain and resolution
VPM_DOMAIN = (-5.0, 15.0, -5.0, 5.0, -6.60, 6.60)
# Keep stable renewal and GBD on one lattice.  A split 0.0375/0.05 lattice
# remeshed every coupled step and added cost/dissipation without resolving more
# VPM physics than the diffusion grid could retain.
VPM_PARTICLE_SPACING = 0.05
GBD_GRID_SPACING = VPM_PARTICLE_SPACING
SAMPLE_SPACING = 2.0 * CELL_SIZE
PARTICLE_LIMIT = 1_000_000

# Coupling
BOUNDARY_CONDITION_MODE = "vorticity_mixed"
TRANSFER_METHOD = "buffered_m4_renewal"
TRANSFER_VORTICITY_CUTOFF = 0.05
TRANSFER_AMPLIFICATION_CAP = 1.8
FVM_CONSISTENCY_WIDTH = 0.0
INTERFACE_ITERATIONS = 3
INTERFACE_TOLERANCE = 1.0e-5

# Time and output
FVM_TIME_STEP_SIZE = 0.004
VPM_TIME_STEP_MULTIPLIER = 10
VPM_TIME_STEP_SIZE = VPM_TIME_STEP_MULTIPLIER * FVM_TIME_STEP_SIZE
END_TIME = 100.0
SAMPLING_INTERVAL_TIME = 0.2
SLICE_INTERVAL_TIME = 0.24
OUTPUT_INTERVAL_TIME = 0.24
# Every schedule must land on a 0.04 s accepted coupling boundary while
# interface iteration is active.
BACKUP_INTERVAL_TIME = 0.24
GBD_VORTICITY_FLOOR = 0.01

FVM_OUTPUT_INTERVAL_STEPS = round(OUTPUT_INTERVAL_TIME / FVM_TIME_STEP_SIZE)
COUPLED_BACKUP_INTERVAL_STEPS = round(BACKUP_INTERVAL_TIME / VPM_TIME_STEP_SIZE)
FVM_SAMPLING_INTERVAL_STEPS = round(SAMPLING_INTERVAL_TIME / FVM_TIME_STEP_SIZE)
VPM_SAMPLING_INTERVAL_STEPS = round(SAMPLING_INTERVAL_TIME / VPM_TIME_STEP_SIZE)
FVM_SLICE_INTERVAL_STEPS = round(SLICE_INTERVAL_TIME / FVM_TIME_STEP_SIZE)
VPM_SLICE_INTERVAL_STEPS = round(SLICE_INTERVAL_TIME / VPM_TIME_STEP_SIZE)
TRANSFER_DIAGNOSTIC_INTERVAL_STEPS = round(2.4 / VPM_TIME_STEP_SIZE)

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
FVM_MESH = msh.CartesianMesher(
    domain=msh.BoxDomain(bounds=FVM_BOX, patches=FVM_PATCHES),
    surfaces=(msh.STLSurface(CYLINDER_STL, patch="cylinder"),),
    # The reference fine mesh has a 0.60 m outer scale. The compact span and
    # boundary closure realize the four fluid levels below it as 0.30 m,
    # 0.15 m, 0.075 m, and 0.0375 m.
    max_cell_size=16.0 * CELL_SIZE,
    cell_size_anchor=CELL_SIZE,
    refinements=(
        msh.BoxRefinement(
            name="nearBody",
            bounds=(
                -1.0,
                FVM_BOX[1] - 2.0 * CELL_SIZE,
                -1.0,
                1.0,
                FVM_BOX[4],
                FVM_BOX[5],
            ),
            cell_size=2.0 * CELL_SIZE,
        ),
    ),
    patch_refinements=(msh.PatchRefinement("cylinder", CELL_SIZE),),
    surface_may_cross_domain_boundary=True,
)

FVM_SAMPLING_SCHEDULE = fvm.RunSchedule(every_n_steps=FVM_SAMPLING_INTERVAL_STEPS)
FVM_SLICE_SCHEDULE = fvm.RunSchedule(every_n_steps=FVM_SLICE_INTERVAL_STEPS)
VPM_SAMPLING_SCHEDULE = vpm.EverySteps(VPM_SAMPLING_INTERVAL_STEPS)
VPM_SLICE_SCHEDULE = vpm.EverySteps(VPM_SLICE_INTERVAL_STEPS)

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
        time_step_size=FVM_TIME_STEP_SIZE,
        end_time=END_TIME,
        # Retained fields and the coupled checkpoint use the same accepted times.
        output_schedule=fvm.RunSchedule(every_n_steps=FVM_OUTPUT_INTERVAL_STEPS),
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
        n_correctors=PIMPLE_CORRECTORS,
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
    transfer_method=TRANSFER_METHOD,
    transfer_region_bounds=TRANSFER_REGION_BOX,
    backup_interval_steps=COUPLED_BACKUP_INTERVAL_STEPS,
    boundary_condition_mode=BOUNDARY_CONDITION_MODE,
    fvm_consistency_width=FVM_CONSISTENCY_WIDTH,
    interface_iterations=INTERFACE_ITERATIONS,
    interface_normal_tolerance=INTERFACE_TOLERANCE,
    interface_gradient_tolerance=INTERFACE_TOLERANCE,
    eta_blend_width=6.0 * VPM_PARTICLE_SPACING,
    vpm_only_width=2.0 * VPM_PARTICLE_SPACING,
    transfer_vorticity_cutoff=TRANSFER_VORTICITY_CUTOFF,
    transfer_amplification_cap=TRANSFER_AMPLIFICATION_CAP,
    transfer_diagnostic_interval_steps=TRANSFER_DIAGNOSTIC_INTERVAL_STEPS,
)

VPM_SAMPLERS = (
    # Panel-source gradients are undefined on the cylinder surface. Keep VPM
    # diagnostics on exterior transverse lines; FVM owns the body centreline.
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
        time_step_size=VPM_TIME_STEP_SIZE,
        freestream_velocity=FREESTREAM_VELOCITY,
        viscous=vpm.ViscousConfig.gbd(
            particle_spacing=VPM_PARTICLE_SPACING,
            gbd_grid_spacing=GBD_GRID_SPACING,
            padding=5.0,
            kinematic_viscosity=KINEMATIC_VISCOSITY,
            threshold_mode="absolute",
            threshold=GBD_VORTICITY_FLOOR * VPM_PARTICLE_SPACING**3,
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
                reference_area=PANEL_REFERENCE_AREA,
                diffusion_mask="cylinder_z",
            ),
        ),
    ),
    # Coupler backups contain the FVM state, VPM state, and boundary history
    # atomically; independent native backups would not be restart-consistent.
    backup=Backup(interval_steps=0, directory="solution", log_directory="solution"),
    samplers=Samplers(samples=VPM_SAMPLERS),
    run=vpm.RunPlan(steps=round(END_TIME / VPM_TIME_STEP_SIZE)),
    directory=CASE_DIR,
)


def main() -> int:
    with coupling.create_coupler(FVM_SETUP, VPM_CASE, COUPLER_SETUP, mesh=FVM_MESH) as solver:
        solver.run()
    return 0


if __name__ == "__main__":
    main()
