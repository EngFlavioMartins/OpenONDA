"""Coupled LES FVM–VPM simulation of flow past a cube at Re = 1000.

The FVM mesh is generated directly as solver-native data by OpenONDA's native
surface-driven Cartesian mesher. No external solver case is used. Both
solvers use the same equilibrium Smagorinsky coefficients.

All case parameters are kept below in one explicit configuration block. Edit
them here to define a different case.

Usage:
    ./allrun.sh
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import openonda.coupler as coupling
import openonda.fvm as fvm
import openonda.fvm.mesher as msh
import openonda.vpm as vpm
from openonda.vpm import Backup, Samplers

# Physical problem
CUBE_SIDE = 1.0
FREESTREAM_VELOCITY = (1.0, 0.0, 0.0)
DENSITY = 1.0
REYNOLDS = 1000.0
KINEMATIC_VISCOSITY = np.linalg.norm(FREESTREAM_VELOCITY) * CUBE_SIDE / REYNOLDS
INITIAL_VELOCITY = (1.0, 0.0, 0.0)

# FVM domain and mesh
FVM_CORES = 4
FVM_BOX = (-1.50, 1.50, -1.50, 1.50, -1.50, 1.50)
TRANSFER_REGION_BOX = (-1.25, 1.25, -1.25, 1.25, -1.25, 1.25)
REFERENCE_FINE_DX = 0.06
SURFACE_CELL_SIZE = REFERENCE_FINE_DX
FVM_MAX_CELL_SIZE = 12 * REFERENCE_FINE_DX
PIMPLE_CORRECTORS = 2

# VPM domain and resolution
VPM_DOMAIN = (-4.5, 12.0, -3.0, 3.0, -3.0, 3.0)
PARTICLE_LIMIT = 1_500_000
VPM_CORE_RADIUS_RATIO = 1.1
GBD_VORTICITY_FLOOR = 0.02
VPM_PARTICLE_SPACING = REFERENCE_FINE_DX
ETA_BLEND_WIDTH = 6 * VPM_PARTICLE_SPACING
# Correct strength/curl misalignment before stretching amplifies the divergent
# part of the discrete wake. Express the relaxation as a physical rate so a
# timestep refinement also refines the per-step correction.
VPM_ALIGNMENT_RELAXATION_RATE = 10.0  # 1/s

# Coupling
BOUNDARY_CONDITION_MODE = "vorticity_mixed"
TRANSFER_METHOD = "buffered_m4_renewal"
TRANSFER_VORTICITY_CUTOFF = 0.05
TRANSFER_BOUNDARY_PRUNE_MULTIPLIER = 10.0
TRANSFER_AMPLIFICATION_CAP = 1.8
FVM_CONSISTENCY_WIDTH = 0.0
INTERFACE_ITERATIONS = 12

# Time and output
END_TIME = 20.0
SAMPLING_INTERVAL_TIME = 0.050
WRITE_SOLUTION_BACKUP = 0.5
FVM_TIME_STEP_SIZE = 0.01
# Match the FVM step and keep samples/backups on exact common times. Reducing
# dt alone did not cure the wake's vorticity-consistency instability.
VPM_TIME_STEP_MULTIPLIER = 1
VPM_TIME_STEP_SIZE = VPM_TIME_STEP_MULTIPLIER * FVM_TIME_STEP_SIZE
FVM_WRITE_SOLUTION_BACKUP_INTERVAL_STEPS = round(WRITE_SOLUTION_BACKUP / FVM_TIME_STEP_SIZE)
VPM_WRITE_SOLUTION_BACKUP_INTERVAL_STEPS = round(WRITE_SOLUTION_BACKUP / VPM_TIME_STEP_SIZE)
FVM_SAMPLING_INTERVAL_STEPS = round(SAMPLING_INTERVAL_TIME / FVM_TIME_STEP_SIZE)
VPM_SAMPLING_INTERVAL_STEPS = round(SAMPLING_INTERVAL_TIME / VPM_TIME_STEP_SIZE)

SAMPLE_SPACING = min(0.125, 2 * REFERENCE_FINE_DX)
TRANSFER_DIAGNOSTIC_INTERVAL_STEPS = VPM_WRITE_SOLUTION_BACKUP_INTERVAL_STEPS

# Case files and derived sampling data
CASE_DIR = Path(__file__).resolve().parent
CUBE_STL = CASE_DIR / "assets" / "cube.stl"
BODY_STL = str(CUBE_STL)
CUBE_BOUNDS = (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)
OFFAXIS_Y = 0.75 * CUBE_SIDE
SLICE_BOUNDS = [FVM_BOX[0], FVM_BOX[1], FVM_BOX[2], FVM_BOX[3]]
WAKE_SLICE_BOUNDS = [0.0, 5.0, -1.5, 1.5]

FVM_MESH = msh.CartesianMesher(
    domain=msh.BoxDomain(
        bounds=FVM_BOX,
        patches=msh.BoxPatches(
            xmin="numericalBoundary",
            xmax="numericalBoundary",
            ymin="numericalBoundary",
            ymax="numericalBoundary",
            zmin="numericalBoundary",
            zmax="numericalBoundary",
        ),
    ),
    surfaces=(msh.STLSurface(CUBE_STL, patch="cube"),),
    max_cell_size=FVM_MAX_CELL_SIZE,
    boundary_cell_size=REFERENCE_FINE_DX,
    patch_refinements=(msh.PatchRefinement("cube", SURFACE_CELL_SIZE),),
    refinements=(
        msh.BoxRefinement(
            name="nearBody",
            bounds=FVM_BOX,
            cell_size=REFERENCE_FINE_DX,
        ),
    ),
)

FVM_SAMPLING_SCHEDULE = fvm.RunSchedule(every_n_steps=FVM_SAMPLING_INTERVAL_STEPS)
VPM_SAMPLING_SCHEDULE = vpm.EverySteps(VPM_SAMPLING_INTERVAL_STEPS)

FVM_SAMPLERS = (
    fvm.ForceSampler(
        patch_names=["cube"],
        reference_velocity=np.linalg.norm(FREESTREAM_VELOCITY),
        reference_area=CUBE_SIDE**2,
        reference_length=CUBE_SIDE,
        moment_centre=[0.0, 0.0, 0.0],
        schedule=FVM_SAMPLING_SCHEDULE,
    ),
    fvm.LineSampler(
        start=[FVM_BOX[0], 0.0, 0.0],
        end=[FVM_BOX[1], 0.0, 0.0],
        spacing=SAMPLE_SPACING,
        file_name=f"fvm_centreline",
        schedule=FVM_SAMPLING_SCHEDULE,
    ),
    fvm.LineSampler(
        start=[FVM_BOX[0], OFFAXIS_Y, 0.0],
        end=[FVM_BOX[1], OFFAXIS_Y, 0.0],
        spacing=SAMPLE_SPACING,
        file_name=f"fvm_offaxis_y075",
        schedule=FVM_SAMPLING_SCHEDULE,
    ),
    fvm.SurfaceSampler(
        point=[0.0, 0.0, 0.0],
        normal=[0, 0, 1],
        bounds=SLICE_BOUNDS,
        spacing=SAMPLE_SPACING,
        file_name="fvm_slice_z0",
        schedule=FVM_SAMPLING_SCHEDULE,
        body_bounds=CUBE_BOUNDS,
    ),
)

FVM_SETUP = fvm.FVMSetup(
    case_name="coupled_replacement_flow",
    cores=FVM_CORES,
    execution=fvm.ComputeConfig(operator_backend="numba"),
    output=fvm.OutputConfig(
        format="vtk_xml",
        data_location="cell",
        encoding="appended",
        compression="lz4",
        precision="f32",
        asynchronous=True,
        ghost_layers=0,
    ),
    time=fvm.TimeConfig(
        time_step_size=FVM_TIME_STEP_SIZE,
        start_time=0.0,
        end_time=END_TIME,
        output_schedule=fvm.RunSchedule(every_n_steps=FVM_WRITE_SOLUTION_BACKUP_INTERVAL_STEPS),
    ),
    schemes=fvm.DiscretizationConfig(
        convection_scheme="linearUpwind",
        gradient_scheme="gauss",
        time_scheme="backward",
    ),
    linear=fvm.LinearSolverConfig(
        linear_solver="bicgstab",
        pressure_solver="amg",
        pressure_tolerance=1e-6,
        pressure_relative_tolerance=0.01,
        pressure_final_relative_tolerance=0.0,
        momentum_tolerance=1e-6,
        momentum_relative_tolerance=0.1,
        momentum_final_relative_tolerance=0.0,
        momentum_max_iterations=2000,
        ilu_drop_tolerance=1e-4,
        ilu_fill_factor=10.0,
        ilu_reuse_tolerance=0.05,
    ),
    pimple=fvm.PimpleControl(
        n_correctors=PIMPLE_CORRECTORS,
        n_outer_correctors=2,
        n_nonorthogonal_correctors=1,
        velocity_relaxation=0.7,
        pressure_relaxation=0.3,
    ),
    samplers=FVM_SAMPLERS,
    transport=fvm.TransportConfig(density=DENSITY, kinematic_viscosity=KINEMATIC_VISCOSITY),
    turbulence=fvm.TurbulenceConfig.equilibrium_smagorinsky(),
    boundaries=[
        fvm.BoundaryConfig(
            name="numericalBoundary",
            velocity_type="fixedValue",
            velocity_value=list(FREESTREAM_VELOCITY),
            pressure_type="fixedFluxPressure",
        ),
        fvm.BoundaryConfig.wall("cube"),
    ],
    initial_velocity=list(INITIAL_VELOCITY),
    initial_kinematic_pressure=0.0,
)

COUPLER_SETUP = coupling.CouplerSetup(
    freestream_velocity=list(FREESTREAM_VELOCITY),
    transfer_method=TRANSFER_METHOD,
    transfer_region_bounds=TRANSFER_REGION_BOX,
    backup_interval_steps=VPM_WRITE_SOLUTION_BACKUP_INTERVAL_STEPS,
    boundary_condition_mode=BOUNDARY_CONDITION_MODE,
    fvm_consistency_width=FVM_CONSISTENCY_WIDTH,
    interface_iterations=INTERFACE_ITERATIONS,
    eta_blend_width=ETA_BLEND_WIDTH,
    vpm_only_width=0.0,
    transfer_vorticity_cutoff=TRANSFER_VORTICITY_CUTOFF,
    transfer_boundary_prune_multiplier=TRANSFER_BOUNDARY_PRUNE_MULTIPLIER,
    transfer_amplification_cap=TRANSFER_AMPLIFICATION_CAP,
    transfer_diagnostic_interval_steps=TRANSFER_DIAGNOSTIC_INTERVAL_STEPS,
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
        start=[VPM_DOMAIN[0], OFFAXIS_Y, 0.0],
        end=[VPM_DOMAIN[1], OFFAXIS_Y, 0.0],
        spacing=SAMPLE_SPACING,
        file_name="vpm_offaxis_y075",
        schedule=VPM_SAMPLING_SCHEDULE,
    ),
    vpm.SurfaceSampler(
        point=[0.0, 0.0, 0.0],
        normal=[0, 0, 1],
        bounds=SLICE_BOUNDS,
        spacing=SAMPLE_SPACING,
        file_name="vpm_slice_z0",
        include_derivatives=False,
        schedule=VPM_SAMPLING_SCHEDULE,
    ),
    vpm.SurfaceSampler(
        point=[0.0, 0.0, 0.0],
        normal=[0, 0, 1],
        bounds=WAKE_SLICE_BOUNDS,
        spacing=SAMPLE_SPACING,
        file_name="vpm_wake_slice_z0",
        include_derivatives=False,
        schedule=VPM_SAMPLING_SCHEDULE,
    ),
)

VPM_PANEL_SOLVER = vpm.PanelSolver(
    max_n_panels=128,
    float_dtype="f32",
    linear_solver="SCIPY",
    boundary_condition_type="NEUMANN",
    density=DENSITY,
    freestream_velocity=np.asarray(FREESTREAM_VELOCITY),
    coupling_scope="fvm_vpm",
)
VPM_CASE = vpm.VPMCase(
    name="coupled_replacement_flow",
    numerics=vpm.Numerics(
        time_step_size=VPM_TIME_STEP_SIZE,
        freestream_velocity=list(FREESTREAM_VELOCITY),
        viscous=vpm.ViscousConfig.gbd(
            kinematic_viscosity=KINEMATIC_VISCOSITY,
            particle_spacing=VPM_PARTICLE_SPACING,
            core_radius_ratio=VPM_CORE_RADIUS_RATIO,
            padding=5.0,
            threshold_mode="absolute",
            threshold=GBD_VORTICITY_FLOOR * VPM_PARTICLE_SPACING**3,
            max_nodes=PARTICLE_LIMIT,
        ),
        integrator=vpm.RK2(),
        turbulence=vpm.TurbulenceConfig.equilibrium_smagorinsky(),
        induction=vpm.TreecodeInduction(),
        stabilization=vpm.StabilizationConfig(
            remove_particles_by_bounds=list(VPM_DOMAIN),
            pedrizzetti_relaxation_factor=VPM_ALIGNMENT_RELAXATION_RATE * VPM_TIME_STEP_SIZE,
            pedrizzetti_relaxation_preserve_moments=True,
        ),
        particle_kernel="GAUSSIAN",
        precision="f32",
        compute_device="AUTO",
        max_n_particles=PARTICLE_LIMIT,
        max_evaluation_points=PARTICLE_LIMIT,
        domain_bounds=list(VPM_DOMAIN),
        write_precision="f32",
        panel_solver=VPM_PANEL_SOLVER,
        bodies=(vpm.PanelBodySetup(stl=BODY_STL, uid="body", reference_area=CUBE_SIDE**2),),
    ),
    # Coupled runs use the atomic FVM+VPM restart save owned by COUPLER_SETUP.
    backup=Backup(interval_steps=0, directory="solution", log_directory="solution"),
    samplers=Samplers(samples=VPM_SAMPLERS),
    run=vpm.RunPlan(steps=round(END_TIME / VPM_TIME_STEP_SIZE)),
    directory=CASE_DIR,
)


def main() -> int:
    mesh = msh.CachedMesh(FVM_MESH, CASE_DIR / "constant" / "mesh.npz")
    with coupling.create_coupler(FVM_SETUP, VPM_CASE, COUPLER_SETUP, mesh=mesh) as solver:
        solver.run()
    return 0


if __name__ == "__main__":
    main()
