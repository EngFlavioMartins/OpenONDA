"""Hybrid LES FVM–VPM simulation of flow past a cube at Re = 1000.

The FVM mesh is generated directly as solver-native data by OpenONDA's
adaptive Cartesian mesher. No external solver case is used. Both solvers use
the same equilibrium Smagorinsky coefficients.

All case parameters are kept below in one explicit configuration block. Edit
them here to define a different case.

Usage:
    python cube_flow_setup.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import openonda.fvm as fvm
import openonda.coupler as coupling
import openonda.vpm as vpm

# Physical problem
CUBE_SIDE = 1.0
FREESTREAM_VELOCITY = (1.0, 0.0, 0.0)
DENSITY = 1.0
REYNOLDS = 1000.0
KINEMATIC_VISCOSITY = np.linalg.norm(FREESTREAM_VELOCITY) * CUBE_SIDE / REYNOLDS
SMAGORINSKY_CK = 0.094
SMAGORINSKY_CE = 1.048
INITIAL_VELOCITY = (1.0, 0.0, 0.0)
VPM_SCHEME = "RK2"

# FVM domain and mesh
FVM_CORES = 4
TRANSFER_REGION_BOX = (-1.5, 1.5, -1.5, 1.5, -1.5, 1.5)
FVM_BOX = (-1.5, 1.5, -1.5, 1.5, -1.5, 1.5)
PIMPLE_CORRECTORS = 2

FVM_WAKE_BOX = (-1.25, 1.25, -1.25, 1.25, -1.25, 1.25)
SURFACE_CELL_SIZE = 0.015625

# VPM domain and resolution
VPM_DOMAIN = (-4.5, 12.0, -3.0, 3.0, -3.0, 3.0)
VPM_PARTICLE_SPACING = SURFACE_CELL_SIZE * 2
PARTICLE_LIMIT = 1500000
VPM_CORE_RADIUS_RATIO = 1.05
TRANSFER_PRUNE_VORTICITY_MIN = 0.05
TRANSFER_BOUNDARY_PRUNE_MULTIPLIER = 10.0
GBD_VORTICITY_FLOOR = 0.05
OVERLAP_ZONE_RAMP_WIDTH = SURFACE_CELL_SIZE * 4

# Coupling
VPM_BC_MODE = "vorticity_mixed"
TRANSFER_AMPLIFICATION_CAP = 1.8

# Output and diagnostics
FVM_TIME_STEP_SIZE = 0.01
FORCE_INTERVAL = 0.05
DIAGNOSTIC_INTERVAL = 0.60
CHECKPOINT_INTERVAL = 1.0
FVM_VOLUME_INTERVAL = 1.0
VPM_LOG_PERIOD = 20
VPM_TIME_STEP_SIZE = 0.03
END_TIME = 20
COUPLER_CHECKPOINT_INTERVAL_STEPS = 20
SAMPLE_SPACING = SURFACE_CELL_SIZE * 2
TRANSFER_DIAGNOSTIC_INTERVAL = 12

# Case files and derived sampling data
CASE_DIR = Path(__file__).resolve().parent
CUBE_STL = CASE_DIR / "assets" / "cube.stl"
BODY_STL = str(CUBE_STL)
OFFAXIS_Y = 0.75 * CUBE_SIDE
SLICE_BOUNDS = [FVM_BOX[0], FVM_BOX[1], FVM_BOX[2], FVM_BOX[3]]
WAKE_SLICE_BOUNDS = [0.0, 5.0, -1.5, 1.5]

FVM_MESH = fvm.AdaptiveCartesianMesher(
    domain=FVM_BOX,
    max_cell_size=SURFACE_CELL_SIZE * 4,
    surface_file=CUBE_STL,
    wall_patch_name="cube",
    surface_cell_size=SURFACE_CELL_SIZE,
    refinements=(fvm.BoxRefinement(FVM_WAKE_BOX, SURFACE_CELL_SIZE * 2, "wakeBox"),),
    merge_outer_patch="numericalBoundary",
)

FVM_SAMPLERS = (
    fvm.ForceSampler(
        patch_names=["cube"],
        ref_velocity=np.linalg.norm(FREESTREAM_VELOCITY),
        ref_area=CUBE_SIDE**2,
        ref_length=CUBE_SIDE,
        moment_centre=[0.0, 0.0, 0.0],
        schedule=fvm.SamplingSchedule(every_time=FORCE_INTERVAL),
    ),
    fvm.LineSampler(
        start=[FVM_BOX[0], 0.0, 0.0],
        end=[FVM_BOX[1], 0.0, 0.0],
        spacing=SAMPLE_SPACING,
        file_name="fvm_centerline",
        schedule=fvm.SamplingSchedule(every_time=DIAGNOSTIC_INTERVAL),
    ),
    fvm.LineSampler(
        start=[FVM_BOX[0], OFFAXIS_Y, 0.0],
        end=[FVM_BOX[1], OFFAXIS_Y, 0.0],
        spacing=SAMPLE_SPACING,
        file_name="fvm_offaxis_y075",
        schedule=fvm.SamplingSchedule(every_time=DIAGNOSTIC_INTERVAL),
    ),
    fvm.SurfaceSampler(
        point=[0.0, 0.0, 0.0],
        normal=[0, 0, 1],
        bounds=SLICE_BOUNDS,
        spacing=SAMPLE_SPACING,
        file_name="fvm_slice_z0",
        schedule=fvm.SamplingSchedule(every_time=DIAGNOSTIC_INTERVAL),
        body_bounds=FVM_MESH.surface_bounds,
    ),
)

FVM_SETUP = fvm.FVMSetup(
    case_name="coupled_hybridFlow",
    cores=FVM_CORES,
    execution=fvm.ComputeConfig(operator_backend="numba"),
    output=fvm.OutputConfig(
        format="vtk_xml",
        data_location="cell",
        encoding="appended",
        compression="lz4",
        precision="float32",
        asynchronous=True,
        ghost_layers=0,
    ),
    time=fvm.TimeConfig(
        time_step_size=FVM_TIME_STEP_SIZE,
        start_time=0.0,
        end_time=END_TIME,
        output_interval_steps=10**9,
        output_interval_time=FVM_VOLUME_INTERVAL,
        adjust_time_step=False,
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
        n_outer_correctors=1,
        n_orthogonal_correctors=0,
        velocity_relaxation=0.7,
        pressure_relaxation=0.3,
    ),
    samplers=FVM_SAMPLERS,
    transport=fvm.TransportConfig(density=DENSITY, kinematic_viscosity=KINEMATIC_VISCOSITY),
    turbulence=fvm.TurbulenceConfig.equilibrium_smagorinsky(
        c_k=SMAGORINSKY_CK,
        c_e=SMAGORINSKY_CE,
    ),
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
    transfer_region_bounds=TRANSFER_REGION_BOX,
    bc_resync_after_transfer=True,
    pressure_anchor_to_freestream=False,
    checkpoint_interval_steps=COUPLER_CHECKPOINT_INTERVAL_STEPS,
    boundary_condition_mode=VPM_BC_MODE,
    vpm_particle_spacing=VPM_PARTICLE_SPACING,
    vpm_core_radius_ratio=VPM_CORE_RADIUS_RATIO,
    authority_ramp_width=OVERLAP_ZONE_RAMP_WIDTH,
    vpm_only_width=0.0,
    transfer_vorticity_cutoff=TRANSFER_PRUNE_VORTICITY_MIN,
    transfer_boundary_prune_multiplier=TRANSFER_BOUNDARY_PRUNE_MULTIPLIER,
    transfer_max_particles=PARTICLE_LIMIT,
    transfer_amplification_cap=TRANSFER_AMPLIFICATION_CAP,
    transfer_diagnostic_interval_steps=TRANSFER_DIAGNOSTIC_INTERVAL,
)


VPM_SAMPLERS = (
    vpm.LineSampler(
        start=[VPM_DOMAIN[0], 0.0, 0.0],
        end=[VPM_DOMAIN[1], 0.0, 0.0],
        spacing=SAMPLE_SPACING,
        file_name="vpm_centerline",
    ),
    vpm.LineSampler(
        start=[VPM_DOMAIN[0], OFFAXIS_Y, 0.0],
        end=[VPM_DOMAIN[1], OFFAXIS_Y, 0.0],
        spacing=SAMPLE_SPACING,
        file_name="vpm_offaxis_y075",
    ),
    vpm.SurfaceSampler(
        point=[0.0, 0.0, 0.0],
        normal=[0, 0, 1],
        bounds=SLICE_BOUNDS,
        spacing=SAMPLE_SPACING,
        file_name="vpm_slice_z0",
        include_derivatives=False,
    ),
    vpm.SurfaceSampler(
        point=[0.0, 0.0, 0.0],
        normal=[0, 0, 1],
        bounds=WAKE_SLICE_BOUNDS,
        spacing=SAMPLE_SPACING,
        file_name="vpm_wake_slice_z0",
        include_derivatives=False,
    ),
)
VPM_PANEL_SOLVER = vpm.PanelSolver(
    max_panels=128,
    float_dtype="f32",
    linear_solver="BICGSTAB_GPU",
    bc_type="NEUMANN",
    density=DENSITY,
    freestream_velocity=np.asarray(FREESTREAM_VELOCITY),
    coupling_scope="vpm_bc",
)
VPM_SETUP = vpm.VPMSetup(
    time_step_size=VPM_TIME_STEP_SIZE,
    freestream_velocity=list(FREESTREAM_VELOCITY),
    viscous=vpm.ViscousConfig.gbd(
        particle_spacing=VPM_PARTICLE_SPACING,
        padding=3.0,
        kinematic_viscosity=KINEMATIC_VISCOSITY,
        threshold_mode="absolute",
        threshold=GBD_VORTICITY_FLOOR * VPM_PARTICLE_SPACING**3,
        max_nodes=PARTICLE_LIMIT,
        cap_absolute_fraction=0.95,
        core_radius_ratio=VPM_CORE_RADIUS_RATIO,
    ),
    stretching=vpm.StretchingConfig.transposed(scheme=VPM_SCHEME),
    advection=vpm.AdvectionConfig(scheme=VPM_SCHEME),
    turbulence=vpm.TurbulenceConfig.equilibrium_smagorinsky(
        c_k=SMAGORINSKY_CK,
        c_e=SMAGORINSKY_CE,
    ),
    velocity=vpm.VelocityConfig.treecode(theta=0.3, multipole_order=2),
    stabilization=vpm.StabilizationConfig.bounded_domain(VPM_DOMAIN),
    particle_kernel="GAUSSIAN",
    precision="f32",
    compute_device="AUTO",
    max_particles=PARTICLE_LIMIT,
    max_evaluation_points=PARTICLE_LIMIT,
    domain_bounds=list(VPM_DOMAIN),
    log_mode="file",
    logging_interval_steps=VPM_LOG_PERIOD,
    timing_interval_steps=VPM_LOG_PERIOD,
    checkpoint_interval_steps=int(CHECKPOINT_INTERVAL / VPM_TIME_STEP_SIZE),
    checkpoint_directory=str(CASE_DIR / "solution"),
    export_flow_integrals=False,
    samplers=VPM_SAMPLERS,
    panel_solver=VPM_PANEL_SOLVER,
    body_stl=BODY_STL,
)


def main() -> None:
    fvm_solver = fvm.create_fvm_solver(FVM_SETUP, case_dir=CASE_DIR, mesh=FVM_MESH)
    fvm_solver.write_vtk()
    vpm_solver = vpm.create_vpm_solver(VPM_SETUP, case_dir=CASE_DIR)
    coupled_solver = coupling.create_coupler(fvm_solver, vpm_solver, COUPLER_SETUP)
    coupled_solver.run()


if __name__ == "__main__":
    main()
