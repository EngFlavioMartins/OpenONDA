"""Coupled LES FVM–VPM simulation of flow past a cube at Re = 1000.

The FVM mesh is generated directly as solver-native data by OpenONDA's
adaptive Cartesian mesher. No external solver case is used. Both solvers use
the same equilibrium Smagorinsky coefficients.

All case parameters are kept below in one explicit configuration block. Edit
them here to define a different case.

Usage:
    python cube_flow_setup.py
"""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path

import numpy as np

import openonda.fvm as fvm
import openonda.coupler as coupling
import openonda.vpm as vpm


def resolve_case_timing(
    fvm_time_step_size: float,
    vpm_time_step_multiplier: int,
    write_solution_backup: float,
    sampling_period: float,
) -> tuple[float, float, int, int, int, int]:
    """Resolve compatible solver steps and exact integer output intervals."""
    if (
        fvm_time_step_size <= 0.0
        or vpm_time_step_multiplier < 1
        or int(vpm_time_step_multiplier) != vpm_time_step_multiplier
    ):
        raise ValueError("time-step size must be positive and VPM multiplier must be an integer")
    if write_solution_backup <= 0.0 or sampling_period <= 0.0:
        raise ValueError("backup and sampling periods must be positive")

    vpm_steps_per_sample = max(
        1, round(sampling_period / (vpm_time_step_multiplier * fvm_time_step_size))
    )
    vpm_time_step_size = sampling_period / vpm_steps_per_sample
    fvm_time_step_size = vpm_time_step_size / vpm_time_step_multiplier
    intervals = (
        round(write_solution_backup / fvm_time_step_size),
        round(write_solution_backup / vpm_time_step_size),
        round(sampling_period / fvm_time_step_size),
        round(sampling_period / vpm_time_step_size),
    )
    periods = (write_solution_backup, write_solution_backup, sampling_period, sampling_period)
    steps = (fvm_time_step_size, vpm_time_step_size) * 2
    if any(
        not np.isclose(interval * step, period, rtol=0.0, atol=1.0e-12)
        for interval, step, period in zip(intervals, steps, periods, strict=True)
    ):
        raise ValueError("backup and sampling periods must be mutually compatible")
    return fvm_time_step_size, vpm_time_step_size, *intervals


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
FVM_BOX = (-1.5, 1.5, -1.5, 1.5, -1.5, 1.5)
FVM_WAKE_BOX = (-1.25, 1.25, -1.25, 1.25, -1.25, 1.25)
TRANSFER_REGION_BOX = FVM_WAKE_BOX
SURFACE_CELL_SIZE = 0.015625
FVM_MAX_CELL_SIZE = 0.25
PIMPLE_CORRECTORS = 2

# VPM domain and resolution
VPM_DOMAIN = (-4.5, 12.0, -3.0, 3.0, -3.0, 3.0)
PARTICLE_LIMIT = 1_500_000
VPM_CORE_RADIUS_RATIO = 1.0
GBD_VORTICITY_FLOOR = 0.02
VPM_PARTICLE_SPACING = 2 * SURFACE_CELL_SIZE
ETA_BLEND_WIDTH = 6 * VPM_PARTICLE_SPACING
VPM_VISCOUS_SCHEME = "GBD"

# Coupling
BOUNDARY_CONDITION_MODE = "vorticity_mixed"
TRANSFER_METHOD = "buffered_m4_renewal"
TRANSFER_VORTICITY_CUTOFF = 0.05
TRANSFER_BOUNDARY_PRUNE_MULTIPLIER = 10.0
TRANSFER_AMPLIFICATION_CAP = 1.8
FVM_CONSISTENCY_WIDTH = FVM_BOX[1] - TRANSFER_REGION_BOX[1]

# Time and output
END_TIME = 20.0
SAMPLING_INTERVAL_TIME = 0.050
WRITE_SOLUTION_BACKUP = 0.5
VPM_CHECKPOINT_RETENTION = 2
VPM_TIME_STEP_MULTIPLIER = 5
(
    FVM_TIME_STEP_SIZE,
    VPM_TIME_STEP_SIZE,
    FVM_WRITE_SOLUTION_BACKUP_INTERVAL_STEPS,
    VPM_WRITE_SOLUTION_BACKUP_INTERVAL_STEPS,
    FVM_SAMPLING_INTERVAL_STEPS,
    VPM_SAMPLING_INTERVAL_STEPS,
) = resolve_case_timing(
    fvm_time_step_size=0.010,
    vpm_time_step_multiplier=VPM_TIME_STEP_MULTIPLIER,
    write_solution_backup=WRITE_SOLUTION_BACKUP,
    sampling_period=SAMPLING_INTERVAL_TIME,
)
VPM_LOGGING_INTERVAL_STEPS = 20
SAMPLE_SPACING = VPM_PARTICLE_SPACING
TRANSFER_DIAGNOSTIC_INTERVAL_STEPS = 10

# Case files and derived sampling data
CASE_DIR = Path(__file__).resolve().parent
CUBE_STL = CASE_DIR / "assets" / "cube.stl"
BODY_STL = str(CUBE_STL)
OFFAXIS_Y = 0.75 * CUBE_SIDE
SLICE_BOUNDS = [FVM_BOX[0], FVM_BOX[1], FVM_BOX[2], FVM_BOX[3]]
WAKE_SLICE_BOUNDS = [0.0, 5.0, -1.5, 1.5]

FVM_MESH = fvm.AdaptiveCartesianMesher(
    domain=FVM_BOX,
    max_cell_size=FVM_MAX_CELL_SIZE,
    surface_file=CUBE_STL,
    wall_patch_name="cube",
    surface_cell_size=SURFACE_CELL_SIZE,
    refinements=(fvm.BoxRefinement(FVM_WAKE_BOX, SURFACE_CELL_SIZE * 2, "wakeBox"),),
    merge_outer_patch="numericalBoundary",
)

FVM_SAMPLING_SCHEDULE = fvm.SamplingSchedule(every_n_steps=FVM_SAMPLING_INTERVAL_STEPS)
VPM_SAMPLING_SCHEDULE = vpm.SamplingSchedule(every_n_steps=VPM_SAMPLING_INTERVAL_STEPS)

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
        file_name="fvm_centreline",
        schedule=FVM_SAMPLING_SCHEDULE,
    ),
    fvm.LineSampler(
        start=[FVM_BOX[0], OFFAXIS_Y, 0.0],
        end=[FVM_BOX[1], OFFAXIS_Y, 0.0],
        spacing=SAMPLE_SPACING,
        file_name="fvm_offaxis_y075",
        schedule=FVM_SAMPLING_SCHEDULE,
    ),
    fvm.SurfaceSampler(
        point=[0.0, 0.0, 0.0],
        normal=[0, 0, 1],
        bounds=SLICE_BOUNDS,
        spacing=SAMPLE_SPACING,
        file_name="fvm_slice_z0",
        schedule=FVM_SAMPLING_SCHEDULE,
        body_bounds=FVM_MESH.surface_bounds,
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
        output_interval_steps=FVM_WRITE_SOLUTION_BACKUP_INTERVAL_STEPS,
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
        n_outer_correctors=2,
        n_orthogonal_correctors=1,
        velocity_relaxation=0.7,
        pressure_relaxation=0.3,
    ),
    samplers=FVM_SAMPLERS,
    transport=fvm.TransportConfig(density=DENSITY, kinematic_viscosity=KINEMATIC_VISCOSITY),
    turbulence=fvm.TurbulenceConfig.equilibrium_smagorinsky(
        subgrid_kinetic_energy_coefficient=SMAGORINSKY_CK,
        subgrid_dissipation_coefficient=SMAGORINSKY_CE,
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
    transfer_method=TRANSFER_METHOD,
    transfer_region_bounds=TRANSFER_REGION_BOX,
    checkpoint_interval_steps=VPM_WRITE_SOLUTION_BACKUP_INTERVAL_STEPS,
    boundary_condition_mode=BOUNDARY_CONDITION_MODE,
    fvm_consistency_width=FVM_CONSISTENCY_WIDTH,
    vpm_checkpoint_retention=VPM_CHECKPOINT_RETENTION,
    eta_blend_width=ETA_BLEND_WIDTH,
    vpm_only_width=0.0,
    transfer_vorticity_cutoff=TRANSFER_VORTICITY_CUTOFF,
    transfer_boundary_prune_multiplier=TRANSFER_BOUNDARY_PRUNE_MULTIPLIER,
    transfer_amplification_cap=TRANSFER_AMPLIFICATION_CAP,
    transfer_diagnostic_interval_steps=TRANSFER_DIAGNOSTIC_INTERVAL_STEPS,
)


def make_vpm_viscous_config(scheme: str) -> vpm.ViscousConfig:
    """Build any VPM diffusion scheme with the coupled spatial resolution."""
    name = scheme.upper()
    common = {
        "particle_spacing": VPM_PARTICLE_SPACING,
        "core_radius_ratio": VPM_CORE_RADIUS_RATIO,
    }
    if name == "CS":
        return vpm.ViscousConfig.cs(
            kinematic_viscosity=KINEMATIC_VISCOSITY,
            **common,
        )
    if name == "RWM":
        return vpm.ViscousConfig.rwm(
            kinematic_viscosity=KINEMATIC_VISCOSITY,
            **common,
        )
    if name == "DVH":
        return vpm.ViscousConfig.dvh(
            padding=5.0,
            kinematic_viscosity=KINEMATIC_VISCOSITY,
            threshold_mode="absolute",
            threshold=GBD_VORTICITY_FLOOR * VPM_PARTICLE_SPACING**3,
            max_nodes=PARTICLE_LIMIT,
            **common,
        )
    if name == "GBD":
        return vpm.ViscousConfig.gbd(
            padding=5.0,
            kinematic_viscosity=KINEMATIC_VISCOSITY,
            threshold_mode="absolute",
            threshold=GBD_VORTICITY_FLOOR * VPM_PARTICLE_SPACING**3,
            max_nodes=PARTICLE_LIMIT,
            **common,
        )
    if name == "NONE":
        return vpm.ViscousConfig.inviscid(**common)
    raise ValueError(f"Unsupported VPM viscous scheme {scheme!r}")


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
    coupling_scope="vpm_boundary_condition",
)
VPM_SETUP = vpm.VPMSetup(
    time_step_size=VPM_TIME_STEP_SIZE,
    time_integration="COUPLED",
    coupled_max_strain_increment=None,
    coupled_max_advection_fraction=None,
    freestream_velocity=list(FREESTREAM_VELOCITY),
    viscous=make_vpm_viscous_config(VPM_VISCOUS_SCHEME),
    stretching=vpm.StretchingConfig.transposed(scheme=VPM_SCHEME),
    advection=vpm.AdvectionConfig(scheme=VPM_SCHEME),
    turbulence=vpm.TurbulenceConfig.equilibrium_smagorinsky(
        subgrid_kinetic_energy_coefficient=SMAGORINSKY_CK,
        subgrid_dissipation_coefficient=SMAGORINSKY_CE,
    ),
    velocity=vpm.VelocityConfig.treecode(theta=0.3, multipole_order=2),
    stabilization=vpm.StabilizationConfig.bounded_domain(VPM_DOMAIN),
    particle_kernel="GAUSSIAN",
    precision="f32",
    compute_device="AUTO",
    max_n_particles=PARTICLE_LIMIT,
    max_evaluation_points=PARTICLE_LIMIT,
    domain_bounds=list(VPM_DOMAIN),
    log_mode="file",
    logging_interval_steps=VPM_LOGGING_INTERVAL_STEPS,
    timing_interval_steps=VPM_LOGGING_INTERVAL_STEPS,
    write_precision="f32",
    checkpoint_store_velocity_gradient=False,
    # Coupled runs use the atomic FVM+VPM checkpoint owned by COUPLER_SETUP.
    checkpoint_interval_steps=0,
    checkpoint_directory=str(CASE_DIR / "solution"),
    export_flow_integrals=False,
    samplers=VPM_SAMPLERS,
    panel_solver=VPM_PANEL_SOLVER,
    bodies=(vpm.PanelBodySetup(stl=BODY_STL, uid="body", reference_area=CUBE_SIDE**2),),
)


def main(
    *,
    restart_from: Path | None = None,
    restart_allowed_config_differences: Collection[str] = (),
    max_coupling_steps: int | None = None,
    checkpoint_at_stop: bool = False,
) -> int:
    fvm_solver = fvm.create_fvm_solver(FVM_SETUP, case_dir=CASE_DIR, mesh=FVM_MESH)
    if restart_from is None:
        fvm_solver.write_vtk()
    vpm_solver = vpm.create_vpm_solver(VPM_SETUP, case_dir=CASE_DIR)
    coupled_solver = coupling.create_coupler(fvm_solver, vpm_solver, COUPLER_SETUP)
    return coupled_solver.run(
        restart_from=restart_from,
        restart_allowed_config_differences=restart_allowed_config_differences,
        max_coupling_steps=max_coupling_steps,
        checkpoint_at_stop=checkpoint_at_stop,
    )


if __name__ == "__main__":
    main()
