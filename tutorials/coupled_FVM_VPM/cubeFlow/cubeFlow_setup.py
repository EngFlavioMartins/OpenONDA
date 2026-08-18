"""Hybrid LES FVM–VPM simulation of flow past a cube at Re = 1000.

The FVM mesh is generated directly as solver-native data by OpenONDA's
adaptive Cartesian mesher. No external solver case is used. Both solvers use
the same equilibrium Smagorinsky coefficients.

All case parameters are kept below in one explicit configuration block. Edit
them here to define a different case.

Usage:
    python cubeFlow_setup.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from openonda.coupler import CouplerSetup, FVMVPMCoupler, setup_coupler
from openonda.fvm import (
    AdaptiveCartesianMesher,
    BoundaryConfig,
    BoxRefinement,
    ExecutionConfig,
    ForceSampler,
    FVMSetup,
    LinearSolverConfig,
    LineSampler as FVMLineSampler,
    OutputSetup,
    PimpleControl,
    SchemesConfig,
    SurfaceSampler as FVMSurfaceSampler,
    TimeConfig,
    TransportConfig,
    TurbulenceConfig as FVMTurbulenceConfig,
    setup_fvm_solver,
)
from openonda.fvm import SamplingSchedule

# Physical problem
CUBE_SIDE = 1.0
U_INF = (1.0, 0.0, 0.0)
RHO = 1.0
REYNOLDS = 1000.0
NU = np.linalg.norm(U_INF) * CUBE_SIDE / REYNOLDS
SMAGORINSKY_CK = 0.094
SMAGORINSKY_CE = 1.048
INITIAL_U = (1.0, 0.0, 0.0)

# Time integration.  OPENONDA_SMOKE=1 shortens the run to a couple of coupling
# steps so the configuration can be exercised without committing hours to it.
SMOKE = os.environ.get("OPENONDA_SMOKE", "0") == "1"
DT_FVM = 0.01
DT_VPM = 0.05
T_END = float(os.environ.get("OPENONDA_T_END", "0.10" if SMOKE else "6.0"))
VPM_SCHEME = "RK2"

# FVM domain and mesh
# Partitioned FVM-VPM coupling has not yet been qualified by a collective
# regression, so keep this production tutorial on the supported serial path.
FVM_CORES = 1
HANDOFF_BOX = (-1.5, 3.2, -1.5, 1.5, -1.5, 1.5)
FVM_BOX = (-1.5, 3.5, -1.5, 1.5, -1.5, 1.5)
# Resolve the body, separation region, and near wake; VPM transports the
# downstream wake after the handoff instead of duplicating that cost in FVM.
FVM_WAKE_BOX = (-0.75, 2.0, -0.9, 0.9, -0.9, 0.9)
FVM_CELL_SIZE = 0.0625
FVM_WAKE_CELL_SIZE = 0.03125
SURFACE_CELL_SIZE = 0.015625

# VPM domain and resolution
# Memory: GBD pre-allocates a fixed diffusion grid over the WHOLE VPM domain
# at spacing h, so its cost scales with domain volume and h^-3 -- 686 MB at the
# defaults below, on top of particle arrays sized by PARTICLE_LIMIT.  On a
# memory-constrained machine that allocation dominates start-up: the FVM
# initialises in ~3 s while the VPM can take tens of minutes paging.
# OPENONDA_VPM_COMPACT=1 trims the domain to what the cube wake actually needs
# (256 MB); it still contains the FVM box with 2x margin in y and z.
_VPM_COMPACT = os.environ.get("OPENONDA_VPM_COMPACT", "0") == "1"
VPM_DOMAIN = (
    (-4.5, 8.0, -3.0, 3.0, -3.0, 3.0) if _VPM_COMPACT else (-4.5, 11.0, -4.5, 4.5, -4.5, 4.5)
)
PARTICLE_SPACING = 0.04
PARTICLE_LIMIT = int(
    os.environ.get("OPENONDA_MAX_PARTICLES", "1200000" if _VPM_COMPACT else "2000000")
)
OVERLAP_RADIUS_RATIO = 1.0
PRUNE_VORTICITY_MIN = 0.005
BOUNDARY_PRUNE_MULTIPLIER = 10.0
GBD_THRESHOLD = 0.30
BUFFER_THICKNESS = 0.24

# Coupling
# The mixed condition preserves the tangential vorticity implied by the VPM
# trace without constraining the tangential velocity itself.
VPM_BC_MODE = os.environ.get("OPENONDA_VPM_BC_MODE", "vorticity_mixed")
TRANSFER_AMPLIFICATION_CAP = 1.8

# Output and diagnostics
FORCE_INTERVAL = 0.05
DIAGNOSTIC_INTERVAL = 0.60
CHECKPOINT_INTERVAL = 1.0
FVM_VOLUME_INTERVAL = 1.0
VPM_LOG_PERIOD = 12
BACKUP_PERIOD = 20
SAMPLE_SPACING = 0.04
HANDOFF_DIAGNOSTIC_INTERVAL = 12

# Case files and derived sampling data
CASE_DIR = Path(__file__).resolve().parent
CUBE_STL = CASE_DIR / "assets" / "cube.stl"
BODY_STL = str(CUBE_STL)
OFFAXIS_Y = 0.75 * CUBE_SIDE
SLICE_BOUNDS = [FVM_BOX[0], FVM_BOX[1], FVM_BOX[2], FVM_BOX[3]]
WAKE_SLICE_BOUNDS = [0.0, 5.0, -1.5, 1.5]

FVM_SAMPLERS = (
    ForceSampler(
        patch_names=["cube"],
        ref_velocity=np.linalg.norm(U_INF),
        ref_area=CUBE_SIDE**2,
        ref_length=CUBE_SIDE,
        moment_centre=[0.0, 0.0, 0.0],
        schedule=SamplingSchedule(every_time=FORCE_INTERVAL),
    ),
    FVMLineSampler(
        start=[FVM_BOX[0], 0.0, 0.0],
        end=[FVM_BOX[1], 0.0, 0.0],
        spacing=SAMPLE_SPACING,
        file_name="fvm_centerline",
        schedule=SamplingSchedule(every_time=DIAGNOSTIC_INTERVAL),
    ),
    FVMLineSampler(
        start=[FVM_BOX[0], OFFAXIS_Y, 0.0],
        end=[FVM_BOX[1], OFFAXIS_Y, 0.0],
        spacing=SAMPLE_SPACING,
        file_name="fvm_offaxis_y075",
        schedule=SamplingSchedule(every_time=DIAGNOSTIC_INTERVAL),
    ),
    FVMSurfaceSampler(
        point=[0.0, 0.0, 0.0],
        normal=[0, 0, 1],
        bounds=SLICE_BOUNDS,
        spacing=SAMPLE_SPACING,
        file_name="fvm_slice_z0",
        schedule=SamplingSchedule(every_time=DIAGNOSTIC_INTERVAL),
    ),
)

FVM_MESH = AdaptiveCartesianMesher(
    domain=FVM_BOX,
    max_cell_size=FVM_CELL_SIZE,
    surface_file=CUBE_STL,
    wall_patch_name="cube",
    surface_cell_size=SURFACE_CELL_SIZE,
    refinements=(BoxRefinement(FVM_WAKE_BOX, FVM_WAKE_CELL_SIZE, "wakeBox"),),
    merge_outer_patch="numericalBoundary",
)

FVM_SETUP = FVMSetup(
    case_name="coupled_hybridFlow",
    cores=FVM_CORES,
    execution=ExecutionConfig(operator_backend="numba"),
    output=OutputSetup(
        format="vtk_xml",
        data_location="cell",
        encoding="appended",
        compression="lz4",
        precision="float32",
        asynchronous=True,
        ghost_layers=0,
    ),
    time=TimeConfig(
        delta_t=DT_FVM,
        start_time=0.0,
        end_time=T_END,
        write_interval=10**9,
        write_interval_time=FVM_VOLUME_INTERVAL,
        adjust_timestep=False,
    ),
    schemes=SchemesConfig(
        convection_scheme="linearUpwind",
        gradient_scheme="gauss",
        time_scheme="backward",
    ),
    linear=LinearSolverConfig(
        linear_solver="bicgstab",
        pressure_solver="amg",
        pressure_tol=1e-6,
        pressure_rel_tol=0.01,
        pressure_final_rel_tol=0.0,
        momentum_tol=1e-6,
        momentum_rel_tol=0.1,
        momentum_final_rel_tol=0.0,
        momentum_maxiter=2000,
        ilu_drop_tol=1e-4,
        ilu_fill_factor=10.0,
        ilu_reuse_tol=0.05,
    ),
    pimple=PimpleControl(
        n_correctors=2,
        n_outer_correctors=1,
        n_orthogonal_correctors=0,
        alpha_u=0.7,
        alpha_p=0.3,
    ),
    samplers=FVM_SAMPLERS,
    transport=TransportConfig(density=RHO, nu=NU),
    turbulence=FVMTurbulenceConfig.equilibrium_smagorinsky(
        Ck=SMAGORINSKY_CK,
        Ce=SMAGORINSKY_CE,
    ),
    boundaries=[
        BoundaryConfig(
            name="numericalBoundary",
            type_U="fixedValue",
            value_U=list(U_INF),
            type_p="fixedFluxPressure",
        ),
        BoundaryConfig.wall("cube"),
    ],
    initial_U=list(INITIAL_U),
    initial_p=0.0,
)

COUPLER_SETUP = CouplerSetup(
    u_inf=list(U_INF),
    handoff_box=HANDOFF_BOX,
    vpm_bc_mode=VPM_BC_MODE,
    h=PARTICLE_SPACING,
    buffer_thickness=BUFFER_THICKNESS,
    dead_zone_h=0.0,
    prune_vorticity_min=PRUNE_VORTICITY_MIN,
    boundary_prune_multiplier=BOUNDARY_PRUNE_MULTIPLIER,
    handoff_max_particles=PARTICLE_LIMIT,
    overlap_radius_ratio=OVERLAP_RADIUS_RATIO,
    transfer_amplification_cap=TRANSFER_AMPLIFICATION_CAP,
    handoff_diagnostic_interval=HANDOFF_DIAGNOSTIC_INTERVAL,
    resync_vpm_bc_after_handoff=True,
    # A pressure gauge shift is reporting-only and is intentionally kept out
    # of the coupled numerical hot path.
    anchor_pressure=False,
    backup_period=BACKUP_PERIOD,
)


def make_vpm_setup():
    """Build the VPM setup on the MPI rank that owns the VPM solver."""
    from openonda.vpm import (
        AdvectionConfig,
        LineSampler,
        PanelSolver,
        StabilizationConfig,
        StretchingConfig,
        SurfaceSampler,
        TurbulenceConfig,
        VelocityConfig,
        ViscousConfig,
        VPMSetup,
    )

    samplers = (
        LineSampler(
            start=[VPM_DOMAIN[0], 0.0, 0.0],
            end=[VPM_DOMAIN[1], 0.0, 0.0],
            spacing=SAMPLE_SPACING,
            file_name="vpm_centerline",
        ),
        LineSampler(
            start=[VPM_DOMAIN[0], OFFAXIS_Y, 0.0],
            end=[VPM_DOMAIN[1], OFFAXIS_Y, 0.0],
            spacing=SAMPLE_SPACING,
            file_name="vpm_offaxis_y075",
        ),
        SurfaceSampler(
            point=[0.0, 0.0, 0.0],
            normal=[0, 0, 1],
            bounds=SLICE_BOUNDS,
            spacing=SAMPLE_SPACING,
            file_name="vpm_slice_z0",
            include_derivatives=False,
        ),
        SurfaceSampler(
            point=[0.0, 0.0, 0.0],
            normal=[0, 0, 1],
            bounds=WAKE_SLICE_BOUNDS,
            spacing=SAMPLE_SPACING,
            file_name="vpm_wake_slice_z0",
            include_derivatives=False,
        ),
    )
    panel_solver = PanelSolver(
        max_panels=128,
        float_dtype="f32",
        linear_solver="BICGSTAB_GPU",
        bc_type="NEUMANN",
        density=RHO,
        U_inf=np.asarray(U_INF),
        coupling_scope="vpm_bc",
    )
    return VPMSetup(
        time_step_size=DT_VPM,
        background_velocity=list(U_INF),
        viscous=ViscousConfig.gbd(
            h=PARTICLE_SPACING,
            padding=3.0,
            viscosity=NU,
            threshold_mode="relative_local",
            threshold=GBD_THRESHOLD,
            max_nodes=PARTICLE_LIMIT,
            cap_abs_fraction=0.95,
            regen_radius_ratio=OVERLAP_RADIUS_RATIO,
        ),
        stretching=StretchingConfig.transposed(scheme=VPM_SCHEME),
        advection=AdvectionConfig(scheme=VPM_SCHEME),
        turbulence=TurbulenceConfig.equilibrium_smagorinsky(
            ck=SMAGORINSKY_CK,
            ce=SMAGORINSKY_CE,
        ),
        velocity=VelocityConfig.treecode(theta=0.3, multipole_order=2),
        stabilization=StabilizationConfig.bounded_domain(VPM_DOMAIN),
        particles_kernel="GAUSSIAN",
        precision="f32",
        processing_unit="AUTO",
        max_particles=PARTICLE_LIMIT,
        max_targets=PARTICLE_LIMIT,
        vpm_domain_bounds=list(VPM_DOMAIN),
        log_mode="file",
        logging_frequency=VPM_LOG_PERIOD,
        timing_frequency=VPM_LOG_PERIOD,
        backup_frequency=0,
        backup_directory=str(CASE_DIR / "solution"),
        export_flow_integrals=False,
        samplers=samplers,
        panel_solver=panel_solver,
        body_stl=BODY_STL,
    )


def main() -> None:
    fvm_solver = setup_fvm_solver(FVM_SETUP, case_dir=CASE_DIR, mesh=FVM_MESH)
    fvm_solver.write_vtk()

    is_master = FVMVPMCoupler.is_master_rank()
    vpm_solver = None
    if is_master:
        from openonda.vpm import setup_vpm_solver

        vpm_solver = setup_vpm_solver(make_vpm_setup())
        print("\n===== SIMULATION =====")
        print(
            f"  FVM dt={DT_FVM}s / VPM dt={DT_VPM}s, "
            f"FVM cell={FVM_CELL_SIZE}, particle h={PARTICLE_SPACING}, "
            f"VPM scheme={VPM_SCHEME}, "
            f"sample spacing={SAMPLE_SPACING}, particles<={PARTICLE_LIMIT}"
        )

    coupled_solver = setup_coupler(vpm_solver, fvm_solver, COUPLER_SETUP)

    coupled_solver.run()
    if is_master:
        print("\n===== DONE =====")
        print("Simulation completed successfully. Run ./allplot.sh to make the figures.")


if __name__ == "__main__":
    main()
