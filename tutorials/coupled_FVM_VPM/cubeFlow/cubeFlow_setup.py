"""Hybrid LES FVM–VPM simulation of flow past a cube at Re = 1000.

The FVM mesh is generated directly as solver-native data by OpenONDA's
adaptive Cartesian mesher. No external solver case is used. Both solvers use
the same equilibrium Smagorinsky coefficients.

The case is configured through ``OPENONDA_*`` environment variables. The
recommended, explicit configuration lives in ``allrun.sh`` so every run is
auditable without a second command-line configuration layer.

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
from openonda.vpm import (
    AdvectionConfig,
    LineSampler as VPMLineSampler,
    PanelSolver,
    StabilizationConfig,
    StretchingConfig,
    SurfaceSampler as VPMSurfaceSampler,
    TurbulenceConfig as VPMTurbulenceConfig,
    VelocityConfig,
    ViscousConfig,
    VPMSetup,
    setup_vpm_solver,
)

CASE_DIR = Path(__file__).resolve().parent
SMOKE = os.environ.get("OPENONDA_SMOKE", "0") == "1"
CUBE_STL = CASE_DIR / "assets" / "cube.stl"
BODY_STL = str(CUBE_STL)

CUBE_SIDE = 1.0
U_INF = (1.0, 0.0, 0.0)
RHO = 1.0
REYNOLDS = 1000.0
NU = np.linalg.norm(U_INF) * CUBE_SIDE / REYNOLDS
SMAGORINSKY_CK = 0.094
SMAGORINSKY_CE = 1.048
INITIAL_U = (1.0, 0.0, 0.0)
FVM_XMAX = float(os.environ.get("OPENONDA_FVM_XMAX", "1.5"))
if FVM_XMAX <= 0.5:
    raise ValueError("OPENONDA_FVM_XMAX must lie downstream of the cube")
HANDOFF_BOX = (-1.5, FVM_XMAX, -1.5, 1.5, -1.5, 1.5)
FVM_OVERLAP_BUFFER = float(os.environ.get("OPENONDA_FVM_OVERLAP_BUFFER", "0.0"))
if FVM_OVERLAP_BUFFER < 0.0:
    raise ValueError("OPENONDA_FVM_OVERLAP_BUFFER must be non-negative")
FVM_BOX = (
    HANDOFF_BOX[0] - FVM_OVERLAP_BUFFER,
    HANDOFF_BOX[1] + FVM_OVERLAP_BUFFER,
    HANDOFF_BOX[2] - FVM_OVERLAP_BUFFER,
    HANDOFF_BOX[3] + FVM_OVERLAP_BUFFER,
    HANDOFF_BOX[4] - FVM_OVERLAP_BUFFER,
    HANDOFF_BOX[5] + FVM_OVERLAP_BUFFER,
)
DT_FVM = 0.01
T_END = float(os.environ.get("OPENONDA_T_END", "0.10" if SMOKE else "20.0"))
FVM_CORES = int(os.environ.get("OPENONDA_FVM_CORES", "1" if SMOKE else "4"))

DT_VPM = float(os.environ.get("OPENONDA_DT_VPM", "0.05"))
VPM_SCHEME = os.environ.get("OPENONDA_VPM_SCHEME", "RK2").upper()
if VPM_SCHEME not in {"RK2", "RK3"}:
    raise ValueError("OPENONDA_VPM_SCHEME must be RK2 or RK3")
DONOR_BOUNDARY_MODE = os.environ.get("OPENONDA_DONOR_BOUNDARY_MODE", "dirichlet")
# Keep the historical OPENONDA_SPACING knob as a common fallback, but do not
# force the Eulerian mesh, particle lattice, and diagnostic grid to have the
# same resolution.  Resolution-matched controls show that this distinction is
# essential when comparing drag against the fully meshed reference.
_COMMON_SPACING = float(os.environ.get("OPENONDA_SPACING", "0.20" if SMOKE else "0.04"))
PARTICLE_SPACING = float(os.environ.get("OPENONDA_PARTICLE_SPACING", _COMMON_SPACING))
FVM_CELL_SIZE = float(os.environ.get("OPENONDA_FVM_CELL_SIZE", _COMMON_SPACING))
SAMPLE_SPACING = float(os.environ.get("OPENONDA_SAMPLE_SPACING", _COMMON_SPACING))
# Backward-compatible import for external case scripts; internal setup uses the
# explicit constants above.
SPACING = PARTICLE_SPACING
SURFACE_CELL_SIZE = float(
    os.environ.get("OPENONDA_SURFACE_CELL_SIZE", "0.10" if SMOKE else "0.015")
)
VPM_DOMAIN = (-4.5, 11.0, -4.5, 4.5, -4.5, 4.5)
PARTICLE_LIMIT = int(os.environ.get("OPENONDA_MAX_PARTICLES", "100000" if SMOKE else "1500000"))
OVERLAP_RADIUS_RATIO = 1.0
OVERLAP_SHELL_PRUNE_MULTIPLIER = float(
    os.environ.get("OPENONDA_OVERLAP_SHELL_PRUNE_MULTIPLIER", "1.0")
)
# Force history remains dense enough for Cd/Strouhal analysis. Field samples,
# raw FVM volumes, and restart checkpoints are independent because they have
# very different costs and are not all needed by the plotting scripts.
FORCE_INTERVAL = float(os.environ.get("OPENONDA_FORCE_INTERVAL", str(DT_VPM if SMOKE else 0.15)))
DIAGNOSTIC_INTERVAL = float(
    os.environ.get("OPENONDA_DIAGNOSTIC_INTERVAL", str(DT_VPM if SMOKE else 0.60))
)
CHECKPOINT_INTERVAL = float(
    os.environ.get("OPENONDA_CHECKPOINT_INTERVAL", str(DT_VPM if SMOKE else 1.0))
)
FVM_VOLUME_INTERVAL = float(
    os.environ.get("OPENONDA_VOLUME_INTERVAL", str(DT_VPM if SMOKE else 1.0))
)


def _step_period(name: str, interval: float, time_step: float) -> int:
    if interval <= 0.0 or time_step <= 0.0:
        raise ValueError(f"{name} and its time step must be positive")
    ratio = interval / time_step
    period = int(round(ratio))
    if period < 1 or not np.isclose(ratio, period, rtol=0.0, atol=1.0e-10):
        raise ValueError(f"{name} must be an integer multiple of {time_step:g} s")
    return period


# Backward-compatible name used by external case checks: the primary scalar
# diagnostic is the force history.
WRITE_INTERVAL = FORCE_INTERVAL
VPM_LOG_PERIOD = _step_period("diagnostic interval", DIAGNOSTIC_INTERVAL, DT_VPM)
BACKUP_PERIOD = _step_period("checkpoint interval", CHECKPOINT_INTERVAL, DT_VPM)
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

VPM_SAMPLERS = (
    VPMLineSampler(
        start=[VPM_DOMAIN[0], 0.0, 0.0],
        end=[VPM_DOMAIN[1], 0.0, 0.0],
        spacing=SAMPLE_SPACING,
        file_name="vpm_centerline",
    ),
    VPMLineSampler(
        start=[VPM_DOMAIN[0], OFFAXIS_Y, 0.0],
        end=[VPM_DOMAIN[1], OFFAXIS_Y, 0.0],
        spacing=SAMPLE_SPACING,
        file_name="vpm_offaxis_y075",
    ),
    VPMSurfaceSampler(
        point=[0.0, 0.0, 0.0],
        normal=[0, 0, 1],
        bounds=SLICE_BOUNDS,
        spacing=SAMPLE_SPACING,
        file_name="vpm_slice_z0",
        include_derivatives=False,
    ),
    VPMSurfaceSampler(
        point=[0.0, 0.0, 0.0],
        normal=[0, 0, 1],
        bounds=WAKE_SLICE_BOUNDS,
        spacing=SAMPLE_SPACING,
        file_name="vpm_wake_slice_z0",
        include_derivatives=False,
    ),
)

FVM_MESH = AdaptiveCartesianMesher(
    domain=FVM_BOX,
    max_cell_size=FVM_CELL_SIZE,
    surface_file=CUBE_STL,
    wall_patch_name="cube",
    surface_cell_size=SURFACE_CELL_SIZE,
    merge_outer_patch="numericalBoundary",
)

PANEL_SOLVER = PanelSolver(
    max_panels=128,
    float_dtype="f32",
    linear_solver="BICGSTAB_GPU",
    bc_type="NEUMANN",
    density=RHO,
    U_inf=np.asarray(U_INF),
    coupling_scope="donor",
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
        n_outer_correctors=2,
        n_orthogonal_correctors=1,
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

VPM_SETUP = VPMSetup(
    time_step_size=DT_VPM,
    background_velocity=list(U_INF),
    viscous=ViscousConfig.gbd(
        h=PARTICLE_SPACING,
        padding=3.0,
        viscosity=NU,
        threshold_mode="relative_local",
        threshold=0.30,
        max_nodes=PARTICLE_LIMIT,
        cap_abs_fraction=0.95,
        regen_radius_ratio=OVERLAP_RADIUS_RATIO,
    ),
    stretching=StretchingConfig.transposed(scheme=VPM_SCHEME),
    advection=AdvectionConfig(scheme=VPM_SCHEME),
    turbulence=VPMTurbulenceConfig.equilibrium_smagorinsky(
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
    # The coupler already writes one complete rolling restart. Standalone VPM
    # histories duplicated that state and dominated output-step wall time.
    backup_frequency=0,
    backup_directory=str(CASE_DIR / "solution"),
    export_flow_integrals=False,
    samplers=VPM_SAMPLERS,
    panel_solver=PANEL_SOLVER,
    body_stl=BODY_STL,
)

COUPLER_SETUP = CouplerSetup(
    u_inf=list(U_INF),
    handoff_box=HANDOFF_BOX,
    donor_boundary_mode=DONOR_BOUNDARY_MODE,
    wall_patch_name="cube",
    h=PARTICLE_SPACING,
    buffer_thickness=6 * PARTICLE_SPACING,
    dead_zone_h=0.0,
    prune_vorticity_min=0.005,
    overlap_shell_prune_multiplier=OVERLAP_SHELL_PRUNE_MULTIPLIER,
    handoff_max_particles=PARTICLE_LIMIT,
    overlap_radius_ratio=OVERLAP_RADIUS_RATIO,
    log_period=VPM_LOG_PERIOD,
    backup_period=BACKUP_PERIOD,
)


def main() -> None:
    print("\n===== SIMULATION =====")
    print(
        f"  FVM dt={DT_FVM}s / VPM dt={DT_VPM}s, "
        f"FVM cell={FVM_CELL_SIZE}, particle h={PARTICLE_SPACING}, "
        f"VPM scheme={VPM_SCHEME}, shell prune×{OVERLAP_SHELL_PRUNE_MULTIPLIER:g}, "
        f"overlap buffer={FVM_OVERLAP_BUFFER:g}, sample spacing={SAMPLE_SPACING}, "
        f"particles<={PARTICLE_LIMIT}"
    )
    fvm_solver = setup_fvm_solver(FVM_SETUP, case_dir=CASE_DIR, mesh=FVM_MESH)
    fvm_solver.write_vtk()

    vpm_solver = setup_vpm_solver(VPM_SETUP) if FVMVPMCoupler.is_master_rank() else None

    coupled_solver = setup_coupler(vpm_solver, fvm_solver, COUPLER_SETUP)

    coupled_solver.run()
    print("\n===== DONE =====")
    print("Simulation completed successfully. Run ./allplot.sh to make the figures.")


if __name__ == "__main__":
    main()
