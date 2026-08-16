"""Matched hybrid FVM-VPM benchmark for cylinder LES at Re=500.

The near-body FVM box and the fully meshed sibling reference use the same
uniform spacing, immersed-cylinder forcing, time scheme, and force sampler.
Only the treatment outside the compact FVM box differs, making the onset and
periodic shedding phases a clean coupling diagnostic.

The recommended, explicit configuration is hardcoded in ``allrun.sh``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from openonda.coupler import CouplerSetup, FVMVPMCoupler, setup_coupler
from openonda.fvm import (
    BoundaryConfig,
    ExecutionConfig,
    FVMSetup,
    IBMForceSampler,
    ImmersedBody,
    LinearSolverConfig,
    LineSampler as FVMLineSampler,
    OutputSetup,
    PimpleControl,
    SamplingSchedule,
    SchemesConfig,
    SurfaceSampler as FVMSurfaceSampler,
    TimeConfig,
    TransportConfig,
    TurbulenceConfig as FVMTurbulenceConfig,
    coupling_box_mesh,
    setup_fvm_solver,
)
from openonda.vpm import (
    AdvectionConfig,
    LineSampler as VPMLineSampler,
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

DIAMETER = 1.0
RHO = 1.0
REYNOLDS = 500.0
U_INF = (1.0, 0.0, 0.0)
# A tiny initial cross-flow disturbance seeds the unstable wake identically in
# the reference and hybrid solvers; the boundary condition itself remains U∞.
INITIAL_U = (1.0, 0.01, 0.0)
NU = float(np.linalg.norm(U_INF)) * DIAMETER / REYNOLDS

SPACING = float(os.environ.get("OPENONDA_SPACING", "0.25" if SMOKE else "0.125"))
DT_FVM = float(os.environ.get("OPENONDA_FVM_DT", "0.025"))
DT_VPM = float(os.environ.get("OPENONDA_VPM_DT", "0.10"))
T_END = float(os.environ.get("OPENONDA_T_END", "0.10" if SMOKE else "15.0"))
FVM_CORES = int(os.environ.get("OPENONDA_FVM_CORES", "1" if SMOKE else "4"))
MAX_PARTICLES = int(os.environ.get("OPENONDA_MAX_PARTICLES", "150000" if SMOKE else "800000"))

HANDOFF_BOX = (-1.5, 2.5, -2.0, 2.0, -1.0, 1.0)
DOWNSTREAM_BUFFER = float(
    os.environ.get("OPENONDA_FVM_DOWNSTREAM_BUFFER", "0.4" if SMOKE else "0.5")
)
FVM_BOX = (
    HANDOFF_BOX[0],
    HANDOFF_BOX[1] + DOWNSTREAM_BUFFER,
    HANDOFF_BOX[2],
    HANDOFF_BOX[3],
    HANDOFF_BOX[4],
    HANDOFF_BOX[5],
)
VPM_DOMAIN = (-4.0, 10.0, -3.5, 3.5, -1.5, 1.5)
SPAN = FVM_BOX[5] - FVM_BOX[4]

FORCE_INTERVAL = float(os.environ.get("OPENONDA_FORCE_INTERVAL", "0.05"))
DIAGNOSTIC_INTERVAL = float(os.environ.get("OPENONDA_DIAGNOSTIC_INTERVAL", "0.5"))
CHECKPOINT_INTERVAL = float(os.environ.get("OPENONDA_CHECKPOINT_INTERVAL", "5.0"))
VOLUME_INTERVAL = float(os.environ.get("OPENONDA_VOLUME_INTERVAL", "7.5"))
SAMPLE_SPACING = float(os.environ.get("OPENONDA_SAMPLE_SPACING", "0.125"))


def _period(name: str, interval: float, dt: float) -> int:
    ratio = interval / dt
    period = int(round(ratio))
    if interval <= 0.0 or dt <= 0.0 or period < 1 or not np.isclose(ratio, period, atol=1e-10):
        raise ValueError(f"{name}={interval:g} must be a positive integer multiple of dt={dt:g}")
    return period


FVM_FORCE_PERIOD = _period("force interval", FORCE_INTERVAL, DT_FVM)
FVM_DIAGNOSTIC_PERIOD = _period("diagnostic interval", DIAGNOSTIC_INTERVAL, DT_FVM)
VPM_DIAGNOSTIC_PERIOD = _period("diagnostic interval", DIAGNOSTIC_INTERVAL, DT_VPM)
BACKUP_PERIOD = _period("checkpoint interval", CHECKPOINT_INTERVAL, DT_VPM)

FVM_MESH = coupling_box_mesh(
    FVM_BOX,
    SPACING,
    patch_name="numericalBoundary",
)
CYLINDER = ImmersedBody.extruded_cylinder_z(
    centre=[0.0, 0.0, 0.0],
    diameter=DIAMETER,
    z_bounds=[FVM_BOX[4], FVM_BOX[5]],
    h=SPACING,
    alpha=1.5,
    name="cylinder",
    caps=False,
)

FORCE_SCHEDULE = SamplingSchedule(every_n_steps=FVM_FORCE_PERIOD)
FIELD_SCHEDULE = SamplingSchedule(every_n_steps=FVM_DIAGNOSTIC_PERIOD)
FVM_SAMPLERS = (
    IBMForceSampler(
        ref_velocity=float(np.linalg.norm(U_INF)),
        ref_area=DIAMETER * SPAN,
        schedule=FORCE_SCHEDULE,
    ),
    FVMLineSampler(
        start=[FVM_BOX[0], 0.0, 0.0],
        end=[FVM_BOX[1], 0.0, 0.0],
        spacing=SAMPLE_SPACING,
        file_name="fvm_centerline",
        schedule=FIELD_SCHEDULE,
    ),
    FVMLineSampler(
        start=[FVM_BOX[0], 0.75, 0.0],
        end=[FVM_BOX[1], 0.75, 0.0],
        spacing=SAMPLE_SPACING,
        file_name="fvm_offaxis_y075",
        schedule=FIELD_SCHEDULE,
    ),
    FVMLineSampler(
        start=[1.0, FVM_BOX[2], 0.0],
        end=[1.0, FVM_BOX[3], 0.0],
        spacing=SAMPLE_SPACING,
        file_name="fvm_section_x100",
        schedule=FIELD_SCHEDULE,
    ),
    FVMLineSampler(
        start=[2.0, FVM_BOX[2], 0.0],
        end=[2.0, FVM_BOX[3], 0.0],
        spacing=SAMPLE_SPACING,
        file_name="fvm_section_x200",
        schedule=FIELD_SCHEDULE,
    ),
    FVMSurfaceSampler(
        point=[0.0, 0.0, 0.0],
        normal=[0.0, 0.0, 1.0],
        bounds=[FVM_BOX[0], FVM_BOX[1], FVM_BOX[2], FVM_BOX[3]],
        spacing=SAMPLE_SPACING,
        file_name="fvm_slice_z0",
        schedule=FIELD_SCHEDULE,
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
        start=[VPM_DOMAIN[0], 0.75, 0.0],
        end=[VPM_DOMAIN[1], 0.75, 0.0],
        spacing=SAMPLE_SPACING,
        file_name="vpm_offaxis_y075",
    ),
    VPMLineSampler(
        start=[1.0, VPM_DOMAIN[2], 0.0],
        end=[1.0, VPM_DOMAIN[3], 0.0],
        spacing=SAMPLE_SPACING,
        file_name="vpm_section_x100",
    ),
    VPMLineSampler(
        start=[2.0, VPM_DOMAIN[2], 0.0],
        end=[2.0, VPM_DOMAIN[3], 0.0],
        spacing=SAMPLE_SPACING,
        file_name="vpm_section_x200",
    ),
    VPMSurfaceSampler(
        point=[0.0, 0.0, 0.0],
        normal=[0.0, 0.0, 1.0],
        bounds=[-1.6, 6.0, -2.0, 2.0],
        spacing=SAMPLE_SPACING,
        file_name="vpm_slice_z0",
        include_derivatives=False,
    ),
)

FVM_SETUP = FVMSetup(
    case_name="coupled_cylinderFlow",
    cores=FVM_CORES,
    # Replicated PETSc keeps the complete IBM marker support on every rank
    # while solving the pressure/momentum systems collectively.
    execution=ExecutionConfig(
        operator_backend="numba",
        linear_backend="petsc",
        parallel_mode="petsc_replicated",
    ),
    output=OutputSetup(
        compression="lz4",
        precision="float32",
        asynchronous=False,
        ghost_layers=0,
    ),
    time=TimeConfig(
        delta_t=DT_FVM,
        end_time=T_END,
        write_interval=10**9,
        write_interval_time=VOLUME_INTERVAL,
    ),
    schemes=SchemesConfig(
        convection_scheme="limitedLinear",
        gradient_scheme="gauss",
        time_scheme="backward",
    ),
    linear=LinearSolverConfig(
        pressure_solver="amg",
        pressure_tol=1e-6,
        pressure_rel_tol=0.01,
        momentum_tol=1e-6,
        momentum_rel_tol=0.1,
        momentum_maxiter=2000,
    ),
    pimple=PimpleControl(
        n_correctors=2,
        n_outer_correctors=2,
        alpha_u=0.7,
        alpha_p=0.3,
        ibm_forcing_loops=2,
    ),
    samplers=FVM_SAMPLERS,
    transport=TransportConfig(density=RHO, nu=NU),
    turbulence=FVMTurbulenceConfig.smagorinsky(Cs=0.12),
    boundaries=[
        BoundaryConfig(
            name="numericalBoundary",
            type_U="fixedValue",
            value_U=list(U_INF),
            type_p="fixedFluxPressure",
        ),
    ],
    initial_U=list(INITIAL_U),
    initial_p=0.0,
)

VPM_SETUP = VPMSetup(
    time_step_size=DT_VPM,
    background_velocity=list(U_INF),
    viscous=ViscousConfig.cs(viscosity=NU, characteristic_distance=SPACING),
    stretching=StretchingConfig.transposed(scheme="RK2"),
    advection=AdvectionConfig(scheme="RK2"),
    turbulence=VPMTurbulenceConfig.les_smagorinsky(cs=0.12),
    velocity=VelocityConfig.treecode(theta=0.3, multipole_order=2),
    stabilization=StabilizationConfig.bounded_domain(VPM_DOMAIN),
    particles_kernel="GAUSSIAN",
    precision="f32",
    processing_unit="AUTO",
    max_particles=MAX_PARTICLES,
    max_targets=MAX_PARTICLES,
    vpm_domain_bounds=list(VPM_DOMAIN),
    log_mode="file",
    logging_frequency=VPM_DIAGNOSTIC_PERIOD,
    timing_frequency=VPM_DIAGNOSTIC_PERIOD,
    backup_frequency=0,
    backup_directory=str(CASE_DIR / "solution"),
    samplers=VPM_SAMPLERS,
)

COUPLER_SETUP = CouplerSetup(
    u_inf=list(U_INF),
    handoff_box=HANDOFF_BOX,
    donor_boundary_mode="pressure_gradient",
    wall_patch_name=None,
    h=SPACING,
    buffer_thickness=6 * SPACING,
    dead_zone_h=0.0,
    prune_vorticity_min=0.01,
    overlap_shell_prune_multiplier=10.0,
    handoff_max_particles=MAX_PARTICLES,
    log_period=VPM_DIAGNOSTIC_PERIOD,
    backup_period=BACKUP_PERIOD,
)


def main() -> None:
    fvm_solver = setup_fvm_solver(FVM_SETUP, case_dir=CASE_DIR, mesh=FVM_MESH)
    if fvm_solver.parallel.is_root:
        print("\n===== SIMULATION =====")
        print(
            f"  cylinder Re={REYNOLDS:g}, h={SPACING:g}, "
            f"FVM dt={DT_FVM:g}, VPM dt={DT_VPM:g}, t_end={T_END:g}"
        )
    fvm_solver.set_immersed_bodies(CYLINDER, h=SPACING)
    fvm_solver.write_vtk()

    vpm_solver = setup_vpm_solver(VPM_SETUP) if FVMVPMCoupler.is_master_rank() else None
    coupled_solver = setup_coupler(vpm_solver, fvm_solver, COUPLER_SETUP)
    coupled_solver.run()
    if fvm_solver.parallel.is_root:
        print("\n===== DONE =====")
        print("Hybrid cylinder simulation completed. Run ./allplot.sh after the reference case.")


if __name__ == "__main__":
    main()
