"""Cylinder shedding at Re=200 with OpenONDA's native FVM-VPM workflow.

The near field is a solver-native Cartesian FVM mesh with a direct-forcing
immersed cylinder.  No external mesher, OpenFOAM installation, or path setup
is required; every import comes from the installed ``openonda`` package.
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
REYNOLDS = 200.0
U_INF = (1.0, 0.0, 0.0)
NU = np.linalg.norm(U_INF) * DIAMETER / REYNOLDS
SPACING = float(os.environ.get("OPENONDA_SPACING", "0.20" if SMOKE else "0.05"))
DT_FVM = float(os.environ.get("OPENONDA_FVM_DT", "0.01"))
DT_VPM = float(os.environ.get("OPENONDA_VPM_DT", "0.05"))
T_END = float(
    os.environ.get("OPENONDA_T_END", os.environ.get("HYBRID_T_END", "0.10" if SMOKE else "60.0"))
)
FVM_BOX = (-1.5, 2.5, -2.0, 2.0, -1.0, 1.0)
VPM_DOMAIN = (-2.0, 15.0, -3.0, 3.0, -1.5, 1.5)
MAX_PARTICLES = int(os.environ.get("OPENONDA_MAX_PARTICLES", "100000" if SMOKE else "1500000"))
WRITE_INTERVAL = DT_VPM if SMOKE else 1.0
SAMPLE_INTERVAL = min(WRITE_INTERVAL, T_END)
FVM_LOG_PERIOD = max(1, int(round(SAMPLE_INTERVAL / DT_FVM)))
VPM_LOG_PERIOD = max(1, int(round(SAMPLE_INTERVAL / DT_VPM)))
SPAN = FVM_BOX[5] - FVM_BOX[4]

FVM_MESH = coupling_box_mesh(FVM_BOX, SPACING, patch_name="numericalBoundary")
CYLINDER = ImmersedBody.extruded_cylinder_z(
    centre=[0.0, 0.0, 0.0],
    diameter=DIAMETER,
    z_bounds=[FVM_BOX[4], FVM_BOX[5]],
    h=SPACING,
    name="cylinder",
    caps=False,
)

FVM_SAMPLERS = (
    IBMForceSampler(
        ref_velocity=float(np.linalg.norm(U_INF)),
        ref_area=DIAMETER * SPAN,
        schedule=SamplingSchedule(every_n_steps=FVM_LOG_PERIOD),
    ),
    FVMLineSampler(
        start=[FVM_BOX[0], 0.0, 0.0],
        end=[FVM_BOX[1], 0.0, 0.0],
        spacing=SPACING,
        file_name="fvm_centerline",
        schedule=SamplingSchedule(every_n_steps=FVM_LOG_PERIOD),
    ),
    FVMSurfaceSampler(
        point=[0.0, 0.0, 0.0],
        normal=[0.0, 0.0, 1.0],
        bounds=[FVM_BOX[0], FVM_BOX[1], FVM_BOX[2], FVM_BOX[3]],
        spacing=SPACING,
        file_name="fvm_slice_z0",
        schedule=SamplingSchedule(every_n_steps=FVM_LOG_PERIOD),
    ),
)

VPM_SAMPLERS = (
    VPMLineSampler(
        start=[VPM_DOMAIN[0], 0.0, 0.0],
        end=[VPM_DOMAIN[1], 0.0, 0.0],
        spacing=SPACING,
        file_name="vpm_centerline",
    ),
    VPMSurfaceSampler(
        point=[0.0, 0.0, 0.0],
        normal=[0.0, 0.0, 1.0],
        bounds=[VPM_DOMAIN[0], VPM_DOMAIN[1], VPM_DOMAIN[2], VPM_DOMAIN[3]],
        spacing=SPACING,
        file_name="vpm_slice_z0",
    ),
)

FVM_SETUP = FVMSetup(
    case_name="cylinderSheddingFlow",
    cores=1,
    execution=ExecutionConfig(operator_backend="numba"),
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
        write_interval_time=WRITE_INTERVAL,
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
    turbulence=FVMTurbulenceConfig.smagorinsky(Cs=0.17),
    boundaries=[
        BoundaryConfig(
            name="numericalBoundary",
            type_U="fixedValue",
            value_U=list(U_INF),
            type_p="fixedFluxPressure",
        )
    ],
    initial_U=list(U_INF),
    initial_p=0.0,
)

VPM_SETUP = VPMSetup(
    time_step_size=DT_VPM,
    background_velocity=list(U_INF),
    viscous=ViscousConfig.cs(viscosity=NU, characteristic_distance=SPACING),
    stretching=StretchingConfig.transposed(scheme="RK2"),
    advection=AdvectionConfig(scheme="RK2"),
    turbulence=VPMTurbulenceConfig.les_smagorinsky(cs=0.17),
    velocity=VelocityConfig.treecode(theta=0.3, multipole_order=2),
    stabilization=StabilizationConfig.bounded_domain(VPM_DOMAIN),
    particles_kernel="GAUSSIAN",
    precision="f32",
    processing_unit="AUTO",
    max_particles=MAX_PARTICLES,
    max_targets=MAX_PARTICLES,
    vpm_domain_bounds=list(VPM_DOMAIN),
    log_mode="file",
    logging_frequency=VPM_LOG_PERIOD,
    backup_frequency=VPM_LOG_PERIOD,
    backup_directory=str(CASE_DIR / "solution"),
    samplers=VPM_SAMPLERS,
)

COUPLER_SETUP = CouplerSetup(
    u_inf=list(U_INF),
    wall_patch_name=None,
    h=SPACING,
    buffer_thickness=6 * SPACING,
    dead_zone_h=3.0,
    prune_vorticity_min=0.01,
    handoff_max_particles=MAX_PARTICLES,
    log_period=VPM_LOG_PERIOD,
    backup_period=VPM_LOG_PERIOD,
)


def main() -> None:
    fvm_solver = setup_fvm_solver(FVM_SETUP, case_dir=CASE_DIR, mesh=FVM_MESH)
    fvm_solver.set_immersed_bodies(CYLINDER, h=SPACING)
    fvm_solver.write_vtk()

    vpm_solver = setup_vpm_solver(VPM_SETUP) if FVMVPMCoupler.is_master_rank() else None
    coupled_solver = setup_coupler(vpm_solver, fvm_solver, COUPLER_SETUP)
    coupled_solver.run()


if __name__ == "__main__":
    main()
