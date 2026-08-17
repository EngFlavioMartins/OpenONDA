"""Hybrid FVM-VPM sphere at Re = 300: the unsteady coupling benchmark.

Why a sphere and not a cylinder
-------------------------------
A vortex-particle method represents vorticity as a finite set of blobs with no
images and no periodicity, so it can only represent a field whose vortex lines
close inside the particle cloud.  A spanwise-uniform ("quasi-2D") cylinder wake
violates that by construction: its vortex lines run straight out through the
spanwise faces, and a straight tube of span L induces only ``a/sqrt(a^2+r^2)``
of the two-dimensional value at distance r (``a = L/2``).  For a 2D span that is
0.71 at r = 1D and 0.32 at r = 3D -- the donor boundary condition is then 30-70%
too weak wherever the wake matters, and no coupling scheme can repair it.

A sphere at Re = 300 sheds a planar-symmetric train of hairpin vortices.  It is
genuinely unsteady and periodic, so Strouhal number, mean drag and lift
amplitude all converge and a 1% target is meaningful -- and its vortex lines
close.  The coupler's ``vortex_line_closure`` diagnostic reports this per face
every step; all entries should stay well below 0.25.

Usage:
    ./allrun.sh                 # production
    OPENONDA_SMOKE=1 ./allrun.sh
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
    VelocityConfig,
    ViscousConfig,
    VPMSetup,
    setup_vpm_solver,
)

import _geometry as G

CASE_DIR = Path(__file__).resolve().parent
FVM_CORES = int(os.environ.get("OPENONDA_FVM_CORES", "1"))
MAX_PARTICLES = int(os.environ.get("OPENONDA_MAX_PARTICLES", "200000" if G.SMOKE else "3000000"))

FVM_FORCE_PERIOD = G.step_period("force interval", G.FORCE_INTERVAL, G.DT_FVM)
FVM_FIELD_PERIOD = G.step_period("diagnostic interval", G.DIAGNOSTIC_INTERVAL, G.DT_FVM)
VPM_FIELD_PERIOD = G.step_period("diagnostic interval", G.DIAGNOSTIC_INTERVAL, G.DT_VPM)
BACKUP_PERIOD = G.step_period("checkpoint interval", G.CHECKPOINT_INTERVAL, G.DT_VPM)

MESH = coupling_box_mesh(G.FVM_BOX, G.SPACING, patch_name="numericalBoundary")
SPHERE = ImmersedBody.sphere(
    centre=[0.0, 0.0, 0.0],
    diameter=G.DIAMETER,
    h=G.SPACING,
    alpha=1.0,
    name="sphere",
)

FORCE_SCHEDULE = SamplingSchedule(every_n_steps=FVM_FORCE_PERIOD)
FIELD_SCHEDULE = SamplingSchedule(every_n_steps=FVM_FIELD_PERIOD)

FVM_SAMPLERS = (
    IBMForceSampler(
        ref_velocity=float(np.linalg.norm(G.U_INF)),
        ref_area=0.25 * np.pi * G.DIAMETER**2,
        schedule=FORCE_SCHEDULE,
    ),
    FVMLineSampler(
        start=[G.FVM_BOX[0], 0.0, 0.0],
        end=[G.FVM_BOX[1], 0.0, 0.0],
        spacing=G.SAMPLE_SPACING,
        file_name="fvm_centerline",
        schedule=FIELD_SCHEDULE,
    ),
    FVMLineSampler(
        start=[2.0, G.FVM_BOX[2], 0.0],
        end=[2.0, G.FVM_BOX[3], 0.0],
        spacing=G.SAMPLE_SPACING,
        file_name="fvm_section_x200",
        schedule=FIELD_SCHEDULE,
    ),
    FVMSurfaceSampler(
        point=[0.0, 0.0, 0.0],
        normal=[0.0, 0.0, 1.0],
        bounds=[G.FVM_BOX[0], G.FVM_BOX[1], G.FVM_BOX[2], G.FVM_BOX[3]],
        spacing=G.SAMPLE_SPACING,
        file_name="fvm_slice_z0",
        schedule=FIELD_SCHEDULE,
    ),
)

VPM_SAMPLERS = (
    VPMLineSampler(
        start=[G.VPM_DOMAIN[0], 0.0, 0.0],
        end=[G.VPM_DOMAIN[1], 0.0, 0.0],
        spacing=G.SAMPLE_SPACING,
        file_name="vpm_centerline",
    ),
    VPMLineSampler(
        start=[2.0, G.VPM_DOMAIN[2], 0.0],
        end=[2.0, G.VPM_DOMAIN[3], 0.0],
        spacing=G.SAMPLE_SPACING,
        file_name="vpm_section_x200",
    ),
    VPMSurfaceSampler(
        point=[0.0, 0.0, 0.0],
        normal=[0.0, 0.0, 1.0],
        bounds=[G.FVM_BOX[0], 12.0, G.FVM_BOX[2], G.FVM_BOX[3]],
        spacing=G.SAMPLE_SPACING,
        file_name="vpm_slice_z0",
        include_derivatives=False,
    ),
)

FVM_SETUP = FVMSetup(
    case_name="coupled_sphereFlow",
    cores=FVM_CORES,
    execution=ExecutionConfig(
        operator_backend="numba",
        linear_backend="petsc" if FVM_CORES > 1 else "scipy",
        parallel_mode="petsc_replicated" if FVM_CORES > 1 else "serial",
    ),
    output=OutputSetup(compression="lz4", precision="float32", asynchronous=False, ghost_layers=0),
    time=TimeConfig(
        delta_t=G.DT_FVM,
        end_time=G.T_END,
        write_interval=10**9,
        write_interval_time=G.VOLUME_INTERVAL,
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
    transport=TransportConfig(density=G.RHO, nu=G.NU),
    # Re = 300 is laminar: no subgrid model on either side, so the comparison is
    # not contaminated by two different SGS discretisations.
    turbulence=FVMTurbulenceConfig(),  # model="None": laminar
    boundaries=[
        BoundaryConfig(
            name="numericalBoundary",
            type_U="fixedValue",
            value_U=list(G.U_INF),
            type_p="fixedFluxPressure",
        ),
    ],
    initial_U=list(G.U_INF),
    initial_p=0.0,
)

VPM_SETUP = VPMSetup(
    time_step_size=G.DT_VPM,
    background_velocity=list(G.U_INF),
    viscous=ViscousConfig.gbd(
        h=G.SPACING,
        padding=3.0,
        viscosity=G.NU,
        threshold_mode="relative_local",
        threshold=0.10,
        max_nodes=MAX_PARTICLES,
        regen_radius_ratio=1.0,
    ),
    stretching=StretchingConfig.transposed(scheme="RK2"),
    advection=AdvectionConfig(scheme="RK2"),
    velocity=VelocityConfig.treecode(theta=0.3, multipole_order=2),
    stabilization=StabilizationConfig.bounded_domain(G.VPM_DOMAIN),
    particles_kernel="GAUSSIAN",
    precision="f32",
    processing_unit="AUTO",
    max_particles=MAX_PARTICLES,
    max_targets=MAX_PARTICLES,
    vpm_domain_bounds=list(G.VPM_DOMAIN),
    log_mode="file",
    logging_frequency=VPM_FIELD_PERIOD,
    timing_frequency=VPM_FIELD_PERIOD,
    backup_frequency=0,
    backup_directory=str(CASE_DIR / "solution"),
    samplers=VPM_SAMPLERS,
)

COUPLER_SETUP = CouplerSetup(
    u_inf=list(G.U_INF),
    handoff_box=G.HANDOFF_BOX,
    donor_boundary_mode="pressure_gradient",
    wall_patch_name=None,
    h=G.SPACING,
    buffer_thickness=6 * G.SPACING,
    dead_zone_h=0.0,
    prune_vorticity_min=0.005,
    handoff_max_particles=MAX_PARTICLES,
    overlap_radius_ratio=1.0,
    transfer_amplification_cap=2.0,
    resync_donor_after_handoff=True,
    anchor_pressure=True,
    log_period=VPM_FIELD_PERIOD,
    backup_period=BACKUP_PERIOD,
)


def main() -> None:
    fvm_solver = setup_fvm_solver(FVM_SETUP, case_dir=CASE_DIR, mesh=MESH)
    if fvm_solver.parallel.is_root:
        print("\n===== HYBRID SPHERE Re=300 =====")
        print(
            f"  h/D={G.SPACING / G.DIAMETER:g}  dt_FVM={G.DT_FVM:g}  dt_VPM={G.DT_VPM:g}  "
            f"t_end={G.T_END:g}  cells={fvm_solver.mesh_data['n_elements']}"
        )
    fvm_solver.set_immersed_bodies(SPHERE, h=G.SPACING)
    fvm_solver.write_vtk()

    vpm_solver = setup_vpm_solver(VPM_SETUP) if FVMVPMCoupler.is_master_rank() else None
    coupled_solver = setup_coupler(vpm_solver, fvm_solver, COUPLER_SETUP)
    coupled_solver.run()

    if fvm_solver.parallel.is_root:
        print("\n===== DONE =====")
        print("Run referenceFlow/allrun.sh, then ./allplot.sh.")


if __name__ == "__main__":
    main()
