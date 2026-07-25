"""Hybrid FVM–VPM simulation of flow past a cube at Re = 1000.

The FVM mesh is built beforehand by ``assets/create_mesh.py`` and read from
``constant/polyMesh/``; this file holds only the case physics and coupling.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys


REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "openonda_bootstrap.py").is_file()
)
sys.path.insert(0, str(REPO_ROOT))
from openonda_bootstrap import activate  # noqa: E402

activate(__file__)

import numpy as np  # noqa: E402

from source.coupler import CouplerSetup, setup_coupler  # noqa: E402
from source.solvers.FVM import (  # noqa: E402
    BoundaryConfig,
    ForcesConfig,
    FVMSetup,
    LinearSolverConfig,
    OutputSetup,
    PimpleControl,
    SchemesConfig,
    TimeConfig,
    TransportConfig,
    setup_fvm_solver,
)
from source.solvers.VPM import (  # noqa: E402
    AdvectionConfig,
    StabilizationConfig,
    StretchingConfig,
    TurbulenceConfig,
    VelocityConfig,
    ViscousConfig,
    VPMSetup,
    setup_vpm_solver,
)

CASE_DIR = Path(__file__).resolve().parent
MESH = str(CASE_DIR / "assets" / "mesh.msh")  # built by assets/create_mesh.py

# Physical problem
CUBE_SIDE = 1.0
CORE_BOX = (-1.5, 1.5, -1.5, 1.5, -1.5, 1.5)  # FVM common-region extent
U_INF = (1.0, 0.0, 0.0)
RHO = 1.0
REYNOLDS = 1000.0
NU = np.linalg.norm(U_INF) * CUBE_SIDE / REYNOLDS
INITIAL_U = (1.0, 0.0, 0.0)

# Time integration
DT_FVM = 0.0125
T_END = 40.0
WRITE_INTERVAL = 0.15

DT_VPM = 0.05
VPM_SPACING = 0.05
VPM_DOMAIN = (-2.0, 10.0, -2.0, 2.0, -2.0, 2.0)
MAX_PARTICLES = 1_000_000
OVERLAP_RADIUS_RATIO = 1.0
BACKUP_PERIOD = round(WRITE_INTERVAL / DT_VPM)


FVM_SETUP = FVMSetup(
    case_name="coupled_hybridFlow",
    cores=4,
    output=OutputSetup(
        format="vtk_xml",
        data_location="cell",
        encoding="appended",
        compression="lz4",
        precision="float64",
        asynchronous=True,
        ghost_layers=1,
    ),
    time=TimeConfig(
        delta_t=DT_FVM,
        start_time=0.0,
        end_time=T_END,
        write_interval=10**9,
        write_interval_time=WRITE_INTERVAL,
        adjust_timestep=False,
    ),
    schemes=SchemesConfig(
        convection_scheme="lust",
        gradient_scheme="lsq",
        time_scheme="backward",
    ),
    linear=LinearSolverConfig(
        linear_solver="bicgstab",
        pressure_solver="amg",
        pressure_tol=1e-8,
        momentum_maxiter=2000,
        ilu_drop_tol=1e-4,
        ilu_fill_factor=10.0,
        ilu_reuse_tol=0.05,
    ),
    pimple=PimpleControl(n_correctors=2, n_outer_correctors=2),
    forces=ForcesConfig(
        force_patches=["cube"],
        ref_velocity=np.linalg.norm(U_INF),
        ref_area=CUBE_SIDE**2,
        ref_length=CUBE_SIDE,
        moment_centre=[0.0, 0.0, 0.0],
        force_log_interval=1,
    ),
    transport=TransportConfig(density=RHO, nu=NU),
    turbulence=None,
    boundaries=[
        # Must match the donor_bc_mode="dirichlet" contract (the same pair the
        # OFW reference case uses, and the one coupling_patch_boundaries()
        # returns): the coupler imposes the donor velocity on EVERY face, so
        # the pressure must be all-Neumann.  Declaring freestream pressure here
        # additionally pins p=0 on whichever faces it calls outflow, which
        # over-determines those faces and lets the pressure correction violate
        # the prescribed donor flux -- and the pinned set flips with the flux
        # sign every step, injecting per-step noise into the forces.
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
        h=VPM_SPACING,
        padding=3.0,
        viscosity=NU,
        threshold_mode="relative_max",
        threshold=5e-3,
        regen_radius_ratio=OVERLAP_RADIUS_RATIO,
    ),
    stretching=StretchingConfig.transposed(scheme="RK2"),
    advection=AdvectionConfig(scheme="RK2"),
    # Matches the OFW reference case.  DNS applies no subgrid dissipation and
    # assumes the particle spacing resolves every scale; at h = 0.05 and
    # Re = 1000 the wake is under-resolved, so subgrid energy piles up and the
    # vortical structures break down instead of convecting coherently.
    turbulence=TurbulenceConfig.les_smagorinsky(),
    velocity=VelocityConfig.treecode(theta=0.3),
    stabilization=replace(
        StabilizationConfig.disabled(),
        remove_particles_by_bounds=list(VPM_DOMAIN),
    ),
    particles_kernel="GAUSSIAN",
    precision="f32",
    processing_unit="GPU",
    max_particles=MAX_PARTICLES,
    max_targets=MAX_PARTICLES,
    vpm_domain_bounds=list(VPM_DOMAIN),
    logging_frequency=BACKUP_PERIOD,
    backup_frequency=BACKUP_PERIOD,
    backup_file_name="vpm_solution",
    backup_directory=str(CASE_DIR / "solution"),
)

COUPLER_SETUP = CouplerSetup(
    u_inf=list(U_INF),
    wall_patch_name="cube",
    h=VPM_SPACING,
    buffer_thickness=6 * VPM_SPACING,
    dead_zone_h=3.0,
    prune_vorticity_min=0.005,
    overlap_radius_ratio=OVERLAP_RADIUS_RATIO,
    overlap_velocity_forcing=False,
    strength_correction_iterations=1,
    strength_correction_relax=1.0,
    donor_bc_mode="dirichlet",
    donor_interior_source="particles",
    bc_coupling_iterations=2,
    donor_bc_relax=0.5,
    donor_interior_warmup_time=0.45,
    log_period=2,
    backup_period=BACKUP_PERIOD,
)


def main() -> None:
    fvm_solver = setup_fvm_solver(FVM_SETUP, case_dir=CASE_DIR, mesh=MESH)
    vpm_solver = setup_vpm_solver(VPM_SETUP)
    coupled_solver = setup_coupler(vpm_solver, fvm_solver, COUPLER_SETUP)
    coupled_solver.run()


if __name__ == "__main__":
    main()
