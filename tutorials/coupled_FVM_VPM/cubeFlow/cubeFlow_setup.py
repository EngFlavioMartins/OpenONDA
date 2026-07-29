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

CUBE_SIDE = 1.0
U_INF = (1.0, 0.0, 0.0)
RHO = 1.0
REYNOLDS = 1000.0
NU = np.linalg.norm(U_INF) * CUBE_SIDE / REYNOLDS
INITIAL_U = (1.0, 0.0, 0.0)

DT_FVM = 0.0125
T_END = 20.0
WRITE_INTERVAL = 0.5
PERTURBATION = 1.0e-3

DT_VPM = 0.05
VPM_SPACING = 0.05
CORE_BOX = (-1.8, 1.8, -1.8, 1.8, -1.8, 1.8)
VPM_DOMAIN = (-4.5, 11.0, -4.5, 4.5, -4.5, 4.5)
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
        precision="float32",
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
        threshold_mode="relative_local",
        threshold=0.30,
        threshold_window=3,
        max_nodes=200_000,
        cap_abs_fraction=0.99,
        regen_radius_ratio=OVERLAP_RADIUS_RATIO,
    ),
    stretching=StretchingConfig.transposed(scheme="RK2"),
    advection=AdvectionConfig(scheme="RK2"),
    turbulence=TurbulenceConfig.dns(),
    velocity=VelocityConfig.treecode(theta=0.3, multipole_order=2),
    stabilization=StabilizationConfig.bounded_domain(VPM_DOMAIN),
    particles_kernel="GAUSSIAN",
    precision="f32",
    processing_unit="METAL",
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
    prune_vorticity_min=0.01,
    handoff_max_particles=250_000,
    overlap_radius_ratio=OVERLAP_RADIUS_RATIO,
    overlap_velocity_forcing=False,
    strength_correction_iterations=1,
    strength_correction_relax=1.0,
    donor_bc_mode="dirichlet",
    donor_interior_source="particles",
    donor_interior_treecode_theta=0.3,
    bc_coupling_iterations=1,
    donor_bc_relax=1.0,
    donor_interior_warmup_time=0.0,
    log_period=2,
    backup_period=BACKUP_PERIOD,
)


def _break_symmetry(solver) -> None:
    centroids = solver.geo_data["element_centroids"]
    n_cells = solver.mesh_data["n_elements"]
    x, y, z = centroids[:n_cells].T
    near_wake = (x > 0.5) & (x < 2.5) & (np.abs(y) < 1.0) & (np.abs(z) < 1.0)
    kick = PERTURBATION * np.sign(z + 1e-12) * np.exp(-((x - 1.0) ** 2))
    solver.U[:n_cells, 1] += np.where(near_wake, kick, 0.0)


def _parse_args():
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--restart",
        action="store_true",
        help="Resume from solution/coupled_checkpoint instead of an impulsive start.",
    )
    p.add_argument("--t-end", type=float, help="Override the end time [s].")
    p.add_argument("--cores", type=int, help="Override FVM MPI ranks.")
    p.add_argument(
        "--fringe-strength",
        type=float,
        help="Fringe relaxation strength A (lambda_max = A*U/buffer_thickness).",
    )
    p.add_argument("--donor-bc-mode", choices=("dirichlet", "mixed", "characteristic"))
    p.add_argument("--vpm-turbulence", choices=("dns", "smagorinsky"))
    p.add_argument(
        "--overlap-velocity-forcing",
        action="store_true",
        help="Advect overlap particles with the eta-blended FVM velocity (two-sided).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    fvm_setup = FVM_SETUP
    if args.cores is not None:
        fvm_setup = replace(fvm_setup, cores=args.cores)
    if args.t_end is not None:
        fvm_setup = replace(fvm_setup, time=replace(fvm_setup.time, end_time=args.t_end))

    coupler_setup = COUPLER_SETUP
    overrides = {
        k: v
        for k, v in (
            ("fringe_strength", args.fringe_strength),
            ("donor_bc_mode", args.donor_bc_mode),
            ("overlap_velocity_forcing", args.overlap_velocity_forcing or None),
        )
        if v is not None
    }
    if overrides:
        coupler_setup = replace(coupler_setup, **overrides)

    vpm_setup = VPM_SETUP
    if args.vpm_turbulence == "dns":
        vpm_setup = replace(vpm_setup, turbulence=TurbulenceConfig.dns())
    elif args.vpm_turbulence == "smagorinsky":
        vpm_setup = replace(vpm_setup, turbulence=TurbulenceConfig.les_smagorinsky())

    fvm_solver = setup_fvm_solver(fvm_setup, case_dir=CASE_DIR, mesh=MESH)
    if not args.restart:
        _break_symmetry(fvm_solver)
    vpm_solver = setup_vpm_solver(vpm_setup)
    coupled_solver = setup_coupler(vpm_solver, fvm_solver, coupler_setup)

    if args.restart:
        coupled_solver.run(
            restart_from=CASE_DIR / "solution" / "coupled_checkpoint",
            allow_config_change=True,
        )
    else:
        coupled_solver.run()


if __name__ == "__main__":
    main()
