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
        # The regen threshold MUST use a local reference here.  This field spans
        # ~4 decades of |Γ|: the maximum is the cube's wall vortex sheet, the
        # wake at x≈1.5 is ~10⁻³ of it.  Every global-reference mode therefore
        # cuts along one iso-|Γ| surface and deletes the wake to keep the
        # boundary layer — measured, one GBD regen removed 100% of the
        # particles below |ω| = 0.243 s⁻¹ (67% of the cloud) with
        # relative_max=5e-3, and 100% below 0.03 s⁻¹ with budget=1e-2.  That is
        # what shredded the vortical structures leaving the FVM box, and because
        # the reference lives on the body the cut level jittered with the
        # near-wall solution and fed straight back through the donor BC.
        #
        # Value picked on the quantity the coupling actually consumes: the error
        # the pruning introduces in the donor trace (the Biot-Savart velocity the
        # cloud induces on the FVM box faces), measured against a keep-everything
        # reference on the t=2.4 field:
        #
        #     threshold   N/N_ref   donor |Δu|/U∞ (max / rms)
        #       0.15       0.82      1.0e-3 / 2.6e-4
        #       0.30       0.75      3.2e-3 / 7.1e-4     <- knee, used here
        #       0.50       0.66      1.2e-2 / 1.3e-3
        #       0.80       0.49      4.0e-2 / 4.0e-3
        #
        # 0.30 drops a quarter of the particles for a 0.07% rms perturbation of
        # the boundary condition — far below the FVM's own discretisation error —
        # and still keeps ≥64% of the nodes in EVERY |ω| decade (no scale is
        # amputated).  Past the knee the error grows ~4x per step in threshold
        # while the particle saving flattens.  Lower to 0.15 for maximum fidelity;
        # raise toward 0.5 only if the far-field cost becomes limiting.
        threshold_mode="relative_local",
        threshold=0.30,
        threshold_window=3,
        regen_radius_ratio=OVERLAP_RADIUS_RATIO,
    ),
    stretching=StretchingConfig.transposed(scheme="RK2"),
    advection=AdvectionConfig(scheme="RK2"),
    turbulence=TurbulenceConfig.les_smagorinsky(),
    velocity=VelocityConfig.treecode(theta=0.3, multipole_order=2),
    stabilization=StabilizationConfig.bounded_domain(VPM_DOMAIN),
    particles_kernel="GAUSSIAN",
    precision="f32",
    processing_unit="AUTO",
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
    # "fvm" is the OFW reference's setting and is now affordable (the interior
    # Biot-Savart runs as a monopole+dipole treecode, ~37x faster for ~0.1%
    # error).  It was switched to "particles" because flux_ratio swung between
    # 0.7 and 16.9 instead of sitting near 1.
    #
    # CAVEAT: that evidence does not hold.  flux_ratio was |ΣΓ_VPM| / |ΣΓ_FVM|
    # over the outflow band — a ratio of two near-cancelling vector sums (ω
    # integrates to ~0 over a wake cross-section), so it swung on cancellation
    # noise regardless of the donor mode; it swung 0.02–29 with "particles"
    # selected too.  It is now an L1 ratio and is meaningful.  The real cause of
    # the near-boundary noise was the VPM regen threshold (see the viscous block
    # above).  The structural argument for "particles" still stands on its own —
    # in "fvm" mode the boundary is driven by FVM interior vorticity while the
    # VPM carries its own particles inside the box, i.e. two different models of
    # the same field — so the setting is kept, but "fvm" has NOT been re-tested
    # since the regen fix and may now be viable.
    donor_interior_source="particles",
    donor_interior_treecode_theta=0.3,
    bc_coupling_iterations=2,
    donor_bc_relax=0.5,
    donor_interior_warmup_time=0.0,
    log_period=2,
    backup_period=BACKUP_PERIOD,
)


def _parse_args():
    """Run-control overrides for coupling A/B studies.

    Defaults reproduce the case exactly as written above; every flag exists so a
    variant can be run from a shared checkpoint without editing this file.
    ``openonda_bootstrap`` forwards ``sys.argv[1:]`` through both the Conda and
    the MPI re-exec, so these reach every rank.
    """
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--restart",
        action="store_true",
        help="Resume from solution/coupled_checkpoint instead of an impulsive start.",
    )
    p.add_argument("--t-end", type=float, help="Override the end time [s].")
    p.add_argument(
        "--fringe-strength",
        type=float,
        help="Fringe relaxation strength A (lambda_max = A*U/buffer_thickness).",
    )
    p.add_argument("--donor-bc-mode", choices=("dirichlet", "mixed", "characteristic"))
    p.add_argument(
        "--overlap-velocity-forcing",
        action="store_true",
        help="Advect overlap particles with the eta-blended FVM velocity (two-sided).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    fvm_setup = FVM_SETUP
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

    fvm_solver = setup_fvm_solver(fvm_setup, case_dir=CASE_DIR, mesh=MESH)
    vpm_solver = setup_vpm_solver(VPM_SETUP)
    coupled_solver = setup_coupler(vpm_solver, fvm_solver, coupler_setup)

    if args.restart:
        # allow_config_change: the coupling knobs under test (donor_bc_mode,
        # overlap_velocity_forcing) are part of the checkpoint's config digest,
        # so an A/B restart is rejected without it.  The changed keys are logged.
        coupled_solver.run(
            restart_from=CASE_DIR / "solution" / "coupled_checkpoint",
            allow_config_change=True,
        )
    else:
        coupled_solver.run()


if __name__ == "__main__":
    main()
