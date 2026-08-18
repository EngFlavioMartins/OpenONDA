"""Isolate what the VPM side does to the FVM interior solution.

The coupled cube case loses roughly half the wake's velocity-profile amplitude
by t=3 while the same FVM solver, same box and same boundary *formulation* fed
an exact trace keeps it to within 1%.  The damage therefore comes from the data
the VPM supplies, and this harness finds which part of the VPM/coupling stack
does the damage by turning the stack down to almost nothing and adding one
component back at a time.

The baseline is deliberately crude: particles are injected and *all* of them are
used to compute the boundary condition by direct O(N^2) summation, with no
treecode approximation, no particle regeneration or thresholding, no stretching,
no subgrid model, and f64 arithmetic.  At h=0.15 that is only ~12k particles, so
the exact sum is affordable and the VPM cost is negligible next to the FVM.

Variants are compared **against each other at fixed h**, never against the
reference in absolute terms: a coarse lattice cannot resolve the wake no matter
which modules are on, and that resolution floor cancels in a variant-to-variant
comparison.  Phase C measures the floor separately.

The tutorial module is never imported: it runs a module-level bootstrap that can
re-exec the case under MPI.  The constants below restate it and are asserted
against it in tests/coupler/test_cube_benchmark_parity.py.

    python scripts/experiments/cube_vpm_matrix.py --list
    python scripts/experiments/cube_vpm_matrix.py --check
    python scripts/experiments/cube_vpm_matrix.py --variant A0_bare --t-end 2.4
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CUBE = ROOT / "tutorials/coupled_FVM_VPM/cubeFlow"

# --- restated from cubeFlow_setup.py -----------------------------------------
CUBE_SIDE = 1.0
U_INF = (1.0, 0.0, 0.0)
RHO = 1.0
NU = 1.0e-3
SMAGORINSKY_CK = 0.094
SMAGORINSKY_CE = 1.048
INITIAL_U = (1.0, 0.0, 0.0)
DT_FVM = 0.01
DT_VPM = 0.05
HANDOFF_BOX = (-1.5, 3.2, -1.5, 1.5, -1.5, 1.5)
FVM_BOX = (-1.5, 3.5, -1.5, 1.5, -1.5, 1.5)
FVM_WAKE_BOX = (-1.25, 3.2, -1.25, 1.25, -1.25, 1.25)
FVM_CELL_SIZE = 0.0625
FVM_WAKE_CELL_SIZE = 0.03125
SURFACE_CELL_SIZE = 0.015625
VPM_DOMAIN = (-4.5, 11.0, -4.5, 4.5, -4.5, 4.5)
SAMPLE_SPACING = 0.04
OFFAXIS_Y = 0.75 * CUBE_SIDE
FORCE_INTERVAL = 0.05

# Production values, reproduced only by the D-phase variants.
PROD_H = 0.03
PROD_CAP = 1.8
PROD_BUFFER = 0.24
PROD_PRUNE_MIN = 0.005
PROD_PRUNE_MULT = 10.0
PROD_GBD_THRESHOLD = 0.30


@dataclass(frozen=True)
class Variant:
    """One point of the matrix.  Defaults are the stripped-down baseline."""

    label: str = ""
    h: float = 0.15
    velocity: str = "direct"  # direct | treecode
    viscous: str = "cs"  # cs | gbd | inviscid
    stretching: str = "off"  # off | transposed
    turbulence: str = "off"  # off | les
    # f32 on the GPU is the baseline because the macOS Metal backend rejects
    # f64 outright, and the treecode is f32 end-to-end regardless: production
    # BC velocity is single precision no matter what.  f64 is reachable only
    # on the CPU, so B5 changes device as well and is a sanity check, not an
    # single-variable point of the matrix.
    precision: str = "f32"  # f32 | f64 (f64 requires device="CPU")
    device: str = "AUTO"  # AUTO | CPU
    prune: bool = False  # boundary particle pruning
    cap: float = 1.0e9  # transfer_amplification_cap (1e9 == uncapped)
    buffer: float = PROD_BUFFER
    panel: bool = True  # body panels contribute to the VPM BC
    resync: bool = False

    @property
    def particles(self) -> int:
        vol = (
            (HANDOFF_BOX[1] - HANDOFF_BOX[0])
            * (HANDOFF_BOX[3] - HANDOFF_BOX[2])
            * (HANDOFF_BOX[5] - HANDOFF_BOX[4])
        )
        return int(vol / self.h**3)


_BARE = Variant(label="stripped baseline: inject, advect, all particles, exact sum")

VARIANTS: dict[str, Variant] = {
    # -- Phase A: the floor.  Everything the stack can do is switched off.
    "A0_bare": _BARE,
    # -- Phase B: one component back at a time, from A0.
    "B1_treecode": replace(_BARE, velocity="treecode", label="Barnes-Hut BC evaluation"),
    "B2_gbd": replace(_BARE, viscous="gbd", label="GBD regeneration + thresholding"),
    "B3_stretch": replace(_BARE, stretching="transposed", label="vortex stretching"),
    "B4_les": replace(_BARE, turbulence="les", label="VPM subgrid model"),
    "B5_f64cpu": replace(
        _BARE, precision="f64", device="CPU", label="f64 on CPU (also changes device)"
    ),
    "B6_prune": replace(_BARE, prune=True, label="boundary particle pruning"),
    "B7_cap": replace(_BARE, cap=PROD_CAP, label="transfer amplification cap"),
    # buffer_thickness must be positive, so the blending zone is bracketed
    # rather than switched off: one FVM cell against production against double.
    "B8_thinbuf": replace(_BARE, buffer=0.0625, label="blending buffer 1 FVM cell"),
    "B8b_thickbuf": replace(_BARE, buffer=0.48, label="blending buffer doubled"),
    "B9_nopanel": replace(_BARE, panel=False, label="body panels off"),
    "B10_resync": replace(_BARE, resync=True, label="BC resync after hand-off"),
    # -- Phase C: resolution floor, bare stack.  Direct summation stops being
    #    affordable below h=0.10, so the finer points use the treecode; B1
    #    prices that substitution at fixed h.
    "C1_h020": replace(_BARE, h=0.20, label="resolution floor h=0.20"),
    "C2_h010": replace(_BARE, h=0.10, label="resolution floor h=0.10"),
    "C3_h006": replace(_BARE, h=0.06, velocity="treecode", label="resolution floor h=0.06"),
    "C4_h003": replace(_BARE, h=PROD_H, velocity="treecode", label="resolution floor h=0.03"),
    # -- Phase D: everything on, to confirm the toggles compose to production.
    "D1_full_h015": replace(
        _BARE,
        velocity="treecode",
        viscous="gbd",
        stretching="transposed",
        turbulence="les",
        prune=True,
        cap=PROD_CAP,
        label="production stack at h=0.15",
    ),
    "D2_production": replace(
        _BARE,
        h=PROD_H,
        velocity="treecode",
        viscous="gbd",
        stretching="transposed",
        turbulence="les",
        prune=True,
        cap=PROD_CAP,
        label="production stack at production h",
    ),
}


def build_fvm(case_dir: Path, t_end: float, cores: int):
    from openonda.fvm import (
        AdaptiveCartesianMesher,
        BoundaryConfig,
        BoxRefinement,
        ExecutionConfig,
        ForceSampler,
        FVMSetup,
        LinearSolverConfig,
        LineSampler,
        OutputSetup,
        PimpleControl,
        SamplingSchedule,
        SchemesConfig,
        TimeConfig,
        TransportConfig,
    )
    from openonda.fvm import TurbulenceConfig as FVMTurbulenceConfig

    schedule = SamplingSchedule(every_time=FORCE_INTERVAL)
    samplers = (
        ForceSampler(
            patch_names=["cube"],
            ref_velocity=1.0,
            ref_area=CUBE_SIDE**2,
            ref_length=CUBE_SIDE,
            moment_centre=[0.0, 0.0, 0.0],
            schedule=schedule,
        ),
        LineSampler(
            start=[FVM_BOX[0], 0.0, 0.0],
            end=[FVM_BOX[1], 0.0, 0.0],
            spacing=SAMPLE_SPACING,
            file_name="centerline",
            schedule=schedule,
        ),
        LineSampler(
            start=[FVM_BOX[0], OFFAXIS_Y, 0.0],
            end=[FVM_BOX[1], OFFAXIS_Y, 0.0],
            spacing=SAMPLE_SPACING,
            file_name="offaxis_y075",
            schedule=schedule,
        ),
    )
    mesh = AdaptiveCartesianMesher(
        domain=FVM_BOX,
        max_cell_size=FVM_CELL_SIZE,
        surface_file=CUBE / "assets" / "cube.stl",
        wall_patch_name="cube",
        surface_cell_size=SURFACE_CELL_SIZE,
        refinements=(BoxRefinement(FVM_WAKE_BOX, FVM_WAKE_CELL_SIZE, "wakeBox"),),
        merge_outer_patch="numericalBoundary",
    )
    setup = FVMSetup(
        case_name="matrix",
        cores=cores,
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
            end_time=t_end,
            write_interval=10**9,
            write_interval_time=1.0e9,
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
        samplers=samplers,
        transport=TransportConfig(density=RHO, nu=NU),
        turbulence=FVMTurbulenceConfig.equilibrium_smagorinsky(
            Ck=SMAGORINSKY_CK, Ce=SMAGORINSKY_CE
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
    return setup, mesh


def build_vpm(v: Variant, case_dir: Path):
    from openonda.vpm import (
        AdvectionConfig,
        PanelSolver,
        StabilizationConfig,
        StretchingConfig,
        TurbulenceConfig,
        VelocityConfig,
        ViscousConfig,
        VPMSetup,
    )

    limit = max(200_000, 8 * v.particles)

    if v.viscous == "gbd":
        viscous = ViscousConfig.gbd(
            h=v.h,
            padding=3.0,
            viscosity=NU,
            threshold_mode="relative_local",
            threshold=PROD_GBD_THRESHOLD,
            max_nodes=limit,
            cap_abs_fraction=0.95,
            regen_radius_ratio=1.0,
        )
    elif v.viscous == "cs":
        # Core spreading: deterministic diffusion, no regeneration and no
        # thresholding, so no particle is ever discarded.
        viscous = ViscousConfig.cs(viscosity=NU, characteristic_distance=v.h)
    else:
        viscous = ViscousConfig.inviscid()

    velocity = (
        VelocityConfig.direct()
        if v.velocity == "direct"
        else VelocityConfig.treecode(theta=0.3, multipole_order=2)
    )
    stretching = (
        StretchingConfig.disabled()
        if v.stretching == "off"
        else StretchingConfig.transposed(scheme="RK2")
    )
    turbulence = (
        TurbulenceConfig.inviscid()
        if v.turbulence == "off"
        else TurbulenceConfig.equilibrium_smagorinsky(ck=SMAGORINSKY_CK, ce=SMAGORINSKY_CE)
    )

    panel_solver = None
    if v.panel:
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
        viscous=viscous,
        stretching=stretching,
        advection=AdvectionConfig(scheme="RK2"),
        turbulence=turbulence,
        velocity=velocity,
        stabilization=StabilizationConfig.bounded_domain(VPM_DOMAIN),
        particles_kernel="GAUSSIAN",
        precision=v.precision,
        processing_unit=v.device,
        max_particles=limit,
        max_targets=limit,
        vpm_domain_bounds=list(VPM_DOMAIN),
        log_mode="file",
        logging_frequency=50,
        timing_frequency=50,
        backup_frequency=0,
        backup_directory=str(case_dir / "solution"),
        export_flow_integrals=False,
        samplers=(),
        panel_solver=panel_solver,
        body_stl=str(CUBE / "assets" / "cube.stl"),
    )


def build_coupler(v: Variant):
    from openonda.coupler import CouplerSetup

    return CouplerSetup(
        u_inf=list(U_INF),
        handoff_box=HANDOFF_BOX,
        vpm_bc_mode="dirichlet",
        h=v.h,
        buffer_thickness=v.buffer,
        dead_zone_h=0.0,
        prune_vorticity_min=PROD_PRUNE_MIN if v.prune else 0.0,
        boundary_prune_multiplier=PROD_PRUNE_MULT if v.prune else 1.0,
        handoff_max_particles=max(200_000, 8 * v.particles),
        overlap_radius_ratio=1.0,
        transfer_amplification_cap=v.cap,
        resync_vpm_bc_after_handoff=v.resync,
        anchor_pressure=True,
        backup_period=10**9,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant")
    ap.add_argument("--t-end", type=float, default=2.4)
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--out", type=Path, default=CUBE / "matrix")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--check", action="store_true", help="construct every variant, run none")
    args = ap.parse_args()

    if args.list:
        print(f"{'variant':16s} {'h':>6s} {'N':>9s}  description")
        for name, v in VARIANTS.items():
            print(f"{name:16s} {v.h:6.3f} {v.particles:9,d}  {v.label}")
        return

    if args.check:
        bad = 0
        for name, v in VARIANTS.items():
            try:
                build_vpm(v, args.out / name)
                build_coupler(v)
                print(f"  ok    {name:16s} h={v.h:.3f} N~{v.particles:,d}")
            except Exception as exc:  # noqa: BLE001 - report, do not abort the sweep
                bad += 1
                print(f"  FAIL  {name:16s} {type(exc).__name__}: {exc}")
        print(f"\n{len(VARIANTS) - bad}/{len(VARIANTS)} variants construct")
        raise SystemExit(1 if bad else 0)

    if not args.variant:
        raise SystemExit("give --variant NAME (or --list / --check)")
    if args.variant not in VARIANTS:
        raise SystemExit(f"unknown variant {args.variant!r}; --list to see them")

    v = VARIANTS[args.variant]
    case_dir = args.out / args.variant
    case_dir.mkdir(parents=True, exist_ok=True)

    from openonda.coupler import FVMVPMCoupler, setup_coupler
    from openonda.fvm import setup_fvm_solver

    setup, mesh = build_fvm(case_dir, args.t_end, args.cores)
    fvm_solver = setup_fvm_solver(setup, case_dir=case_dir, mesh=mesh)

    vpm_solver = None
    if FVMVPMCoupler.is_master_rank():
        from openonda.vpm import setup_vpm_solver

        print(f"[matrix] {args.variant}: {v.label}", flush=True)
        print(
            f"[matrix] h={v.h} N~{v.particles:,d} velocity={v.velocity} viscous={v.viscous} "
            f"stretch={v.stretching} turb={v.turbulence} prec={v.precision}/{v.device} "
            f"prune={v.prune} cap={v.cap:g} buffer={v.buffer} panel={v.panel}",
            flush=True,
        )
        vpm_solver = setup_vpm_solver(build_vpm(v, case_dir))

    setup_coupler(vpm_solver, fvm_solver, build_coupler(v)).run()

    if FVMVPMCoupler.is_master_rank():
        print(f"[matrix] {args.variant} done -> {case_dir / 'samples'}", flush=True)


if __name__ == "__main__":
    main()
