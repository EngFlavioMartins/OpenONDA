#!/usr/bin/env python3
"""Run Lamb-Oseen vortex experiments: single vortex, dipole, or co-rotating merger.

The case is determined by --gamma1 and --gamma2:
  gamma2 == 0          → single vortex (only vortex 1 is placed, at the origin)
  gamma1 * gamma2 < 0  → counter-rotating dipole (self-propels)
  gamma1 * gamma2 > 0  → co-rotating pair (merging)
"""

from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path

from source.solvers.VPM.utils import LambOseenVPM, LineSampler, SurfaceSampler
from source.solvers.VPM import ParticleDistributor, Solver, SolverConfig
from source.solvers.VPM.config.types import StretchingConfig, ViscousConfig

# ── Shared physics (all cases use the same Re/b₀/a₀) ──────────────────────────
# Physical Parameters:
#   Gamma       = 1.0 m²/s
#   Re          = Gamma/nu = 530.0  =>  nu = 1/530 ≈ 1.887e-3 m²/s
#   a0/b0       = 0.125, b0 = 1.0 m  =>  rc(0) = 0.125 m
#   t0          = rc(0)² / (4*nu) ~ 2.07 s (initial vortex age)
#   spacing h   = 0.27 * rc = 0.03375 m
#   Uc_ref      = Gamma / (2*pi*rc) ~ 1.2732 m/s
#   wc_ref      = Gamma / (pi*rc²) ~ 20.37 s⁻¹
#
# Buckingham-Pi Normalization (used in plots):
#   r* = r / rc
#   u* = u / Uc_ref
#   w* = w / wc_ref
#   G* = (dv/dx) * (rc / Uc_ref)
#
TUTORIAL_DIR = Path(__file__).resolve().parent
DEFAULT_SOLUTION_DIR = TUTORIAL_DIR / "solution"

RE = 530.0  # Re_Γ = Γ/nu — matches C&W 2003 reference data
RC = 0.125  # initial core radius a0 [m]
B0 = 1.0    # center-to-center separation b0 [m]  (a0/b0 = 0.125)
TOTAL_TIME = 20.0
LENGTH = 50


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------
def build_viscous_config(scheme: str, nu: float, args: argparse.Namespace, spacing: float):
    if scheme in {"cs", "rwm"}:
        return ViscousConfig(scheme=scheme.upper(), viscosity=nu)
    if scheme == "gbd":
        return ViscousConfig.gbd(
            h=spacing,
            padding=5.0,
            threshold=args.viscous_threshold,
            threshold_mode=args.viscous_threshold_mode,
            viscosity=nu,
        )
    elif scheme == "dvh":
        return ViscousConfig.dvh(
            h=spacing,
            padding=5.0,
            threshold=args.viscous_threshold,
            threshold_mode=args.viscous_threshold_mode,
            dvh_rd_ratio=args.dvh_rd_ratio,
            viscosity=nu,
            max_nodes=args.dvh_max_nodes,
        )


# ---------------------------------------------------------------------------
# Unified case runner
# ---------------------------------------------------------------------------
def run_case(args: argparse.Namespace, scheme: str, solution_dir: Path) -> None:
    """Run one viscous scheme for the case given by --gamma1/--gamma2."""
    gamma1, gamma2 = args.gamma1, args.gamma2
    nu = 1.0 / args.re
    t0 = RC**2 / (4.0 * nu)
    spacing = args.spacing_factor * RC

    # Determine case label for folder naming
    case_type = "vortex" if gamma2 == 0 else ("dipole" if gamma1 * gamma2 < 0 else "merging")
    output_dir = solution_dir / f"{case_type}_{scheme}{args.tag}"

    # Grid bounds:
    margin = 7.0 * RC  # clearance beyond vortex cores
    y_offset = 0.5 * B0 if gamma2 != 0 else 0.0  # half-separation for pairs
    domain_half = y_offset + margin  # single: 1.5*rc, pair: b0/2 + 1.5*rc

    if gamma1 * gamma2 < 0:
        # Dipole: extra room in +x for self-propulsion
        bounds_x_max = domain_half + 3.0 * B0
    else:
        bounds_x_max = domain_half

    domain_bounds = [
        -domain_half,
        bounds_x_max,
        -domain_half,
        domain_half,
        -LENGTH * RC / 2,
        LENGTH * RC / 2,
    ]

    positions, volumes, radii = ParticleDistributor.hexagonal_distribution(domain_bounds, spacing)
    mean_particle_radius = float(radii.mean())

    # Vortex 1 initialization: center must be np.array to avoid casting errors
    v1_vel, v1_visc, v1_circ = LambOseenVPM(
        viscosity=nu,
        avg_particle_radius=mean_particle_radius,
        positions=positions,
        volumes=volumes,
        vortex_center=np.array([0.0, y_offset, 0.0]),
        vortex_strength=gamma1,
        vortex_time=t0,
        anti_diffuse_flag=True,
    )

    # Vortex 2 initialization
    v2_vel, _, v2_circ = LambOseenVPM(
        viscosity=nu,
        avg_particle_radius=mean_particle_radius,
        positions=positions,
        volumes=volumes,
        vortex_center=np.array([0.0, -y_offset, 0.0]),
        vortex_strength=gamma2,
        vortex_time=t0,
        anti_diffuse_flag=True,
    )

    # All-case samplers: output defaults to backup_directory/samples/ (solver default)
    samplers = [
        LineSampler(
            start=[-10, 0, 0], end=[10, 0, 0], spacing=spacing, file_name=f"{case_type}_{scheme}_x"
        ),
        LineSampler(
            start=[0, -5, 0], end=[0, 5, 0], spacing=spacing, file_name=f"{case_type}_{scheme}_y"
        ),
        SurfaceSampler(
            point=[0, 0, 0],
            normal=[0, 0, 1],
            bounds=domain_bounds[:4],
            spacing=spacing,
            file_name=f"{case_type}_{scheme}_z0",
        ),
    ]

    config = SolverConfig.dns_simulation(
        time_step_size=args.dt,
        viscous=build_viscous_config(scheme, nu, args, spacing),
        stretching=StretchingConfig.transposed(),
        processing_unit="GPU_VULKAN",
        backup_frequency=10,
        logging_frequency=10,
        backup_file_name=f"{case_type}_{scheme}",
        solution_name=str(output_dir),
        backup_directory=str(output_dir),
        samplers=samplers,
        clean=args.clean,
        vpm_domain_bounds=domain_bounds,
    )

    solver = Solver(config=config)
    solver.physics._resize_temp_fields(250_000)

    total_circ = v1_circ + v2_circ
    total_vel = v1_vel + v2_vel
    circ_mag = np.linalg.norm(total_circ, axis=1)
    max_circ = float(circ_mag.max()) if len(circ_mag) > 0 else 0.0

    mask = circ_mag >= 0.01 * max_circ
    positions_add = positions[mask]
    vel_add = total_vel[mask]
    circ_add = total_circ[mask]
    radii_add = radii[mask]
    volumes_add = volumes[mask]


    solver.add_vortex_particles(positions_add, vel_add, circ_add, radii_add, volumes_add)

    # We need to get the time-step size due the DVH scheme changing it:
    dt_actual = solver.get_time_step_size()

    # Determine number of steps: explicit --num-steps overrides --total-time
    num_steps = int(np.ceil(args.total_time / dt_actual))

    solver.info()
    for _ in range(num_steps):
        solver.update_state()
    solver.reset_gpu()  # This should clean up the GPU


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Lamb-Oseen vortex experiments.\n"
            "Case is inferred from --gamma1/--gamma2:\n"
            "  gamma2=0 → single vortex | gamma1*gamma2<0 → dipole | same sign → merging"
        )
    )

    parser.add_argument(
        "--gamma1", type=float, required=True, help="Circulation of vortex 1 [m²/s]."
    )
    parser.add_argument(
        "--gamma2",
        type=float,
        default=0.0,
        help="Circulation of vortex 2 [m²/s]. "
        "0 → single vortex; opposite sign → dipole; same sign → merging.",
    )
    parser.add_argument(
        "--schemes", default="cs", help="Comma-separated viscous schemes: cs, rwm, dvh, gbd."
    )
    parser.add_argument(
        "--solution-dir",
        default=str(DEFAULT_SOLUTION_DIR),
        help="Root output directory for all scheme sub-folders.",
    )
    parser.add_argument(
        "--clean", action="store_true", help="Delete existing output sub-folder before running."
    )
    parser.add_argument(
        "--total-time", type=float, default=TOTAL_TIME, help="Total simulation time [s]."
    )
    parser.add_argument("--dt", type=float, default=0.05, help="Time step size [s].")
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Exact number of time steps (overrides --total-time when given).",
    )
    parser.add_argument(
        "--re",
        type=float,
        default=RE,
        help="Reynolds number Re_Γ = Γ/nu (default: 530, matching C&W 2003 reference).",
    )
    # ── Viscous-scheme sweep parameters (defaults = historical tutorial values) ──
    parser.add_argument(
        "--viscous-threshold-mode",
        choices=["budget", "relative_max", "absolute"],
        default="budget",
        help="Grid-regen node survival criterion for DVH/GBD (default: budget).",
    )
    parser.add_argument(
        "--viscous-threshold",
        type=float,
        default=1e-4,
        help="Threshold value: budget → fractional Σ|Γ| loss allowed per firing; "
        "absolute → node |Γ| floor [m³/s] (default: 1e-4).",
    )
    parser.add_argument(
        "--dvh-rd-ratio", type=int, default=3, choices=[3, 4, 5], help="DVH R_d/h ratio."
    )
    parser.add_argument(
        "--dvh-max-nodes",
        type=int,
        default=250_000,
        help="Hard cap on surviving DVH regen nodes (budget-by-count).",
    )
    parser.add_argument(
        "--spacing-factor",
        type=float,
        default=0.33,
        help="Particle/grid spacing as a fraction of the core radius rc (default 0.33).",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Suffix appended to the output directory name (for parameter sweeps).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(args: argparse.Namespace) -> int:
    schemes = [s.strip().lower() for s in args.schemes.split(",") if s.strip()]

    solution_dir = Path(args.solution_dir).resolve()
    for scheme in schemes:
        run_case(args, scheme, solution_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
