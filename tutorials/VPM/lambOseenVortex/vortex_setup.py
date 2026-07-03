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
from source.solvers.VPM.config.types import (
    AdvectionConfig,
    StretchingConfig,
    VelocityConfig,
    ViscousConfig,
)

# =========================================================
# Buckingham-Pi normalisation
# =========================================================
# Dimensional parameters:
#   $\Gamma$       = 1.0 m^2/s          circulation per vortex
#   $\nu$          = 1/530 m^2/s        kinematic viscosity
#   $b_0$          = 1.0 m              centre-to-centre separation
#   $r_{c,0}$     = 0.125 m            initial core radius (= a_0)
#   $a_0/b_0$     = 0.125              core-to-separation ratio
#
# Derived reference quantities:
#   $Re = \Gamma/\nu$                    = 530        Reynolds number
#   $U_{c,0} = \Gamma/(2\pi r_{c,0})$   = 1.273 m/s  reference velocity
#   $\omega_{c,0} = \Gamma/(\pi r_{c,0}^2)$ = 20.37 1/s  reference vorticity
#   $G_{c,0} = U_{c,0}/r_{c,0}$        = 10.19 1/s  reference velocity gradient
#   $t_0 = r_{c,0}^2/(4\nu)$           = 2.07 s     initial vortex age
#   $h$ (particle spacing)              = 0.04125 m  (0.33 * r_{c,0})
#
# Buckingham-Pi groups used in figures:
#   $\tau = \nu t / b_0^2$                              time
#   $r^* = r / r_{c,0}$                                 radius
#   $u^* = u_\theta / U_{c,0}$                          azimuthal velocity
#   $\omega^* = \omega_z / \omega_{c,0}$                vorticity
#   $G^* = (\partial u_y/\partial x)\,r_{c,0}/U_{c,0}$  velocity gradient
#   $x_c^* = x_c / b_0$                                 dipole trajectory
#   $r_c^* = r_c / r_{c,0}$                             core radius (dipole)
#   $\sigma^2 / b_0^2$                                  core area (merging)
#   $b^* = b / b_0$                                     separation (merging)
#   $P^* = (dE/dt) / (\nu\Gamma^2/b_0^2)$               energy dissipation
#
TUTORIAL_DIR = Path(__file__).resolve().parent
DEFAULT_SOLUTION_DIR = TUTORIAL_DIR / "solution"

RE = 530.0  # Re_Γ = Γ/nu — matches C&W 2003 reference data
RC = 0.125  # initial core radius a0 [m]
B0 = 1.0    # center-to-center separation b0 [m]  (a0/b0 = 0.125)
TOTAL_TIME = 16.0
LENGTH = 20  # vortex column span in z, in units of RC (default; override with --length)
VISCOUS_THRESHOLD_MODE = "budget"
VISCOUS_THRESHOLD = 2.0e-4
DVH_RD_RATIO = 3
GBD_MAX_NODES = 120_000


# =========================================================
# Shared utilities
# =========================================================
def build_viscous_config(scheme: str, nu: float, args: argparse.Namespace, spacing: float):
    if scheme in {"cs", "rwm"}:
        return ViscousConfig(scheme=scheme.upper(), viscosity=nu)
    if scheme == "gbd":
        return ViscousConfig.gbd(
            h=spacing,
            threshold=VISCOUS_THRESHOLD,
            threshold_mode=VISCOUS_THRESHOLD_MODE,
            viscosity=nu,
            max_nodes=GBD_MAX_NODES,
        )
    elif scheme == "dvh":
        return ViscousConfig.dvh(
            h=spacing,
            threshold=VISCOUS_THRESHOLD,
            threshold_mode=VISCOUS_THRESHOLD_MODE,
            dvh_rd_ratio=DVH_RD_RATIO,
            viscosity=nu,
            max_nodes=args.dvh_max_nodes,
        )


# =========================================================
# Unified case runner
# =========================================================
def run_case(args: argparse.Namespace, scheme: str, solution_dir: Path) -> None:
    """Run one viscous scheme for the case given by --gamma1/--gamma2."""
    gamma1, gamma2 = args.gamma1, args.gamma2
    nu = 1.0 / args.re
    t0 = RC**2 / (4.0 * nu)
    spacing_factor = args.spacing_factor
    if scheme in {"dvh", "gbd"} and args.grid_spacing_factor is not None:
        spacing_factor = args.grid_spacing_factor
    spacing = spacing_factor * RC

    # Determine case label for folder naming
    case_type = "vortex" if gamma2 == 0 else ("dipole" if gamma1 * gamma2 < 0 else "merging")
    output_dir = solution_dir / f"{case_type}_{scheme}{args.tag}"

    # Grid bounds:
    margin = 7.0 * RC  # clearance beyond vortex cores
    y_offset = 0.5 * B0 if gamma2 != 0 else 0.0  # half-separation for pairs
    domain_half = y_offset + margin  # single: 1.5*rc, pair: b0/2 + 1.5*rc

    if gamma1 * gamma2 < 0:
        # Dipole: extra room in +x for self-propulsion
        bounds_x_max = domain_half + 8.0 * B0
    else:
        bounds_x_max = domain_half

    domain_bounds = [
        -domain_half,
        bounds_x_max,
        -domain_half,
        domain_half,
        -args.length * RC / 2,
        args.length * RC / 2,
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

    # Vortex 2 initialization (only when a second vortex is present)
    if gamma2 != 0.0:
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

    # ================================================
    # Time integration
    # ================================================
    advection = AdvectionConfig(scheme="RK3")
    stretching = StretchingConfig.transposed(scheme="RK3")

    # ================================================
    # Core-size control (CS only)
    # ================================================
    config = SolverConfig.dns_simulation(
        time_step_size=args.dt,
        viscous=build_viscous_config(scheme, nu, args, spacing),
        advection=advection,
        stretching=stretching,
        # CUDA avoids the Taichi 1.7.x Vulkan field-lifetime growth seen in
        # long DVH/GBD runs. Override only for backend debugging.
        processing_unit=args.processing_unit,
        device_memory_fraction=args.device_memory_fraction,
        velocity=VelocityConfig.treecode(
            theta=0.35,
            sort_particle_targets=True,
            traversal_block_dim=128,
        ),
        backup_frequency=args.backup_frequency,
        logging_frequency=args.backup_frequency,
        timing_frequency=50,
        backup_file_name=f"{case_type}_{scheme}",
        solution_name=str(output_dir),
        backup_directory=str(output_dir),
        samplers=samplers,
        clean=args.clean,
        vpm_domain_bounds=domain_bounds,
    )

    solver = Solver(config=config)
    # Pre-size temp fields to the bounded particle count,
    # not 500k: on a 6 GB laptop GPU the oversized pre-allocation wastes VRAM
    # headroom that the DVH/GBD grid + treecode + per-step staging buffers need.
    # Fields still grow on demand if a run exceeds this.
    solver.physics._resize_temp_fields(args.temp_field_capacity)

    n = len(positions)
    solver.add_vortex_particles(
        positions, v1_vel, v1_circ, radii, volumes,
        group_id=np.zeros(n, dtype=np.int32),
    )
    if gamma2 != 0.0:
        solver.add_vortex_particles(
            positions, v2_vel, v2_circ, radii, volumes,
            group_id=np.ones(n, dtype=np.int32),
        )
    solver.remove_weak_particles(percent=1.0, per_group=True)

    # DVH overrides the user-set time step, so we query the actual dt used.
    dt_actual = solver.get_time_step_size()

    # Keep a consistent physical backup interval (~0.3s matching CS/RWM at
    # args.dt=0.03, backup_frequency=10) regardless of which scheme is used.
    fixed_interval = max(1, round(10.0 * args.dt / dt_actual))
    solver.update_config(backup_frequency=fixed_interval)
    solver.update_config(logging_frequency=fixed_interval)

    # Determine number of steps: explicit --num-steps overrides --total-time
    num_steps = (
        int(args.num_steps)
        if args.num_steps is not None
        else int(np.ceil(args.total_time / dt_actual))
    )

    solver.info()
    for _ in range(num_steps):
        solver.update_state()
    solver.reset_gpu()  # This should clean up the GPU


# =========================================================
# Argument parsing
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Lamb-Oseen vortex experiments.\n"
            "  gamma2=0 → single vortex | gamma1*gamma2<0 → dipole | same sign → merging"
        )
    )

    parser.add_argument(
        "--gamma1", type=float, required=True, help="Circulation of vortex 1 [m²/s]."
    )
    parser.add_argument(
        "--gamma2", type=float, default=0.0, help="Circulation of vortex 2 [m²/s]. "
    )
    parser.add_argument(
        "--schemes", default="cs", help="Comma-separated viscous schemes: cs, rwm, dvh, gbd.",
    )
    parser.add_argument(
        "--solution-dir", default=str(DEFAULT_SOLUTION_DIR), help="Root output directory for all scheme sub-folders.",
    )
    parser.add_argument(
        "--clean", action="store_true", help="Delete existing output sub-folder before running.",
    )
    parser.add_argument(
        "--total-time", type=float, default=TOTAL_TIME, help="Total simulation time [s].",
    )
    parser.add_argument(
        "--dt", type=float, default=0.04, help="Time step size [s].",
    )
    parser.add_argument(
        "--num-steps", type=int, default=None, help="Exact number of time steps (overrides --total-time when given).",
    )
    parser.add_argument(
        "--length", type=float, default=LENGTH, help="Vortex column span in z, in units of RC.",
    )
    parser.add_argument( 
        "--re", type=float, default=RE, help="Reynolds number Re_Γ = Γ/nu (default: 530, matching C&W 2003 reference).",
    )
    parser.add_argument(
        "--dvh-max-nodes", type=int, default=120_000, help="Hard cap on surviving DVH regen nodes (budget-by-count).",
    )
    parser.add_argument(
        "--spacing-factor", type=float, default=0.45, help="Particle/grid spacing as a fraction of the core radius rc.",
    )
    parser.add_argument(
        "--grid-spacing-factor",
        type=float,
        default=0.60,
        help="DVH/GBD spacing as a fraction of rc. Use 0.45-0.50 for higher-resolution sweeps.",
    )
    parser.add_argument(
        "--processing-unit",
        default="CUDA",
        choices=["CPU", "GPU", "GPU_VULKAN", "VULKAN", "CUDA", "GPU_METAL", "METAL"],
        help="Compute backend. Default is CUDA to avoid Vulkan DVH/GBD field-retention issues.",
    )
    parser.add_argument(
        "--device-memory-fraction",
        type=float,
        default=0.55,
        help="Fraction of GPU memory reserved by Taichi.",
    )
    parser.add_argument(
        "--temp-field-capacity",
        type=int,
        default=160_000,
        help="Initial temporary-field capacity; fields still grow on demand.",
    )
    parser.add_argument(
        "--backup-frequency",
        type=int,
        default=15,
        help="Initial backup/log interval. Grid schemes are adjusted to preserve physical cadence.",
    )
    parser.add_argument(
        "--tag", default="", help="Suffix appended to the output directory name (for parameter sweeps).",
    )
    return parser.parse_args()

# =========================================================
# Entry point
# =========================================================
def main(args: argparse.Namespace) -> int:
    schemes = [s.strip().lower() for s in args.schemes.split(",") if s.strip()]

    solution_dir = Path(args.solution_dir).resolve()
    for scheme in schemes:
        run_case(args, scheme, solution_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
