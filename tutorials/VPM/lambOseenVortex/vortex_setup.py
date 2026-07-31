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

from openonda.vpm import LambOseenVPM, LineSampler, SurfaceSampler
from openonda.vpm import ParticleDistributor, Solver, VPMSetup
from openonda.vpm import (
    AdvectionConfig,
    StretchingConfig,
    VelocityConfig,
    ViscousConfig,
)

# =========================================================
# Buckingham-Pi normalisation
# =========================================================
# Dimensional parameters:
#   $\Gamma$          = 1.0 m^2/s          circulation per vortex
#   $\nu$             = 1/530 m^2/s        kinematic viscosity
#   $b_0$             = 1.0 m              centre-to-centre separation
#   $a_{c,0}$         = 0.125 m            initial core radius (radius of peak
#                                          azimuthal velocity — C&W convention)
#   $a_{c,0}/b_0$     = 0.125              core-to-separation ratio
#   $\sigma_0 = a_{c,0}/1.12091$          = 0.1115 m   diffused Gaussian width
#
# Derived reference quantities:
#   $Re = \Gamma/\nu$                            = 530        Reynolds number
#   $U_{c,0} = \Gamma/(2\pi a_{c,0})$           = 1.273 m/s  reference velocity
#   $\omega_{c,0} = \Gamma/(\pi a_{c,0}^2)$     = 20.37 1/s  reference vorticity
#   $G_{c,0} = U_{c,0}/a_{c,0}$                = 10.19 1/s  reference velocity gradient
#   $t_0 = \sigma_0^2/(4\nu)$                  = 1.65 s     initial vortex age
#   $h$ (particle spacing)                      = 0.3 * a_{c,0}
#
# Buckingham-Pi groups used in figures:
#   $\tau = \nu t / a_{c,0}^2$                              time
#   $r^* = r / a_{c,0}$                                     radius
#   $u^* = u_\theta / U_{c,0}$                              azimuthal velocity
#   $\omega^* = \omega_z / \omega_{c,0}$                    vorticity
#   $G^* = (\partial u_y/\partial x)\,a_{c,0}/U_{c,0}$      velocity gradient
#   $x_c^* = x_c / b_0$                                     dipole trajectory
#   $a_c^* = a_c / a_{c,0}$                                 core radius (dipole)
#   $\sigma^2 / b_0^2$                                      core area (merging)
#   $b^* = b / b_0$                                         separation (merging)
#   $P^* = (dE/dt) / (\nu\Gamma^2/a_{c,0}^2)$               energy dissipation
#
TUTORIAL_DIR = Path(__file__).resolve().parent
DEFAULT_SOLUTION_DIR = TUTORIAL_DIR / "solution"

RE = 530.0  # Re_Γ = Γ/nu — matches C&W 2003 reference data
AC0 = 0.125  # initial C&W core radius a_{c,0} [m] — radius of PEAK azimuthal velocity
# C&W 2003 eq. (4.1): the co-rotating merger experiment starts at a0/b0 = 0.125
# +/- 0.007 — the SAME core-to-separation ratio as the single-vortex case, and the
# ratio at which their reference fig. 5 data (Re=530) were taken. A larger a0/b0
# seeds the pair much closer to the convective-merger threshold a_crit/b0 = 0.29
# (eq. 4.2), so it merges far too early. (Was 0.17, which merged ~2x too soon.)
MERGING_AC0 = 0.125  # C&W co-rotating merger benchmark core radius a_{c,0} [m] (a0/b0=0.125)
B0 = 1.0  # center-to-center separation b0 [m]  (a0/b0 = 0.125)

# C&W 2003 define a0 as the radius of PEAK azimuthal velocity. For a Lamb-Oseen
# vortex r_max = BETA_RMAX*sigma, BETA_RMAX = nonzero root of e^x = 1 + 2x.
BETA_RMAX = 1.1209064227785341
TOTAL_TIME = 20.0

# z-column span in units of AC0. Kept short (was 50) so the capped DVH/GBD regen
# nodes are not spread thin across redundant z-layers: the cap only affords an
# in-plane disk of radius sqrt(max_nodes/(nz*pi))*h (nz=LENGTH/spacing_factor);
# too-long a column shrinks that disk below the domain and hard-clips the
# diffusing tail (spurious under-diffusion). 16 keeps the mid-plane 3-D while
# max_nodes=250k keeps the clip radius past the domain on every case.
LENGTH = 16
SPACING_FACTOR = 0.3
GRID_SPACING_FACTOR = 0.3
VISCOUS_THRESHOLD_MODE = "budget"
VISCOUS_THRESHOLD = 1.0e-5
REGEN_RADIUS_RATIO = 1.5
DVH_RD_RATIO = 3
GBD_MAX_NODES = 250_000


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
            regen_radius_ratio=REGEN_RADIUS_RATIO,
        )
    elif scheme == "dvh":
        return ViscousConfig.dvh(
            h=spacing,
            threshold=VISCOUS_THRESHOLD,
            threshold_mode=VISCOUS_THRESHOLD_MODE,
            dvh_rd_ratio=DVH_RD_RATIO,
            viscosity=nu,
            max_nodes=args.dvh_max_nodes,
            regen_radius_ratio=REGEN_RADIUS_RATIO,
        )


# =========================================================
# Unified case runner
# =========================================================
def run_case(args: argparse.Namespace, scheme: str, solution_dir: Path) -> None:
    """Run one viscous scheme for the case given by --gamma1/--gamma2."""
    gamma1, gamma2 = args.gamma1, args.gamma2
    nu = 1.0 / args.re
    rc = MERGING_AC0 if gamma1 * gamma2 > 0 else args.core_radius
    b0 = args.separation
    if rc <= 0.0:
        raise ValueError("--core-radius must be positive")
    if b0 <= 0.0:
        raise ValueError("--separation must be positive")
    # rc is the C&W peak-velocity radius; diffuse the matching Gaussian width.
    sigma0 = rc / BETA_RMAX
    t0 = sigma0**2 / (4.0 * nu)
    spacing_factor = args.spacing_factor
    if scheme in {"dvh", "gbd"} and args.grid_spacing_factor is not None:
        spacing_factor = args.grid_spacing_factor
    spacing = spacing_factor * rc

    # Determine case label for folder naming
    case_type = "vortex" if gamma2 == 0 else ("dipole" if gamma1 * gamma2 < 0 else "merging")
    output_dir = solution_dir / f"{case_type}_{scheme}{args.tag}"

    # Grid bounds:
    margin = 7.0 * rc  # clearance beyond vortex cores
    y_offset = 0.5 * b0 if gamma2 != 0 else 0.0  # half-separation for pairs
    domain_half = y_offset + margin  # single: 1.5*rc, pair: b0/2 + 1.5*rc

    if gamma1 * gamma2 < 0:
        # Dipole: extra room in +x for self-propulsion
        bounds_x_max = domain_half + 8.0 * b0
    else:
        bounds_x_max = domain_half

    domain_bounds = [
        -domain_half,
        bounds_x_max,
        -domain_half,
        domain_half,
        -args.length * rc / 2,
        args.length * rc / 2,
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
    viscous = build_viscous_config(scheme, nu, args, spacing)
    dt_actual = viscous.dvh_required_dt() if scheme == "dvh" else args.dt
    output_interval = max(1, round(args.backup_frequency * args.dt / dt_actual))

    # ================================================
    # Core-size control (CS only)
    # ================================================
    config = VPMSetup.dns_simulation(
        time_step_size=dt_actual,
        viscous=viscous,
        advection=advection,
        stretching=stretching,
        processing_unit=args.processing_unit,
        device_memory_fraction=args.device_memory_fraction,
        velocity=VelocityConfig.treecode(
            theta=0.7,
            multipole_order=3,
            sort_particle_targets=True,
            traversal_block_dim=128,
        ),
        backup_frequency=output_interval,
        logging_frequency=output_interval,
        timing_frequency=50,
        backup_file_name=f"{case_type}_{scheme}",
        backup_directory=str(output_dir),
        samplers=samplers,
        clean=args.clean,
        vpm_domain_bounds=domain_bounds,
    )

    solver = Solver(setup=config)

    n = len(positions)
    solver.add_vortex_particles(
        positions,
        v1_vel,
        v1_circ,
        radii,
        volumes,
        group_id=np.zeros(n, dtype=np.int32),
    )
    if gamma2 != 0.0:
        solver.add_vortex_particles(
            positions,
            v2_vel,
            v2_circ,
            radii,
            volumes,
            group_id=np.ones(n, dtype=np.int32),
        )
    solver.remove_weak_particles(percent=1.0, per_group=True)

    if args.num_steps is not None:
        num_steps = int(args.num_steps)
    else:
        steps_float = args.total_time / dt_actual
        num_steps = max(1, round(steps_float))
        actual_time = num_steps * dt_actual
        if not np.isclose(actual_time, args.total_time, rtol=1e-12, atol=1e-9):
            print(
                f"  [{scheme}] dt={dt_actual:.6g}s does not evenly divide "
                f"--total-time={args.total_time:g}s; running {num_steps} steps "
                f"to t={actual_time:.6g}s instead ({actual_time - args.total_time:+.4g}s off target)."
            )

    solver.info()
    for _ in range(num_steps):
        solver.update_state()
    if not np.isclose(solver.flow_time, num_steps * dt_actual, rtol=0.0, atol=1e-9):
        raise RuntimeError(
            f"{scheme.upper()} finished at t={solver.flow_time:.12g}s, "
            f"expected {num_steps * dt_actual:.12g}s."
        )
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
        "--schemes",
        default="cs",
        help="Comma-separated viscous schemes: cs, rwm, dvh, gbd.",
    )
    parser.add_argument(
        "--solution-dir",
        default=str(DEFAULT_SOLUTION_DIR),
        help="Root output directory for all scheme sub-folders.",
    )
    parser.add_argument(
        "--clean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete existing output sub-folder before running.",
    )
    parser.add_argument(
        "--total-time",
        type=float,
        default=TOTAL_TIME,
        help="Total simulation time [s].",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.04,
        help="Time step size [s].",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Exact number of time steps (overrides --total-time when given).",
    )
    parser.add_argument(
        "--length",
        type=float,
        default=LENGTH,
        help="Vortex column span in z, in units of AC0.",
    )
    parser.add_argument(
        "--re",
        type=float,
        default=RE,
        help="Reynolds number Re_Γ = Γ/nu (default: 530, matching C&W 2003 reference).",
    )
    parser.add_argument(
        "--core-radius",
        type=float,
        default=AC0,
        help="Initial Lamb-Oseen core radius a_{c,0} [m]. Co-rotating merger cases use MERGING_AC0.",
    )
    parser.add_argument(
        "--separation",
        type=float,
        default=B0,
        help="Initial centre-to-centre separation b0 [m] for two-vortex cases.",
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
        default=SPACING_FACTOR,
        help="Particle/grid spacing as a fraction of the core radius rc.",
    )
    parser.add_argument(
        "--grid-spacing-factor",
        type=float,
        default=GRID_SPACING_FACTOR,
        help="DVH/GBD spacing as a fraction of rc.",
    )
    parser.add_argument(
        "--processing-unit",
        default="CUDA",
        choices=["AUTO", "CPU", "VULKAN", "CUDA", "METAL"],
        help="Compute backend. Default is CUDA to avoid Vulkan DVH/GBD field-retention issues.",
    )
    parser.add_argument(
        "--device-memory-fraction",
        type=float,
        default=0.55,
        help="Fraction of GPU memory reserved by Taichi.",
    )
    parser.add_argument(
        "--backup-frequency",
        type=int,
        default=15,
        help="Initial backup/log interval. Grid schemes are adjusted to preserve physical cadence.",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Suffix appended to the output directory name (for parameter sweeps).",
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
