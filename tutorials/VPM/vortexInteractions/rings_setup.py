#!/usr/bin/env python3
"""
Vortex-ring interaction stabilizer benchmark.

Each run uses the same physical formulation:

  * LES Smagorinsky model
  * transposed stretching
  * RK3 advection and stretching

The case matrix changes only the stabilization method so the plots show which
stabilizer improves survival and conservation for a fixed LES baseline.

Usage::

    python rings_setup.py --gamma1 3.14159265358979 --gamma2 -3.14159265358979

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: May 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import argparse
import json
from pathlib import Path

import numpy as np

from source.solvers.VPM import ParticleDistributor, Solver, SolverConfig
from source.solvers.VPM.config.types import (
    AdvectionConfig,
    RVPM_DEFAULT_F,
    RVPM_DEFAULT_G,
    StabilizationConfig,
    StretchingConfig,
    TurbulenceConfig,
    VelocityConfig,
    ViscousConfig,
)
from source.solvers.VPM.utils import VortexRingVPM
from source.solvers.VPM.utils.field_samplers import SurfaceSampler


GAMMA_REF = np.pi
RING_RADIUS = 1.0
CORE_RADIUS = 0.1
REYNOLDS_GAMMA = 3000.0
KINEMATIC_VISCOSITY = GAMMA_REF / REYNOLDS_GAMMA
REGEN_THRESHOLD = 2.0e-4
REGEN_THRESHOLD_MODE = "budget"
MAX_REGEN_NODES = 150_000


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Vortex-ring LES stabilizer benchmark")

    parser.add_argument(
        "--gamma1", type=float, default=GAMMA_REF, help="Ring 1 circulation [m2/s]."
    )
    parser.add_argument(
        "--gamma2", type=float, default=GAMMA_REF, help="Ring 2 circulation [m2/s]."
    )
    parser.add_argument("--name", default="leapfrog_adaptive", help="Output sub-directory name.")
    parser.add_argument("--dt", type=float, default=1.0e-2, help="Time-step size [s].")
    parser.add_argument("--num-steps", type=int, default=720, help="Number of time steps.")
    parser.add_argument(
        "--particle-spacing", type=float, default=0.045, help="Particle spacing [m]."
    )
    parser.add_argument("--output-root", default="solution", help="Parent output directory.")
    parser.add_argument("--backup-frequency", type=int, default=20, help="Backup interval [steps].")
    parser.add_argument(
        "--logging-frequency", type=int, default=10, help="Logging interval [steps]."
    )
    parser.add_argument(
        "--processing-unit",
        default="GPU",
        choices=["CPU", "GPU", "GPU_VULKAN", "VULKAN", "CUDA", "GPU_METAL", "METAL"],
        help="Compute backend. GPU selects Metal on macOS and CUDA/Vulkan elsewhere.",
    )
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="Allow a requested GPU backend to fall back to CPU instead of failing the case.",
    )
    parser.add_argument(
        "--device-memory-fraction",
        type=float,
        default=0.5,
        help="Fraction of GPU memory reserved by Taichi; lower values leave room for remesh transfers.",
    )
    parser.add_argument(
        "--blowup-check-frequency",
        type=int,
        default=10,
        help="Peak-circulation blow-up check interval [steps]; 0 disables the check.",
    )
    parser.add_argument(
        "--stabilization",
        choices=[
            "control",
            "les",
            "rvpm",
            "relax",
            "remesh",
            "projection",
            "split",
            "energy",
            "adaptive",
        ],
        default="adaptive",
        help=(
            "'control' is the true unstabilized baseline (no stage limiter, no "
            "energy-budget ADM). Every other choice, including 'les', additionally "
            "runs the common RK-stage safeguard and the dormant energy-budget "
            "escape filter — so 'les' is NOT unstabilized. Use 'control' as the "
            "reference for what the safeguards and each named stabilizer change."
        ),
    )
    parser.add_argument(
        "--disable-stage-safety",
        action="store_true",
        help=(
            "Disable the common conservative RK-stage safety limiter. "
            "Useful only for reproducing historical blow-ups."
        ),
    )
    parser.add_argument(
        "--epsilon-w",
        type=float,
        default=0.05,
        help="Dimensionless Widnall centreline-perturbation amplitude.",
    )
    parser.add_argument(
        "--perturbation-model",
        choices=["solenoidal", "legacy"],
        default="solenoidal",
        help="Use the divergence-free perturbation or reproduce the historical field.",
    )
    parser.add_argument(
        "--perturbation-modes",
        type=int,
        default=24,
        help="Number of azimuthal perturbation modes.",
    )
    parser.add_argument(
        "--parallel-strain-f",
        type=float,
        default=RVPM_DEFAULT_F,
        help=f"rVPM correction parameter f (FLOWVPM default: {RVPM_DEFAULT_F:g}).",
    )
    parser.add_argument(
        "--parallel-strain-g",
        type=float,
        default=RVPM_DEFAULT_G,
        help=f"rVPM correction parameter g (FLOWVPM default: {RVPM_DEFAULT_G:g}).",
    )
    parser.add_argument(
        "--relaxation-factor",
        type=float,
        default=0.01,
        help="Constant factor for conservative ADM residual relaxation.",
    )
    parser.add_argument(
        "--adaptive-parallel-increment",
        type=float,
        default=0.04,
        help="Maximum positive S_parallel*dt admitted by the adaptive limiter.",
    )
    parser.add_argument(
        "--adaptive-rotation-increment",
        type=float,
        default=0.12,
        help="Maximum strength-vector rotation admitted per adaptive RK stage.",
    )
    parser.add_argument(
        "--adaptive-budget-r-max",
        type=float,
        default=0.02,
        help="Maximum windowed ADM-relaxation factor for adaptive stabilization.",
    )
    parser.add_argument(
        "--remesh-frequency",
        type=int,
        default=100,
        help=(
            "Remeshing interval for remesh/projection variants. "
            "The longer interval avoids repeated M4 support dilation."
        ),
    )
    parser.add_argument(
        "--remesh-relative-threshold",
        type=float,
        default=0.005,
        help="Relative deposited-circulation cutoff for sparse remeshing.",
    )
    parser.add_argument(
        "--remesh-max-particles",
        type=int,
        default=30000,
        help="Maximum particles retained by a remesh/projection rebuild.",
    )
    parser.add_argument(
        "--remesh-max-particle-growth",
        type=float,
        default=1.5,
        help="Maximum rebuilt/previous particle-count ratio for one event.",
    )
    parser.add_argument(
        "--split-radius",
        type=float,
        default=0.16,
        help="Core-radius threshold for particle splitting.",
    )
    parser.add_argument(
        "--split-strength",
        type=float,
        default=0.5,
        help="Particle-circulation threshold for particle splitting [m3/s].",
    )
    parser.add_argument(
        "--viscous",
        choices=["cs", "gbd", "dvh"],
        default="cs",
        help="Viscous scheme.",
    )

    return parser


def build_viscous_config(scheme: str, particle_spacing: float) -> ViscousConfig:
    if scheme == "cs":
        return ViscousConfig.cs(
            viscosity=KINEMATIC_VISCOSITY,
            characteristic_distance=particle_spacing,
        )
    if scheme == "gbd":
        return ViscousConfig.gbd(
            h=particle_spacing,
            viscosity=KINEMATIC_VISCOSITY,
            threshold=REGEN_THRESHOLD,
            threshold_mode=REGEN_THRESHOLD_MODE,
            max_nodes=MAX_REGEN_NODES,
        )
    return ViscousConfig.dvh(
        h=particle_spacing,
        dvh_rd_ratio=3,
        viscosity=KINEMATIC_VISCOSITY,
        threshold=REGEN_THRESHOLD,
        threshold_mode=REGEN_THRESHOLD_MODE,
        max_nodes=MAX_REGEN_NODES,
    )


def build_stabilization_config(
    args: argparse.Namespace, particle_spacing: float
) -> StabilizationConfig:
    """Return one benchmark stabilizer plus the common integration safeguard."""
    if args.stabilization == "control":
        # True unstabilized baseline: no stage limiter, no energy-budget ADM,
        # no relaxation. Exposes the raw spurious-stretching instability so the
        # effect of the safeguards and every named stabilizer can be measured
        # against it. Returns before the common-safeguard block below.
        return StabilizationConfig.disabled()
    if args.stabilization == "les":
        stabilization = StabilizationConfig.disabled()
    elif args.stabilization == "rvpm":
        stabilization = StabilizationConfig.parallel_strain_relaxation(
            f=args.parallel_strain_f,
            g=args.parallel_strain_g,
        )
    elif args.stabilization == "relax":
        stabilization = StabilizationConfig.strength_relaxation(
            mode="blend",
            gate="constant",
            factor=args.relaxation_factor,
            conserve=True,
            constraint="both",
            deconv=1,
        )
    elif args.stabilization == "energy":
        stabilization = StabilizationConfig.energy_budget()
    elif args.stabilization == "adaptive":
        stabilization = StabilizationConfig.adaptive_rvpm(
            parallel_increment=args.adaptive_parallel_increment,
            rotation_increment=args.adaptive_rotation_increment,
            budget_r_max=args.adaptive_budget_r_max,
        )
    elif args.stabilization == "split":
        stabilization = StabilizationConfig.particle_splitting(
            radius=args.split_radius,
            max_strength=args.split_strength,
            weak_threshold_percent=0.5,
        )
    else:
        project = args.stabilization == "projection"
        stabilization = StabilizationConfig.conservative_remeshing(
            frequency=args.remesh_frequency,
            spacing=particle_spacing,
            relative_threshold=args.remesh_relative_threshold,
            absolute_threshold=0.0,
            conserve_impulse=True,
            conserve_energy=True,
            delta_correction=False,
            radius=2.0 * particle_spacing,
            preserve_radius_profile=True,
            max_particles=args.remesh_max_particles,
            max_particle_growth=args.remesh_max_particle_growth,
            project_solenoidal=project,
        )

    if not args.disable_stage_safety:
        stabilization.stretching_limiter_enabled = True
        stabilization.stretching_limiter_parallel_increment = args.adaptive_parallel_increment
        stabilization.stretching_limiter_rotation_increment = args.adaptive_rotation_increment
        stabilization.stretching_limiter_conserve = True
        stabilization.stretching_limiter_constraint = "both"
        stabilization.stretching_limiter_project_step_invariants = True
        stabilization.stretching_limiter_project_step_angular_impulse = True

        if args.stabilization in {"les", "rvpm", "relax", "remesh", "projection", "split"}:
            stabilization.energy_budget_enabled = True
            stabilization.energy_budget_frequency = 5
            stabilization.energy_budget_gain = 0.25
            stabilization.energy_budget_tolerance = 0.5
            stabilization.energy_budget_r_max = 0.05
            stabilization.energy_budget_r_seed = 0.001
            stabilization.energy_budget_smoothing = 0.25
            stabilization.energy_budget_max_log_change = 0.35
            stabilization.relaxation_enabled = True
            stabilization.relaxation_mode = "blend"
            stabilization.relaxation_deconv = 1
            stabilization.relaxation_gate = "constant"
            if args.stabilization != "relax":
                stabilization.relaxation_factor = 0.0
            stabilization.relaxation_conserve = True
            stabilization.relaxation_constraint = "both"

        if args.stabilization in {"remesh", "projection"}:
            stabilization.relaxation_deconv = 0
            stabilization.energy_budget_gain = 0.5
            stabilization.energy_budget_tolerance = 0.2
            stabilization.energy_budget_r_max = 0.3
            stabilization.energy_budget_r_seed = 0.002
            stabilization.energy_budget_smoothing = 0.4
            stabilization.energy_budget_max_log_change = 0.7

        if args.stabilization == "energy":
            stabilization.relaxation_deconv = 0
            stabilization.energy_budget_frequency = 2
            stabilization.energy_budget_gain = 0.5
            stabilization.energy_budget_tolerance = 0.1
            stabilization.energy_budget_r_max = 0.4
            stabilization.energy_budget_r_seed = 0.002
            stabilization.energy_budget_r_inject = 0.15
            stabilization.energy_budget_smoothing = 0.7
            stabilization.energy_budget_max_log_change = 0.7

    return stabilization


def make_surface_sampler(
    case_label: str, particle_spacing: float, output_dir: Path
) -> SurfaceSampler:
    if case_label == "collide":
        bounds = [-7.0, 7.0, -4.0, 4.0]
    else:
        bounds = [-0.5, 11.5, -2.0, 2.0]

    return SurfaceSampler(
        point=[0.0, 0.0, 0.0],
        normal=[0, 1, 0],
        bounds=bounds,
        spacing=particle_spacing,
        file_name="xz_slice",
        output_dir=str(output_dir / "samples"),
    )


def ring_centers_and_strengths(
    gamma1: float, gamma2: float
) -> tuple[list[list[float]], list[float]]:
    if gamma1 * gamma2 >= 0.0:
        return [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], [gamma1, gamma2]

    ring_separation = 2.5 * (2.0 * RING_RADIUS)
    return [[0.5 * ring_separation, 0.0, 0.0], [-0.5 * ring_separation, 0.0, 0.0]], [
        gamma1,
        gamma2,
    ]


def initialize_vortex_rings(
    solver: Solver,
    positions: np.ndarray,
    volumes: np.ndarray,
    radii: np.ndarray,
    gamma1: float,
    gamma2: float,
    epsilon_w: float,
    perturbation_model: str,
    perturbation_modes: int,
) -> None:
    centers, strengths = ring_centers_and_strengths(gamma1, gamma2)

    for group_index, (center, strength) in enumerate(zip(centers, strengths)):
        velocity, viscosity, circulation = VortexRingVPM(
            viscosity=KINEMATIC_VISCOSITY,
            ring_center=np.zeros(3),
            ring_radius=RING_RADIUS,
            ring_strength=strength,
            ring_thickness=CORE_RADIUS,
            avg_particle_radius=float(radii.mean()),
            positions=positions,
            volumes=volumes,
            epsilon_W=epsilon_w,
            max_modes=perturbation_modes,
            anti_diffuse_flag=True,
            perturbation_model=perturbation_model,
        )

        solver.add_vortex_particles(
            position=positions - np.asarray(center),
            velocity=velocity,
            circulation=circulation,
            radius=radii,
            volume=volumes,
            viscosity=viscosity,
            group_id=np.full(len(positions), group_index, dtype=np.int32),
        )
        solver.remove_weak_particles(percent=0.1, per_group=True)


def write_manifest(
    args: argparse.Namespace,
    case_label: str,
    output_dir: Path,
    actual_processing_unit: str,
) -> None:
    manifest = {
        "case": output_dir.name,
        "family": case_label,
        "model": "LES",
        "stabilization": args.stabilization,
        "advection_scheme": "RK3",
        "stretching_mode": "TRANSPOSED",
        "stretching_scheme": "RK3",
        "viscous_scheme": args.viscous,
        "processing_unit": actual_processing_unit,
        "processing_unit_requested": args.processing_unit,
        "device_memory_fraction": args.device_memory_fraction,
        "dt": args.dt,
        "num_steps": args.num_steps,
        "particle_spacing": args.particle_spacing,
        "epsilon_W": args.epsilon_w,
        "perturbation_model": args.perturbation_model,
        "perturbation_modes": args.perturbation_modes,
        "gamma1": args.gamma1,
        "gamma2": args.gamma2,
        "parallel_strain_f": args.parallel_strain_f,
        "parallel_strain_g": args.parallel_strain_g,
        "adaptive_parallel_increment": args.adaptive_parallel_increment,
        "adaptive_rotation_increment": args.adaptive_rotation_increment,
        "adaptive_budget_r_max": args.adaptive_budget_r_max,
        "remesh_frequency": args.remesh_frequency,
        "remesh_relative_threshold": args.remesh_relative_threshold,
        "remesh_preserve_radius_profile": (args.stabilization in {"remesh", "projection"}),
        "remesh_max_particles": args.remesh_max_particles,
        "remesh_max_particle_growth": args.remesh_max_particle_growth,
        "split_strength": args.split_strength,
        "stage_safety_enabled": not args.disable_stage_safety,
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def run_case(args: argparse.Namespace) -> None:
    case_label = "leapfrog" if args.gamma1 * args.gamma2 >= 0.0 else "collide"

    # ================================================
    # 1. Physical Parameters
    # ================================================
    gamma1 = args.gamma1
    gamma2 = args.gamma2

    # ================================================
    # 2. Numerical Parameters
    # ================================================
    time_step = args.dt
    num_steps = args.num_steps
    particle_spacing = args.particle_spacing

    if time_step <= 0.0:
        raise ValueError("--dt must be positive.")
    if num_steps < 0:
        raise ValueError("--num-steps must be non-negative.")
    if particle_spacing <= 0.0:
        raise ValueError("--particle-spacing must be positive.")
    if args.blowup_check_frequency < 0:
        raise ValueError("--blowup-check-frequency must be non-negative.")
    if not 0.1 <= args.device_memory_fraction <= 0.7:
        raise ValueError("--device-memory-fraction must be between 0.1 and 0.7.")
    if args.remesh_frequency <= 0:
        raise ValueError("--remesh-frequency must be positive.")
    if not 0.0 < args.remesh_relative_threshold < 1.0:
        raise ValueError("--remesh-relative-threshold must be in (0, 1).")
    if args.remesh_max_particles <= 0:
        raise ValueError("--remesh-max-particles must be positive.")
    if args.remesh_max_particle_growth < 1.0:
        raise ValueError("--remesh-max-particle-growth must be at least one.")
    if args.split_radius <= 0.0:
        raise ValueError("--split-radius must be positive.")
    if args.split_strength <= 0.0:
        raise ValueError("--split-strength must be positive.")
    if args.epsilon_w < 0.0:
        raise ValueError("--epsilon-w must be non-negative.")
    if args.perturbation_modes < 1:
        raise ValueError("--perturbation-modes must be at least 1.")
    if args.adaptive_parallel_increment <= 0.0:
        raise ValueError("--adaptive-parallel-increment must be positive.")
    if args.adaptive_rotation_increment <= 0.0:
        raise ValueError("--adaptive-rotation-increment must be positive.")
    if not 0.0 < args.adaptive_budget_r_max <= 1.0:
        raise ValueError("--adaptive-budget-r-max must be in (0, 1].")

    # ================================================
    # 3. Create Initial Particle Distribution
    # ================================================
    domain_bounds = [-0.15, 0.15, -1.5, 1.5, -1.5, 1.5]
    positions, volumes, radii = ParticleDistributor.hexagonal_distribution(
        domain_bounds,
        particle_spacing,
    )

    # ================================================
    # 4. Configure VPM Solver
    # ================================================
    output_dir = Path(args.output_root) / args.name
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty result directory: {output_dir}. "
            "Choose a new --name or --output-root."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    sampler = make_surface_sampler(case_label, particle_spacing, output_dir)

    advection = AdvectionConfig(scheme="RK3")

    turbulence = TurbulenceConfig.les_smagorinsky(cs=0.16, ce=1.048)

    stretching = StretchingConfig.transposed(scheme="RK3")
    stabilization = build_stabilization_config(args, particle_spacing)

    velocity = VelocityConfig.treecode(
        theta=0.35,
        sort_particle_targets=True,
        traversal_block_dim=128,
    )

    viscous = build_viscous_config(args.viscous, particle_spacing)

    solver_config = SolverConfig(
        time_step_size=time_step,
        processing_unit=args.processing_unit,
        device_memory_fraction=args.device_memory_fraction,
        advection=advection,
        turbulence=turbulence,
        stretching=stretching,
        stabilization=stabilization,
        velocity=velocity,
        viscous=viscous,
        samplers=[(sampler, "xz_slice")],
        sampler_output_format="legacy",
        backup_file_name=args.name,
        backup_directory=str(output_dir),
        solution_name=str(output_dir),
        backup_frequency=args.backup_frequency,
        logging_frequency=args.logging_frequency,
        timing_frequency=40,
    )
    solver = Solver(config=solver_config)
    requested_gpu = args.processing_unit.upper() != "CPU"
    if requested_gpu and solver.processing_unit == "CPU" and not args.allow_cpu_fallback:
        raise RuntimeError(
            f"Requested GPU backend {args.processing_unit!r}, but Taichi initialized CPU. "
            "Use --processing-unit CPU explicitly or --allow-cpu-fallback."
        )
    write_manifest(args, case_label, output_dir, solver.processing_unit)
    # Plot scripts read flow diagnostics from the log; skip duplicate CSV output here.
    solver._export_flow_integrals_csv = lambda: None

    # ================================================
    # 5. Initialize Two Vortex Rings
    # ================================================
    initialize_vortex_rings(
        solver,
        positions,
        volumes,
        radii,
        gamma1,
        gamma2,
        args.epsilon_w,
        args.perturbation_model,
        args.perturbation_modes,
    )
    solver.info()

    # ================================================
    # 6. Run Simulation
    # ================================================
    blowup_threshold = np.inf
    if args.blowup_check_frequency > 0:
        initial_max_norm = float(np.linalg.norm(solver.particles_circulation, axis=1).max())
        blowup_threshold = max(50.0 * initial_max_norm, 0.1)
        print(
            "BLOWUP CHECK "
            f"step=0 time={solver.flow_time:.6e} max_gamma={initial_max_norm:.6e} "
            f"threshold={blowup_threshold:.6e} "
            f"n_particles={solver.particles.number_of_particles}",
            flush=True,
        )

    for step in range(num_steps):
        solver.update_state()

        check_due = (
            args.blowup_check_frequency > 0 and (step + 1) % args.blowup_check_frequency == 0
        )
        if not check_due:
            continue

        max_norm = float(np.linalg.norm(solver.particles_circulation, axis=1).max())
        print(
            "BLOWUP CHECK "
            f"step={step + 1} time={solver.flow_time:.6e} max_gamma={max_norm:.6e} "
            f"threshold={blowup_threshold:.6e} n_particles={solver.particles.number_of_particles}",
            flush=True,
        )

        if not np.isfinite(max_norm) or max_norm > blowup_threshold:
            print(
                f"\n*** BLOWUP DETECTED at step {step + 1} "
                f"(t={solver.flow_time:.2f} s): max|Gamma|={max_norm:.4f} "
                f"> {blowup_threshold:.4f} ***",
                flush=True,
            )
            solver.save_state(str(output_dir / "pre_blowup"))
            break
    else:
        print(f"Simulation completed {num_steps} steps without blowup.")


def main() -> int:
    args = build_arg_parser().parse_args()
    run_case(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
