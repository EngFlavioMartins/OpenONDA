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
MAX_REGEN_NODES = 250_000


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Vortex-ring LES stabilizer benchmark")

    parser.add_argument("--gamma1", type=float, default=GAMMA_REF, help="Ring 1 circulation [m2/s].")
    parser.add_argument("--gamma2", type=float, default=GAMMA_REF, help="Ring 2 circulation [m2/s].")
    parser.add_argument("--name", default="leapfrog_les", help="Output sub-directory name.")
    parser.add_argument("--dt", type=float, default=2.0e-2, help="Time-step size [s].")
    parser.add_argument("--num-steps", type=int, default=450, help="Number of time steps.")
    parser.add_argument("--particle-spacing", type=float, default=0.030, help="Particle spacing [m].")
    parser.add_argument("--output-root", default="solution", help="Parent output directory.")
    parser.add_argument("--backup-frequency", type=int, default=20, help="Backup interval [steps].")
    parser.add_argument("--logging-frequency", type=int, default=10, help="Logging interval [steps].")
    parser.add_argument(
        "--processing-unit",
        default="GPU",
        choices=["CPU", "GPU", "GPU_VULKAN", "VULKAN", "CUDA", "GPU_METAL", "METAL"],
        help="Compute backend. Default GPU selects CUDA on NVIDIA systems and Vulkan otherwise.",
    )
    parser.add_argument(
        "--device-memory-fraction",
        type=float,
        default=0.4,
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
        choices=["les", "rvpm", "relax", "remesh", "projection", "split", "energy"],
        default="les",
        help="Stabilization variant. 'les' is the unstabilized LES baseline.",
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
        default=0.3,
        help="Constant factor for strength relaxation.",
    )
    parser.add_argument(
        "--remesh-frequency",
        type=int,
        default=20,
        help="Remeshing interval for remesh/projection variants.",
    )
    parser.add_argument(
        "--split-radius",
        type=float,
        default=0.16,
        help="Core-radius threshold for particle splitting.",
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


def build_stabilization_config(args: argparse.Namespace, particle_spacing: float) -> StabilizationConfig:
    """Return exactly one benchmark stabilizer for the requested variant."""
    if args.stabilization == "les":
        return StabilizationConfig.disabled()
    if args.stabilization == "rvpm":
        return StabilizationConfig.parallel_strain_relaxation(
            f=args.parallel_strain_f,
            g=args.parallel_strain_g,
        )
    if args.stabilization == "relax":
        return StabilizationConfig.strength_relaxation(
            mode="pedrizzetti",
            gate="constant",
            factor=args.relaxation_factor,
            conserve=True,
            constraint="both",
        )
    if args.stabilization == "energy":
        return StabilizationConfig.energy_budget()
    if args.stabilization == "split":
        return StabilizationConfig.particle_splitting(
            radius=args.split_radius,
            weak_threshold_percent=0.5,
        )

    project = args.stabilization == "projection"
    return StabilizationConfig.conservative_remeshing(
        frequency=args.remesh_frequency,
        spacing=particle_spacing,
        relative_threshold=0.01,
        absolute_threshold=1.0e-8,
        conserve_impulse=True,
        delta_correction=False,
        radius=particle_spacing,
        project_solenoidal=project,
    )


def make_surface_sampler(case_label: str, particle_spacing: float, output_dir: Path) -> SurfaceSampler:
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


def ring_centers_and_strengths(gamma1: float, gamma2: float) -> tuple[list[list[float]], list[float]]:
    if gamma1 * gamma2 >= 0.0:
        return [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], [gamma1, gamma2]

    ring_separation = 4.0 * (2.0 * RING_RADIUS)
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
            epsilon_W=0.05,
            anti_diffuse_flag=True,
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


def write_manifest(args: argparse.Namespace, case_label: str, output_dir: Path) -> None:
    manifest = {
        "case": output_dir.name,
        "family": case_label,
        "model": "LES",
        "stabilization": args.stabilization,
        "advection_scheme": "RK3",
        "stretching_mode": "TRANSPOSED",
        "stretching_scheme": "RK3",
        "viscous_scheme": args.viscous,
        "processing_unit": args.processing_unit,
        "device_memory_fraction": args.device_memory_fraction,
        "dt": args.dt,
        "num_steps": args.num_steps,
        "particle_spacing": args.particle_spacing,
        "gamma1": args.gamma1,
        "gamma2": args.gamma2,
        "parallel_strain_f": args.parallel_strain_f,
        "parallel_strain_g": args.parallel_strain_g,
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
    if args.split_radius <= 0.0:
        raise ValueError("--split-radius must be positive.")

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

    velocity = VelocityConfig.treecode(theta=0.3)

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
        backup_file_name=args.name,
        backup_directory=str(output_dir),
        solution_name=str(output_dir),
        backup_frequency=args.backup_frequency,
        logging_frequency=args.logging_frequency,
        timing_frequency=40,
    )
    write_manifest(args, case_label, output_dir)

    solver = Solver(config=solver_config)
    # Plot scripts read flow diagnostics from the log; skip duplicate CSV output here.
    solver._export_flow_integrals_csv = lambda: None

    # ================================================
    # 5. Initialize Two Vortex Rings
    # ================================================
    initialize_vortex_rings(solver, positions, volumes, radii, gamma1, gamma2)
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
            args.blowup_check_frequency > 0
            and (step + 1) % args.blowup_check_frequency == 0
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
