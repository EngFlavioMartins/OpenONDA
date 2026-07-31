#!/usr/bin/env python3
"""Vortex-ring interaction controls for stabilization development.

Each interaction family exposes two untreated controls and one candidate:

* ``baseline``: molecular-viscosity DNS with the fractional RK3 core;
* ``les``: the same numerical core plus Smagorinsky LES;
* ``les_stabilized``: LES plus conservative filament subdivision and
  reference-restoring constrained Winckelmans divergence relaxation.

The two controls run until the solution actually falls apart.  The stabilized
candidate also stops on any declared conservation or field-transfer gate.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from openonda.vpm import ParticleDistributor, Solver, VPMSetup
from openonda.vpm import (
    AdvectionConfig,
    BackupSystem,
    DivergenceRelaxationConfig,
    DivergenceRelaxationError,
    FilamentRefinementConfig,
    FilamentRefinementError,
    StabilizationConfig,
    StretchingConfig,
    TurbulenceConfig,
    VelocityConfig,
    ViscousConfig,
)
from openonda.vpm import VortexRingVPM
from openonda.vpm import SurfaceSampler

# The configuration module keeps library tracebacks terse. A long batch runner
# needs the full traceback in its per-case log so infrastructure failures are
# diagnosable instead of appearing as a single unexplained exception line.
sys.tracebacklimit = None


GAMMA_REF = np.pi
RING_RADIUS = 1.0
CORE_RADIUS = 0.1
REYNOLDS_GAMMA = 3000.0
KINEMATIC_VISCOSITY = GAMMA_REF / REYNOLDS_GAMMA
PAPER_SPACING = 0.2 * CORE_RADIUS
PAPER_DT = 20.0 * PAPER_SPACING**2 / GAMMA_REF
CONTROL_LES_CS = 0.16


class NonphysicalState(RuntimeError):
    """The particle field went non-finite or its strengths ran away."""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Baseline, LES, and physics-gated vortex-ring runs"
    )
    parser.add_argument("--gamma1", type=float, default=GAMMA_REF)
    parser.add_argument("--gamma2", type=float, default=GAMMA_REF)
    parser.add_argument("--name", default="leapfrog_les")
    parser.add_argument(
        "--method",
        choices=["baseline", "les", "les_stabilized"],
        default="les",
    )
    parser.add_argument("--dt", type=float, default=PAPER_DT, help="Macro time step [s].")
    parser.add_argument("--num-steps", type=int, default=2800)
    parser.add_argument("--restart-from", type=Path)
    parser.add_argument("--particle-spacing", type=float, default=PAPER_SPACING)
    parser.add_argument("--output-root", default="solution")
    parser.add_argument("--backup-frequency", type=int, default=100)
    parser.add_argument("--logging-frequency", type=int, default=20)
    parser.add_argument(
        "--guard-frequency",
        type=int,
        default=1,
        help="Check for a runaway solution every N steps.",
    )
    parser.add_argument(
        "--processing-unit",
        default="AUTO",
        choices=["AUTO", "CPU", "VULKAN", "CUDA", "METAL"],
    )
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--device-memory-fraction", type=float, default=0.5)
    parser.add_argument("--epsilon-w", type=float, default=0.025)
    parser.add_argument("--perturbation-modes", type=int, default=24)
    parser.add_argument(
        "--allow-underresolved",
        action="store_true",
        help="Permit h/a0 or dt above the reference limits for convergence studies.",
    )
    parser.add_argument(
        "--blowup-factor",
        type=float,
        default=50.0,
        help="Stop a non-physical control after max|Gamma| exceeds this multiple of its initial value.",
    )
    parser.add_argument("--refinement-frequency", type=int, default=1)
    parser.add_argument("--refinement-strength-factor", type=float, default=2.0)
    parser.add_argument("--refinement-offset-fraction", type=float, default=0.25)
    parser.add_argument("--refinement-max-particles", type=int, default=200_000)
    parser.add_argument("--relaxation-frequency", type=int, default=10)
    parser.add_argument("--relaxation-start-step", type=int, default=50)
    parser.add_argument(
        "--relaxation-grid-spacing",
        type=float,
        default=None,
        help="Projection-grid spacing [m]; default is 1.5 times particle spacing.",
    )
    parser.add_argument("--relaxation-regularization", type=float, default=0.1)
    parser.add_argument("--relaxation-max-grid-nodes", type=int, default=12_000_000)
    return parser


def ring_centers_and_strengths(
    gamma1: float, gamma2: float
) -> tuple[list[list[float]], list[float]]:
    if gamma1 * gamma2 >= 0.0:
        # Centreline separation x/R0 = 1 in the reference leapfrog case.
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
    perturbation_modes: int,
    diffusivity_constant: float,
) -> None:
    """Create two Gaussian-core rings with a solenoidal centreline perturbation."""
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
            diffusivity_constant=diffusivity_constant,
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
        # Match the reference setup's 5% initial Gaussian-tail cutoff.  No
        # particles are removed after time integration starts.
        solver.remove_weak_particles(percent=5.0, per_group=True)


def make_surface_sampler(
    case_label: str, particle_spacing: float, output_dir: Path
) -> SurfaceSampler:
    bounds = [-7.0, 7.0, -4.0, 4.0] if case_label == "collide" else [-0.5, 11.5, -2.0, 2.0]
    return SurfaceSampler(
        point=[0.0, 0.0, 0.0],
        normal=[0, 1, 0],
        bounds=bounds,
        spacing=particle_spacing,
        file_name="xz_slice",
        output_dir=str(output_dir / "samples"),
    )


def validate_resolution(args: argparse.Namespace) -> None:
    if args.dt <= 0.0 or args.particle_spacing <= 0.0:
        raise ValueError("--dt and --particle-spacing must be positive.")
    if args.num_steps < 0:
        raise ValueError("--num-steps must be non-negative.")
    if args.guard_frequency < 1:
        raise ValueError("--guard-frequency must be at least one.")
    if args.perturbation_modes < 1 or args.epsilon_w < 0.0:
        raise ValueError("Perturbation modes must be positive and epsilon non-negative.")
    if not 0.1 <= args.device_memory_fraction <= 0.7:
        raise ValueError("--device-memory-fraction must be between 0.1 and 0.7.")
    if args.blowup_factor <= 1.0:
        raise ValueError("--blowup-factor must be greater than one.")
    if args.refinement_frequency < 1 or args.relaxation_frequency < 1:
        raise ValueError("stabilization frequencies must be at least one.")
    if args.relaxation_start_step < 0:
        raise ValueError("--relaxation-start-step must be non-negative.")
    if args.refinement_max_particles < 1:
        raise ValueError("--refinement-max-particles must be positive.")
    if args.relaxation_grid_spacing is not None and args.relaxation_grid_spacing <= 0.0:
        raise ValueError("--relaxation-grid-spacing must be positive.")
    if args.relaxation_regularization <= 0.0:
        raise ValueError("--relaxation-regularization must be positive.")
    if args.relaxation_max_grid_nodes < 1:
        raise ValueError("--relaxation-max-grid-nodes must be positive.")
    if args.allow_underresolved:
        return
    spacing_ratio = args.particle_spacing / CORE_RADIUS
    dt_limit = (
        20.0
        * args.particle_spacing**2
        / max(abs(args.gamma1), abs(args.gamma2), np.finfo(float).tiny)
    )
    violations = []
    if spacing_ratio > 0.2 * (1.0 + 1.0e-12):
        violations.append(f"h/a0={spacing_ratio:.3f} > 0.2")
    if args.dt > dt_limit * (1.0 + 1.0e-12):
        violations.append(f"dt={args.dt:.4e} > 20h^2/Gamma={dt_limit:.4e}")
    if violations:
        raise ValueError(
            "Stabilized VPM refused an under-resolved setup: "
            + "; ".join(violations)
            + ". Refine h/dt, or pass --allow-underresolved only for a convergence study."
        )


def build_solver_config(args: argparse.Namespace, output_dir: Path, case_label: str) -> VPMSetup:
    sampler = make_surface_sampler(case_label, args.particle_spacing, output_dir)
    viscous = ViscousConfig.cs(
        viscosity=KINEMATIC_VISCOSITY,
        characteristic_distance=args.particle_spacing,
    )
    common = dict(
        time_step_size=args.dt,
        processing_unit=args.processing_unit,
        device_memory_fraction=args.device_memory_fraction,
        stabilization=StabilizationConfig.disabled(),
        viscous=viscous,
        samplers=[(sampler, "xz_slice")],
        backup_file_name=args.name,
        backup_directory=str(output_dir),
        backup_frequency=args.backup_frequency,
        logging_frequency=args.logging_frequency,
        timing_frequency=100,
        export_flow_integrals=True,
    )
    stabilized = args.method == "les_stabilized"
    relaxation_spacing = (
        args.relaxation_grid_spacing
        if args.relaxation_grid_spacing is not None
        else 1.5 * args.particle_spacing
    )
    return VPMSetup(
        **common,
        time_integration="FRACTIONAL",
        advection=AdvectionConfig(scheme="RK3"),
        stretching=StretchingConfig.transposed(scheme="RK3"),
        turbulence=(
            TurbulenceConfig.les_smagorinsky(cs=CONTROL_LES_CS, ce=1.048)
            if args.method in {"les", "les_stabilized"}
            else TurbulenceConfig.dns()
        ),
        filament_refinement=(
            FilamentRefinementConfig.adaptive(
                frequency=args.refinement_frequency,
                max_strength_factor=args.refinement_strength_factor,
                offset_fraction=args.refinement_offset_fraction,
                max_particles=args.refinement_max_particles,
            )
            if stabilized
            else FilamentRefinementConfig.disabled()
        ),
        divergence_relaxation=(
            DivergenceRelaxationConfig.constrained(
                frequency=args.relaxation_frequency,
                start_step=args.relaxation_start_step,
                grid_spacing=relaxation_spacing,
                regularization=args.relaxation_regularization,
                max_grid_nodes=args.relaxation_max_grid_nodes,
                max_correction_norm=2e-2,
                max_residual_ratio=0.9,
                max_direct_divergence_ratio=0.98,
                energy_tolerance=1e-6,
                enstrophy_tolerance=1.5e-4,
                helicity_tolerance=1e-4,
                variation_tolerance=1e-3,
            )
            if stabilized
            else DivergenceRelaxationConfig.disabled()
        ),
        velocity=VelocityConfig.treecode(
            theta=0.35,
            sort_particle_targets=True,
            traversal_block_dim=128,
        ),
        particles_kernel="GAUSSIAN",
    )


def enforce_numerical_bound(
    solver: Solver,
    initial_max_strength: float,
    blowup_factor: float,
) -> float:
    """Stop only when the strengths have actually run away or gone non-finite."""
    circulation = solver.particles_circulation
    magnitudes = np.linalg.norm(circulation, axis=1)
    maximum = float(magnitudes.max()) if len(magnitudes) else 0.0
    threshold = blowup_factor * max(initial_max_strength, np.finfo(float).tiny)
    print(
        "BLOWUP CHECK "
        f"step={solver.time_step} time={solver.flow_time:.6e} "
        f"max_gamma={maximum:.8e} threshold={threshold:.8e} "
        f"n_particles={len(magnitudes)}",
        flush=True,
    )
    if not np.isfinite(circulation).all():
        raise NonphysicalState(
            f"NONPHYSICAL at step {solver.time_step} (t={solver.flow_time:.4f}): "
            "particle strengths went to NaN or infinity."
        )
    if maximum > threshold:
        raise NonphysicalState(
            f"NONPHYSICAL at step {solver.time_step} (t={solver.flow_time:.4f}): "
            f"peak particle strength ran away to {maximum:.4e}, which is "
            f"{maximum / max(initial_max_strength, np.finfo(float).tiny):.0f}x its "
            f"starting value of {initial_max_strength:.4e}."
        )
    return maximum


def export_diagnostic_snapshot(solver: Solver) -> None:
    """Export the current state once when periodic logging did not just do so."""
    if (
        solver.logging_frequency > 0
        and solver.time_step > 0
        and solver.time_step % solver.logging_frequency == 0
    ):
        return
    solver._update_all_flow_integrals()
    solver._export_flow_integrals_csv()


def write_manifest(
    args: argparse.Namespace,
    case_label: str,
    output_dir: Path,
    solver: Solver,
    *,
    status: str,
    termination_reason: str | None = None,
) -> None:
    cfg = solver.config
    manifest = {
        "case": output_dir.name,
        "family": case_label,
        "method": args.method,
        "status": status,
        "completed_steps": solver.time_step,
        "requested_steps": args.num_steps,
        "termination_reason": termination_reason,
        "time_integration": cfg.time_integration,
        "model": cfg.turbulence.flow_model,
        "kernel": cfg.particles_kernel,
        "velocity_method": cfg.velocity.method,
        "advection_scheme": cfg.advection.scheme,
        "stretching_mode": cfg.stretching.mode,
        "stretching_scheme": cfg.stretching.scheme,
        "viscous_scheme": cfg.viscous.scheme,
        "filament_refinement_frequency": cfg.filament_refinement.frequency,
        "filament_refinement_max_strength_factor": (cfg.filament_refinement.max_strength_factor),
        "filament_refinement_offset_fraction": cfg.filament_refinement.offset_fraction,
        "filament_refinement_max_particles": cfg.filament_refinement.max_particles,
        "filament_refinement_energy_injection_tolerance": (
            cfg.filament_refinement.energy_injection_tolerance
        ),
        "filament_refinement_energy_dissipation_tolerance": (
            cfg.filament_refinement.energy_dissipation_tolerance
        ),
        "filament_refinement_enstrophy_transfer_tolerance": (
            cfg.filament_refinement.enstrophy_transfer_tolerance
        ),
        "filament_refinement_helicity_transfer_tolerance": (
            cfg.filament_refinement.helicity_transfer_tolerance
        ),
        "filament_refinement_cumulative_energy_tolerance": (
            cfg.filament_refinement.cumulative_energy_tolerance
        ),
        "filament_refinement_cumulative_enstrophy_tolerance": (
            cfg.filament_refinement.cumulative_enstrophy_tolerance
        ),
        "filament_refinement_cumulative_helicity_tolerance": (
            cfg.filament_refinement.cumulative_helicity_tolerance
        ),
        "filament_refinement_cumulative_moment_tolerance": (
            cfg.filament_refinement.cumulative_moment_tolerance
        ),
        "divergence_relaxation_frequency": cfg.divergence_relaxation.frequency,
        "divergence_relaxation_start_step": cfg.divergence_relaxation.start_step,
        "divergence_relaxation_grid_spacing": cfg.divergence_relaxation.grid_spacing,
        "divergence_relaxation_regularization": (cfg.divergence_relaxation.regularization),
        "divergence_relaxation_max_grid_nodes": (cfg.divergence_relaxation.max_grid_nodes),
        "divergence_relaxation_max_correction_norm": (
            cfg.divergence_relaxation.max_correction_norm
        ),
        "divergence_relaxation_max_residual_ratio": (cfg.divergence_relaxation.max_residual_ratio),
        "divergence_relaxation_max_direct_divergence_ratio": (
            cfg.divergence_relaxation.max_direct_divergence_ratio
        ),
        "divergence_relaxation_energy_tolerance": (cfg.divergence_relaxation.energy_tolerance),
        "divergence_relaxation_enstrophy_tolerance": (
            cfg.divergence_relaxation.enstrophy_tolerance
        ),
        "divergence_relaxation_helicity_tolerance": (cfg.divergence_relaxation.helicity_tolerance),
        "divergence_relaxation_variation_tolerance": (
            cfg.divergence_relaxation.variation_tolerance
        ),
        "divergence_relaxation_spectral_convergence_fraction": (
            cfg.divergence_relaxation.spectral_convergence_fraction
        ),
        "divergence_relaxation_cumulative_energy_tolerance": (
            cfg.divergence_relaxation.cumulative_energy_tolerance
        ),
        "divergence_relaxation_cumulative_enstrophy_tolerance": (
            cfg.divergence_relaxation.cumulative_enstrophy_tolerance
        ),
        "divergence_relaxation_cumulative_helicity_tolerance": (
            cfg.divergence_relaxation.cumulative_helicity_tolerance
        ),
        "divergence_relaxation_cumulative_variation_tolerance": (
            cfg.divergence_relaxation.cumulative_variation_tolerance
        ),
        "divergence_relaxation_cumulative_moment_tolerance": (
            cfg.divergence_relaxation.cumulative_moment_tolerance
        ),
        "smagorinsky_cs": cfg.turbulence.cs if cfg.turbulence.flow_model == "LES" else None,
        "molecular_viscosity": cfg.viscous.viscosity,
        "characteristic_distance": cfg.viscous.characteristic_distance,
        "field_modification": (
            "conservative axial split plus reference-restoring constrained relaxation"
            if args.method == "les_stabilized"
            else "none"
        ),
        "retention_bounds": cfg.stabilization.remove_particles_by_bounds,
        "diagnostics": (
            "conservation, transfer, and divergence reduction are hard gated"
            if args.method == "les_stabilized"
            else "conservation and resolution recorded per logging step; not gated"
        ),
        "processing_unit": solver.processing_unit,
        "processing_unit_requested": args.processing_unit,
        "dt": args.dt,
        "dt_limit_20h2_over_gamma": 20.0 * args.particle_spacing**2 / GAMMA_REF,
        "num_steps": args.num_steps,
        "particle_spacing": args.particle_spacing,
        "h_over_a0": args.particle_spacing / CORE_RADIUS,
        "epsilon_W": args.epsilon_w,
        "perturbation_modes": args.perturbation_modes,
        "gamma1": args.gamma1,
        "gamma2": args.gamma2,
        "guard_frequency": args.guard_frequency,
        "blowup_factor": args.blowup_factor,
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def run_case(args: argparse.Namespace) -> str:
    validate_resolution(args)
    case_label = "leapfrog" if args.gamma1 * args.gamma2 >= 0.0 else "collide"
    domain_bounds = [-0.15, 0.15, -1.5, 1.5, -1.5, 1.5]
    positions, volumes, radii = ParticleDistributor.hexagonal_distribution(
        domain_bounds, args.particle_spacing
    )

    output_dir = Path(args.output_root) / args.name
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty result directory: {output_dir}. "
            "Choose a new --name or --output-root."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    solver = Solver(setup=build_solver_config(args, output_dir, case_label))
    requested_gpu = args.processing_unit.upper() != "CPU"
    if requested_gpu and solver.processing_unit == "CPU" and not args.allow_cpu_fallback:
        raise RuntimeError(
            f"Requested backend {args.processing_unit!r}, but Taichi initialized CPU. "
            "Use --processing-unit CPU explicitly or --allow-cpu-fallback."
        )
    initialize_vortex_rings(
        solver,
        positions,
        volumes,
        radii,
        args.gamma1,
        args.gamma2,
        args.epsilon_w,
        args.perturbation_modes,
        4.0,
    )
    initial_max_strength = float(np.linalg.norm(solver.particles_circulation, axis=1).max())
    solver._capture_filament_refinement_reference()
    if args.restart_from is not None:
        BackupSystem.load_numerical_state(solver, args.restart_from)
        if solver.time_step >= args.num_steps:
            raise ValueError(
                f"checkpoint step {solver.time_step} must be below target step {args.num_steps}"
            )
    solver.info()

    solver._update_all_flow_integrals()
    solver._export_flow_integrals_csv()
    write_manifest(args, case_label, output_dir, solver, status="running")

    try:
        for step in range(solver.time_step, args.num_steps):
            solver.update_state()
            if (step + 1) % args.guard_frequency == 0:
                enforce_numerical_bound(solver, initial_max_strength, args.blowup_factor)
    except (DivergenceRelaxationError, FilamentRefinementError) as error:
        export_diagnostic_snapshot(solver)
        solver.save_state(str(output_dir / f"vpm_{args.name}_rejected_state"))
        write_manifest(
            args,
            case_label,
            output_dir,
            solver,
            status="rejected",
            termination_reason=str(error),
        )
        print(f"\nPHYSICS GATE REJECTED: {error}", flush=True)
        print(
            f"Ran {solver.time_step} of {args.num_steps} steps before rejection. "
            f"State saved to vpm_{args.name}_rejected_state.",
            flush=True,
        )
        return "rejected"
    except NonphysicalState as error:
        export_diagnostic_snapshot(solver)
        solver.save_state(str(output_dir / f"vpm_{args.name}_nonphysical_state"))
        write_manifest(
            args,
            case_label,
            output_dir,
            solver,
            status="terminated_nonphysical",
            termination_reason=str(error),
        )
        print(f"\n{error}", flush=True)
        print(
            f"Ran {solver.time_step} of {args.num_steps} steps before the "
            "solution became nonphysical. State saved to "
            f"vpm_{args.name}_nonphysical_state.",
            flush=True,
        )
        return "terminated_nonphysical"

    export_diagnostic_snapshot(solver)
    write_manifest(args, case_label, output_dir, solver, status="completed")
    print(f"Finished all {args.num_steps} steps.", flush=True)
    return "completed"


def main() -> int:
    run_case(build_arg_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
