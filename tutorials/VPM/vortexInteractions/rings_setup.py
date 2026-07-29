#!/usr/bin/env python3
"""Six-case vortex-ring interaction comparison.

Each interaction family is run with three deliberately distinct methods:

* ``baseline``: molecular-viscosity DNS with the legacy fractional RK3 core;
* ``les``: the same legacy core plus Smagorinsky LES;
* ``les_stabilized``: numerically identical to ``les`` -- same fractional RK3
  core, Gaussian kernel and treecode -- plus a coarse-grid Smagorinsky
  coefficient and the enstrophy envelope, so the comparison isolates the
  envelope rather than a pile of numerical differences.

Every case runs to the end unless the solution actually falls apart, and then
it says so.  Conservation and resolution are recorded every logging step into
samples/flow_integrals.csv for the figures; they are diagnostics, never gates.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from source.solvers.VPM import ParticleDistributor, Solver, VPMSetup
from source.solvers.VPM.config.types import (
    AdvectionConfig,
    EnvelopeConfig,
    StabilizationConfig,
    StretchingConfig,
    TurbulenceConfig,
    VelocityConfig,
    ViscousConfig,
)
from source.solvers.VPM.utils import VortexRingVPM
from source.solvers.VPM.utils.field_samplers import SurfaceSampler

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
STABILIZED_METHOD = "les_stabilized"
CONTROL_LES_CS = 0.16
STABILIZED_LES_CS = 0.20


class SimulationCrashed(RuntimeError):
    """The particle field went non-finite or its strengths ran away."""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Baseline, LES, and stabilized-LES vortex rings")
    parser.add_argument("--gamma1", type=float, default=GAMMA_REF)
    parser.add_argument("--gamma2", type=float, default=GAMMA_REF)
    parser.add_argument("--name", default="leapfrog_les_stabilized")
    parser.add_argument(
        "--method",
        choices=["baseline", "les", STABILIZED_METHOD],
        default=STABILIZED_METHOD,
    )
    parser.add_argument("--dt", type=float, default=PAPER_DT, help="Macro time step [s].")
    parser.add_argument("--num-steps", type=int, default=2800)
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
        help="Permit h/a0 or dt above the stabilized-method limits for convergence studies.",
    )
    parser.add_argument(
        "--rho-max",
        type=float,
        default=2.0,
        help=(
            "Largest credible Z_Delta/Z_2Delta before fine-scale growth counts as "
            "anomalous. Calibrate with assets/calibrate_envelope.py; the shipped "
            "default is a placeholder, not a measurement."
        ),
    )
    parser.add_argument(
        "--envelope-growth",
        type=float,
        default=1.0,
        help="Coarse-scale exponential growth allowance b_L [1/s] (calibrated).",
    )
    parser.add_argument(
        "--r-loc-max",
        type=float,
        default=15.0,
        help=(
            "Local barrier: correct a particle whose |Gamma|/sigma^3 exceeds this "
            "multiple of its 20-neighbour median. This is the criterion that "
            "actually fires; the global enstrophy bound is an L2 certificate and "
            "cannot see a sparse runaway."
        ),
    )
    parser.add_argument(
        "--envelope-kappa",
        type=float,
        default=1.0,
        help="Barrier relaxation rate [1/s].",
    )
    parser.add_argument(
        "--omega-hard",
        type=float,
        default=None,
        help=(
            "Hard ceiling on max |Gamma_p|/sigma_p^3. Reaching it stops the run as "
            "under-resolved rather than dissipating it into looking fine."
        ),
    )
    parser.add_argument(
        "--blowup-factor",
        type=float,
        default=50.0,
        help="Stop a non-physical control after max|Gamma| exceeds this multiple of its initial value.",
    )
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
    if args.rho_max <= 0.0 or args.envelope_growth < 0.0 or args.envelope_kappa <= 0.0:
        raise ValueError("--rho-max and --envelope-kappa must be positive, growth >= 0.")

    if args.method != STABILIZED_METHOD or args.allow_underresolved:
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
    common = dict(
        time_step_size=args.dt,
        processing_unit=args.processing_unit,
        device_memory_fraction=args.device_memory_fraction,
        stabilization=StabilizationConfig.disabled(),
        viscous=ViscousConfig.cs(
            viscosity=KINEMATIC_VISCOSITY,
            characteristic_distance=args.particle_spacing,
        ),
        samplers=[(sampler, "xz_slice")],
        backup_file_name=args.name,
        backup_directory=str(output_dir),
        backup_frequency=args.backup_frequency,
        logging_frequency=args.logging_frequency,
        timing_frequency=100,
        export_flow_integrals=True,
    )
    if args.method == STABILIZED_METHOD:
        return VPMSetup(
            **common,
            # Numerically identical to the `les` control -- same integration,
            # same kernel, same treecode -- so the three-way comparison isolates
            # exactly two things: the coarse-grid Smagorinsky constant and the
            # enstrophy envelope.
            time_integration="FRACTIONAL",
            advection=AdvectionConfig(scheme="RK3"),
            stretching=StretchingConfig.transposed(scheme="RK3"),
            turbulence=TurbulenceConfig.les_smagorinsky(cs=STABILIZED_LES_CS, ce=1.048),
            velocity=VelocityConfig.treecode(
                theta=0.35, sort_particle_targets=True, traversal_block_dim=128
            ),
            particles_kernel="GAUSSIAN",
            envelope=EnvelopeConfig.bounded(
                rho_max=args.rho_max,
                b_l=args.envelope_growth,
                kappa=args.envelope_kappa,
                r_loc_max=args.r_loc_max,
                omega_hard=args.omega_hard,
            ),
        )
    return VPMSetup(
        **common,
        time_integration="FRACTIONAL",
        advection=AdvectionConfig(scheme="RK3"),
        stretching=StretchingConfig.transposed(scheme="RK3"),
        turbulence=(
            TurbulenceConfig.les_smagorinsky(cs=CONTROL_LES_CS, ce=1.048)
            if args.method == "les"
            else TurbulenceConfig.dns()
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
        raise SimulationCrashed(
            f"CRASHED at step {solver.time_step} (t={solver.flow_time:.4f}): "
            "particle strengths went to NaN or infinity."
        )
    if maximum > threshold:
        raise SimulationCrashed(
            f"CRASHED at step {solver.time_step} (t={solver.flow_time:.4f}): "
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
        "smagorinsky_cs": cfg.turbulence.cs if cfg.turbulence.flow_model == "LES" else None,
        "molecular_viscosity": cfg.viscous.viscosity,
        "characteristic_distance": cfg.viscous.characteristic_distance,
        "field_modification": "none",
        "retention_bounds": cfg.stabilization.remove_particles_by_bounds,
        "diagnostics": "conservation and resolution recorded per logging step; not gated",
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
        "rho_max": args.rho_max,
        "envelope_growth": args.envelope_growth,
        "envelope_kappa": args.envelope_kappa,
        "r_loc_max": args.r_loc_max,
        "omega_hard": args.omega_hard,
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
    solver.info()

    solver._update_all_flow_integrals()
    solver._export_flow_integrals_csv()
    initial_max_strength = float(np.linalg.norm(solver.particles_circulation, axis=1).max())
    write_manifest(args, case_label, output_dir, solver, status="running")

    try:
        for step in range(args.num_steps):
            solver.update_state()
            if (step + 1) % args.guard_frequency == 0:
                enforce_numerical_bound(solver, initial_max_strength, args.blowup_factor)
    except SimulationCrashed as error:
        export_diagnostic_snapshot(solver)
        solver.save_state(str(output_dir / f"vpm_{args.name}_crash_state"))
        write_manifest(
            args, case_label, output_dir, solver, status="crashed", termination_reason=str(error)
        )
        print(f"\n{error}", flush=True)
        print(
            f"Ran {solver.time_step} of {args.num_steps} steps before crashing. "
            f"State at the crash saved to vpm_{args.name}_crash_state.",
            flush=True,
        )
        return "crashed"

    export_diagnostic_snapshot(solver)
    write_manifest(args, case_label, output_dir, solver, status="completed")
    print(f"Finished all {args.num_steps} steps.", flush=True)
    return "completed"


def main() -> int:
    run_case(build_arg_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
