#!/usr/bin/env python3
"""Six-case vortex-ring interaction comparison.

Each interaction family is run with three deliberately distinct methods:

* ``baseline``: molecular-viscosity DNS with the legacy fractional RK3 core;
* ``les``: the same legacy core plus Smagorinsky LES;
* ``les_stabilized``: the same LES physics with direct symmetric interactions,
  coupled RK2 stages, conservative stretching, Strang-split core spreading,
  and strain/displacement subcycling.

Only ``les_stabilized`` is subject to the strict closed-system physical
acceptance contract.  Baseline and LES are controls: their invariant errors
are measured and plotted, not used to reject an otherwise useful comparison.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from source.solvers.VPM import ParticleDistributor, Solver, VPMSetup
from source.solvers.VPM.config.types import (
    AdvectionConfig,
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


class PhysicalContractError(RuntimeError):
    """A finite solution violated the stabilized method's acceptance contract."""


class NumericalBlowupError(RuntimeError):
    """A control run became non-finite or exceeded its declared blow-up scale."""


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
        help="Check the physical contract every N steps; 1 checks every accepted step.",
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
        "--energy-increase-tolerance",
        type=float,
        default=2.0e-5,
        help="Allowed relative diagnostic roundoff before E(t) is considered to increase.",
    )
    parser.add_argument(
        "--energy-budget-tolerance",
        type=float,
        default=0.30,
        help="Allowed relative mismatch between dE/dt and the viscous enstrophy sink.",
    )
    parser.add_argument(
        "--invariant-drift-tolerance",
        type=float,
        default=5.0e-3,
        help="Maximum normalized drift of circulation and the two impulses.",
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
    if (
        args.energy_increase_tolerance < 0.0
        or args.energy_budget_tolerance <= 0.0
        or args.invariant_drift_tolerance <= 0.0
        or args.blowup_factor <= 1.0
    ):
        raise ValueError("Physical-contract tolerances and --blowup-factor are invalid.")

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
            "Stabilized VPM contract rejected an under-resolved setup: "
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
            time_integration="COUPLED",
            coupled_max_strain_increment=0.08,
            coupled_max_advection_fraction=0.25,
            coupled_max_substeps=128,
            advection=AdvectionConfig(scheme="RK2"),
            stretching=StretchingConfig.conservative(scheme="RK2"),
            turbulence=TurbulenceConfig.les_smagorinsky(cs=0.16, ce=1.048),
            velocity=VelocityConfig.direct(),
            particles_kernel="WINCKELMANS",
        )
    return VPMSetup(
        **common,
        time_integration="FRACTIONAL",
        advection=AdvectionConfig(scheme="RK3"),
        stretching=StretchingConfig.transposed(scheme="RK3"),
        turbulence=(
            TurbulenceConfig.les_smagorinsky(cs=0.16, ce=1.048)
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


def _contract_scales(solver: Solver, initial: dict[str, object]) -> dict[str, float]:
    pos = solver.particles_positions
    gamma = solver.particles_circulation
    circ_scale = max(float(np.linalg.norm(gamma, axis=1).sum()), np.finfo(float).tiny)
    linear_scale = max(
        float(0.5 * np.linalg.norm(np.cross(pos, gamma), axis=1).sum()),
        float(np.linalg.norm(initial["linear_impulse"])),
        np.finfo(float).tiny,
    )
    angular_terms = np.cross(pos, np.cross(pos, gamma)) / 3.0
    angular_scale = max(
        float(np.linalg.norm(angular_terms, axis=1).sum()),
        float(np.linalg.norm(initial["angular_impulse"])),
        np.finfo(float).tiny,
    )
    return {
        "strength": circ_scale,
        "linear_impulse": linear_scale,
        "angular_impulse": angular_scale,
    }


def _normalized_vector_drift(
    current: dict[str, object], initial: dict[str, object], scales: dict[str, float], key: str
) -> float:
    return float(np.linalg.norm(np.asarray(current[key]) - np.asarray(initial[key])) / scales[key])


def enforce_physical_contract(
    solver: Solver,
    initial: dict[str, object],
    previous_energy: float,
    previous_sink: float,
    scales: dict[str, float],
    args: argparse.Namespace,
) -> tuple[float, float, dict[str, float]]:
    """Reject the first step that would make the numerical solution non-physical."""
    current = solver.field_diagnostics.compute_flow_integrals(
        solver.particles, solver.flow_time, record_history=False
    )
    energy = float(current["kinetic_energy"])
    sink = float(current["vorticity_dissipation_rate"])
    energy_growth = (energy - previous_energy) / max(abs(previous_energy), np.finfo(float).tiny)
    elapsed = args.guard_frequency * args.dt
    observed_dedt = (energy - previous_energy) / elapsed
    target_dedt = 0.5 * (previous_sink + sink)
    budget_error = abs(observed_dedt - target_dedt) / max(abs(target_dedt), np.finfo(float).tiny)
    drifts = {
        key: _normalized_vector_drift(current, initial, scales, key)
        for key in ("strength", "linear_impulse", "angular_impulse")
    }
    failures = []
    if not np.isfinite(energy) or energy_growth > args.energy_increase_tolerance:
        failures.append(
            f"energy increased by {energy_growth:.3e} (limit {args.energy_increase_tolerance:.3e})"
        )
    if not np.isfinite(budget_error) or budget_error > args.energy_budget_tolerance:
        failures.append(
            f"|dE/dt-sink|/|sink|={budget_error:.3e} (limit {args.energy_budget_tolerance:.3e})"
        )
    for key, drift in drifts.items():
        if not np.isfinite(drift) or drift > args.invariant_drift_tolerance:
            failures.append(f"{key} drift={drift:.3e} (limit {args.invariant_drift_tolerance:.3e})")
    if failures:
        raise PhysicalContractError(
            "Physical acceptance contract failed at "
            f"step={solver.time_step}, t={solver.flow_time:.6e}: "
            + "; ".join(failures)
            + ". The run was stopped rather than changing Gamma."
        )
    return energy, sink, drifts


def enforce_numerical_bound(
    solver: Solver,
    initial_max_strength: float,
    blowup_factor: float,
) -> float:
    """Log and reject only a genuine strength blow-up or non-finite field."""
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
    if not np.isfinite(circulation).all() or not np.isfinite(maximum) or maximum > threshold:
        raise NumericalBlowupError(
            f"Numerical blow-up at step={solver.time_step}, t={solver.flow_time:.6e}: "
            f"max|Gamma|={maximum:.6e}, threshold={threshold:.6e}."
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
        "molecular_viscosity": cfg.viscous.viscosity,
        "characteristic_distance": cfg.viscous.characteristic_distance,
        "coupled_max_strain_increment": cfg.coupled_max_strain_increment,
        "coupled_max_advection_fraction": cfg.coupled_max_advection_fraction,
        "coupled_max_substeps": cfg.coupled_max_substeps,
        "field_modification": "none",
        "retention_bounds": cfg.stabilization.remove_particles_by_bounds,
        "physical_acceptance": (
            "closed-system energy/circulation/linear-impulse/angular-impulse contract"
            if args.method == STABILIZED_METHOD
            else "diagnostic control; invariant errors are measured but do not reject the run"
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
        "energy_increase_tolerance": args.energy_increase_tolerance,
        "energy_budget_tolerance": args.energy_budget_tolerance,
        "invariant_drift_tolerance": args.invariant_drift_tolerance,
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
        256.0 / 45.0 if args.method == STABILIZED_METHOD else 4.0,
    )
    solver.info()

    solver._update_all_flow_integrals()
    solver._export_flow_integrals_csv()
    solver.backup_solution(str(output_dir / f"vpm_{args.name}"))
    initial = dict(solver._flow_integrals)
    scales = _contract_scales(solver, initial)
    previous_energy = float(initial["kinetic_energy"])
    previous_sink = float(initial["vorticity_dissipation_rate"])
    initial_max_strength = float(np.linalg.norm(solver.particles_circulation, axis=1).max())
    write_manifest(args, case_label, output_dir, solver, status="running")

    try:
        for step in range(args.num_steps):
            solver.update_state()
            if (step + 1) % args.guard_frequency != 0:
                continue
            enforce_numerical_bound(solver, initial_max_strength, args.blowup_factor)
            if args.method != STABILIZED_METHOD:
                continue
            previous_energy, previous_sink, drifts = enforce_physical_contract(
                solver, initial, previous_energy, previous_sink, scales, args
            )
            print(
                "PHYSICS CHECK "
                f"step={step + 1} time={solver.flow_time:.6e} "
                f"energy={previous_energy:.8e} "
                f"circ_drift={drifts['strength']:.3e} "
                f"impulse_drift={drifts['linear_impulse']:.3e} "
                f"angular_drift={drifts['angular_impulse']:.3e}",
                flush=True,
            )
    except (PhysicalContractError, NumericalBlowupError) as error:
        export_diagnostic_snapshot(solver)
        solver.save_state(str(output_dir / "rejected_state"))
        status = (
            "rejected_physical_contract"
            if isinstance(error, PhysicalContractError)
            else "terminated_nonphysical"
        )
        write_manifest(
            args,
            case_label,
            output_dir,
            solver,
            status=status,
            termination_reason=str(error),
        )
        print(f"Simulation ended with status={status}: {error}", flush=True)
        return status

    export_diagnostic_snapshot(solver)
    write_manifest(args, case_label, output_dir, solver, status="completed")
    qualifier = "physically accepted " if args.method == STABILIZED_METHOD else ""
    print(f"Simulation completed {args.num_steps} {qualifier}steps.", flush=True)
    return "completed"


def main() -> int:
    run_case(build_arg_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
