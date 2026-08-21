#!/usr/bin/env python3
"""Vortex ring evolution under different stretching and turbulence models (VPM).

An initially-Gaussian vortex ring is advanced with the vortex particle method.
Four physics variants are provided:

  * ``DNS_direct`` / ``DNS_transposed`` / ``DNS_mixed``: the three vortex
    stretching formulations at DNS resolution;
  * ``LES_transposed``: the transposed stretching with a Smagorinsky model.

Each case samples the ring motion, energy, and circulation that the
``allplot.sh`` figures compare with the theory.

Usage:
    python ring_setup.py --variant DNS_transposed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from assets.ring_diagnostics import RingDiagnosticsSampler, RingModeDiagnosticsSampler
from assets.ring_initialization import initialize_single_mode_toroidal_ring
import openonda.vpm as vpm

TUTORIAL_DIR = Path(__file__).resolve().parent
SOLUTION_DIR = TUTORIAL_DIR / "solution"

# ---- Physics -------------------------------------------------------------
RING_RADIUS = 1.0
RING_STRENGTH = np.pi
REYNOLDS_NUMBER = 3000.0
CORE_RADIUS = 0.1

# ---- Numerics ------------------------------------------------------------
PARTICLE_SPACING = 0.035
TIME_STEP = 0.02
NUMBER_OF_STEPS = 600
DOMAIN_BOUNDS = (-0.15, 0.15, -1.5, 1.5, -1.5, 1.5)
SAMPLE_INTERVAL_TIME = 0.1  # write a snapshot every this many seconds
CHECKPOINT_INTERVAL_TIME = 0.5  # keep an animation frame every this many seconds
WIDNALL_MODES = 24
DEFAULT_WIDNALL_AMPLITUDE = 0.05
TOROIDAL_TAIL_FRACTION = 0.05
RESOLUTION_DIVERGENCE_LIMIT = 0.12
RESOLUTION_MISALIGNMENT_LIMIT_DEG = 45.0


def cadence_steps(period: float, time_step: float = TIME_STEP) -> int:
    """Convert a physical output period to solver steps."""
    return max(1, round(period / time_step))


def stretching_setup(name: str) -> vpm.StretchingConfig:
    """Build the selected vortex-stretching formulation."""
    return {
        "direct": vpm.StretchingConfig.direct,
        "transposed": vpm.StretchingConfig.transposed,
        "mixed": vpm.StretchingConfig.mixed,
    }[name](scheme="RK3")


def run_case(
    name: str,
    *,
    widnall_amplitude: float = DEFAULT_WIDNALL_AMPLITUDE,
    widnall_modes: int = WIDNALL_MODES,
    widnall_single_mode: int | None = None,
    number_of_steps: int = NUMBER_OF_STEPS,
    output_directory: Path = SOLUTION_DIR,
    output_label: str | None = None,
    particle_distribution: str = "hexagonal",
    time_step: float = TIME_STEP,
    time_integration: str = "FRACTIONAL",
    velocity_method: str = "TREECODE",
    treecode_theta: float = 0.3,
    compute_device: str = "AUTO",
) -> None:
    if widnall_amplitude < 0.0:
        raise ValueError("widnall_amplitude must be non-negative")
    if widnall_modes < 1:
        raise ValueError("widnall_modes must be positive")
    if widnall_single_mode is not None and widnall_single_mode < 1:
        raise ValueError("widnall_single_mode must be positive")
    if number_of_steps < 1:
        raise ValueError("number_of_steps must be positive")
    if time_step <= 0.0:
        raise ValueError("time_step must be positive")
    time_integration = time_integration.upper()
    if time_integration not in {"FRACTIONAL", "COUPLED"}:
        raise ValueError("time_integration must be FRACTIONAL or COUPLED")
    velocity_method = velocity_method.upper()
    if velocity_method not in {"DIRECT", "TREECODE"}:
        raise ValueError("velocity_method must be DIRECT or TREECODE")
    if not 0.0 < treecode_theta < 2.0:
        raise ValueError("treecode_theta must be in (0, 2)")

    mode, stretching = name.lower().split("_", maxsplit=1)
    label = output_label or name
    if Path(label).name != label:
        raise ValueError("output_label must be a file name, not a path")
    output_directory = Path(output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    sample_subdirectory = label if output_directory.name == "solution" else None
    sample_directory = (
        output_directory.parent / "samples" / label
        if sample_subdirectory
        else output_directory / "samples"
    )
    existing = [
        *output_directory.glob(f"vpm_{label}_*.h5"),
        *output_directory.glob(f"vpm_{label}_*.xdmf"),
        output_directory / f"run_manifest_{label}.json",
        sample_directory / "ring_modes.csv",
    ]
    existing = [path for path in existing if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to mix with an existing run; choose a new --output-label or "
            f"--output-directory. First conflict: {existing[0]}"
        )

    viscosity = RING_STRENGTH / REYNOLDS_NUMBER
    if particle_distribution == "hexagonal":
        positions, volumes, radii = vpm.ParticleDistributor.hexagonal_distribution(
            DOMAIN_BOUNDS,
            PARTICLE_SPACING,
        )
    elif particle_distribution == "toroidal":
        particle_radius = 2.0 * PARTICLE_SPACING
        represented_core_sq = CORE_RADIUS**2 - particle_radius**2
        if represented_core_sq <= 0.0:
            raise ValueError("particle radius must be smaller than the physical core radius")
        tube_radius = np.sqrt(represented_core_sq) * np.sqrt(-np.log(TOROIDAL_TAIL_FRACTION))
        positions, volumes, radii = vpm.ParticleDistributor.toroidal_distribution(
            RING_RADIUS,
            tube_radius,
            PARTICLE_SPACING,
            epsilon_w=0.0 if widnall_single_mode is not None else widnall_amplitude,
            seed=42,
            max_modes=widnall_modes,
        )
        radii.fill(particle_radius)
    else:
        raise ValueError(f"unknown particle_distribution {particle_distribution!r}")

    single_mode_phase = None
    if widnall_single_mode is not None:
        if particle_distribution != "toroidal":
            raise ValueError("widnall_single_mode requires the toroidal distribution")
        (
            positions,
            volumes,
            radii,
            velocity,
            particle_viscosity,
            circulation,
            single_mode_phase,
        ) = initialize_single_mode_toroidal_ring(
            positions,
            volumes,
            radii,
            viscosity=viscosity,
            ring_radius=RING_RADIUS,
            ring_strength=RING_STRENGTH,
            ring_thickness=CORE_RADIUS,
            amplitude=widnall_amplitude,
            mode=widnall_single_mode,
            seed=42,
        )
    else:
        velocity, particle_viscosity, circulation = vpm.VortexRingVPM(
            kinematic_viscosity=viscosity,
            ring_center=[0, 0, 0],
            ring_radius=RING_RADIUS,
            ring_strength=RING_STRENGTH,
            ring_thickness=CORE_RADIUS,
            avg_particle_radius=float(radii.mean()),
            positions=positions,
            volumes=volumes,
            epsilon_W=widnall_amplitude,
            max_modes=widnall_modes,
            anti_diffuse_flag=True,
            normalize_circulation=particle_distribution == "toroidal",
        )
    mode_sampler = RingModeDiagnosticsSampler(
        maximum_mode=40,
        azimuthal_bins=128,
        reference_radius=RING_RADIUS,
        transverse_origin=(0.0, 0.0),
    )
    initial_modes = np.asarray(mode_sampler._sample_group(positions, circulation), dtype=float)
    seeded_modes = (
        np.asarray([widnall_single_mode], dtype=int)
        if widnall_single_mode is not None
        else np.arange(1, widnall_modes + 1)
    )
    theoretical_seed_amplitude = widnall_amplitude / np.sqrt(len(seeded_modes))
    seeded_indices = seeded_modes - 1
    unseeded_indices = np.setdiff1d(np.arange(len(initial_modes)), seeded_indices)
    if theoretical_seed_amplitude > 0.0:
        initial_seed_relative_l2 = float(
            np.linalg.norm(initial_modes[seeded_indices, 1] - theoretical_seed_amplitude)
            / (np.sqrt(len(seeded_modes)) * theoretical_seed_amplitude)
        )
        initial_unseeded_to_seeded_rms = float(
            np.sqrt(np.mean(initial_modes[unseeded_indices, 1] ** 2))
            / np.sqrt(np.mean(initial_modes[seeded_indices, 1] ** 2))
        )
    else:
        initial_seed_relative_l2 = 0.0
        initial_unseeded_to_seeded_rms = 0.0
    if particle_distribution == "toroidal" and (
        initial_seed_relative_l2 > 0.05 or initial_unseeded_to_seeded_rms > 0.10
    ):
        raise RuntimeError(
            "Discrete Widnall seed failed its representation gate: "
            f"seed_error={initial_seed_relative_l2:.3%}, "
            f"unseeded_noise={initial_unseeded_to_seeded_rms:.3%}"
        )

    solver = vpm.VPMSolver(
        setup=vpm.VPMSetup(
            time_step_size=time_step,
            compute_device=compute_device,
            time_integration=time_integration,
            advection=vpm.AdvectionConfig(scheme="RK3"),
            turbulence=(
                vpm.TurbulenceConfig.dns()
                if mode == "dns"
                else vpm.TurbulenceConfig.les_smagorinsky(c_s=0.20)
            ),
            stretching=stretching_setup(stretching),
            stabilization=vpm.StabilizationConfig.disabled(),
            velocity=(
                vpm.VelocityConfig.direct()
                if velocity_method == "DIRECT"
                else vpm.VelocityConfig.treecode(
                    theta=treecode_theta,
                    sort_particle_targets=True,
                    traversal_block_dim=128,
                )
            ),
            viscous=vpm.ViscousConfig.cs(),
            logging_interval_steps=cadence_steps(SAMPLE_INTERVAL_TIME, time_step),
            checkpoint_interval_steps=cadence_steps(CHECKPOINT_INTERVAL_TIME, time_step),
            checkpoint_name=label,
            checkpoint_directory=str(output_directory),
            sample_subdirectory=sample_subdirectory,
            samplers=(RingDiagnosticsSampler(), mode_sampler),
            max_particles=100_000,
        ),
        case_dir=output_directory.parent if sample_subdirectory else output_directory,
    )
    solver.add_vortex_particles(
        position=positions,
        velocity=velocity,
        vortex_strength=circulation,
        core_radius=radii,
        volume=volumes,
        kinematic_viscosity=particle_viscosity,
        group_id=0,
    )
    if particle_distribution == "hexagonal":
        solver.remove_weak_particles(percent=0.1, per_group=True)

    initial_strength = np.abs(solver.particles.vortex_strength_cpu()).max()
    manifest = {
        "status": "running",
        "variant": name,
        "output_label": label,
        "requested_steps": number_of_steps,
        "time_step": time_step,
        "time_integration": time_integration,
        "velocity_method": velocity_method,
        "treecode_theta": treecode_theta if velocity_method == "TREECODE" else None,
        "compute_device": compute_device,
        "ring_radius": RING_RADIUS,
        "core_radius": CORE_RADIUS,
        "ring_circulation": RING_STRENGTH,
        "circulation_reynolds_number": REYNOLDS_NUMBER,
        "particle_spacing": PARTICLE_SPACING,
        "particle_distribution": particle_distribution,
        "toroidal_tail_fraction": (
            TOROIDAL_TAIL_FRACTION if particle_distribution == "toroidal" else None
        ),
        "widnall_amplitude": widnall_amplitude,
        "widnall_modes": widnall_modes,
        "widnall_single_mode": widnall_single_mode,
        "widnall_mode_numbers": seeded_modes.tolist(),
        "widnall_single_mode_phase": single_mode_phase,
        "theoretical_radial_seed_amplitude_per_mode": theoretical_seed_amplitude,
        "discrete_radial_seed_relative_l2": initial_seed_relative_l2,
        "discrete_unseeded_to_seeded_rms": initial_unseeded_to_seeded_rms,
        "discrete_seed_relative_l2_limit": 0.05,
        "discrete_unseeded_noise_limit": 0.10,
        "theoretical_gaussian_dominant_mode_estimate": 2.26 * RING_RADIUS / CORE_RADIUS,
        "resolution_divergence_limit": RESOLUTION_DIVERGENCE_LIMIT,
        "resolution_misalignment_limit_deg": RESOLUTION_MISALIGNMENT_LIMIT_DEG,
        "molecular_diffusion": "core_spreading",
        "sgs_model": "none" if mode == "dns" else "legacy_smagorinsky",
        "claim_scope": "VPM Widnall challenge; structural DIAD is not active",
    }
    manifest_path = output_directory / f"run_manifest_{label}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    solver.record_diagnostics(refresh_fields=True)
    solver.save_state(str(output_directory / f"vpm_{label}"))
    termination_reason = None
    for _ in range(number_of_steps):
        solver.advance()
        if np.abs(solver.particles.vortex_strength_cpu()).max() > 50 * initial_strength:
            termination_reason = "peak particle strength exceeded 50 times its initial value"
            break
        if solver.step % cadence_steps(SAMPLE_INTERVAL_TIME, time_step):
            continue
        health = solver._discretization_health
        divergence = float(health["vorticity_divergence_error"])
        misalignment = float(health["strength_misalignment_deg"])
        if (
            divergence > RESOLUTION_DIVERGENCE_LIMIT
            or misalignment > RESOLUTION_MISALIGNMENT_LIMIT_DEG
        ):
            termination_reason = (
                "particle resolution lost: "
                f"divergence={divergence:.6g}, misalignment_deg={misalignment:.6g}"
            )
            break

    solver.save_state(str(output_directory / f"vpm_{label}_final"))
    manifest.update(
        status="resolution_lost" if termination_reason else "completed",
        completed_steps=solver.step,
        completed_time=solver.time,
        final_particles=len(solver.particles),
    )
    if termination_reason:
        manifest["termination_reason"] = termination_reason
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        required=True,
        choices=("DNS_direct", "DNS_transposed", "DNS_mixed", "LES_transposed"),
        help="physics variant to run",
    )
    parser.add_argument(
        "--particle-distribution",
        choices=("hexagonal", "toroidal"),
        default="hexagonal",
        help="Toroidal uses complete azimuthal orbits for modal-instability studies.",
    )
    parser.add_argument(
        "--widnall-amplitude",
        type=float,
        default=DEFAULT_WIDNALL_AMPLITUDE,
        help="Broadband centreline perturbation amplitude.",
    )
    parser.add_argument(
        "--widnall-modes",
        type=int,
        default=WIDNALL_MODES,
        help="Number of equally seeded azimuthal modes.",
    )
    parser.add_argument(
        "--widnall-single-mode",
        type=int,
        help="Seed one azimuthal mode instead of broadband modes 1...N.",
    )
    parser.add_argument("--number-of-steps", type=int, default=NUMBER_OF_STEPS)
    parser.add_argument("--time-step", type=float, default=TIME_STEP)
    parser.add_argument(
        "--time-integration",
        choices=("FRACTIONAL", "COUPLED"),
        default="FRACTIONAL",
        help="COUPLED advances positions and strengths at common RK stages.",
    )
    parser.add_argument(
        "--velocity-method",
        choices=("DIRECT", "TREECODE"),
        default="TREECODE",
    )
    parser.add_argument("--treecode-theta", type=float, default=0.3)
    parser.add_argument(
        "--compute-device",
        choices=("AUTO", "CPU", "METAL", "VULKAN", "CUDA"),
        default="AUTO",
    )
    parser.add_argument(
        "--final-time-star",
        type=float,
        help="Override --number-of-steps to reach t*=t Gamma/R^2.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=SOLUTION_DIR,
        help="Directory for raw restart states and run manifest.",
    )
    parser.add_argument(
        "--output-label",
        help="Unique run label; defaults to the physics variant.",
    )
    args = parser.parse_args()

    print("\n===== SIMULATION =====")
    print(f"---- vortex ring variant: {args.variant} ----")
    number_of_steps = args.number_of_steps
    if args.final_time_star is not None:
        if args.final_time_star <= 0.0:
            parser.error("--final-time-star must be positive")
        physical_end_time = args.final_time_star * RING_RADIUS**2 / RING_STRENGTH
        number_of_steps = int(np.ceil(physical_end_time / args.time_step))

    run_case(
        args.variant,
        widnall_amplitude=args.widnall_amplitude,
        widnall_modes=args.widnall_modes,
        widnall_single_mode=args.widnall_single_mode,
        number_of_steps=number_of_steps,
        output_directory=args.output_directory,
        output_label=args.output_label,
        particle_distribution=args.particle_distribution,
        time_step=args.time_step,
        time_integration=args.time_integration,
        velocity_method=args.velocity_method,
        treecode_theta=args.treecode_theta,
        compute_device=args.compute_device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
