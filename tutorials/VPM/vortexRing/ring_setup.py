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
from openonda.vpm import (
    AdvectionConfig,
    ParticleDistributor,
    Solver,
    StabilizationConfig,
    StretchingConfig,
    TurbulenceConfig,
    VelocityConfig,
    ViscousConfig,
    VortexRingVPM,
    VPMSetup,
)

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
SAMPLE_PERIOD = 0.1  # write a snapshot every this many seconds
BACKUP_PERIOD = 0.5  # keep an animation frame every this many seconds
WIDNALL_MODES = 24
DEFAULT_WIDNALL_AMPLITUDE = 0.05
RESOLUTION_DIVERGENCE_LIMIT = 0.12
RESOLUTION_MISALIGNMENT_LIMIT_DEG = 45.0


def cadence_steps(period: float) -> int:
    """Convert a physical output period to solver steps."""
    return max(1, round(period / TIME_STEP))


def stretching_setup(name: str) -> StretchingConfig:
    """Build the selected vortex-stretching formulation."""
    return {
        "direct": StretchingConfig.direct,
        "transposed": StretchingConfig.transposed,
        "mixed": StretchingConfig.mixed,
    }[name](scheme="RK3")


def run_case(
    name: str,
    *,
    widnall_amplitude: float = DEFAULT_WIDNALL_AMPLITUDE,
    widnall_modes: int = WIDNALL_MODES,
    number_of_steps: int = NUMBER_OF_STEPS,
    output_directory: Path = SOLUTION_DIR,
    output_label: str | None = None,
) -> None:
    if widnall_amplitude < 0.0:
        raise ValueError("widnall_amplitude must be non-negative")
    if widnall_modes < 1:
        raise ValueError("widnall_modes must be positive")
    if number_of_steps < 1:
        raise ValueError("number_of_steps must be positive")

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
    positions, volumes, radii = ParticleDistributor.hexagonal_distribution(
        DOMAIN_BOUNDS,
        PARTICLE_SPACING,
    )

    solver = Solver(
        setup=VPMSetup(
            time_step_size=TIME_STEP,
            processing_unit="AUTO",
            advection=AdvectionConfig(scheme="RK3"),
            turbulence=(
                TurbulenceConfig.dns()
                if mode == "dns"
                else TurbulenceConfig.les_smagorinsky(cs=0.20)
            ),
            stretching=stretching_setup(stretching),
            stabilization=StabilizationConfig.disabled(),
            velocity=VelocityConfig.treecode(
                theta=0.3,
                sort_particle_targets=True,
                traversal_block_dim=128,
            ),
            viscous=ViscousConfig.cs(),
            logging_frequency=cadence_steps(SAMPLE_PERIOD),
            backup_frequency=cadence_steps(BACKUP_PERIOD),
            backup_file_name=label,
            backup_directory=str(output_directory),
            sample_subdirectory=sample_subdirectory,
            samplers=(
                RingDiagnosticsSampler(),
                RingModeDiagnosticsSampler(
                    maximum_mode=40,
                    azimuthal_bins=128,
                    reference_radius=RING_RADIUS,
                ),
            ),
            max_particles=100_000,
        )
    )

    velocity, particle_viscosity, circulation = VortexRingVPM(
        viscosity=viscosity,
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
    )
    solver.add_vortex_particles(
        position=positions,
        velocity=velocity,
        circulation=circulation,
        radius=radii,
        volume=volumes,
        viscosity=particle_viscosity,
        group_id=0,
    )
    solver.remove_weak_particles(percent=0.1, per_group=True)

    initial_strength = np.abs(solver.particles.circulation_cpu()).max()
    theoretical_seed_amplitude = widnall_amplitude / np.sqrt(widnall_modes)
    manifest = {
        "status": "running",
        "variant": name,
        "output_label": label,
        "requested_steps": number_of_steps,
        "time_step": TIME_STEP,
        "ring_radius": RING_RADIUS,
        "core_radius": CORE_RADIUS,
        "ring_circulation": RING_STRENGTH,
        "circulation_reynolds_number": REYNOLDS_NUMBER,
        "particle_spacing": PARTICLE_SPACING,
        "widnall_amplitude": widnall_amplitude,
        "widnall_modes": widnall_modes,
        "theoretical_radial_seed_amplitude_per_mode": theoretical_seed_amplitude,
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
    solver.backup_solution(str(output_directory / f"vpm_{label}"))
    termination_reason = None
    for _ in range(number_of_steps):
        solver.update_state()
        if np.abs(solver.particles.circulation_cpu()).max() > 50 * initial_strength:
            termination_reason = "peak particle strength exceeded 50 times its initial value"
            break
        if solver.time_step % cadence_steps(SAMPLE_PERIOD):
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
        completed_steps=solver.time_step,
        completed_time=solver.flow_time,
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
    parser.add_argument("--number-of-steps", type=int, default=NUMBER_OF_STEPS)
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
    run_case(
        args.variant,
        widnall_amplitude=args.widnall_amplitude,
        widnall_modes=args.widnall_modes,
        number_of_steps=args.number_of_steps,
        output_directory=args.output_directory,
        output_label=args.output_label,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
