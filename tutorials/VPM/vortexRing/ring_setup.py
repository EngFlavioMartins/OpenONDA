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
from pathlib import Path

import numpy as np

from assets.ring_diagnostics import RingDiagnosticsSampler
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


def run_case(name: str) -> None:
    mode, stretching = name.lower().split("_", maxsplit=1)
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
                TurbulenceConfig.dns() if mode == "dns" else TurbulenceConfig.les_smagorinsky()
            ),
            stretching=stretching_setup(stretching),
            stabilization=StabilizationConfig.disabled(),
            velocity=VelocityConfig.treecode(
                theta=0.35,
                sort_particle_targets=True,
                traversal_block_dim=128,
            ),
            viscous=ViscousConfig.cs(),
            logging_frequency=cadence_steps(SAMPLE_PERIOD),
            backup_frequency=cadence_steps(BACKUP_PERIOD),
            backup_file_name=name,
            backup_directory=str(SOLUTION_DIR),
            sample_subdirectory=name,
            samplers=(RingDiagnosticsSampler(),),
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
        epsilon_W=0.025,
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
    for _ in range(NUMBER_OF_STEPS):
        solver.update_state()
        if np.abs(solver.particles.circulation_cpu()).max() > 50 * initial_strength:
            print(
                f">>> {name} became unstable at step {solver.time_step}. Moving on to the next case."
            )
            break


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        required=True,
        choices=("DNS_direct", "DNS_transposed", "DNS_mixed", "LES_transposed"),
        help="physics variant to run",
    )
    args = parser.parse_args()

    print("\n===== SIMULATION =====")
    print(f"---- vortex ring variant: {args.variant} ----")
    run_case(args.variant)
    print("\n===== DONE =====")
    print("Simulation completed successfully. Run ./allplot.sh to make the figures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
