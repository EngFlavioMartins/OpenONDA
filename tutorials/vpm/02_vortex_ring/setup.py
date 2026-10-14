#!/usr/bin/env python3
"""Compare vortex-ring instability onset across stretching formulations.

Examples (from this case directory)::

    python setup.py --variant dns_direct
    python setup.py --variant les_transposed
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import openonda.vpm as vpm

from assets.initial_conditions import ToroidalDistribution, VortexRing, WidnallDisturbance
from assets.ring_diagnostics import (
    RingDiagnosticsSampler,
    RingModeDiagnosticsSampler,
)


# Physics
START_FROM = "latest"  # Resume the latest backup; ./allrun.sh cleans first.

RING_RADIUS = 1.0  # major radius [m]
RING_STRENGTH = np.pi  # filament circulation [m²/s]
REYNOLDS_NUMBER = 3000.0  # Re = Gamma/nu
CORE_RADIUS = 0.1  # physical Gaussian core radius [m]
KINEMATIC_VISCOSITY = RING_STRENGTH / REYNOLDS_NUMBER  # [m²/s]

# Resolution and run length
PARTICLE_SPACING = 0.035  # [m]
TIME_STEP_SIZE = 0.02  # [s]
END_TIME = 60.0  # [s]
SAMPLE_INTERVAL_TIME = 0.1  # [s]
BACKUP_INTERVAL_TIME = 0.5  # [s]
MAX_N_PARTICLES = 100_000

# Perturbation and model controls
WIDNALL_MODES = 24
DIAGNOSTIC_MAX_MODE = 40
DIAGNOSTIC_AZIMUTHAL_BINS = 128
DEFAULT_WIDNALL_AMPLITUDE = 0.005
TOROIDAL_TAIL_FRACTION = 0.05
RANDOM_SEED = 42

# Common resolution-loss limits
MAX_LAGRANGIAN_CFL = 1.0
MAX_VORTICITY_DIVERGENCE = 0.12
MAX_VORTEX_MISALIGNMENT = 25.0

VARIANT_CONFIG = {
    "dns_direct": ("dns", "direct"),
    "dns_transposed": ("dns", "transposed"),
    "dns_mixed": ("dns", "mixed"),
    "les_transposed": ("smagorinsky", "transposed"),
}
VARIANTS = tuple(VARIANT_CONFIG)


TUTORIAL_DIR = Path(__file__).resolve().parent


def cadence_steps(period: float) -> int:
    """Convert a positive output period in seconds to solver steps."""
    return round(period / TIME_STEP_SIZE)


def turbulence_config(variant: str) -> vpm.TurbulenceConfig:
    return {
        "dns": vpm.TurbulenceConfig.dns(),
        "smagorinsky": vpm.TurbulenceConfig.smagorinsky(),
    }[VARIANT_CONFIG[variant][0]]


def stretching_scheme(variant: str) -> str:
    return VARIANT_CONFIG[variant][1]


def build_case(
    variant: str,
    *,
    compute_device: str = "auto",
    end_time: float = END_TIME,
) -> vpm.VPMCase:
    """Build one vortex-ring comparison case without allocating a solver."""

    particle_core_radius = 2.0 * PARTICLE_SPACING
    represented_core_sq = CORE_RADIUS**2 - particle_core_radius**2
    tube_radius = np.sqrt(represented_core_sq) * np.sqrt(-np.log(TOROIDAL_TAIL_FRACTION))
    distribution = ToroidalDistribution(
        ring_radius=RING_RADIUS,
        tube_radius=tube_radius,
        spacing=PARTICLE_SPACING,
        core_radius_ratio=particle_core_radius / PARTICLE_SPACING,
    )
    initial_condition = VortexRing(
        kinematic_viscosity=KINEMATIC_VISCOSITY,
        centre=(0.0, 0.0, 0.0),
        radius=RING_RADIUS,
        circulation=RING_STRENGTH,
        vortex_core_radius=CORE_RADIUS,
        disturbance=WidnallDisturbance(
            amplitude=DEFAULT_WIDNALL_AMPLITUDE,
            number_of_modes=WIDNALL_MODES,
            seed=RANDOM_SEED,
        ),
        distribution=distribution,
        group_id=0,
    )

    sample_steps = cadence_steps(SAMPLE_INTERVAL_TIME)
    return vpm.VPMCase(
        name=variant,
        directory=TUTORIAL_DIR,
        variant=variant,
        numerics=vpm.Numerics(
            time_step_size=TIME_STEP_SIZE,
            compute_device=compute_device,
            turbulence=turbulence_config(variant),
            induction=vpm.DirectInduction(stretching_scheme=stretching_scheme(variant)),
            max_n_particles=MAX_N_PARTICLES,
            random_seed=RANDOM_SEED,
            state_limits=vpm.ParticleStateLimits(
                lagrangian_cfl=vpm.LagrangianCFLLimit(maximum=MAX_LAGRANGIAN_CFL),
                divergence=vpm.DivergenceLimit(maximum=MAX_VORTICITY_DIVERGENCE),
                misalignment=vpm.MisalignmentLimit(maximum_degrees=MAX_VORTEX_MISALIGNMENT),
            ),
        ),
        initial_conditions=(initial_condition,),
        run=vpm.RunPlan(end_time=end_time, state_limit_action="stop"),
        samplers=vpm.Samplers(
            samples=(
                vpm.FlowIntegralsSampler(schedule=vpm.EverySteps(sample_steps)),
                RingDiagnosticsSampler(schedule=vpm.EverySteps(sample_steps)),
                RingModeDiagnosticsSampler(
                    max_mode=DIAGNOSTIC_MAX_MODE,
                    azimuthal_bins=DIAGNOSTIC_AZIMUTHAL_BINS,
                    transverse_origin=(0.0, 0.0),
                    reference_radius=RING_RADIUS,
                    schedule=vpm.EverySteps(sample_steps),
                ),
            ),
        ),
        backup=vpm.Backup(schedule=vpm.EverySteps(cadence_steps(BACKUP_INTERVAL_TIME))),
    )


def run_case(
    variant: str,
    *,
    compute_device: str = "auto",
    end_time: float = END_TIME,
) -> None:
    """Construct and run one vortex-ring comparison case."""
    vpm.VPMSolver(build_case(variant, compute_device=compute_device, end_time=end_time)).run(
        start_from=START_FROM
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="dns_transposed", choices=VARIANTS)
    args = parser.parse_args()
    run_case(args.variant)
