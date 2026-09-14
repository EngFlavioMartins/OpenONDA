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
from openonda.tutorial_runner import case_package

__package__ = case_package(Path(__file__).parent)
from .assets.ring_diagnostics import RingDiagnosticsSampler, vortex_ring_mode_sampler


# Physics
RING_RADIUS = 1.0  # major radius [m]
RING_STRENGTH = np.pi  # filament circulation [m²/s]
REYNOLDS_NUMBER = 3000.0  # Re = Gamma/nu
CORE_RADIUS = 0.1  # physical Gaussian core radius [m]
KINEMATIC_VISCOSITY = RING_STRENGTH / REYNOLDS_NUMBER  # [m²/s]

# Resolution and run length
PARTICLE_SPACING = 0.035  # [m]
TIME_STEP_SIZE = 0.02  # [s]
N_STEPS = 3000
SAMPLE_INTERVAL_TIME = 0.1  # [s]
BACKUP_INTERVAL_TIME = 0.5  # [s]
MAX_N_PARTICLES = 100_000

# Perturbation and model controls
WIDNALL_MODES = 24
DEFAULT_WIDNALL_AMPLITUDE = 0.005
TOROIDAL_TAIL_FRACTION = 0.05
SMAGORINSKY_COEFFICIENT = 0.20
RANDOM_SEED = 42

# Common resolution-loss limits
MAX_LAGRANGIAN_CFL = 1.0
MAX_VORTICITY_DIVERGENCE = 0.12
MAX_VORTEX_MISALIGNMENT = 25.0

VARIANT_CONFIG = {
    "dns_direct": ("DNS", "DIRECT"),
    "dns_transposed": ("DNS", "TRANSPOSED"),
    "dns_mixed": ("DNS", "MIXED"),
    "les_transposed": ("LES_SMAGORINSKY", "TRANSPOSED"),
}
VARIANTS = tuple(VARIANT_CONFIG)


TUTORIAL_DIR = Path(__file__).resolve().parent


def cadence_steps(period: float) -> int:
    """Convert a positive output period in seconds to solver steps."""
    return round(period / TIME_STEP_SIZE)


def turbulence_config(variant: str) -> vpm.TurbulenceConfig:
    return {
        "DNS": vpm.TurbulenceConfig.dns(),
        "LES_SMAGORINSKY": vpm.TurbulenceConfig.les_smagorinsky(
            smagorinsky_coefficient=SMAGORINSKY_COEFFICIENT
        ),
    }[VARIANT_CONFIG[variant][0]]


def stretching_scheme(variant: str) -> str:
    return VARIANT_CONFIG[variant][1]


def build_case(
    variant: str,
    *,
    compute_device: str = "AUTO",
    n_steps: int = N_STEPS,
) -> vpm.VPMCase:
    """Build one vortex-ring comparison case without allocating a solver."""

    particle_core_radius = 2.0 * PARTICLE_SPACING
    represented_core_sq = CORE_RADIUS**2 - particle_core_radius**2
    tube_radius = np.sqrt(represented_core_sq) * np.sqrt(-np.log(TOROIDAL_TAIL_FRACTION))
    distribution = vpm.ToroidalDistribution(
        ring_radius=RING_RADIUS,
        tube_radius=tube_radius,
        spacing=PARTICLE_SPACING,
        core_radius_ratio=particle_core_radius / PARTICLE_SPACING,
    )
    initial_condition = vpm.VortexRing(
        kinematic_viscosity=KINEMATIC_VISCOSITY,
        centre=(0.0, 0.0, 0.0),
        radius=RING_RADIUS,
        circulation=RING_STRENGTH,
        vortex_core_radius=CORE_RADIUS,
        disturbance=vpm.WidnallDisturbance.broadband(
            amplitude=DEFAULT_WIDNALL_AMPLITUDE,
            number_of_modes=WIDNALL_MODES,
            seed=RANDOM_SEED,
        ),
        core_compensation=vpm.ParticleCoreCompensation(),
        distribution=distribution,
        group_id=0,
    )

    sample_steps = cadence_steps(SAMPLE_INTERVAL_TIME)
    return vpm.VPMCase(
        name=variant,
        directory=TUTORIAL_DIR,
        numerics=vpm.Numerics(
            time_step_size=TIME_STEP_SIZE,
            compute_device=compute_device,
            integrator=vpm.SSPRK3(),
            turbulence=turbulence_config(variant),
            stabilization=vpm.StabilizationConfig.disabled(),
            induction=vpm.TreecodeInduction(stretching_scheme=stretching_scheme(variant)),
            viscous=vpm.ViscousConfig.cs(),
            write_precision="f32",
            max_n_particles=MAX_N_PARTICLES,
            random_seed=RANDOM_SEED,
            health_limits=vpm.HealthLimits(
                lagrangian_cfl=vpm.LagrangianCFLLimit(maximum=MAX_LAGRANGIAN_CFL),
                divergence=vpm.DivergenceLimit(maximum=MAX_VORTICITY_DIVERGENCE),
                misalignment=vpm.MisalignmentLimit(maximum_degrees=MAX_VORTEX_MISALIGNMENT),
            ),
        ),
        initial_conditions=(initial_condition,),
        backup=vpm.Backup(
            interval_steps=cadence_steps(BACKUP_INTERVAL_TIME),
            directory=str(Path("solution") / variant),
            log_directory=str(Path("solution") / variant),
        ),
        samplers=vpm.Samplers(
            samples=(
                vpm.FlowIntegralsSampler(schedule=vpm.EverySteps(sample_steps), initial=True),
                RingDiagnosticsSampler(schedule=vpm.EverySteps(sample_steps)),
                vortex_ring_mode_sampler(
                    reference_radius=RING_RADIUS,
                    schedule=vpm.EverySteps(sample_steps),
                ),
            ),
            directory=variant,
        ),
        run=vpm.RunPlan(
            steps=n_steps,
            final_backup=False,
            health_limit_action="STOP",
        ),
    )


def run_case(
    variant: str,
    *,
    compute_device: str = "AUTO",
    n_steps: int = N_STEPS,
) -> None:
    """Construct and run one vortex-ring comparison case."""
    vpm.VPMSolver(build_case(variant, compute_device=compute_device, n_steps=n_steps)).run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="dns_transposed", choices=VARIANTS)
    args = parser.parse_args()
    run_case(args.variant)
