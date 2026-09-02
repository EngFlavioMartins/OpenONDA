#!/usr/bin/env python3
"""Vortex ring evolution under different stretching and turbulence models (VPM).

An initially-Gaussian vortex ring is advanced with the vortex particle method.
Four physics variants are provided:

  * ``dns_direct`` / ``dns_transposed`` / ``dns_mixed``: DNS with the three
    vortex-stretching formulations;
  * ``les_transposed``: the transposed stretching with a Smagorinsky model.

Each case samples the ring motion, energy, and circulation that the
``allplot.sh`` figures compare with the theory.

The explicit variant selector compares the three stretching formulations and
the LES model used in this tutorial.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from assets.ring_diagnostics import RingDiagnosticsSampler, vortex_ring_mode_sampler
import openonda.vpm as vpm
from openonda.vpm import Backup, Samplers

TUTORIAL_DIR = Path(__file__).resolve().parent

# ---- Physics ---------------------------------------------------------------
RING_RADIUS = 1.0  # ring major radius [m]
RING_STRENGTH = np.pi  # circulation [m²/s]
REYNOLDS_NUMBER = 3000.0  # Re = Γ/ν — sets the vortex Reynolds number
CORE_RADIUS = 0.1  # initial Gaussian core radius [m]

# ---- Numerics --------------------------------------------------------------
PARTICLE_SPACING = 0.035  # in-plane particle spacing [m]
TIME_STEP_SIZE = 0.02  # Δt [s]
N_STEPS = 3000  # total number of time steps
SAMPLE_INTERVAL_TIME = 0.1  # write a sample every this many seconds
BACKUP_INTERVAL_TIME = 0.5  # keep an animation frame every this many seconds
WIDNALL_MODES = 24  # number of azimuthal bending modes
DEFAULT_WIDNALL_AMPLITUDE = 0.05  # broadband centreline perturbation amplitude
TOROIDAL_TAIL_FRACTION = 0.05  # toroidal particle distribution tail fraction
MAX_N_PARTICLES = 100_000  # particle count guard
ENABLE_STABILIZATION = False  # enable conservative particle filter
SMAGORINSKY_COEFFICIENT = 0.20  # Smagorinsky coefficient for LES

# -- Derived quantities -------------------------------------------------------
KINEMATIC_VISCOSITY = RING_STRENGTH / REYNOLDS_NUMBER  # ν = Γ/Re


def cadence_steps(period: float, time_step_size: float = TIME_STEP_SIZE) -> int:
    """Convert a physical output period to solver steps."""
    return max(1, round(period / time_step_size))


def run_case(variant: str, compute_device: str = "AUTO") -> None:
    """Run a single vortex-ring variant and write solution/samples."""
    mode, stretching = variant.lower().split("_", maxsplit=1)
    smagorinsky_coefficient = 0.0 if mode == "dns" else SMAGORINSKY_COEFFICIENT
    # -- Particle distribution ------------------------------------------------
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
        ),
        core_compensation=vpm.ParticleCoreCompensation(),
        distribution=distribution,
        group_id=0,
    )

    mode_sampler = vortex_ring_mode_sampler(
        reference_radius=RING_RADIUS,
        schedule=vpm.EverySteps(cadence_steps(SAMPLE_INTERVAL_TIME, TIME_STEP_SIZE)),
    )
    # -- Stabilization --------------------------------------------------------
    stabilization = vpm.StabilizationConfig.disabled()
    if ENABLE_STABILIZATION:
        stabilization = vpm.StabilizationConfig.conservative_filter(
            coefficient=0.25,
            interval_steps=20,
            start_step=20,
            grid_spacing=0.084,
            max_n_particles=MAX_N_PARTICLES,
            tail_budget=0.003,
            total_kinetic_energy_dissipation_limit=0.10,
            total_enstrophy_dissipation_limit=0.10,
            divergence_trigger=0.12,
            misalignment_trigger=25.0,
            capacity_divergence_trigger=0.20,
            capacity_misalignment_trigger=35.0,
            capacity_fraction=0.70,
            capacity_grid_spacing=0.13,
            core_radius=0.15,
            capacity_core_radius=0.15,
            projection_trigger=0.12,
            projection_max_correction=0.10,
        )

    # -- Solver setup ---------------------------------------------------------
    case = vpm.VPMCase(
        numerics=vpm.Numerics(
            time_step_size=TIME_STEP_SIZE,
            compute_device=compute_device,
            integrator=vpm.SSPRK3(),
            turbulence=(
                vpm.TurbulenceConfig.dns()
                if mode == "dns"
                else vpm.TurbulenceConfig.les_smagorinsky(
                    smagorinsky_coefficient=smagorinsky_coefficient
                )
            ),
            stabilization=stabilization,
            induction=vpm.TreecodeInduction(),
            viscous=vpm.ViscousConfig.cs(),
            write_precision="f32",
            max_n_particles=MAX_N_PARTICLES,
        ),
        initial_conditions=(initial_condition,),
        backup=Backup(
            interval_steps=cadence_steps(BACKUP_INTERVAL_TIME, TIME_STEP_SIZE),
            directory=str(Path("solution") / variant),
            log_directory=str(Path("solution") / variant),
        ),
        samplers=Samplers(
            samples=(
                vpm.FlowIntegralsSampler(
                    schedule=vpm.EverySteps(cadence_steps(SAMPLE_INTERVAL_TIME, TIME_STEP_SIZE))
                ),
                RingDiagnosticsSampler(
                    schedule=vpm.EverySteps(cadence_steps(SAMPLE_INTERVAL_TIME, TIME_STEP_SIZE))
                ),
                mode_sampler,
            ),
            directory=variant,
        ),
        run=vpm.RunPlan(steps=N_STEPS),
        directory=TUTORIAL_DIR,
    )
    vpm.VPMSolver(case).run()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        required=True,
        choices=("dns_direct", "dns_transposed", "dns_mixed", "les_transposed"),
    )
    args = parser.parse_args()
    run_case(args.variant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
