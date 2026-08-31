#!/usr/bin/env python3
"""Vortex ring evolution under different stretching and turbulence models (VPM).

An initially-Gaussian vortex ring is advanced with the vortex particle method.
Four physics variants are provided:

  * ``dns_direct`` / ``dns_transposed`` / ``dns_mixed``: DNS with the three
    vortex-stretching formulations;
  * ``les_transposed``: the transposed stretching with a Smagorinsky model.

Each case samples the ring motion, energy, and circulation that the
``allplot.sh`` figures compare with the theory.

Usage:
    python ring_setup.py --variant dns_transposed --compute-device METAL
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from assets.ring_diagnostics import RingDiagnosticsSampler, RingModeDiagnosticsSampler
import openonda.vpm as vpm

TUTORIAL_DIR = Path(__file__).resolve().parent
SOLUTION_DIR = TUTORIAL_DIR / "solution"

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
RESOLUTION_DIVERGENCE_LIMIT = 0.12  # vorticity-divergence health gate
RESOLUTION_MISALIGNMENT_LIMIT_DEG = 45.0  # vortex-strength misalignment health gate
MAX_N_PARTICLES = 100_000  # particle count guard
ENABLE_STABILIZATION = False  # enable conservative particle filter
TREECODE_THETA = 0.30  # treecode accuracy parameter
SMAGORINSKY_COEFFICIENT = 0.20  # Smagorinsky coefficient for LES

# -- Derived quantities -------------------------------------------------------
KINEMATIC_VISCOSITY = RING_STRENGTH / REYNOLDS_NUMBER  # ν = Γ/Re


def cadence_steps(period: float, time_step_size: float = TIME_STEP_SIZE) -> int:
    """Convert a physical output period to solver steps."""
    return max(1, round(period / time_step_size))


def stretching_setup(name: str) -> vpm.StretchingConfig:
    """Build the selected vortex-stretching formulation."""
    return {
        "direct": vpm.StretchingConfig.direct,
        "transposed": vpm.StretchingConfig.transposed,
        "mixed": vpm.StretchingConfig.mixed,
    }[name](scheme="RK3")


def run_case(variant: str, compute_device: str = "AUTO") -> None:
    """Run a single vortex-ring variant and write solution/samples."""
    mode, stretching = variant.lower().split("_", maxsplit=1)
    smagorinsky_coefficient = 0.0 if mode == "dns" else SMAGORINSKY_COEFFICIENT
    output_directory = SOLUTION_DIR.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    # -- Particle distribution ------------------------------------------------
    particle_core_radius = 2.0 * PARTICLE_SPACING
    represented_core_sq = CORE_RADIUS**2 - particle_core_radius**2
    tube_radius = np.sqrt(represented_core_sq) * np.sqrt(-np.log(TOROIDAL_TAIL_FRACTION))
    distribution = vpm.create_toroidal_distribution(
        ring_radius=RING_RADIUS,
        tube_radius=tube_radius,
        spacing=PARTICLE_SPACING,
        core_radius_ratio=particle_core_radius / PARTICLE_SPACING,
    )

    # -- Initial vortex velocity and strength ---------------------------------
    particles = vpm.initialize_vortex_ring(
        distribution,
        kinematic_viscosity=KINEMATIC_VISCOSITY,
        centre=(0.0, 0.0, 0.0),
        radius=RING_RADIUS,
        circulation=RING_STRENGTH,
        vortex_core_radius=CORE_RADIUS,
        disturbance=vpm.WidnallDisturbance.broadband(
            amplitude=DEFAULT_WIDNALL_AMPLITUDE,
            number_of_modes=WIDNALL_MODES,
        ),
        compensate_particle_core=True,
    )
    position = particles.position
    particle_volume = particles.particle_volume
    core_radius = particles.core_radius
    velocity = particles.velocity
    particle_kinematic_viscosity = particles.kinematic_viscosity
    vortex_strength = particles.vortex_strength

    # -- Ring mode diagnostics (seed quality check) ---------------------------
    mode_sampler = RingModeDiagnosticsSampler(
        max_mode=40,
        azimuthal_bins=128,
        reference_radius=RING_RADIUS,
        transverse_origin=(0.0, 0.0),
        schedule=vpm.SamplingSchedule(
            every_n_steps=cadence_steps(SAMPLE_INTERVAL_TIME, TIME_STEP_SIZE)
        ),
    )
    initial_modes = np.asarray(mode_sampler._sample_group(position, vortex_strength), dtype=float)
    seeded_modes = np.arange(1, WIDNALL_MODES + 1)
    theoretical_seed_amplitude = DEFAULT_WIDNALL_AMPLITUDE / np.sqrt(len(seeded_modes))
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
    if initial_seed_relative_l2 > 0.05 or initial_unseeded_to_seeded_rms > 0.10:
        raise RuntimeError(
            "Discrete Widnall seed failed its representation gate: "
            f"seed_error={initial_seed_relative_l2:.3%}, "
            f"unseeded_noise={initial_unseeded_to_seeded_rms:.3%}"
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
    solver = vpm.VPMSolver(
        setup=vpm.VPMSetup(
            time_step_size=TIME_STEP_SIZE,
            compute_device=compute_device,
            time_integration="FRACTIONAL",
            advection=vpm.AdvectionConfig(scheme="RK3"),
            turbulence=(
                vpm.TurbulenceConfig.dns()
                if mode == "dns"
                else vpm.TurbulenceConfig.les_smagorinsky(
                    smagorinsky_coefficient=smagorinsky_coefficient
                )
            ),
            stretching=stretching_setup(stretching),
            stabilization=stabilization,
            velocity=vpm.VelocityConfig.treecode(
                theta=TREECODE_THETA,
                sort_particle_targets=True,
                traversal_block_dim=128,
            ),
            viscous=vpm.ViscousConfig.cs(),
            backup=vpm.Backup(
                interval_steps=cadence_steps(BACKUP_INTERVAL_TIME, TIME_STEP_SIZE),
                directory=str(output_directory / variant),
                log_directory=str(output_directory / variant),
            ),
            samplers=vpm.Samplers(
                vpm.FlowIntegralsSampler(
                    schedule=vpm.SamplingSchedule(
                        every_n_steps=cadence_steps(SAMPLE_INTERVAL_TIME, TIME_STEP_SIZE)
                    )
                ),
                RingDiagnosticsSampler(
                    schedule=vpm.SamplingSchedule(
                        every_n_steps=cadence_steps(SAMPLE_INTERVAL_TIME, TIME_STEP_SIZE)
                    )
                ),
                mode_sampler,
                directory=variant,
            ),
            write_precision="f32",
            max_n_particles=MAX_N_PARTICLES,
        ),
        case_dir=output_directory.parent,
    )
    solver.add_vortex_particles(
        position=position,
        velocity=velocity,
        vortex_strength=vortex_strength,
        core_radius=core_radius,
        particle_volume=particle_volume,
        kinematic_viscosity=particle_kinematic_viscosity,
        group_id=0,
    )

    # -- Reproducibility metadata ---------------------------------------------
    manifest = {
        "status": "running",
        "variant": variant,
        "requested_steps": N_STEPS,
        "time_step_size": TIME_STEP_SIZE,
        "compute_device": solver.compute_device,
        "ring_radius": RING_RADIUS,
        "tube_circulation": RING_STRENGTH,
        "vortex_reynolds_number": REYNOLDS_NUMBER,
        "kinematic_viscosity": KINEMATIC_VISCOSITY,
        "initial_physical_core_radius": CORE_RADIUS,
        "particle_spacing": PARTICLE_SPACING,
        "particle_core_radius": particle_core_radius,
        "particle_count": len(position),
        "sample_interval_time": SAMPLE_INTERVAL_TIME,
        "backup_interval_time": BACKUP_INTERVAL_TIME,
        "backup": {"interval_steps": solver.setup.backup.interval_steps},
        "widnall_modes": WIDNALL_MODES,
        "widnall_rms_amplitude": DEFAULT_WIDNALL_AMPLITUDE,
        "treecode_theta": TREECODE_THETA,
        "smagorinsky_coefficient": smagorinsky_coefficient,
        "time_integration": "FRACTIONAL",
        "advection_scheme": "RK3",
        "stretching_scheme": "RK3",
        "viscous_scheme": "CS",
        "write_precision": "f32",
    }
    manifest_path = output_directory / f"run_manifest_{variant}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    # -- Time integration -----------------------------------------------------
    solver.record_diagnostics(refresh_fields=True)
    solver.save_backup()
    initial_strength = np.abs(solver.particles.vortex_strength_cpu()).max()
    termination_reason = None
    for _ in range(N_STEPS):
        solver.advance()
        if np.abs(solver.particles.vortex_strength_cpu()).max() > 50 * initial_strength:
            termination_reason = "peak particle strength exceeded 50 times its initial value"
            break
        if solver.step % cadence_steps(SAMPLE_INTERVAL_TIME, TIME_STEP_SIZE):
            continue
        health = solver._discretization_health
        divergence = float(health["vorticity_divergence_error"])
        misalignment = float(health["vortex_strength_misalignment_degrees"])
        if (
            divergence > RESOLUTION_DIVERGENCE_LIMIT
            or misalignment > RESOLUTION_MISALIGNMENT_LIMIT_DEG
        ):
            termination_reason = (
                "particle resolution lost: "
                f"divergence={divergence:.6g}, misalignment_deg={misalignment:.6g}"
            )
            break

    solver.save_backup()
    manifest.update(
        status="resolution_lost" if termination_reason else "completed",
        completed_steps=solver.step,
        completed_time=solver.time,
        n_particles_total=len(solver.particles),
    )
    if termination_reason:
        manifest["termination_reason"] = termination_reason
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        required=True,
        choices=("dns_direct", "dns_transposed", "dns_mixed", "les_transposed"),
    )
    parser.add_argument("--compute-device", default="AUTO")
    args = parser.parse_args()
    run_case(args.variant, compute_device=args.compute_device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
