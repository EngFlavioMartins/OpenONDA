#!/usr/bin/env python3
"""DNS, LES and stabilized LES for two interacting vortex rings.

The rings either leapfrog or collide. Stabilized LES adds filament splitting
to the calibrated LES model used by the single-ring case.

Usage:
    python rings_setup.py --case leapfrog_dns
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import openonda.vpm as vpm

CASE_DIR = Path(__file__).resolve().parent

# Physics
RING_RADIUS = 1.0  # ring major radius [m]
RING_CIRCULATION = np.pi  # tube circulation [m²/s]
CORE_RADIUS = 0.1  # initial Gaussian core radius [m]
KINEMATIC_VISCOSITY = RING_CIRCULATION / 3000.0  # ν = Γ/Re [m²/s], Re = 3000

# Numerics match the single-ring calibration.
PARTICLE_SPACING = 0.035  # in-plane particle spacing [m]
PARTICLE_RADIUS = 2.0 * PARTICLE_SPACING  # Gaussian blob radius = 2h [m]
TAIL_FRACTION = 0.05  # toroidal particle distribution tail fraction
TIME_STEP_SIZE = 20.0 * PARTICLE_SPACING**2 / RING_CIRCULATION  # Δt = 20 h²/Γ [s]
NUM_STEPS = 6000  # total number of time steps
DIAGNOSTIC_INTERVAL_STEPS = 5  # flow-integral / ring-diagnostic cadence
CHECKPOINT_INTERVAL_STEPS = 50  # HDF5/XDMF particle-snapshot cadence

WIDNALL_AMPLITUDE = 0.05  # broadband centreline perturbation amplitude
WIDNALL_MODES = 24  # number of azimuthal bending modes
RING_SEEDS = (7, 19)  # reproducible per-ring perturbation phases

LES_COEFFICIENT = {"leapfrog": 0.20, "collide": 0.20}  # Smagorinsky C_s per family
FILAMENT_REFINEMENT_INTERVAL_STEPS = 25  # overshoot-splitting check cadence
FILAMENT_REFINEMENT_STRENGTH_FACTOR = 3.0  # strength multiple that triggers a split
FILAMENT_REFINEMENT_OFFSET_FRACTION = 0.25  # transverse offset of the split cloud
STABILIZED_MAX_PARTICLES = 100_000  # particle-count guard for stabilized LES
BASELINE_MAX_PARTICLES = 20_000  # particle-count guard for fixed-particle baselines

VELOCITY_METHOD = "treecode"
TREECODE_THETA = 0.30  # treecode accuracy parameter

CASES = {
    "leapfrog_dns": ("leapfrog", "dns"),
    "leapfrog_les": ("leapfrog", "les"),
    "leapfrog_les_stabilized": ("leapfrog", "les_stabilized"),
    "collide_dns": ("collide", "dns"),
    "collide_les": ("collide", "les"),
    "collide_les_stabilized": ("collide", "les_stabilized"),
}


def ring_geometry(family: str) -> tuple[tuple[float, float], tuple[float, float]]:
    if family == "leapfrog":
        return (-0.5, 0.5), (RING_CIRCULATION, RING_CIRCULATION)
    return (-2.5, 2.5), (RING_CIRCULATION, -RING_CIRCULATION)


def turbulence(family: str, variant: str) -> vpm.TurbulenceConfig:
    if variant == "dns":
        return vpm.TurbulenceConfig.dns()
    return vpm.TurbulenceConfig.les_smagorinsky(smagorinsky_coefficient=LES_COEFFICIENT[family])


def stabilization(family: str, variant: str) -> vpm.StabilizationConfig:
    del family
    if variant == "les_stabilized":
        return vpm.StabilizationConfig(
            filament_refinement=vpm.FilamentRefinementConfig.adaptive(
                interval_steps=FILAMENT_REFINEMENT_INTERVAL_STEPS,
                max_vortex_strength_factor=FILAMENT_REFINEMENT_STRENGTH_FACTOR,
                offset_fraction=FILAMENT_REFINEMENT_OFFSET_FRACTION,
                max_n_particles=STABILIZED_MAX_PARTICLES,
            )
        )
    return vpm.StabilizationConfig.disabled()


def viscous_diffusion() -> vpm.ViscousConfig:
    return vpm.ViscousConfig.cs(
        kinematic_viscosity=KINEMATIC_VISCOSITY,
        particle_spacing=PARTICLE_SPACING,
    )


def solver_setup(case_name: str, output_dir: Path) -> vpm.VPMSetup:
    family, variant = CASES[case_name]
    return vpm.VPMSetup(
        time_step_size=TIME_STEP_SIZE,
        time_integration="COUPLED",
        coupled_max_strain_increment=0.15,
        coupled_max_advection_fraction=0.5,
        advection=vpm.AdvectionConfig(scheme="RK2"),
        stretching=vpm.StretchingConfig.transposed(scheme="RK2"),
        viscous=viscous_diffusion(),
        turbulence=turbulence(family, variant),
        stabilization=stabilization(family, variant),
        velocity=(
            vpm.VelocityConfig.treecode(
                theta=TREECODE_THETA,
                sort_particle_targets=True,
                traversal_block_dim=128,
            )
            if VELOCITY_METHOD == "treecode"
            else vpm.VelocityConfig.direct()
        ),
        particle_kernel="GAUSSIAN",
        write_precision="f32",
        checkpoint_store_velocity_gradient=False,
        max_n_particles=(
            STABILIZED_MAX_PARTICLES if variant == "les_stabilized" else BASELINE_MAX_PARTICLES
        ),
        checkpoint_directory=str(output_dir),
        checkpoint_name=case_name,
        sample_subdirectory=case_name,
        checkpoint_interval_steps=CHECKPOINT_INTERVAL_STEPS,
        logging_interval_steps=DIAGNOSTIC_INTERVAL_STEPS,
        timing_interval_steps=200,
        export_flow_integrals=True,
        samplers=(vpm.RingDiagnosticsSampler(),),
        log_mode="file",
    )


def ring_particles(
    centre_x: float,
    circulation: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    represented_core = np.sqrt(CORE_RADIUS**2 - PARTICLE_RADIUS**2)
    tube_radius = represented_core * np.sqrt(-np.log(TAIL_FRACTION))
    centre = np.array([centre_x, 0.0, 0.0])
    position, particle_volume, core_radius = vpm.ParticleDistributor.toroidal_distribution(
        RING_RADIUS,
        tube_radius,
        PARTICLE_SPACING,
        centre_position=centre,
        widnall_amplitude=WIDNALL_AMPLITUDE,
        seed=seed,
        n_widnall_modes=WIDNALL_MODES,
    )
    core_radius.fill(PARTICLE_RADIUS)
    _, kinematic_viscosity, vortex_strength = vpm.vortex_ring_vpm(
        kinematic_viscosity=KINEMATIC_VISCOSITY,
        ring_centre=centre,
        tube_circulation=circulation,
        ring_radius=RING_RADIUS,
        ring_core_radius=CORE_RADIUS,
        mean_core_radius=float(core_radius.mean()),
        position=position,
        particle_volume=particle_volume,
        widnall_amplitude=WIDNALL_AMPLITUDE,
        seed=seed,
        n_widnall_modes=WIDNALL_MODES,
        is_anti_diffusion_enabled=True,
        is_circulation_normalization_enabled=True,
    )
    return position, particle_volume, core_radius, kinematic_viscosity, vortex_strength


def run_case(case_name: str, *, n_steps: int = NUM_STEPS) -> None:
    family, variant = CASES[case_name]
    output_dir = CASE_DIR / "solution" / case_name
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing results in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n---- " + case_name + " ----")
    print(f"  family={family}, model={variant}, steps={n_steps}")

    solver = vpm.VPMSolver(setup=solver_setup(case_name, output_dir), case_dir=CASE_DIR)
    centres, circulations = ring_geometry(family)
    for group, (centre, circulation, seed) in enumerate(
        zip(centres, circulations, RING_SEEDS, strict=True)
    ):
        position, particle_volume, core_radius, kinematic_viscosity, vortex_strength = (
            ring_particles(centre, circulation, seed)
        )
        solver.add_vortex_particles(
            position=position,
            velocity=np.zeros_like(position),
            vortex_strength=vortex_strength,
            core_radius=core_radius,
            particle_volume=particle_volume,
            kinematic_viscosity=kinematic_viscosity,
            group_id=np.full(len(position), group, dtype=np.int32),
        )

    manifest = {
        "status": "running",
        "case": case_name,
        "family": family,
        "model": variant,
        "requested_steps": n_steps,
        "diagnostic_interval_steps": DIAGNOSTIC_INTERVAL_STEPS,
        "checkpoint_interval_steps": CHECKPOINT_INTERVAL_STEPS,
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    solver.record_diagnostics(refresh_fields=True)
    solver.save_state(str(output_dir / f"vpm_{case_name}"))
    initial_strength = np.linalg.norm(solver.particles.vortex_strength_cpu(), axis=1).max()
    stopped = False
    for _ in range(n_steps):
        solver.advance()
        if solver.step % DIAGNOSTIC_INTERVAL_STEPS:
            continue
        peak_strength = np.linalg.norm(solver.particles.vortex_strength_cpu(), axis=1).max()
        if not np.isfinite(peak_strength) or peak_strength > 50 * initial_strength:
            print(f"Stopping {case_name} at step {solver.step}: particle strength diverged.")
            stopped = True
            break
    if solver.step % DIAGNOSTIC_INTERVAL_STEPS:
        solver.record_diagnostics(refresh_fields=True)
    solver.save_state(str(output_dir / f"vpm_{case_name}_final"))

    manifest.update(
        status="stopped" if stopped else "completed",
        completed_steps=solver.step,
        end_time=solver.time,
        n_particles_total=len(solver.particles),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        required=True,
        choices=tuple(CASES),
        help="vortex-ring interaction case to run",
    )
    args = parser.parse_args()
    run_case(args.case)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
