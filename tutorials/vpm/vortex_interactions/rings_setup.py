#!/usr/bin/env python3
"""Transposed DNS/LES vortex-ring interactions and conditional stabilization.

The four primary cases compare co-rotating leapfrogging rings and
counter-rotating colliding rings with transposed DNS and calibrated transposed
LES.  A stabilized LES retry exists for each family, but ``allrun.sh`` launches
it only when the corresponding plain LES fails before the requested turbulent
transition horizon.  Stabilization is conservative filament refinement: it
splits only particles whose strength shows a clear lineage-relative overshoot;
it does not remesh or continuously relax the full cloud.

Usage:
    python rings_setup.py --case leapfrog_dns
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

import openonda.vpm as vpm

CASE_DIR = Path(__file__).resolve().parent

RING_RADIUS = 1.0
RING_CIRCULATION = np.pi
CORE_RADIUS = 0.1
KINEMATIC_VISCOSITY = RING_CIRCULATION / 3000.0

# Match the calibrated single-ring campaign at the same circulation Reynolds
# number.  This gives 5.71 intervals across the physical core diameter and
# about 7.9 particle spacings per wavelength near the predicted m=23 Widnall
# mode, while retaining complete azimuthal particle orbits.
PARTICLE_SPACING = 0.035
PARTICLE_RADIUS = 2.0 * PARTICLE_SPACING
TAIL_FRACTION = 0.05
# The nondimensional step is 20 h^2=0.0245.  The final horizon
# t Gamma/R^2=147 is long enough to include loss of coherent leapfrogging and
# the subsequent disordered motion if the numerical method survives it.
TIME_STEP_SIZE = 20.0 * PARTICLE_SPACING**2 / RING_CIRCULATION
NUM_STEPS = 6000
END_TIME = NUM_STEPS * TIME_STEP_SIZE
# Integral histories resolve the energy budget; particle snapshots resolve the
# ring motion for visualization. Their cadences are intentionally independent.
DIAGNOSTIC_INTERVAL_STEPS = 5
CHECKPOINT_INTERVAL_STEPS = 50

# The two fixed-particle baselines stop when vortex-line alignment or the
# reconstructed divergence says the cloud is no longer resolved.  Peak
# particle strength is intentionally not a stop: physical filament stretching
# can increase it.  Stabilized LES must reach END_TIME without this allowance.
BASELINE_MISALIGNMENT_LIMIT = 45.0
BASELINE_DIVERGENCE_LIMIT = 0.12

# Use the exact broadband perturbation calibrated by the single-ring case.
WIDNALL_AMPLITUDE = 0.05
WIDNALL_MODES = 24
RING_SEEDS = (7, 19)

LES_COEFFICIENT = {"leapfrog": 0.20, "collide": 0.20}
FILAMENT_REFINEMENT_INTERVAL_STEPS = 5
FILAMENT_REFINEMENT_STRENGTH_FACTOR = 2.0
FILAMENT_REFINEMENT_OFFSET_FRACTION = 0.25
STABILIZED_MAX_PARTICLES = 100_000
BASELINE_MAX_PARTICLES = 20_000

COMPUTE_DEVICE = os.environ.get("OPENONDA_COMPUTE_DEVICE", "METAL").upper()
# Metal does not expose f64 kernels in Taichi.  Keep the historical f64 path
# for CPU diagnostics, while making the canonical GPU campaign explicit and
# recording the reduced precision in each run manifest.
PRECISION = "f32" if COMPUTE_DEVICE == "METAL" else "f64"
NUM_STEPS = int(os.environ.get("OPENONDA_INTERACTIONS_NUM_STEPS", str(NUM_STEPS)))
END_TIME = NUM_STEPS * TIME_STEP_SIZE
VELOCITY_METHOD = os.environ.get("OPENONDA_INTERACTIONS_VELOCITY_METHOD", "treecode")
TREECODE_THETA = float(os.environ.get("OPENONDA_INTERACTIONS_TREECODE_THETA", "0.30"))

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
        stretching=vpm.StretchingConfig.transposed(
            scheme="RK2",
            conserve_moments=True,
            conserve_energy=True,
        ),
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
        precision=PRECISION,
        compute_device=COMPUTE_DEVICE,
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

    particle_spacings_per_wavelength = 2.0 * np.pi * RING_RADIUS / (22.6 * PARTICLE_SPACING)
    particle_spacings_across_core = 2.0 * CORE_RADIUS / PARTICLE_SPACING
    if (
        len(solver.particles) < 17_000
        or particle_spacings_per_wavelength < 7.5
        or particle_spacings_across_core < 5.5
    ):
        raise RuntimeError(
            "The particle cloud does not meet the Re=3000 Widnall-resolution gate: "
            f"N={len(solver.particles)}, wavelength/h={particle_spacings_per_wavelength:.3f}, "
            f"core_diameter/h={particle_spacings_across_core:.3f}"
        )

    manifest = {
        "status": "running",
        "case": case_name,
        "family": family,
        "model": variant,
        "requested_steps": n_steps,
        "requested_end_time": n_steps * TIME_STEP_SIZE,
        "diagnostic_interval_steps": DIAGNOSTIC_INTERVAL_STEPS,
        "checkpoint_interval_steps": CHECKPOINT_INTERVAL_STEPS,
        "baseline_misalignment_limit_deg": BASELINE_MISALIGNMENT_LIMIT,
        "baseline_divergence_limit": BASELINE_DIVERGENCE_LIMIT,
        "particle_spacing": PARTICLE_SPACING,
        "particle_core_radius": PARTICLE_RADIUS,
        "initial_n_particles_total": len(solver.particles),
        "precision": PRECISION,
        "compute_device": COMPUTE_DEVICE,
        "velocity_method": VELOCITY_METHOD,
        "treecode_theta": TREECODE_THETA if VELOCITY_METHOD == "treecode" else None,
        "widnall_amplitude": WIDNALL_AMPLITUDE,
        "widnall_modes": WIDNALL_MODES,
        "widnall_dominant_mode_estimate": 22.6,
        "particle_spacings_per_dominant_wavelength": particle_spacings_per_wavelength,
        "particle_spacings_across_core_diameter": particle_spacings_across_core,
        "smagorinsky_coefficient": 0.0 if variant == "dns" else LES_COEFFICIENT[family],
        "stabilization_mechanism": (
            "overshoot_gated_filament_refinement" if variant == "les_stabilized" else "disabled"
        ),
        "filament_refinement_interval_steps": (
            FILAMENT_REFINEMENT_INTERVAL_STEPS if variant == "les_stabilized" else 0
        ),
        "filament_refinement_strength_factor": (
            FILAMENT_REFINEMENT_STRENGTH_FACTOR if variant == "les_stabilized" else None
        ),
        "filament_refinement_offset_fraction": (
            FILAMENT_REFINEMENT_OFFSET_FRACTION if variant == "les_stabilized" else None
        ),
        "remeshing_enabled": False,
        "pedrizzetti_relaxation_enabled": False,
        "diffusion_scheme": "CS",
        "core_spreading_moment_projection": True,
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    solver.record_diagnostics(refresh_fields=True)
    solver.save_state(str(output_dir / f"vpm_{case_name}"))
    resolution_lost = False
    for _ in range(n_steps):
        solver.advance()
        if variant == "les_stabilized" or solver.step % DIAGNOSTIC_INTERVAL_STEPS:
            continue
        health = solver._discretization_health
        misalignment = float(health["vortex_strength_misalignment_degrees"])
        divergence = float(health["vorticity_divergence_error"])
        if misalignment <= BASELINE_MISALIGNMENT_LIMIT and divergence <= BASELINE_DIVERGENCE_LIMIT:
            continue
        resolution_lost = True
        print(
            "RESOLUTION LIMIT "
            f"step={solver.step} time={solver.time:.8e} "
            f"misalignment_deg={misalignment:.8e} "
            f"divergence={divergence:.8e}",
            flush=True,
        )
        break
    if not resolution_lost and solver.step % DIAGNOSTIC_INTERVAL_STEPS:
        solver.record_diagnostics(refresh_fields=True)
    solver.save_state(str(output_dir / f"vpm_{case_name}_final"))

    manifest.update(
        status="resolution_lost" if resolution_lost else "completed",
        completed_steps=solver.step,
        end_time=solver.time,
        particles=len(solver.particles),
    )
    if resolution_lost:
        manifest["termination_reason"] = (
            "vortex-line alignment or divergence exceeded the baseline resolution limit"
        )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        required=True,
        choices=tuple(CASES),
        help="vortex-ring interaction case to run",
    )
    args = parser.parse_args()
    run_case(args.case)


if __name__ == "__main__":
    main()
