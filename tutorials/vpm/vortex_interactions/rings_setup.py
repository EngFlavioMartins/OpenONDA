#!/usr/bin/env python3
"""DNS, LES, and stabilized-LES vortex-ring interactions (VPM).

Six cases, hard-coded as a matrix of two ring families (leapfrog co-rotating
rings, colliding counter-rotating rings) times three turbulence models
(DNS, LES, stabilized LES). The stabilized cases use a conservative
regularization filter to preserve resolution when the stretched cores
threaten to under-resolve the flow. ``plot_all.sh`` turns the sampled
diagnostics into the comparison figures.

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

RING_RADIUS = 1.0
RING_CIRCULATION = np.pi
CORE_RADIUS = 0.1
KINEMATIC_VISCOSITY = RING_CIRCULATION / 3000.0

# Five intervals across the physical core diameter produce 19 complete
# cross-section orbits per ring without changing the Gaussian core width.
PARTICLE_SPACING = 0.04
PARTICLE_RADIUS = 2.0 * PARTICLE_SPACING
TAIL_FRACTION = 0.05
# The nondimensional step is 0.032 and the final time is t Gamma/R^2 = 36.48.
TIME_STEP_SIZE = 20.0 * PARTICLE_SPACING**2 / RING_CIRCULATION
NUM_STEPS = 1140
END_TIME = NUM_STEPS * TIME_STEP_SIZE
# Integral histories resolve the energy budget; particle snapshots resolve the
# ring motion for visualization. Their cadences are intentionally independent.
DIAGNOSTIC_INTERVAL_STEPS = 5
CHECKPOINT_INTERVAL_STEPS = 10

# The two fixed-particle baselines stop when vortex-line alignment or the
# reconstructed divergence says the cloud is no longer resolved.  Peak
# particle strength is intentionally not a stop: physical filament stretching
# can increase it.  Stabilized LES must reach END_TIME without this allowance.
BASELINE_MISALIGNMENT_LIMIT = 45.0
BASELINE_DIVERGENCE_LIMIT = 0.12

# Modes 1--12 include the unstable Widnall band of this slender ring.
WIDNALL_AMPLITUDE = 0.01
WIDNALL_MODES = 12
RING_SEEDS = (7, 19)

# The classical coefficient resolves the sustained leapfrogging deformation;
# the more violent head-on collision needs the stronger coarse-LES filter.
LES_COEFFICIENT = {"leapfrog": 0.16, "collide": 0.32}
STABILIZATION_COEFFICIENT = 0.5
REGULARIZATION_INTERVAL_STEPS = 20
REGULARIZATION_START_STEP = 380
REGULARIZATION_SPACING = 0.084
REGULARIZATION_CAPACITY_SPACING = 0.13
REGULARIZATION_CORE_RADIUS = {"leapfrog": 0.23, "collide": 0.195}
REGULARIZATION_CAPACITY_CORE_RADIUS = 0.195
REGULARIZATION_TAIL_BUDGET = 0.003
REGULARIZATION_DIVERGENCE_TRIGGER = 0.20
# Act as vortex-line alignment first degrades; a solenoidal projection is only
# warranted if the rebuilt field remains above the baseline divergence warning.
REGULARIZATION_MISALIGNMENT_TRIGGER = 4.0
REGULARIZATION_CAPACITY_DIVERGENCE_TRIGGER = 0.20
REGULARIZATION_CAPACITY_MISALIGNMENT_TRIGGER = 25.0
REGULARIZATION_CAPACITY_FRACTION = 0.70
REGULARIZATION_ENERGY_LIMIT = 0.20
REGULARIZATION_ENSTROPHY_LIMIT = 0.15
REGULARIZATION_PROJECTION_TRIGGER = 0.12
REGULARIZATION_PROJECTION_LIMIT = {"leapfrog": 0.05, "collide": 0.10}
STABILIZED_MAX_PARTICLES = 20_000
BASELINE_MAX_PARTICLES = 8_000

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
    if variant == "les_stabilized":
        return vpm.StabilizationConfig.conservative_filter(
            coefficient=STABILIZATION_COEFFICIENT,
            interval_steps=REGULARIZATION_INTERVAL_STEPS,
            start_step=REGULARIZATION_START_STEP,
            grid_spacing=REGULARIZATION_SPACING,
            max_n_particles=STABILIZED_MAX_PARTICLES,
            tail_budget=REGULARIZATION_TAIL_BUDGET,
            divergence_trigger=REGULARIZATION_DIVERGENCE_TRIGGER,
            misalignment_trigger=REGULARIZATION_MISALIGNMENT_TRIGGER,
            capacity_divergence_trigger=REGULARIZATION_CAPACITY_DIVERGENCE_TRIGGER,
            capacity_misalignment_trigger=REGULARIZATION_CAPACITY_MISALIGNMENT_TRIGGER,
            capacity_fraction=REGULARIZATION_CAPACITY_FRACTION,
            capacity_grid_spacing=REGULARIZATION_CAPACITY_SPACING,
            core_radius=REGULARIZATION_CORE_RADIUS[family],
            capacity_core_radius=REGULARIZATION_CAPACITY_CORE_RADIUS,
            total_kinetic_energy_dissipation_limit=REGULARIZATION_ENERGY_LIMIT,
            total_enstrophy_dissipation_limit=REGULARIZATION_ENSTROPHY_LIMIT,
            projection_trigger=REGULARIZATION_PROJECTION_TRIGGER,
            projection_max_correction=REGULARIZATION_PROJECTION_LIMIT[family],
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
        velocity=vpm.VelocityConfig.direct(),
        particle_kernel="GAUSSIAN",
        precision="f64",
        compute_device="CPU",
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
    position, particle_volume, radius = vpm.ParticleDistributor.toroidal_distribution(
        RING_RADIUS,
        tube_radius,
        PARTICLE_SPACING,
        centre_position=centre,
        widnall_amplitude=WIDNALL_AMPLITUDE,
        seed=seed,
        n_widnall_modes=WIDNALL_MODES,
    )
    radius.fill(PARTICLE_RADIUS)
    _, kinematic_viscosity, strength = vpm.vortex_ring_vpm(
        kinematic_viscosity=KINEMATIC_VISCOSITY,
        ring_centre=centre,
        tube_circulation=circulation,
        ring_radius=RING_RADIUS,
        ring_core_radius=CORE_RADIUS,
        mean_core_radius=float(radius.mean()),
        position=position,
        particle_volume=particle_volume,
        widnall_amplitude=WIDNALL_AMPLITUDE,
        seed=seed,
        n_widnall_modes=WIDNALL_MODES,
        is_anti_diffusion_enabled=True,
        is_circulation_normalization_enabled=True,
    )
    return position, particle_volume, radius, kinematic_viscosity, strength


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
        position, particle_volume, radius, kinematic_viscosity, strength = ring_particles(
            centre, circulation, seed
        )
        solver.add_vortex_particles(
            position=position,
            velocity=np.zeros_like(position),
            vortex_strength=strength,
            core_radius=radius,
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
        "requested_end_time": n_steps * TIME_STEP_SIZE,
        "diagnostic_interval_steps": DIAGNOSTIC_INTERVAL_STEPS,
        "checkpoint_interval_steps": CHECKPOINT_INTERVAL_STEPS,
        "baseline_misalignment_limit_deg": BASELINE_MISALIGNMENT_LIMIT,
        "baseline_divergence_limit": BASELINE_DIVERGENCE_LIMIT,
        "particle_spacing": PARTICLE_SPACING,
        "particle_core_radius": PARTICLE_RADIUS,
        "initial_n_particles_total": len(solver.particles),
        "precision": "f64",
        "widnall_amplitude": WIDNALL_AMPLITUDE,
        "widnall_modes": WIDNALL_MODES,
        "smagorinsky_coefficient": 0.0 if variant == "dns" else LES_COEFFICIENT[family],
        "stabilization_coefficient": (
            STABILIZATION_COEFFICIENT if variant == "les_stabilized" else 0.0
        ),
        "regularization_interval_steps": (
            REGULARIZATION_INTERVAL_STEPS if variant == "les_stabilized" else 0
        ),
        "regularization_grid_spacing": (
            REGULARIZATION_SPACING if variant == "les_stabilized" else None
        ),
        "regularization_capacity_grid_spacing": (
            REGULARIZATION_CAPACITY_SPACING if variant == "les_stabilized" else None
        ),
        "regularization_core_radius": (
            REGULARIZATION_CORE_RADIUS[family] if variant == "les_stabilized" else None
        ),
        "regularization_capacity_core_radius": (
            REGULARIZATION_CAPACITY_CORE_RADIUS if variant == "les_stabilized" else None
        ),
        "regularization_capacity_fraction": (
            REGULARIZATION_CAPACITY_FRACTION if variant == "les_stabilized" else None
        ),
        "regularization_tail_budget": (
            REGULARIZATION_TAIL_BUDGET if variant == "les_stabilized" else 0.0
        ),
        "regularization_misalignment_trigger_deg": (
            REGULARIZATION_MISALIGNMENT_TRIGGER if variant == "les_stabilized" else None
        ),
        "regularization_projection_trigger": (
            REGULARIZATION_PROJECTION_TRIGGER if variant == "les_stabilized" else None
        ),
        "regularization_projection_limit": (
            REGULARIZATION_PROJECTION_LIMIT[family] if variant == "les_stabilized" else None
        ),
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
