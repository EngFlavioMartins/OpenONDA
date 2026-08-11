"""DNS, LES, and stabilized-LES vortex-ring interactions."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

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

CASE_DIR = Path(__file__).resolve().parent

RING_RADIUS = 1.0
RING_CIRCULATION = np.pi
CORE_RADIUS = 0.1
KINEMATIC_VISCOSITY = RING_CIRCULATION / 3000.0

PARTICLE_SPACING = 0.045
PARTICLE_RADIUS = 2.0 * PARTICLE_SPACING
TAIL_FRACTION = 0.05
TIME_STEP = 20.0 * PARTICLE_SPACING**2 / RING_CIRCULATION
END_TIME = 11.55
NUM_STEPS = int(np.ceil(END_TIME / TIME_STEP))
OUTPUT_FREQUENCY = 20

# The two fixed-particle baselines stop when vortex-line alignment or the
# reconstructed divergence says the cloud is no longer resolved.  Peak
# particle strength is intentionally not a stop: physical filament stretching
# can increase it.  Stabilized LES must reach END_TIME without this allowance.
BASELINE_MISALIGNMENT_LIMIT = 45.0
BASELINE_DIVERGENCE_LIMIT = 0.25

# Modes 1--12 include the unstable Widnall band of this slender ring.
WIDNALL_AMPLITUDE = 0.01
WIDNALL_MODES = 12
RING_SEEDS = (7, 19)

# Deliberately strong coarse-LES damping makes the model hierarchy observable.
LES_COEFFICIENT = 0.32
STABILIZATION_COEFFICIENT = 0.5
REGULARIZATION_FREQUENCY = OUTPUT_FREQUENCY
REGULARIZATION_START_STEP = 300
REGULARIZATION_SPACING = 0.084
REGULARIZATION_TAIL_BUDGET = 0.003
REGULARIZATION_DIVERGENCE_TRIGGER = 0.20
REGULARIZATION_CAPACITY_DIVERGENCE_TRIGGER = 0.25
REGULARIZATION_CAPACITY_MISALIGNMENT_TRIGGER = 25.0
REGULARIZATION_ENERGY_LIMIT = 0.30
REGULARIZATION_ENSTROPHY_LIMIT = 0.25
STABILIZED_MAX_PARTICLES = 20_000

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


def turbulence(variant: str) -> TurbulenceConfig:
    if variant == "dns":
        return TurbulenceConfig.dns()
    return TurbulenceConfig.les_smagorinsky(cs=LES_COEFFICIENT)


def stabilization(variant: str) -> StabilizationConfig:
    if variant == "les_stabilized":
        return StabilizationConfig.conservative_filter(
            coefficient=STABILIZATION_COEFFICIENT,
            frequency=REGULARIZATION_FREQUENCY,
            start_step=REGULARIZATION_START_STEP,
            grid_spacing=REGULARIZATION_SPACING,
            max_particles=STABILIZED_MAX_PARTICLES,
            tail_budget=REGULARIZATION_TAIL_BUDGET,
            divergence_trigger=REGULARIZATION_DIVERGENCE_TRIGGER,
            capacity_divergence_trigger=REGULARIZATION_CAPACITY_DIVERGENCE_TRIGGER,
            capacity_misalignment_trigger=REGULARIZATION_CAPACITY_MISALIGNMENT_TRIGGER,
            energy_dissipation_limit=REGULARIZATION_ENERGY_LIMIT,
            enstrophy_dissipation_limit=REGULARIZATION_ENSTROPHY_LIMIT,
        )
    return StabilizationConfig.disabled()


def viscous_diffusion(variant: str) -> ViscousConfig:
    return ViscousConfig.cs(
        viscosity=KINEMATIC_VISCOSITY,
        characteristic_distance=PARTICLE_SPACING,
    )


def solver_setup(case_name: str, output_dir: Path) -> VPMSetup:
    _, variant = CASES[case_name]
    return VPMSetup(
        time_step_size=TIME_STEP,
        time_integration="COUPLED",
        coupled_max_strain_increment=0.15,
        coupled_max_advection_fraction=0.5,
        advection=AdvectionConfig(scheme="RK2"),
        stretching=StretchingConfig.transposed(
            scheme="RK2",
            conserve_moments=True,
            conserve_energy=True,
        ),
        viscous=viscous_diffusion(variant),
        turbulence=turbulence(variant),
        stabilization=stabilization(variant),
        velocity=VelocityConfig.direct(),
        particles_kernel="GAUSSIAN",
        processing_unit="CPU",
        max_particles=STABILIZED_MAX_PARTICLES if variant == "les_stabilized" else 5_000,
        backup_directory=str(output_dir),
        backup_file_name=case_name,
        backup_frequency=5 * OUTPUT_FREQUENCY,
        logging_frequency=OUTPUT_FREQUENCY,
        timing_frequency=10 * OUTPUT_FREQUENCY,
        export_flow_integrals=True,
        log_mode="file",
    )


def ring_particles(
    center_x: float,
    circulation: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    represented_core = np.sqrt(CORE_RADIUS**2 - PARTICLE_RADIUS**2)
    tube_radius = represented_core * np.sqrt(-np.log(TAIL_FRACTION))
    center = np.array([center_x, 0.0, 0.0])
    position, volume, radius = ParticleDistributor.toroidal_distribution(
        RING_RADIUS,
        tube_radius,
        PARTICLE_SPACING,
        center=center,
        epsilon_w=WIDNALL_AMPLITUDE,
        seed=seed,
        max_modes=WIDNALL_MODES,
    )
    radius.fill(PARTICLE_RADIUS)
    _, viscosity, strength = VortexRingVPM(
        viscosity=KINEMATIC_VISCOSITY,
        ring_center=center,
        ring_strength=circulation,
        ring_radius=RING_RADIUS,
        ring_thickness=CORE_RADIUS,
        avg_particle_radius=float(radius.mean()),
        positions=position,
        volumes=volume,
        epsilon_W=WIDNALL_AMPLITUDE,
        seed=seed,
        max_modes=WIDNALL_MODES,
        anti_diffuse_flag=True,
        normalize_circulation=True,
    )
    return position, volume, radius, viscosity, strength


def run_case(case_name: str, *, num_steps: int = NUM_STEPS) -> None:
    family, variant = CASES[case_name]
    output_dir = CASE_DIR / "solution" / case_name
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing results in {output_dir}")

    solver = Solver(setup=solver_setup(case_name, output_dir))
    centers, circulations = ring_geometry(family)
    for group, (center, circulation, seed) in enumerate(
        zip(centers, circulations, RING_SEEDS, strict=True)
    ):
        position, volume, radius, viscosity, strength = ring_particles(center, circulation, seed)
        solver.add_vortex_particles(
            position=position,
            velocity=np.zeros_like(position),
            circulation=strength,
            radius=radius,
            volume=volume,
            viscosity=viscosity,
            group_id=np.full(len(position), group, dtype=np.int32),
        )

    manifest = {
        "status": "running",
        "case": case_name,
        "family": family,
        "model": variant,
        "requested_steps": num_steps,
        "requested_end_time": num_steps * TIME_STEP,
        "baseline_misalignment_limit_deg": BASELINE_MISALIGNMENT_LIMIT,
        "baseline_divergence_limit": BASELINE_DIVERGENCE_LIMIT,
        "particle_spacing": PARTICLE_SPACING,
        "widnall_amplitude": WIDNALL_AMPLITUDE,
        "widnall_modes": WIDNALL_MODES,
        "smagorinsky_coefficient": 0.0 if variant == "dns" else LES_COEFFICIENT,
        "stabilization_coefficient": (
            STABILIZATION_COEFFICIENT if variant == "les_stabilized" else 0.0
        ),
        "regularization_frequency": (
            REGULARIZATION_FREQUENCY if variant == "les_stabilized" else 0
        ),
        "regularization_grid_spacing": (
            REGULARIZATION_SPACING if variant == "les_stabilized" else None
        ),
        "regularization_tail_budget": (
            REGULARIZATION_TAIL_BUDGET if variant == "les_stabilized" else 0.0
        ),
        "diffusion_scheme": "CS",
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    solver.record_diagnostics(refresh_fields=True)
    resolution_lost = False
    for _ in range(num_steps):
        solver.update_state()
        if variant == "les_stabilized" or solver.time_step % OUTPUT_FREQUENCY:
            continue
        health = solver._discretization_health
        misalignment = float(health["strength_misalignment_deg"])
        divergence = float(health["vorticity_divergence_error"])
        if (
            misalignment <= BASELINE_MISALIGNMENT_LIMIT
            and divergence <= BASELINE_DIVERGENCE_LIMIT
        ):
            continue
        resolution_lost = True
        print(
            "RESOLUTION LIMIT "
            f"step={solver.time_step} time={solver.flow_time:.8e} "
            f"misalignment_deg={misalignment:.8e} "
            f"divergence={divergence:.8e}",
            flush=True,
        )
        break
    if not resolution_lost and solver.time_step % OUTPUT_FREQUENCY:
        solver.record_diagnostics(refresh_fields=True)
    solver.save_state(str(output_dir / f"vpm_{case_name}_final"))

    manifest.update(
        status="resolution_lost" if resolution_lost else "completed",
        completed_steps=solver.time_step,
        end_time=solver.flow_time,
        particles=len(solver.particles),
    )
    if resolution_lost:
        manifest["termination_reason"] = (
            "vortex-line alignment or divergence exceeded the baseline resolution limit"
        )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in CASES:
        choices = " | ".join(CASES)
        raise SystemExit(f"Usage: python rings_setup.py CASE\nCASE = {choices}")
    run_case(sys.argv[1])


if __name__ == "__main__":
    main()
