#!/usr/bin/env python3
"""LES stabilization comparison for two interacting vortex rings.

Leapfrogging compares filament splitting, vortex realignment, sparse
conservative remeshing, and their combined use. The best method is then
applied to the colliding rings. DNS cases remain available as historical
baselines but are not included in the reported comparison.

Usage:
    python rings_setup.py --case leapfrog_les_realignment
"""

from __future__ import annotations

import argparse
from functools import cache
import json
import os
from pathlib import Path

import numpy as np

# Keep one energy definition through the combined case's particle ceiling.
os.environ.setdefault("OPENONDA_VPM_DIRECT_INTEGRAL_LIMIT", "120000")
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
NUM_STEPS = 1200  # resolves coherent passes and the breakdown near x/R0 = 7
DIAGNOSTIC_INTERVAL_STEPS = 5  # flow-integral / ring-diagnostic cadence
CHECKPOINT_INTERVAL_STEPS = 50  # HDF5/XDMF particle-snapshot cadence

WIDNALL_AMPLITUDE = 0.025  # validated broadband centreline perturbation amplitude
WIDNALL_MODES = 24  # number of azimuthal bending modes
RING_SEEDS = (7, 19)  # reproducible per-ring perturbation phases

LES_COEFFICIENT = {"leapfrog": 0.20, "collide": 0.20}  # Smagorinsky C_s per family
VORTEX_STRETCHING_SFS_COEFFICIENT = 0.001  # three-level no-backscatter scale
RVPM_RELAXATION_FACTOR = 0.05  # Pedrizzetti direction relaxation
RVPM_RELAXATION_INTERVAL_STEPS = 5
SPLITTING_INTERVAL_STEPS = 25
SPLITTING_STRENGTH_FACTOR = 3.0
SPLITTING_OFFSET_FRACTION = 0.25
REALIGNMENT_FACTOR = 0.005
REALIGNMENT_INTERVAL_STEPS = 25
COMBINED_REALIGNMENT_FACTOR = 0.005
COMBINED_REALIGNMENT_INTERVAL_STEPS = 25
COMBINED_REALIGNMENT_START_STEP = 0
COMBINED_REALIGNMENT_END_STEP = 675
COMBINED_SPLITTING_INTERVAL_STEPS = 25
COMBINED_SPLITTING_LATE_INTERVAL_STEPS = 25
COMBINED_SPLITTING_LATE_START_STEP = 750
COMBINED_ABSOLUTE_SPLITTING_FACTOR = 40.0
COMBINED_STRETCHING_VISCOSITY_COEFFICIENT = 0.0
COMBINED_STRETCHING_VISCOSITY_START_STEP = 550
COMBINED_STRETCHING_VISCOSITY_FEEDBACK_GAIN = 0.0
COMBINED_STRETCHING_VISCOSITY_FEEDBACK_GROWTH_LIMIT = 0.50
COMBINED_STRETCHING_VISCOSITY_MAX_COEFFICIENT = 0.0
REMESH_START_STEP = {"leapfrog": 200, "collide": 550}
REMESH_INTERVAL_STEPS = {"leapfrog": 150, "collide": 250}
REMESH_GRID_SPACING = 0.040
REMESH_CAPACITY_GRID_SPACING = 0.045
REMESH_CAPACITY_FRACTION = 0.40
REMESH_CORE_RADIUS = None  # use the grid-consistent Gaussian core
REMESH_TAIL_BUDGET = 0.001
REMESH_DIVERGENCE_TRIGGER = {"leapfrog": 0.075, "collide": 0.25}
REMESH_MISALIGNMENT_TRIGGER = {"leapfrog": 26.0, "collide": 20.0}
REMESH_MAX_PARTICLES = 60_000
COMBINED_REMESH_START_STEP = 475
COMBINED_REMESH_INTERVAL_STEPS = 25
COMBINED_REMESH_GRID_SPACING = 0.055
COMBINED_CAPACITY_GRID_SPACING = 0.055
COMBINED_REMESH_MAX_PARTICLES = 30_000
COMBINED_CAPACITY_MAX_PARTICLES = 45_000
COMBINED_REMESH_MAX_EVENTS = 2
COMBINED_SPLITTING_MAX_PARTICLES = 120_000
COMBINED_SPLITTING_END_STEP = 1_500
COMBINED_CAPACITY_FRACTION = 0.90
COMBINED_REMESH_CORE_RADIUS = None
COMBINED_DIVERGENCE_TRIGGER = 0.22
COMBINED_MISALIGNMENT_TRIGGER = 35.0
COMBINED_CAPACITY_DIVERGENCE_TRIGGER = 0.30
COMBINED_CAPACITY_MISALIGNMENT_TRIGGER = 35.0
COMBINED_CAPACITY_ENERGY_RATE_TRIGGER = 4.0
COMBINED_ENERGY_DISSIPATION_LIMIT = 0.05
COMBINED_PROJECTION_TRIGGER = 10.0
STABILIZED_MAX_PARTICLES = 100_000
BASELINE_MAX_PARTICLES = 20_000  # particle-count guard for fixed-particle baselines

VELOCITY_METHOD = "treecode"
TREECODE_THETA = 0.30  # treecode accuracy parameter

CASES = {
    "leapfrog_dns": ("leapfrog", "dns"),
    "leapfrog_les": ("leapfrog", "les"),
    "leapfrog_les_splitting": ("leapfrog", "les_splitting"),
    "leapfrog_les_realignment": ("leapfrog", "les_realignment"),
    "leapfrog_les_rvpm": ("leapfrog", "les_rvpm"),
    "leapfrog_les_rvpm_sfs": ("leapfrog", "les_rvpm_sfs"),
    "leapfrog_les_remeshing": ("leapfrog", "les_remeshing"),
    "leapfrog_les_combined": ("leapfrog", "les_combined"),
    "collide_dns": ("collide", "dns"),
    "collide_les": ("collide", "les"),
    "collide_les_splitting": ("collide", "les_splitting"),
    "collide_les_realignment": ("collide", "les_realignment"),
    "collide_les_rvpm": ("collide", "les_rvpm"),
    "collide_les_rvpm_sfs": ("collide", "les_rvpm_sfs"),
    "collide_les_combined": ("collide", "les_combined"),
}


def ring_geometry(family: str) -> tuple[tuple[float, float], tuple[float, float]]:
    if family == "leapfrog":
        return (-0.5, 0.5), (RING_CIRCULATION, RING_CIRCULATION)
    return (-2.5, 2.5), (RING_CIRCULATION, -RING_CIRCULATION)


def turbulence(family: str, variant: str) -> vpm.TurbulenceConfig:
    if variant == "dns":
        return vpm.TurbulenceConfig.dns()
    if variant == "les_rvpm_sfs":
        return vpm.TurbulenceConfig.les_smagorinsky(
            smagorinsky_coefficient=0.0,
            vortex_stretching_sfs_coefficient=VORTEX_STRETCHING_SFS_COEFFICIENT,
        )
    return vpm.TurbulenceConfig.les_smagorinsky(smagorinsky_coefficient=LES_COEFFICIENT[family])


def stabilization(family: str, variant: str) -> vpm.StabilizationConfig:
    if variant == "les_rvpm_sfs":
        return vpm.StabilizationConfig.pedrizzetti_relaxation(
            factor=RVPM_RELAXATION_FACTOR,
            interval_steps=RVPM_RELAXATION_INTERVAL_STEPS,
            preserve_vortex_strength=True,
            preserve_moments=True,
        )
    if variant == "les_splitting":
        return vpm.StabilizationConfig(
            filament_refinement=vpm.FilamentRefinementConfig.adaptive(
                interval_steps=SPLITTING_INTERVAL_STEPS,
                max_vortex_strength_factor=SPLITTING_STRENGTH_FACTOR,
                offset_fraction=SPLITTING_OFFSET_FRACTION,
                max_n_particles=STABILIZED_MAX_PARTICLES,
            )
        )
    if variant == "les_realignment":
        return vpm.StabilizationConfig.pedrizzetti_relaxation(
            factor=REALIGNMENT_FACTOR,
            interval_steps=REALIGNMENT_INTERVAL_STEPS,
            preserve_vortex_strength=True,
        )
    if variant == "les_remeshing":
        return vpm.StabilizationConfig.conservative_filter(
            coefficient=0.0,
            interval_steps=REMESH_INTERVAL_STEPS[family],
            start_step=REMESH_START_STEP[family],
            grid_spacing=REMESH_GRID_SPACING,
            max_n_particles=REMESH_MAX_PARTICLES,
            tail_budget=REMESH_TAIL_BUDGET,
            total_kinetic_energy_dissipation_limit=0.10,
            total_enstrophy_dissipation_limit=0.10,
            divergence_trigger=REMESH_DIVERGENCE_TRIGGER[family],
            misalignment_trigger=REMESH_MISALIGNMENT_TRIGGER[family],
            capacity_fraction=REMESH_CAPACITY_FRACTION,
            capacity_grid_spacing=REMESH_CAPACITY_GRID_SPACING,
            core_radius=REMESH_CORE_RADIUS,
            projection_trigger=0.12,
            projection_max_correction=0.10,
        )
    if variant == "les_combined":
        return vpm.StabilizationConfig(
            stretching_viscosity_coefficient=(
                COMBINED_STRETCHING_VISCOSITY_COEFFICIENT
            ),
            stretching_viscosity_start_step=(
                COMBINED_STRETCHING_VISCOSITY_START_STEP
            ),
            stretching_viscosity_feedback_gain=(
                COMBINED_STRETCHING_VISCOSITY_FEEDBACK_GAIN
            ),
            stretching_viscosity_feedback_interval_steps=(
                DIAGNOSTIC_INTERVAL_STEPS
            ),
            stretching_viscosity_feedback_growth_limit=(
                COMBINED_STRETCHING_VISCOSITY_FEEDBACK_GROWTH_LIMIT
            ),
            stretching_viscosity_max_coefficient=(
                COMBINED_STRETCHING_VISCOSITY_MAX_COEFFICIENT
            ),
            pedrizzetti_relaxation_factor=COMBINED_REALIGNMENT_FACTOR,
            pedrizzetti_relaxation_interval_steps=(
                COMBINED_REALIGNMENT_INTERVAL_STEPS
            ),
            pedrizzetti_relaxation_start_step=COMBINED_REALIGNMENT_START_STEP,
            pedrizzetti_relaxation_end_step=COMBINED_REALIGNMENT_END_STEP,
            pedrizzetti_relaxation_preserve_vortex_strength=True,
            filament_refinement=vpm.FilamentRefinementConfig.adaptive(
                interval_steps=COMBINED_SPLITTING_INTERVAL_STEPS,
                max_vortex_strength_factor=SPLITTING_STRENGTH_FACTOR,
                offset_fraction=SPLITTING_OFFSET_FRACTION,
                max_n_particles=COMBINED_SPLITTING_MAX_PARTICLES,
                max_absolute_vortex_strength=(
                    COMBINED_ABSOLUTE_SPLITTING_FACTOR * initial_peak_strength(family)
                ),
                late_interval_steps=COMBINED_SPLITTING_LATE_INTERVAL_STEPS,
                late_start_step=COMBINED_SPLITTING_LATE_START_STEP,
                late_absolute_only=True,
                end_step=COMBINED_SPLITTING_END_STEP,
            ),
            regularization_interval_steps=COMBINED_REMESH_INTERVAL_STEPS,
            regularization_start_step=COMBINED_REMESH_START_STEP,
            regularization_grid_spacing=COMBINED_REMESH_GRID_SPACING,
            regularization_tail_budget=REMESH_TAIL_BUDGET,
            regularization_max_particles=COMBINED_REMESH_MAX_PARTICLES,
            regularization_capacity_max_particles=(
                COMBINED_CAPACITY_MAX_PARTICLES
            ),
            regularization_max_events=COMBINED_REMESH_MAX_EVENTS,
            regularization_total_kinetic_energy_dissipation_limit=(
                COMBINED_ENERGY_DISSIPATION_LIMIT
            ),
            regularization_total_enstrophy_dissipation_limit=0.10,
            regularization_divergence_trigger=COMBINED_DIVERGENCE_TRIGGER,
            regularization_misalignment_trigger=COMBINED_MISALIGNMENT_TRIGGER,
            regularization_capacity_divergence_trigger=(
                COMBINED_CAPACITY_DIVERGENCE_TRIGGER
            ),
            regularization_capacity_misalignment_trigger=(
                COMBINED_CAPACITY_MISALIGNMENT_TRIGGER
            ),
            regularization_capacity_energy_rate_trigger=(
                COMBINED_CAPACITY_ENERGY_RATE_TRIGGER
            ),
            regularization_capacity_fraction=COMBINED_CAPACITY_FRACTION,
            regularization_capacity_grid_spacing=COMBINED_CAPACITY_GRID_SPACING,
            regularization_core_radius=COMBINED_REMESH_CORE_RADIUS,
            regularization_capacity_core_radius=COMBINED_REMESH_CORE_RADIUS,
            regularization_projection_trigger=COMBINED_PROJECTION_TRIGGER,
            regularization_projection_max_correction=0.10,
        )
    return vpm.StabilizationConfig.disabled()


def viscous_diffusion() -> vpm.ViscousConfig:
    return vpm.ViscousConfig.cs(
        kinematic_viscosity=KINEMATIC_VISCOSITY,
        particle_spacing=PARTICLE_SPACING,
    )


def solver_setup(case_name: str, output_dir: Path) -> vpm.VPMSetup:
    family, variant = CASES[case_name]
    max_particles = {
        "les_splitting": STABILIZED_MAX_PARTICLES,
        "les_remeshing": REMESH_MAX_PARTICLES,
        "les_combined": COMBINED_SPLITTING_MAX_PARTICLES,
    }.get(variant, BASELINE_MAX_PARTICLES)
    projected_stretching = variant in {"les_combined", "les_rvpm", "les_rvpm_sfs"}
    projected_energy = variant == "les_combined"
    reformulated = variant in {"les_rvpm", "les_rvpm_sfs"}
    time_scheme = "RK3"
    return vpm.VPMSetup(
        time_step_size=TIME_STEP_SIZE,
        time_integration="COUPLED",
        coupled_max_strain_increment=0.15,
        coupled_max_advection_fraction=0.5,
        advection=vpm.AdvectionConfig(scheme=time_scheme),
        stretching=vpm.StretchingConfig.transposed(
            scheme=time_scheme,
            conserve_moments=projected_stretching,
            conserve_energy=projected_energy,
            reformulated=reformulated,
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
        write_precision="f32",
        checkpoint_store_velocity_gradient=False,
        max_n_particles=max_particles,
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


@cache
def initial_peak_strength(family: str) -> float:
    """Return the initial particle-strength scale for one interaction family."""
    centres, circulations = ring_geometry(family)
    return max(
        float(np.linalg.norm(ring_particles(centre, circulation, seed)[4], axis=1).max())
        for centre, circulation, seed in zip(centres, circulations, RING_SEEDS, strict=True)
    )


def run_case(
    case_name: str,
    *,
    n_steps: int = NUM_STEPS,
    restart_from: Path | None = None,
) -> None:
    family, variant = CASES[case_name]
    reformulated = variant in {"les_rvpm", "les_rvpm_sfs"}
    time_scheme = "RK3"
    output_dir = CASE_DIR / "solution" / case_name
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing results in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n---- " + case_name + " ----")
    print(f"  family={family}, model={variant}, steps={n_steps}")

    solver = vpm.VPMSolver(setup=solver_setup(case_name, output_dir), case_dir=CASE_DIR)
    initial_strength = initial_peak_strength(family)
    if restart_from is not None:
        solver.load_numerical_state(str(restart_from))
    else:
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
        "ring_radius": RING_RADIUS,
        "tube_circulation": RING_CIRCULATION,
        "core_radius": CORE_RADIUS,
        "vortex_reynolds_number": RING_CIRCULATION / KINEMATIC_VISCOSITY,
        "particle_spacing": PARTICLE_SPACING,
        "particle_core_radius": PARTICLE_RADIUS,
        "initial_particle_count": len(solver.particles),
        "time_step_size": TIME_STEP_SIZE,
        "time_integration": "COUPLED",
        "advection_scheme": time_scheme,
        "stretching_scheme": time_scheme,
        "stretching_formulation": "REFORMULATED" if reformulated else "CLASSIC",
        "treecode_theta": TREECODE_THETA,
        "widnall_amplitude": WIDNALL_AMPLITUDE,
        "widnall_modes": WIDNALL_MODES,
        "ring_seeds": list(RING_SEEDS),
        "smagorinsky_coefficient": (
            0.0 if variant in {"dns", "les_rvpm_sfs"} else LES_COEFFICIENT[family]
        ),
        "vortex_stretching_sfs_coefficient": (
            VORTEX_STRETCHING_SFS_COEFFICIENT if variant == "les_rvpm_sfs" else 0.0
        ),
        "pedrizzetti_relaxation_factor": (
            RVPM_RELAXATION_FACTOR if variant == "les_rvpm_sfs" else 0.0
        ),
        "pedrizzetti_relaxation_interval_steps": (
            RVPM_RELAXATION_INTERVAL_STEPS if variant == "les_rvpm_sfs" else 0
        ),
        "diagnostic_interval_steps": DIAGNOSTIC_INTERVAL_STEPS,
        "checkpoint_interval_steps": CHECKPOINT_INTERVAL_STEPS,
    }
    if restart_from is not None:
        manifest["restart_from"] = str(restart_from)
    if variant in {"les_remeshing", "les_combined"}:
        manifest.update(
            remesh_start_step=(
                COMBINED_REMESH_START_STEP
                if variant == "les_combined"
                else REMESH_START_STEP[family]
            ),
            remesh_interval_steps=(
                COMBINED_REMESH_INTERVAL_STEPS
                if variant == "les_combined"
                else REMESH_INTERVAL_STEPS[family]
            ),
            remesh_grid_spacing=(
                COMBINED_REMESH_GRID_SPACING
                if variant == "les_combined"
                else REMESH_GRID_SPACING
            ),
            remesh_capacity_grid_spacing=(
                COMBINED_CAPACITY_GRID_SPACING
                if variant == "les_combined"
                else REMESH_CAPACITY_GRID_SPACING
            ),
            remesh_capacity_fraction=(
                COMBINED_CAPACITY_FRACTION
                if variant == "les_combined"
                else REMESH_CAPACITY_FRACTION
            ),
            remesh_core_radius=(
                COMBINED_REMESH_CORE_RADIUS
                if variant == "les_combined"
                else REMESH_CORE_RADIUS
            ),
            remesh_tail_budget=REMESH_TAIL_BUDGET,
        )
    if variant == "les_combined":
        manifest.update(
            splitting_strength_factor=SPLITTING_STRENGTH_FACTOR,
            splitting_interval_steps=COMBINED_SPLITTING_INTERVAL_STEPS,
            splitting_late_interval_steps=COMBINED_SPLITTING_LATE_INTERVAL_STEPS,
            splitting_late_start_step=COMBINED_SPLITTING_LATE_START_STEP,
            splitting_end_step=COMBINED_SPLITTING_END_STEP,
            remesh_max_particles=COMBINED_REMESH_MAX_PARTICLES,
            remesh_capacity_max_particles=COMBINED_CAPACITY_MAX_PARTICLES,
            remesh_max_events=COMBINED_REMESH_MAX_EVENTS,
            realignment_factor=COMBINED_REALIGNMENT_FACTOR,
            realignment_interval_steps=COMBINED_REALIGNMENT_INTERVAL_STEPS,
            realignment_start_step=COMBINED_REALIGNMENT_START_STEP,
            realignment_end_step=COMBINED_REALIGNMENT_END_STEP,
            stretching_viscosity_coefficient=(
                COMBINED_STRETCHING_VISCOSITY_COEFFICIENT
            ),
            stretching_viscosity_start_step=(
                COMBINED_STRETCHING_VISCOSITY_START_STEP
            ),
            stretching_viscosity_feedback_gain=(
                COMBINED_STRETCHING_VISCOSITY_FEEDBACK_GAIN
            ),
            stretching_viscosity_feedback_growth_limit=(
                COMBINED_STRETCHING_VISCOSITY_FEEDBACK_GROWTH_LIMIT
            ),
            stretching_viscosity_max_coefficient=(
                COMBINED_STRETCHING_VISCOSITY_MAX_COEFFICIENT
            ),
        )
    manifest_path = output_dir / "run_manifest.json"

    manifest["initial_peak_vortex_strength"] = float(initial_strength)
    manifest["absolute_splitting_strength"] = float(
        COMBINED_ABSOLUTE_SPLITTING_FACTOR * initial_strength
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    solver.record_diagnostics(refresh_fields=True)
    solver.save_state(str(output_dir / f"vpm_{case_name}"))
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
