#!/usr/bin/env python3
"""LES stabilization comparison for two leapfrogging vortex rings.

Usage:
    python interactions_setup.py --case leapfrog_les
"""

from __future__ import annotations

import argparse
from functools import cache
import json
from pathlib import Path

import numpy as np

import openonda.vpm as vpm

TUTORIAL_DIR = Path(__file__).resolve().parent

# ---- Cheng, Lou & Lim (2015) leapfrogging reference -----------------------
RING_RADIUS = 1.0  # R0 [m]
RING_CIRCULATION = np.pi  # Gamma0 [m^2/s]
REYNOLDS_NUMBER = 3000.0  # Re_Gamma = Gamma0/nu
CORE_RADIUS = 0.1 * RING_RADIUS  # a0/R0 = 0.1 [m]
RING_SEPARATION = 1.0 * RING_RADIUS  # h0/R0 = 1
KINEMATIC_VISCOSITY = RING_CIRCULATION / REYNOLDS_NUMBER

# The LBM comparison uses one sinusoidal disturbance with epsilon/R0 = 0.05
# and azimuthal mode n = 8. A zero phase is sufficient because the unbounded
# coaxial problem is rotationally invariant.
DISTURBANCE_AMPLITUDE = 0.05 * RING_RADIUS
DISTURBANCE_MODE = 8
DISTURBANCE_PHASE = 0.0

# ---- VPM discretization ----------------------------------------------------
PARTICLE_SPACING = 0.035 * RING_RADIUS
PARTICLE_CORE_RADIUS = 2.0 * PARTICLE_SPACING
TOROIDAL_TAIL_FRACTION = 0.05
TIME_STEP_SIZE = 20.0 * PARTICLE_SPACING**2 / RING_CIRCULATION
NUMBER_OF_STEPS = 1200
DIAGNOSTIC_INTERVAL_STEPS = 5
CHECKPOINT_INTERVAL_STEPS = 50
TREECODE_THETA = 0.30
MAXIMUM_PARTICLES = 120_000

# ---- LES and stabilization -------------------------------------------------
SMAGORINSKY_COEFFICIENT = 0.20
VORTEX_STRETCHING_SFS_COEFFICIENT = 0.001

# Split when |alpha_p| exceeds twice the largest initial |alpha_p|. Checking
# every step makes the absolute criterion independent of output cadence.
SPLITTING_STRENGTH_FACTOR = 2.0
SPLITTING_INTERVAL_STEPS = 1
SPLITTING_OFFSET_FRACTION = 0.25

# Gaussian CS gives sigma^2(t) = sigma_0^2 + 4 nu t. The first time at which
# sigma = 2 sigma_0 is 3 sigma_0^2/(4 nu). With the time step above this is
# exactly 450 steps. Each remesh restores the initial spacing and core radius.
REMESH_CORE_RADIUS_FACTOR = 2.0
REMESH_CORE_RADIUS_TRIGGER = REMESH_CORE_RADIUS_FACTOR * PARTICLE_CORE_RADIUS
REMESH_INTERVAL_STEPS = round(
    3.0 * PARTICLE_CORE_RADIUS**2 / (4.0 * KINEMATIC_VISCOSITY * TIME_STEP_SIZE)
)
REMESH_GRID_SPACING = PARTICLE_SPACING
REMESH_TAIL_BUDGET = 1.0e-3

CASES = {
    "leapfrog_les": (False, False, False),
    "leapfrog_les_splitting": (True, False, False),
    "leapfrog_les_sfs": (False, True, False),
    "leapfrog_les_splitting_remeshing": (True, False, True),
}


def _single_mode_ring(
    centre_x: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Discretize one Gaussian ring with the LBM single-mode disturbance."""
    represented_core_squared = CORE_RADIUS**2 - PARTICLE_CORE_RADIUS**2
    tube_radius = np.sqrt(represented_core_squared) * np.sqrt(-np.log(TOROIDAL_TAIL_FRACTION))
    centre = np.array([centre_x, 0.0, 0.0])
    position, particle_volume, core_radius = vpm.ParticleDistributor.toroidal_distribution(
        RING_RADIUS,
        tube_radius,
        PARTICLE_SPACING,
        centre_position=centre,
        widnall_amplitude=0.0,
    )
    core_radius.fill(PARTICLE_CORE_RADIUS)

    relative_x = position[:, 0] - centre_x
    azimuth = np.arctan2(position[:, 2], position[:, 1])
    unperturbed_radius = np.hypot(position[:, 1], position[:, 2])
    phase = DISTURBANCE_MODE * azimuth + DISTURBANCE_PHASE
    displacement = DISTURBANCE_AMPLITUDE * np.sin(phase)
    centreline_radius = RING_RADIUS + displacement
    centreline_slope = DISTURBANCE_AMPLITUDE * DISTURBANCE_MODE * np.cos(phase)
    radial_position = unperturbed_radius + displacement

    # Preserve the toroidal quadrature Jacobian under the centreline shift.
    particle_volume *= radial_position / unperturbed_radius
    position[:, 1] = radial_position * np.cos(azimuth)
    position[:, 2] = radial_position * np.sin(azimuth)

    core_distance_squared = (radial_position - centreline_radius) ** 2 + relative_x**2
    vorticity_magnitude = (
        RING_CIRCULATION
        / (np.pi * represented_core_squared)
        * np.exp(-core_distance_squared / represented_core_squared)
    )
    radial_vorticity = vorticity_magnitude * centreline_slope / radial_position
    vortex_strength = np.zeros_like(position)
    vortex_strength[:, 1] = -vorticity_magnitude * np.sin(azimuth) + radial_vorticity * np.cos(
        azimuth
    )
    vortex_strength[:, 2] = vorticity_magnitude * np.cos(azimuth) + radial_vorticity * np.sin(
        azimuth
    )
    vortex_strength *= particle_volume[:, None]

    tangent = np.column_stack((np.zeros_like(azimuth), -np.sin(azimuth), np.cos(azimuth)))
    represented_circulation = np.sum(
        np.einsum("ij,ij->i", vortex_strength, tangent) / radial_position
    ) / (2.0 * np.pi)
    vortex_strength *= RING_CIRCULATION / represented_circulation

    kinematic_viscosity = np.full(len(position), KINEMATIC_VISCOSITY)
    return position, particle_volume, core_radius, kinematic_viscosity, vortex_strength


@cache
def initial_peak_strength() -> float:
    """Return the maximum strength magnitude in the original particle cloud."""
    vortex_strength = _single_mode_ring(-0.5 * RING_SEPARATION)[4]
    return float(np.linalg.norm(vortex_strength, axis=1).max())


def turbulence(case_name: str) -> vpm.TurbulenceConfig:
    """Return the common LES closure, optionally with stretching SFS."""
    _, has_sfs, _ = CASES[case_name]
    return vpm.TurbulenceConfig.les_smagorinsky(
        smagorinsky_coefficient=SMAGORINSKY_COEFFICIENT,
        vortex_stretching_sfs_coefficient=(VORTEX_STRETCHING_SFS_COEFFICIENT if has_sfs else 0.0),
    )


def stabilization(case_name: str) -> vpm.StabilizationConfig:
    """Arm the requested splitting and core-growth remeshing mechanisms."""
    has_splitting, _, has_remeshing = CASES[case_name]
    filament_refinement = (
        vpm.FilamentRefinementConfig.adaptive(
            interval_steps=SPLITTING_INTERVAL_STEPS,
            max_vortex_strength_factor=np.inf,
            max_absolute_vortex_strength=(SPLITTING_STRENGTH_FACTOR * initial_peak_strength()),
            offset_fraction=SPLITTING_OFFSET_FRACTION,
            max_n_particles=MAXIMUM_PARTICLES,
        )
        if has_splitting
        else vpm.FilamentRefinementConfig.disabled()
    )
    return vpm.StabilizationConfig(
        filament_refinement=filament_refinement,
        regularization_interval_steps=(REMESH_INTERVAL_STEPS if has_remeshing else 0),
        regularization_start_step=(REMESH_INTERVAL_STEPS if has_remeshing else 0),
        regularization_grid_spacing=(REMESH_GRID_SPACING if has_remeshing else None),
        regularization_tail_budget=REMESH_TAIL_BUDGET,
        regularization_max_particles=(MAXIMUM_PARTICLES if has_remeshing else None),
        regularization_divergence_trigger=None,
        regularization_misalignment_trigger=None,
        regularization_core_radius_trigger=(REMESH_CORE_RADIUS_TRIGGER if has_remeshing else None),
        regularization_core_radius=(PARTICLE_CORE_RADIUS if has_remeshing else None),
    )


def solver_setup(case_name: str, output_directory: Path) -> vpm.VPMSetup:
    """Build one of the four explicitly supported comparison setups."""
    _, has_sfs, _ = CASES[case_name]
    return vpm.VPMSetup(
        time_step_size=TIME_STEP_SIZE,
        time_integration="COUPLED",
        coupled_max_strain_increment=0.15,
        coupled_max_advection_fraction=0.5,
        advection=vpm.AdvectionConfig(scheme="RK2"),
        stretching=vpm.StretchingConfig.transposed(
            scheme="RK2",
            reformulated=has_sfs,
        ),
        viscous=vpm.ViscousConfig.cs(
            kinematic_viscosity=KINEMATIC_VISCOSITY,
            particle_spacing=PARTICLE_SPACING,
        ),
        turbulence=turbulence(case_name),
        stabilization=stabilization(case_name),
        velocity=vpm.VelocityConfig.treecode(
            theta=TREECODE_THETA,
            sort_particle_targets=True,
            traversal_block_dim=128,
        ),
        particle_kernel="GAUSSIAN",
        write_precision="f32",
        checkpoint_store_velocity_gradient=False,
        max_n_particles=MAXIMUM_PARTICLES,
        checkpoint_directory=str(output_directory),
        checkpoint_name=case_name,
        sample_subdirectory=case_name,
        checkpoint_interval_steps=CHECKPOINT_INTERVAL_STEPS,
        logging_interval_steps=DIAGNOSTIC_INTERVAL_STEPS,
        timing_interval_steps=200,
        export_flow_integrals=True,
        samplers=(vpm.RingDiagnosticsSampler(),),
        log_mode="file",
    )


def run_case(case_name: str) -> None:
    """Run one comparison case into solution/<case> and samples/<case>."""
    has_splitting, has_sfs, has_remeshing = CASES[case_name]
    output_directory = TUTORIAL_DIR / "solution" / case_name
    output_directory.mkdir(parents=True, exist_ok=True)

    print(f"\n---- {case_name} ----")
    solver = vpm.VPMSolver(
        setup=solver_setup(case_name, output_directory),
        case_dir=TUTORIAL_DIR,
    )
    for group, centre_x in enumerate((-0.5 * RING_SEPARATION, 0.5 * RING_SEPARATION)):
        position, particle_volume, core_radius, viscosity, vortex_strength = _single_mode_ring(
            centre_x
        )
        solver.add_vortex_particles(
            position=position,
            velocity=np.zeros_like(position),
            vortex_strength=vortex_strength,
            core_radius=core_radius,
            particle_volume=particle_volume,
            kinematic_viscosity=viscosity,
            group_id=np.full(len(position), group, dtype=np.int32),
        )

    peak_strength = initial_peak_strength()
    manifest = {
        "status": "running",
        "case": case_name,
        "requested_steps": NUMBER_OF_STEPS,
        "ring_radius": RING_RADIUS,
        "ring_separation": RING_SEPARATION,
        "tube_circulation": RING_CIRCULATION,
        "vortex_reynolds_number": REYNOLDS_NUMBER,
        "physical_core_radius": CORE_RADIUS,
        "particle_spacing": PARTICLE_SPACING,
        "initial_particle_core_radius": PARTICLE_CORE_RADIUS,
        "initial_particle_count": len(solver.particles),
        "disturbance_amplitude": DISTURBANCE_AMPLITUDE,
        "disturbance_mode": DISTURBANCE_MODE,
        "disturbance_phase": DISTURBANCE_PHASE,
        "time_step_size": TIME_STEP_SIZE,
        "time_integration": "COUPLED",
        "advection_scheme": "RK2",
        "stretching_scheme": "RK2",
        "stretching_formulation": "REFORMULATED" if has_sfs else "CLASSIC",
        "stretching_discretization": "TRANSPOSED",
        "viscous_scheme": "CS",
        "smagorinsky_coefficient": SMAGORINSKY_COEFFICIENT,
        "vortex_stretching_sfs_coefficient": (
            VORTEX_STRETCHING_SFS_COEFFICIENT if has_sfs else 0.0
        ),
        "particle_splitting": has_splitting,
        "initial_peak_vortex_strength": peak_strength,
        "splitting_strength_threshold": (
            SPLITTING_STRENGTH_FACTOR * peak_strength if has_splitting else None
        ),
        "particle_remeshing": has_remeshing,
        "remesh_core_radius_trigger": (REMESH_CORE_RADIUS_TRIGGER if has_remeshing else None),
        "remesh_interval_steps": REMESH_INTERVAL_STEPS if has_remeshing else None,
        "remesh_grid_spacing": REMESH_GRID_SPACING if has_remeshing else None,
    }
    manifest_path = output_directory / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    solver.record_diagnostics(refresh_fields=True)
    solver.save_state(str(output_directory / f"vpm_{case_name}"))
    termination_reason = None
    for _ in range(NUMBER_OF_STEPS):
        solver.advance()
        if solver.step % DIAGNOSTIC_INTERVAL_STEPS:
            continue
        current_peak = np.linalg.norm(solver.particles.vortex_strength_cpu(), axis=1).max()
        if not np.isfinite(current_peak) or current_peak > 50.0 * peak_strength:
            termination_reason = "peak particle strength exceeded 50 times its initial value"
            break

    if solver.step % DIAGNOSTIC_INTERVAL_STEPS:
        solver.record_diagnostics(refresh_fields=True)
    solver.save_state(str(output_directory / f"vpm_{case_name}_final"))
    manifest.update(
        status="stopped" if termination_reason else "completed",
        termination_reason=termination_reason,
        completed_steps=solver.step,
        end_time=solver.time,
        final_particle_count=len(solver.particles),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=tuple(CASES))
    arguments = parser.parse_args()
    run_case(arguments.case)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
