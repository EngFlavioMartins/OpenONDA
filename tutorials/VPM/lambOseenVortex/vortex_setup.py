#!/usr/bin/env python3
"""Run a one- or two-vortex Lamb--Oseen benchmark case.

Usage:
    python vortex_setup.py vortex_cs +1
    python vortex_setup.py dipole_dvh +1 -1
    python vortex_setup.py merging_gbd +1 +1
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from assets.pair_diagnostics import PairDiagnosticsSampler
from openonda.vpm import (
    LambOseenVPM,
    LineSampler,
    ParticleDistributor,
    Solver,
    SurfaceSampler,
    VelocityConfig,
    ViscousConfig,
    VPMSetup,
)
from source.solvers.VPM.io.sampling import resolve_samples_dir

TUTORIAL_DIR = Path(__file__).resolve().parent
SOLUTION_DIR = TUTORIAL_DIR / "solution"

# Lamb--Oseen benchmark definition.
RE_GAMMA = 530.0
SEPARATION = 1.0
CORE_RADIUS = 0.125
COLUMN_LENGTH = 50.0 * CORE_RADIUS

# Common numerical resolution.
SPACING = 0.48 * CORE_RADIUS
FIELD_SPACING = 0.30 * CORE_RADIUS
COLUMN_SPACING = 0.80 * CORE_RADIUS
PARTICLE_RADIUS = 1.50 * SPACING
PROFILE_SPACING = 0.10 * CORE_RADIUS

# Common run control. DVH overrides TIME_STEP with its required viscous step.
TIME_STEP = 0.05
TOTAL_TIME = 30.0
SAMPLE_PERIOD = 1.0
BACKUP_PERIOD = 2.5  # Keep intermediate states while diagnosing interrupted runs.

# Keep the established Lamb--Oseen DVH discretisation while removing case tuning.
BETA_RMAX = 1.12
DVH_RD_RATIO = 3
REGEN_RADIUS_RATIO = PARTICLE_RADIUS / SPACING

SCHEMES = {"cs", "rwm", "dvh", "gbd"}


def parse_case(arguments: list[str]) -> tuple[str, str, tuple[float, ...]]:
    """Return ``(physics, scheme, circulations)`` from the compact CLI."""
    if len(arguments) not in {2, 3}:
        raise SystemExit(
            "usage: python vortex_setup.py CASE GAMMA [GAMMA]\n"
            "examples:\n"
            "  python vortex_setup.py vortex_cs +1\n"
            "  python vortex_setup.py dipole_dvh +1 -1\n"
            "  python vortex_setup.py merging_gbd +1 +1"
        )

    case_name = arguments[0]

    physics_name, scheme = case_name.rsplit("_", 1)

    circulations = tuple(float(value) for value in arguments[1:])

    if len(circulations) == 1:
        physics = "vortex"
    else:
        physics = "dipole" if circulations[0] * circulations[1] < 0.0 else "merging"

    return physics, scheme, circulations


def viscous_config(scheme: str, viscosity: float) -> ViscousConfig:
    """Build one viscous model using the same spatial resolution for every case."""
    if scheme == "cs":
        return ViscousConfig.cs(
            viscosity=viscosity,
            characteristic_distance=SPACING,
        )
    if scheme == "rwm":
        return ViscousConfig.rwm(
            viscosity=viscosity,
            characteristic_distance=SPACING,
        )
    if scheme == "dvh":
        return ViscousConfig.dvh(
            h=SPACING,
            viscosity=viscosity,
            dvh_rd_ratio=DVH_RD_RATIO,
            regen_radius_ratio=REGEN_RADIUS_RATIO,
        )
    return ViscousConfig.gbd(
        h=SPACING,
        viscosity=viscosity,
        regen_radius_ratio=REGEN_RADIUS_RATIO,
    )


def cadence_steps(period: float, time_step: float) -> int:
    return max(1, round(period / time_step))


def column_distribution(bounds: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extrude a triangular in-plane lattice through the finite vortex column."""
    plane_bounds = [*bounds[:4], 0.0, 0.0]
    plane_positions, areas, _ = ParticleDistributor.hexagonal_distribution(
        plane_bounds,
        SPACING,
    )

    length = bounds[5] - bounds[4]
    number_of_layers = max(3, 2 * round(length / (2 * COLUMN_SPACING)) + 1)
    layer_spacing = length / number_of_layers
    z_positions = bounds[4] + layer_spacing * (np.arange(number_of_layers) + 0.5)

    positions = np.repeat(plane_positions, number_of_layers, axis=0)
    positions[:, 2] = np.tile(z_positions, len(plane_positions))
    volumes = np.repeat(areas * layer_spacing, number_of_layers)
    radii = np.full(len(positions), PARTICLE_RADIUS)
    return positions, volumes, radii


def run(physics: str, scheme: str, circulations: tuple[float, ...]) -> None:
    """Run one benchmark case."""
    started_at = perf_counter()
    case_name = f"{physics}_{scheme}"

    gamma = abs(circulations[0])
    viscosity = gamma / RE_GAMMA
    viscous = viscous_config(scheme, viscosity)
    time_step = float(f"{viscous.dvh_required_dt():.3g}") if scheme == "dvh" else TIME_STEP

    sample_steps = cadence_steps(SAMPLE_PERIOD, time_step)
    backup_steps = cadence_steps(BACKUP_PERIOD, time_step)

    y_positions = (0.0,) if physics == "vortex" else (SEPARATION / 2, -SEPARATION / 2)
    initial_half_width = max(abs(y) for y in y_positions) + 7.0 * CORE_RADIUS
    column_half_length = COLUMN_LENGTH / 2.0

    final_core_radius = np.sqrt(CORE_RADIUS**2 + 4.0 * BETA_RMAX**2 * viscosity * TOTAL_TIME)
    padding = 0.0 if physics == "vortex" else 4.0 * final_core_radius
    lateral_half_width = (
        initial_half_width if physics == "vortex" else max(abs(y) for y in y_positions) + padding
    )
    axial_half_length = column_half_length + padding
    downstream_length = 8.0 * SEPARATION if physics == "dipole" else 0.0

    domain_bounds = [
        -lateral_half_width,
        lateral_half_width + downstream_length,
        -lateral_half_width,
        lateral_half_width,
        -axial_half_length,
        axial_half_length,
    ]
    initial_bounds = [
        -initial_half_width,
        initial_half_width,
        -initial_half_width,
        initial_half_width,
        -column_half_length,
        column_half_length,
    ]
    positions, volumes, radii = column_distribution(initial_bounds)

    scheduled_samplers = []
    final_samplers = []
    if physics == "vortex":
        final_samplers.extend(
            (
                SurfaceSampler(
                    point=[0, 0, 0],
                    normal=[0, 0, 1],
                    bounds=domain_bounds[:4],
                    spacing=FIELD_SPACING,
                    file_name=f"{case_name}_z0",
                ),
                LineSampler(
                    [domain_bounds[0], 0, 0],
                    [domain_bounds[1], 0, 0],
                    PROFILE_SPACING,
                    f"{case_name}_x",
                ),
                LineSampler(
                    [0, domain_bounds[2], 0],
                    [0, domain_bounds[3], 0],
                    PROFILE_SPACING,
                    f"{case_name}_y",
                ),
            )
        )
    else:
        pair_sampler = PairDiagnosticsSampler(
            physics,
            SEPARATION,
            COLUMN_LENGTH,
            slab_half_width=1.5 * COLUMN_SPACING,
        )
        scheduled_samplers.append(pair_sampler)
        final_samplers.append(pair_sampler)

    solver = Solver(
        setup=VPMSetup.viscous_flow_simulation(
            time_step_size=time_step,
            viscous=viscous,
            velocity=VelocityConfig.treecode(
                theta=0.7,
                multipole_order=3,
                sort_particle_targets=True,
            ),
            logging_frequency=sample_steps,
            backup_frequency=backup_steps,
            backup_file_name=case_name,
            backup_directory=str(SOLUTION_DIR),
            sample_subdirectory=case_name,
            samplers=scheduled_samplers,
            final_samplers=final_samplers,
            vpm_domain_bounds=domain_bounds,
        )
    )

    samples_dir = resolve_samples_dir(SOLUTION_DIR, case_name)
    samples_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "case": physics,
        "scheme": scheme,
        "circulations": circulations,
        "viscosity": viscosity,
        "core_radius": CORE_RADIUS,
        "separation": SEPARATION,
        "column_half_length": column_half_length,
        "total_time": TOTAL_TIME,
        "time_step": time_step,
        "in_plane_spacing": SPACING,
        "column_spacing": COLUMN_SPACING,
        "particle_radius": PARTICLE_RADIUS,
        "particle_replicas": 1,
        "diffusion_grid_spacing": SPACING,
        "field_spacing": FIELD_SPACING,
        "sample_interval": sample_steps * time_step,
        "raw_backup_interval": backup_steps * time_step,
    }
    metadata_path = samples_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    vortex_age = (CORE_RADIUS / BETA_RMAX) ** 2 / (4.0 * viscosity)
    for group_id, (circulation, y_position) in enumerate(
        zip(circulations, y_positions, strict=True)
    ):
        velocity, _, particle_circulation = LambOseenVPM(
            viscosity=viscosity,
            avg_particle_radius=float(radii.mean()),
            positions=positions,
            volumes=volumes,
            vortex_center=np.array([0.0, y_position, 0.0]),
            vortex_strength=circulation,
            vortex_time=vortex_age,
            anti_diffuse_flag=True,
        )
        strength = np.linalg.norm(particle_circulation, axis=1)
        keep = strength >= 0.01 * strength.max()
        solver.add_vortex_particles(
            positions[keep],
            velocity[keep],
            particle_circulation[keep],
            radii[keep],
            volumes[keep],
            group_id=np.full(np.count_nonzero(keep), group_id, dtype=np.int32),
        )

    if physics != "vortex":
        solver.execute_final_samplers()  # Record the exact t=0 pair state.

    number_of_steps = round(TOTAL_TIME / time_step)
    try:
        for _ in range(solver.time_step, number_of_steps):
            solver.update_state()
        solver.execute_final_samplers()
    finally:
        solver.reset_gpu()

    metadata["wall_time_seconds"] = perf_counter() - started_at
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main(arguments: list[str] | None = None) -> int:
    physics, scheme, circulations = parse_case(sys.argv[1:] if arguments is None else arguments)
    run(physics, scheme, circulations)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
