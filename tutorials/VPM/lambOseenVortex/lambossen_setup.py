#!/usr/bin/env python3
"""Lamb--Oseen vortex benchmark: single vortex, vortex dipole, and merging pair.

The Lamb--Oseen vortex is an exact solution of the two-dimensional
Navier-Stokes equations for an isolated vortex, so the computed flow can be
compared directly with the analytic profile. This script runs the chosen
benchmark case (single vortex, vortex dipole, or merging vortex pair) with the
chosen diffusion schemes and stores the snapshots under ``solution`` while the
sampled z=L/4 velocity/vorticity fields and flow integrals are written under
``samples``.

Example:
    python lambossen_setup.py --gamma1 +1 --gamma2 -1 --viscous_scheme CS \
        --case_name dipole_cs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from openonda.vpm import (
    AdvectionConfig,
    LambOseenVPM,
    ParticleDistributor,
    VPMSolver,
    SurfaceSampler,
    VelocityConfig,
    ViscousConfig,
    VPMSetup,
)
from source.solvers.VPM.io.sampling import resolve_samples_dir

TUTORIAL_DIR = Path(__file__).resolve().parent
SOLUTION_DIR = TUTORIAL_DIR / "solution"

# ---- Physics (Lamb--Oseen benchmark) ------------------------------------
RE_GAMMA = 530.0  # vortex Reynolds number Gamma / nu
SEPARATION = 1.0  # distance between the two vortices [m]
BETA_RMAX = 1.12  # r(u_theta,max) / Gaussian 1/e-vorticity radius
CORE_RADIUS = 0.125  # initial velocity-peak core radius a0 [m], as in the paper
GAUSSIAN_CORE_RADIUS = CORE_RADIUS / BETA_RMAX
COLUMN_LENGTH = 50.0 * CORE_RADIUS  # length of the finite vortex column [m]

# ---- Numerical setup shared by every viscous scheme ----------------------
SPACING = 0.3375 * CORE_RADIUS
COLUMN_SPACING = 0.80 * CORE_RADIUS
PARTICLE_RADIUS = 1.50 * SPACING
FIELD_SPACING = 0.15 * CORE_RADIUS
TIME_STEP = 0.291 / 9.0
TOTAL_TIME = 103.0 * 0.291
SAMPLE_PERIOD = 2.0 * 0.291
MERGING_SAMPLE_PERIOD = 0.291
BACKUP_PERIOD = 10.0 * 0.291
TREECODE_THETA = 0.30
TREECODE_MULTIPOLE_ORDER = 3
ADVECTION_SCHEME = "RK3"
DVH_RD_RATIO = 4
DVH_PADDING = 5.0
DVH_THRESHOLD = 1.0e-4
DVH_MAX_NODES = 300_000
GBD_MAX_NODES = 300_000
CORE_RADIUS_RATIO = 1.50
VISCOUS_SCHEMES = ("CS", "RWM", "DVH", "GBD")


def viscous_config(scheme: str, viscosity: float, spacing: float) -> ViscousConfig:
    """Build one viscous model using the same spatial resolution for every case."""
    if scheme == "cs":
        return ViscousConfig.cs(
            kinematic_viscosity=viscosity,
            particle_spacing=spacing,
        )
    if scheme == "rwm":
        return ViscousConfig.rwm(
            kinematic_viscosity=viscosity,
            particle_spacing=spacing,
        )
    if scheme == "dvh":
        return ViscousConfig.dvh(
            particle_spacing=spacing,
            padding=DVH_PADDING,
            kinematic_viscosity=viscosity,
            dvh_support_radius_ratio=DVH_RD_RATIO,
            threshold=DVH_THRESHOLD,
            threshold_mode="budget",
            max_nodes=DVH_MAX_NODES,
            cap_absolute_fraction=0.99,
            core_radius_ratio=CORE_RADIUS_RATIO,
        )
    return ViscousConfig.gbd(
        particle_spacing=spacing,
        kinematic_viscosity=viscosity,
        max_nodes=GBD_MAX_NODES,
        cap_absolute_fraction=0.99,
        core_radius_ratio=CORE_RADIUS_RATIO,
    )


def normalize_retained_circulation(
    particle_circulation: np.ndarray,
    keep: np.ndarray,
    requested_circulation_per_length: float,
    column_length: float,
) -> tuple[np.ndarray, float, float]:
    """Preserve the requested circulation after truncating weak particles."""
    retained = particle_circulation[keep].copy()
    raw_per_length = float(retained[:, 2].sum() / column_length)
    if abs(raw_per_length) <= np.finfo(float).tiny:
        raise ValueError("retained particle circulation is zero")
    scale = requested_circulation_per_length / raw_per_length
    retained *= scale
    return retained, raw_per_length, scale


def column_distribution(
    bounds: list[float],
    spacing: float,
    particle_radius: float,
    column_spacing: float = COLUMN_SPACING,
):
    """Extrude a triangular in-plane lattice through the finite vortex column."""
    plane_bounds = [*bounds[:4], 0.0, 0.0]
    plane_positions, areas, _ = ParticleDistributor.hexagonal_distribution(
        plane_bounds,
        spacing,
    )

    length = bounds[5] - bounds[4]
    number_of_layers = max(3, 2 * round(length / (2 * column_spacing)) + 1)
    layer_spacing = length / number_of_layers
    z_positions = bounds[4] + layer_spacing * (np.arange(number_of_layers) + 0.5)

    positions = np.repeat(plane_positions, number_of_layers, axis=0)
    positions[:, 2] = np.tile(z_positions, len(plane_positions))
    volumes = np.repeat(areas * layer_spacing, number_of_layers)
    radii = np.full(len(positions), particle_radius)
    return positions, volumes, radii


def run_case(
    physics: str,
    scheme: str,
    circulations: tuple[float, ...],
    case_name: str,
) -> None:
    """Run one benchmark case."""
    gamma = abs(circulations[0])
    viscosity = gamma / RE_GAMMA
    sample_period = MERGING_SAMPLE_PERIOD if physics == "merging" else SAMPLE_PERIOD
    viscous = viscous_config(scheme, viscosity, SPACING)
    number_of_steps = round(TOTAL_TIME / TIME_STEP)
    sample_steps = round(sample_period / TIME_STEP)
    backup_steps = round(BACKUP_PERIOD / TIME_STEP)

    y_positions = (0.0,) if physics == "vortex" else (SEPARATION / 2, -SEPARATION / 2)
    initial_half_width = max(abs(y) for y in y_positions) + 7.0 * CORE_RADIUS
    column_half_length = COLUMN_LENGTH / 2.0

    # GPU DVH/GBD must pre-allocate its diffusion grid once for the whole domain
    final_core_radius = BETA_RMAX * np.sqrt(GAUSSIAN_CORE_RADIUS**2 + 4.0 * viscosity * TOTAL_TIME)
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
    positions, volumes, radii = column_distribution(
        initial_bounds, SPACING, PARTICLE_RADIUS, COLUMN_SPACING
    )

    solution_dir = SOLUTION_DIR

    field_sampler = SurfaceSampler(
        point=[0, 0, 0.25 * COLUMN_LENGTH],
        normal=[0, 0, 1],
        bounds=domain_bounds[:4],
        spacing=FIELD_SPACING,
        file_name=f"{case_name}_zq",
    )
    scheduled_samplers = [field_sampler]
    final_samplers = [field_sampler]

    solver = VPMSolver(
        setup=VPMSetup.viscous_flow_simulation(
            time_step_size=TIME_STEP,
            viscous=viscous,
            advection=AdvectionConfig(scheme=ADVECTION_SCHEME),
            velocity=VelocityConfig.treecode(
                theta=TREECODE_THETA,
                multipole_order=TREECODE_MULTIPOLE_ORDER,
                sort_particle_targets=True,
            ),
            logging_interval_steps=sample_steps,
            checkpoint_interval_steps=backup_steps,
            checkpoint_name=case_name,
            checkpoint_directory=str(solution_dir),
            sample_subdirectory=case_name,
            samplers=scheduled_samplers,
            final_samplers=final_samplers,
            domain_bounds=domain_bounds,
            compute_device="AUTO",
            random_seed=42,
        )
    )

    # Only values needed by the plotting scripts are recorded.
    samples_dir = resolve_samples_dir(solution_dir, case_name)
    samples_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "case": physics,
        "scheme": scheme,
        "circulations": circulations,
        "viscosity": viscosity,
        "core_radius": GAUSSIAN_CORE_RADIUS,
        "core_radius_definition": "gaussian_1_over_e_vorticity_radius",
        "velocity_peak_radius": CORE_RADIUS,
        "separation": SEPARATION,
        "column_half_length": column_half_length,
        "total_time": TOTAL_TIME,
        "time_step": TIME_STEP,
        "in_plane_spacing": SPACING,
        "field_spacing": FIELD_SPACING,
        "sample_plane_z": 0.25 * COLUMN_LENGTH,
        "status": "running",
        "completed": False,
    }
    metadata_path = samples_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Add the vortex particles
    vortex_age = GAUSSIAN_CORE_RADIUS**2 / (4.0 * viscosity)
    for group_id, (circulation, y_position) in enumerate(
        zip(circulations, y_positions, strict=True)
    ):
        velocity, _, particle_circulation = LambOseenVPM(
            kinematic_viscosity=viscosity,
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
        retained_circulation, _, _ = normalize_retained_circulation(
            particle_circulation,
            keep,
            circulation,
            COLUMN_LENGTH,
        )
        group_ids = np.full(np.count_nonzero(keep), group_id, dtype=np.int32)
        solver.add_vortex_particles(
            positions[keep],
            velocity[keep],
            retained_circulation,
            radii[keep],
            volumes[keep],
            kinematic_viscosity=np.full(int(np.count_nonzero(keep)), viscosity),
            group_id=group_ids,
        )

    try:
        solver.execute_final_samplers()  # Record the exact t=0 state.
        for _ in range(number_of_steps):
            solver.advance()
        solver.execute_final_samplers()
    except BaseException:
        metadata["status"] = "failed"
        metadata["completed"] = False
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        raise
    finally:
        solver.reset_gpu()

    metadata["status"] = "complete"
    metadata["completed"] = True
    metadata["final_time"] = number_of_steps * solver.time_step_size
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"  {case_name} finished")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gamma1", type=float, required=True)
    parser.add_argument("--gamma2", type=float, required=True)
    parser.add_argument("--viscous_scheme", choices=VISCOUS_SCHEMES, required=True)
    parser.add_argument("--case_name", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if abs(args.gamma2) < 1e-12:
        physics, circulations = "vortex", (args.gamma1,)
    elif args.gamma1 * args.gamma2 < 0.0:
        physics, circulations = "dipole", (args.gamma1, args.gamma2)
    else:
        physics, circulations = "merging", (args.gamma1, args.gamma2)
    run_case(physics, args.viscous_scheme.lower(), circulations, args.case_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
