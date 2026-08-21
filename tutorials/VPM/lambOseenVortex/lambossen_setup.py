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
    python lambossen_setup.py --gamma1 +1 --gamma2 -1 --viscous-scheme CS \
        --case-name dipole_cs
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

TUTORIAL_DIR = Path(__file__).resolve().parent

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
SAMPLE_INTERVAL_TIME = 2.0 * 0.291
MERGING_SAMPLE_INTERVAL_TIME = 0.291
CHECKPOINT_INTERVAL_TIME = 10.0 * 0.291
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
    particle_vortex_strength: np.ndarray,
    keep: np.ndarray,
    requested_circulation_per_length: float,
    column_length: float,
) -> tuple[np.ndarray, float, float]:
    """Preserve the requested circulation after truncating weak particles."""
    retained = particle_vortex_strength[keep].copy()
    raw_per_length = float(retained[:, 2].sum() / column_length)
    if abs(raw_per_length) <= np.finfo(float).tiny:
        raise ValueError("retained particle vortex strength is zero")
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
    *,
    spacing_ratio: float = SPACING / CORE_RADIUS,
    column_spacing_ratio: float | None = None,
    field_spacing_ratio: float = FIELD_SPACING / CORE_RADIUS,
    end_time: float = TOTAL_TIME,
    time_step_size: float = TIME_STEP,
    sample_plane_fraction: float = 0.25,
    case_dir: Path = TUTORIAL_DIR,
    compute_device: str = "AUTO",
) -> None:
    """Run one benchmark case."""
    spacing = spacing_ratio * CORE_RADIUS
    column_spacing = (
        COLUMN_SPACING if column_spacing_ratio is None else column_spacing_ratio * CORE_RADIUS
    )
    particle_radius = 1.5 * spacing
    field_spacing = field_spacing_ratio * CORE_RADIUS
    gamma = abs(circulations[0])
    viscosity = gamma / RE_GAMMA
    sample_interval_time = (
        MERGING_SAMPLE_INTERVAL_TIME if physics == "merging" else SAMPLE_INTERVAL_TIME
    )
    viscous = viscous_config(scheme, viscosity, spacing)
    number_of_steps = round(end_time / time_step_size)
    sample_steps = round(sample_interval_time / time_step_size)
    checkpoint_interval_steps = round(CHECKPOINT_INTERVAL_TIME / time_step_size)

    y_positions = (0.0,) if physics == "vortex" else (SEPARATION / 2, -SEPARATION / 2)
    initial_half_width = max(abs(y) for y in y_positions) + 7.0 * CORE_RADIUS
    column_half_length = COLUMN_LENGTH / 2.0

    # GPU DVH/GBD must pre-allocate its diffusion grid once for the whole domain
    final_core_radius = BETA_RMAX * np.sqrt(GAUSSIAN_CORE_RADIUS**2 + 4.0 * viscosity * end_time)
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
        initial_bounds, spacing, particle_radius, column_spacing
    )

    case_dir = Path(case_dir).resolve()
    solution_dir = case_dir / "solution"

    field_sampler = SurfaceSampler(
        point=[0, 0, sample_plane_fraction * COLUMN_LENGTH],
        normal=[0, 0, 1],
        bounds=domain_bounds[:4],
        spacing=field_spacing,
        file_name=f"{case_name}_zq",
    )
    scheduled_samplers = [field_sampler]
    final_samplers = [field_sampler]

    solver = VPMSolver(
        setup=VPMSetup.viscous_flow_simulation(
            time_step_size=time_step_size,
            viscous=viscous,
            advection=AdvectionConfig(scheme=ADVECTION_SCHEME),
            velocity=VelocityConfig.treecode(
                theta=TREECODE_THETA,
                multipole_order=TREECODE_MULTIPOLE_ORDER,
                sort_particle_targets=True,
            ),
            logging_interval_steps=sample_steps,
            checkpoint_interval_steps=checkpoint_interval_steps,
            checkpoint_name=case_name,
            checkpoint_directory=str(solution_dir),
            sample_subdirectory=case_name,
            samplers=scheduled_samplers,
            final_samplers=final_samplers,
            domain_bounds=domain_bounds,
            compute_device=compute_device,
            random_seed=42,
        ),
        case_dir=case_dir,
    )

    # Only values needed by the plotting scripts are recorded.
    samples_dir = case_dir / "samples" / case_name
    samples_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "case": physics,
        "scheme": scheme,
        "circulations": circulations,
        "kinematic_viscosity": viscosity,
        "core_radius": GAUSSIAN_CORE_RADIUS,
        "core_radius_definition": "gaussian_1_over_e_vorticity_radius",
        "velocity_peak_radius": CORE_RADIUS,
        "separation": SEPARATION,
        "column_half_length": column_half_length,
        "end_time": end_time,
        "time_step_size": time_step_size,
        "in_plane_spacing": spacing,
        "field_spacing": field_spacing,
        "sample_plane_fraction": sample_plane_fraction,
        "sample_plane_z": sample_plane_fraction * COLUMN_LENGTH,
        "requested_compute_device": compute_device,
        "advection_scheme": ADVECTION_SCHEME,
        "treecode_theta": TREECODE_THETA,
        "treecode_multipole_order": TREECODE_MULTIPOLE_ORDER,
        "circulation_normalization": "per_vortex_after_strength_cutoff",
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
        velocity, _, particle_vortex_strength = LambOseenVPM(
            kinematic_viscosity=viscosity,
            avg_particle_radius=float(radii.mean()),
            positions=positions,
            volumes=volumes,
            vortex_center=np.array([0.0, y_position, 0.0]),
            circulation=circulation,
            vortex_time=vortex_age,
            anti_diffuse_flag=True,
        )
        strength = np.linalg.norm(particle_vortex_strength, axis=1)
        keep = strength >= 0.01 * strength.max()
        retained_vortex_strength, _, _ = normalize_retained_circulation(
            particle_vortex_strength,
            keep,
            circulation,
            COLUMN_LENGTH,
        )
        group_ids = np.full(np.count_nonzero(keep), group_id, dtype=np.int32)
        solver.add_vortex_particles(
            position=positions[keep],
            velocity=velocity[keep],
            vortex_strength=retained_vortex_strength,
            core_radius=radii[keep],
            volume=volumes[keep],
            kinematic_viscosity=np.full(int(np.count_nonzero(keep)), viscosity),
            group_id=group_ids,
        )

    metadata["initial_particle_count"] = len(solver.particles)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

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
    metadata["resolved_compute_device"] = solver.compute_device
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"  {case_name} finished")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gamma1", type=float, required=True)
    parser.add_argument("--gamma2", type=float, default=0.0)
    parser.add_argument("--viscous-scheme", choices=VISCOUS_SCHEMES, required=True)
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--spacing-ratio", type=float, default=SPACING / CORE_RADIUS)
    parser.add_argument("--column-spacing-ratio", type=float)
    parser.add_argument("--field-spacing-ratio", type=float, default=FIELD_SPACING / CORE_RADIUS)
    parser.add_argument("--end-time", type=float, default=TOTAL_TIME)
    parser.add_argument("--time-step-size", type=float, default=TIME_STEP)
    parser.add_argument("--sample-plane-fraction", type=float, default=0.25)
    parser.add_argument("--case-dir", type=Path, default=TUTORIAL_DIR)
    parser.add_argument(
        "--compute-device",
        choices=("AUTO", "CPU", "VULKAN", "CUDA", "METAL"),
        default="AUTO",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if abs(args.gamma2) < 1e-12:
        physics, circulations = "vortex", (args.gamma1,)
    elif args.gamma1 * args.gamma2 < 0.0:
        physics, circulations = "dipole", (args.gamma1, args.gamma2)
    else:
        physics, circulations = "merging", (args.gamma1, args.gamma2)
    run_case(
        physics,
        args.viscous_scheme.lower(),
        circulations,
        args.case_name,
        spacing_ratio=args.spacing_ratio,
        column_spacing_ratio=args.column_spacing_ratio,
        field_spacing_ratio=args.field_spacing_ratio,
        end_time=args.end_time,
        time_step_size=args.time_step_size,
        sample_plane_fraction=args.sample_plane_fraction,
        case_dir=args.case_dir,
        compute_device=args.compute_device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
