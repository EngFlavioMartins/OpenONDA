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
    python lamb_oseen_setup.py --circulation1 +1 --circulation2 -1 --viscous-scheme CS \
        --case-name dipole_cs
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

import openonda.vpm as vpm

TUTORIAL_DIR = Path(__file__).resolve().parent

# ---- Physics (Lamb--Oseen benchmark) ------------------------------------
CIRCULATION_REYNOLDS_NUMBER = 530.0  # vortex Reynolds number circulation / kinematic_viscosity
SEPARATION = 1.0  # distance between the two vortices [m]
BETA_RMAX = 1.12  # r(u_theta,max) / Gaussian 1/e-vorticity radius
CORE_RADIUS = 0.125  # initial velocity-peak core radius a0 [m], as in the paper
GAUSSIAN_CORE_RADIUS = CORE_RADIUS / BETA_RMAX
COLUMN_LENGTH = 50.0 * CORE_RADIUS  # length of the finite vortex column [m]

# ---- Numerical setup shared by every viscous scheme ----------------------
# The full-length CS convergence study supports h/a0=0.45: its differences
# from the next finer solution are below 0.22% for velocity, vorticity, and
# velocity gradient.  The former h/a0=0.3375 setting made the isotropic
# DVH/GBD regeneration volume 2.37x larger without a material profile gain.
SPACING = 0.45 * CORE_RADIUS
COLUMN_SPACING = 0.45 * CORE_RADIUS
PARTICLE_RADIUS = 1.50 * SPACING
FIELD_SPACING = 0.15 * CORE_RADIUS
TIME_STEP_SIZE = 0.291 / 9.0
TOTAL_TIME = 103.0 * 0.291
SAMPLE_INTERVAL_TIME = 2.0 * 0.291
MERGING_SAMPLE_INTERVAL_TIME = 0.291
CHECKPOINT_INTERVAL_TIME = 10.0 * 0.291
TREECODE_THETA = 0.30
TREECODE_MULTIPOLE_ORDER = 3
ADVECTION_SCHEME = "RK3"
DVH_RD_RATIO = 4
DVH_PADDING = 5.0
GBD_PADDING = 5.0
DVH_THRESHOLD = 1.0e-4
GBD_THRESHOLD = 5.0e-5
# At h/a0=0.45 the former 500k ceiling is unnecessary.  A 300k guard leaves
# margin above the particle count predicted from the completed h/a0=0.3375
# campaigns while preventing an accidentally unbounded tutorial run.
DVH_MAX_NODES = 300_000
GBD_MAX_NODES = 300_000
INITIAL_STRENGTH_CUTOFF = 0.01
CORE_RADIUS_RATIO = 1.50
VISCOUS_SCHEMES = ("CS", "RWM", "DVH", "GBD")


def _regeneration_node_cap(environment_name: str, default: int) -> int:
    """Return an absolute override or the calibrated capacity ceiling."""
    override = os.environ.get(environment_name)
    return int(override) if override is not None else default


def viscous_config(
    scheme: str,
    kinematic_viscosity: float,
    spacing: float,
    *,
    dvh_rd_ratio: int = DVH_RD_RATIO,
    dvh_padding: float = DVH_PADDING,
    gbd_padding: float = GBD_PADDING,
    dvh_threshold: float = DVH_THRESHOLD,
    gbd_threshold: float = GBD_THRESHOLD,
    dvh_max_nodes: int = DVH_MAX_NODES,
    gbd_max_nodes: int = GBD_MAX_NODES,
) -> vpm.ViscousConfig:
    """Build one viscous model using the same spatial resolution for every case."""
    if scheme == "cs":
        return vpm.ViscousConfig.cs(
            kinematic_viscosity=kinematic_viscosity,
            particle_spacing=spacing,
        )
    if scheme == "rwm":
        return vpm.ViscousConfig.rwm(
            kinematic_viscosity=kinematic_viscosity,
            particle_spacing=spacing,
        )
    if scheme == "dvh":
        return vpm.ViscousConfig.dvh(
            particle_spacing=spacing,
            padding=dvh_padding,
            kinematic_viscosity=kinematic_viscosity,
            dvh_support_radius_ratio=int(
                os.environ.get("OPENONDA_LAMB_DVH_RD_RATIO", dvh_rd_ratio)
            ),
            threshold=float(os.environ.get("OPENONDA_LAMB_DVH_THRESHOLD", dvh_threshold)),
            threshold_mode="budget",
            max_nodes=_regeneration_node_cap(
                "OPENONDA_LAMB_DVH_MAX_NODES",
                dvh_max_nodes,
            ),
            core_radius_ratio=CORE_RADIUS_RATIO,
        )
    return vpm.ViscousConfig.gbd(
        particle_spacing=spacing,
        padding=gbd_padding,
        kinematic_viscosity=kinematic_viscosity,
        threshold=float(os.environ.get("OPENONDA_LAMB_GBD_THRESHOLD", gbd_threshold)),
        threshold_mode="budget",
        max_nodes=_regeneration_node_cap(
            "OPENONDA_LAMB_GBD_MAX_NODES",
            gbd_max_nodes,
        ),
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
    particle_core_radius: float,
    column_spacing: float = COLUMN_SPACING,
):
    """Extrude a triangular in-plane lattice through the finite vortex column."""
    plane_bounds = [*bounds[:4], 0.0, 0.0]
    plane_positions, areas, _ = vpm.ParticleDistributor.hexagonal_distribution(
        plane_bounds,
        spacing,
    )

    length = bounds[5] - bounds[4]
    number_of_layers = max(3, 2 * round(length / (2 * column_spacing)) + 1)
    layer_spacing = length / number_of_layers
    z_positions = bounds[4] + layer_spacing * (np.arange(number_of_layers) + 0.5)

    position = np.repeat(plane_positions, number_of_layers, axis=0)
    position[:, 2] = np.tile(z_positions, len(plane_positions))
    particle_volume = np.repeat(areas * layer_spacing, number_of_layers)
    core_radius = np.full(len(position), particle_core_radius)
    return position, particle_volume, core_radius


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
    time_step_size: float = TIME_STEP_SIZE,
    sample_plane_fraction: float = 0.25,
    case_dir: Path = TUTORIAL_DIR,
    compute_device: str = "AUTO",
    anti_diffusion: bool = True,
    strength_cutoff: float = INITIAL_STRENGTH_CUTOFF,
    dvh_rd_ratio: int = DVH_RD_RATIO,
    dvh_padding: float = DVH_PADDING,
    gbd_padding: float = GBD_PADDING,
    dvh_threshold: float = DVH_THRESHOLD,
    gbd_threshold: float = GBD_THRESHOLD,
    dvh_max_nodes: int = DVH_MAX_NODES,
    gbd_max_nodes: int = GBD_MAX_NODES,
) -> None:
    """Run one benchmark case."""
    spacing = spacing_ratio * CORE_RADIUS
    column_spacing = (
        COLUMN_SPACING if column_spacing_ratio is None else column_spacing_ratio * CORE_RADIUS
    )
    particle_core_radius = 1.5 * spacing
    field_spacing = field_spacing_ratio * CORE_RADIUS
    circulation = abs(circulations[0])
    kinematic_viscosity = circulation / CIRCULATION_REYNOLDS_NUMBER
    sample_interval_time = (
        MERGING_SAMPLE_INTERVAL_TIME if physics == "merging" else SAMPLE_INTERVAL_TIME
    )
    if not 0.0 <= strength_cutoff < 1.0:
        raise ValueError("strength_cutoff must be in [0, 1)")
    if dvh_padding < 0.0 or gbd_padding < 0.0:
        raise ValueError("diffusion-grid padding must be non-negative")
    if not 0.0 <= dvh_threshold < 1.0 or not 0.0 <= gbd_threshold < 1.0:
        raise ValueError("diffusion strength budgets must be in [0, 1)")
    viscous = viscous_config(
        scheme,
        kinematic_viscosity,
        spacing,
        dvh_rd_ratio=dvh_rd_ratio,
        dvh_padding=dvh_padding,
        gbd_padding=gbd_padding,
        dvh_threshold=dvh_threshold,
        gbd_threshold=gbd_threshold,
        dvh_max_nodes=dvh_max_nodes,
        gbd_max_nodes=gbd_max_nodes,
    )
    n_steps = round(end_time / time_step_size)
    sample_steps = round(sample_interval_time / time_step_size)
    checkpoint_interval_steps = round(CHECKPOINT_INTERVAL_TIME / time_step_size)

    y_positions = (0.0,) if physics == "vortex" else (SEPARATION / 2, -SEPARATION / 2)
    initial_half_width = max(abs(y) for y in y_positions) + 7.0 * CORE_RADIUS
    column_half_length = COLUMN_LENGTH / 2.0

    # GPU DVH/GBD must pre-allocate its diffusion grid once for the whole domain
    final_core_radius = BETA_RMAX * np.sqrt(
        GAUSSIAN_CORE_RADIUS**2 + 4.0 * kinematic_viscosity * end_time
    )
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
    position, particle_volume, core_radius = column_distribution(
        initial_bounds, spacing, particle_core_radius, column_spacing
    )

    case_dir = Path(case_dir).resolve()
    solution_dir = case_dir / "solution"

    field_sampler = vpm.SurfaceSampler(
        point=[0, 0, sample_plane_fraction * COLUMN_LENGTH],
        normal=[0, 0, 1],
        bounds=domain_bounds[:4],
        spacing=field_spacing,
        file_name=f"{case_name}_zq",
    )
    scheduled_samplers = [field_sampler]
    final_samplers = [field_sampler]

    solver = vpm.VPMSolver(
        setup=vpm.VPMSetup.viscous_flow_simulation(
            time_step_size=time_step_size,
            viscous=viscous,
            advection=vpm.AdvectionConfig(scheme=ADVECTION_SCHEME),
            velocity=vpm.VelocityConfig.treecode(
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
        "kinematic_viscosity": kinematic_viscosity,
        "core_radius": GAUSSIAN_CORE_RADIUS,
        "core_radius_definition": "gaussian_1_over_e_vorticity_radius",
        "velocity_peak_radius": CORE_RADIUS,
        "vortex_separation": SEPARATION,
        "column_half_length": column_half_length,
        "end_time": end_time,
        "time_step_size": time_step_size,
        "in_plane_spacing": spacing,
        "in_plane_spacing_ratio": spacing_ratio,
        "column_spacing": column_spacing,
        "column_spacing_ratio": column_spacing / CORE_RADIUS,
        "field_spacing": field_spacing,
        "sample_plane_fraction": sample_plane_fraction,
        "sample_plane_z": sample_plane_fraction * COLUMN_LENGTH,
        "compute_device": compute_device,
        "precision": solver.precision,
        "smagorinsky_coefficient": 0.0,
        "anti_diffusion_enabled": anti_diffusion,
        "advection_scheme": ADVECTION_SCHEME,
        "treecode_theta": TREECODE_THETA,
        "treecode_multipole_order": TREECODE_MULTIPOLE_ORDER,
        "random_seed": 42,
        "dvh_rd_ratio": viscous.dvh_support_radius_ratio if scheme == "dvh" else None,
        "dvh_threshold": viscous.dvh_threshold if scheme == "dvh" else None,
        "dvh_threshold_mode": viscous.dvh_threshold_mode if scheme == "dvh" else None,
        "dvh_padding_cells": viscous.dvh_domain_padding if scheme == "dvh" else None,
        "dvh_max_nodes": viscous.dvh_max_nodes if scheme == "dvh" else None,
        "gbd_threshold": viscous.gbd_threshold if scheme == "gbd" else None,
        "gbd_threshold_mode": viscous.gbd_threshold_mode if scheme == "gbd" else None,
        "gbd_padding_cells": viscous.gbd_domain_padding if scheme == "gbd" else None,
        "gbd_max_nodes": viscous.gbd_max_nodes if scheme == "gbd" else None,
        "regeneration_cap_basis": "calibrated_resource_ceiling",
        "initial_strength_cutoff_fraction": strength_cutoff,
        "circulation_normalization": "per_vortex_after_strength_cutoff",
        "status": "running",
        "completed": False,
    }
    metadata_path = samples_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Add the vortex particles
    vortex_age = GAUSSIAN_CORE_RADIUS**2 / (4.0 * kinematic_viscosity)
    normalization_records = []
    for group_id, (circulation, y_position) in enumerate(
        zip(circulations, y_positions, strict=True)
    ):
        velocity, _, particle_vortex_strength = vpm.lamb_oseen_vpm(
            kinematic_viscosity=kinematic_viscosity,
            mean_core_radius=float(core_radius.mean()),
            position=position,
            particle_volume=particle_volume,
            vortex_centre_position=np.array([0.0, y_position, 0.0]),
            circulation=circulation,
            vortex_age=vortex_age,
            is_anti_diffusion_enabled=anti_diffusion,
        )
        vortex_strength_magnitude = np.linalg.norm(particle_vortex_strength, axis=1)
        keep = vortex_strength_magnitude >= strength_cutoff * vortex_strength_magnitude.max()
        retained_vortex_strength, raw_per_length, normalization_scale = (
            normalize_retained_circulation(
                particle_vortex_strength,
                keep,
                circulation,
                COLUMN_LENGTH,
            )
        )
        normalization_records.append(
            {
                "group_id": group_id,
                "requested_circulation_per_length": circulation,
                "raw_retained_circulation_per_length": raw_per_length,
                "normalization_scale": normalization_scale,
                "retained_particle_fraction": float(np.count_nonzero(keep) / len(keep)),
            }
        )
        group_id = np.full(np.count_nonzero(keep), group_id, dtype=np.int32)
        solver.add_vortex_particles(
            position=position[keep],
            velocity=velocity[keep],
            vortex_strength=retained_vortex_strength,
            core_radius=core_radius[keep],
            particle_volume=particle_volume[keep],
            kinematic_viscosity=np.full(int(np.count_nonzero(keep)), kinematic_viscosity),
            group_id=group_id,
        )

    metadata["initial_n_particles_total"] = len(solver.particles)
    metadata["circulation_normalization_records"] = normalization_records
    metadata["raw_retained_circulation_fraction"] = min(
        abs(record["raw_retained_circulation_per_length"])
        / abs(record["requested_circulation_per_length"])
        for record in normalization_records
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    try:
        solver.execute_final_samplers()  # Record the exact t=0 state.
        for _ in range(n_steps):
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
    metadata["final_time"] = n_steps * solver.time_step_size
    metadata["compute_device"] = solver.compute_device
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"  {case_name} finished")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circulation1", type=float, required=True)
    parser.add_argument("--circulation2", type=float, default=0.0)
    parser.add_argument("--viscous-scheme", choices=VISCOUS_SCHEMES, required=True)
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--spacing-ratio", type=float, default=SPACING / CORE_RADIUS)
    parser.add_argument("--column-spacing-ratio", type=float)
    parser.add_argument("--field-spacing-ratio", type=float, default=FIELD_SPACING / CORE_RADIUS)
    parser.add_argument("--end-time", type=float, default=TOTAL_TIME)
    parser.add_argument("--time-step-size", type=float, default=TIME_STEP_SIZE)
    parser.add_argument("--sample-plane-fraction", type=float, default=0.25)
    parser.add_argument("--strength-cutoff", type=float, default=INITIAL_STRENGTH_CUTOFF)
    parser.add_argument("--dvh-rd-ratio", type=int, choices=(3, 4, 5), default=DVH_RD_RATIO)
    parser.add_argument("--dvh-padding", type=float, default=DVH_PADDING)
    parser.add_argument("--gbd-padding", type=float, default=GBD_PADDING)
    parser.add_argument("--dvh-threshold", type=float, default=DVH_THRESHOLD)
    parser.add_argument("--gbd-threshold", type=float, default=GBD_THRESHOLD)
    parser.add_argument("--dvh-max-nodes", type=int, default=DVH_MAX_NODES)
    parser.add_argument("--gbd-max-nodes", type=int, default=GBD_MAX_NODES)
    parser.add_argument("--case-dir", type=Path, default=TUTORIAL_DIR)
    parser.add_argument(
        "--compute-device",
        choices=("AUTO", "CPU", "VULKAN", "CUDA", "METAL"),
        default="AUTO",
    )
    parser.add_argument(
        "--disable-anti-diffusion",
        action="store_true",
        help="disable particle-core initialization correction for grid studies",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if abs(args.circulation2) < 1e-12:
        physics, circulations = "vortex", (args.circulation1,)
    elif args.circulation1 * args.circulation2 < 0.0:
        physics, circulations = "dipole", (args.circulation1, args.circulation2)
    else:
        physics, circulations = "merging", (args.circulation1, args.circulation2)
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
        anti_diffusion=not args.disable_anti_diffusion,
        strength_cutoff=args.strength_cutoff,
        dvh_rd_ratio=args.dvh_rd_ratio,
        dvh_padding=args.dvh_padding,
        gbd_padding=args.gbd_padding,
        dvh_threshold=args.dvh_threshold,
        gbd_threshold=args.gbd_threshold,
        dvh_max_nodes=args.dvh_max_nodes,
        gbd_max_nodes=args.gbd_max_nodes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
