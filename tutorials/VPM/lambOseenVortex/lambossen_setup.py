#!/usr/bin/env python3
"""Lamb--Oseen vortex benchmark: single vortex, vortex dipole, and merging pair.

The Lamb--Oseen vortex is an exact solution of the two-dimensional
Navier-Stokes equations for an isolated vortex, so the computed flow can be
compared directly with the analytic profile. This script runs the chosen
benchmark case (single vortex, vortex dipole, or merging vortex pair) with the
chosen diffusion schemes and stores the snapshots under ``solution`` while the
sampled profiles, fields, trajectories, diagnostics, and flow integrals are
written under ``samples``.

Examples:
    python vortex_setup.py --gamma1 +1                            # single vortex, all schemes
    python vortex_setup.py --gamma1 +1 --gamma2 -1 --schemes cs   # vortex dipole
    python vortex_setup.py --gamma1 +1 --gamma2 +1 --schemes cs   # merging pair
    python vortex_setup.py --gamma1 +1 --schemes cs rwm dvh gbd
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
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

# ---- Physics (Lamb--Oseen benchmark) ------------------------------------
RE_GAMMA = 530.0  # vortex Reynolds number Gamma / nu
SEPARATION = 1.0  # distance between the two vortices [m]
CORE_RADIUS = 0.125  # initial core radius a0 [m]
COLUMN_LENGTH = 50.0 * CORE_RADIUS  # length of the finite vortex column [m]

# ---- Numerical resolution ------------------------------------------------
SPACING = 0.4 * CORE_RADIUS  # default particle spacing near the core [m]
COLUMN_SPACING = 0.80 * CORE_RADIUS  # spacing between vortex layers [m]
PARTICLE_RADIUS = 1.50 * SPACING  # particle core radius [m]
FIELD_SPACING = 0.30 * CORE_RADIUS  # spacing of the surface sampling grid [m]
PROFILE_SPACING = 0.10 * CORE_RADIUS  # spacing of the line samples [m]

# ---- Run control ----------------------------------------------------------
TIME_STEP = 0.05  # time step for cs/rwm/gbd [s]
TOTAL_TIME = 30.0  # total simulation time [s]
SAMPLE_PERIOD = 0.5  # write a snapshot every this many seconds
BACKUP_PERIOD = 2.5  # keep intermediate states while diagnosing runs

# ---- Established Lamb--Oseen discretisation -------------------------------
BETA_RMAX = 1.12
DVH_RD_RATIO = 3
REGEN_RADIUS_RATIO = 1.50

SCHEMES = ("cs", "rwm", "dvh", "gbd")


def viscous_config(scheme: str, viscosity: float, spacing: float) -> ViscousConfig:
    """Build one viscous model using the same spatial resolution for every case."""
    if scheme == "cs":
        return ViscousConfig.cs(
            viscosity=viscosity,
            characteristic_distance=spacing,
        )
    if scheme == "rwm":
        return ViscousConfig.rwm(
            viscosity=viscosity,
            characteristic_distance=spacing,
        )
    if scheme == "dvh":
        return ViscousConfig.dvh(
            h=spacing,
            viscosity=viscosity,
            dvh_rd_ratio=DVH_RD_RATIO,
            regen_radius_ratio=REGEN_RADIUS_RATIO,
        )
    return ViscousConfig.gbd(
        h=spacing,
        viscosity=viscosity,
        regen_radius_ratio=REGEN_RADIUS_RATIO,
    )


def cadence_steps(period: float, time_step: float) -> int:
    return max(1, round(period / time_step))


def column_distribution(bounds: list[float], spacing: float, particle_radius: float):
    """Extrude a triangular in-plane lattice through the finite vortex column."""
    plane_bounds = [*bounds[:4], 0.0, 0.0]
    plane_positions, areas, _ = ParticleDistributor.hexagonal_distribution(
        plane_bounds,
        spacing,
    )

    length = bounds[5] - bounds[4]
    number_of_layers = max(3, 2 * round(length / (2 * COLUMN_SPACING)) + 1)
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
    args: argparse.Namespace,
) -> None:
    """Run one benchmark case."""
    started_at = perf_counter()
    case_name = f"{physics}_{scheme}"

    gamma = abs(circulations[0])
    viscosity = gamma / RE_GAMMA
    spacing = args.spacing_factor * CORE_RADIUS
    particle_radius = 1.50 * spacing
    viscous = viscous_config(scheme, viscosity, spacing)
    time_step = float(f"{viscous.dvh_required_dt():.3g}") if scheme == "dvh" else TIME_STEP

    sample_steps = cadence_steps(SAMPLE_PERIOD, time_step)
    backup_steps = (
        args.backup_frequency
        if args.backup_frequency is not None
        else cadence_steps(BACKUP_PERIOD, time_step)
    )

    number_of_steps = (
        args.num_steps if args.num_steps is not None else round(TOTAL_TIME / time_step)
    )
    y_positions = (0.0,) if physics == "vortex" else (SEPARATION / 2, -SEPARATION / 2)
    initial_half_width = max(abs(y) for y in y_positions) + 7.0 * CORE_RADIUS
    column_half_length = args.length / 2.0

    # GPU DVH/GBD must pre-allocate its diffusion grid once for the whole domain
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
    positions, volumes, radii = column_distribution(initial_bounds, spacing, particle_radius)

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
            backup_directory=str(Path(args.solution_dir)),
            sample_subdirectory=case_name,
            samplers=scheduled_samplers,
            final_samplers=final_samplers,
            vpm_domain_bounds=domain_bounds,
        )
    )

    samples_dir = resolve_samples_dir(Path(args.solution_dir), case_name)
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
        "in_plane_spacing": spacing,
        "column_spacing": COLUMN_SPACING,
        "particle_radius": particle_radius,
        "particle_replicas": 1,
        "diffusion_grid_spacing": spacing,
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

    try:
        for _ in range(solver.time_step, number_of_steps):
            solver.update_state()
        solver.execute_final_samplers()
    finally:
        solver.reset_gpu()

    metadata["wall_time_seconds"] = perf_counter() - started_at
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"  {case_name} finished in {perf_counter() - started_at:.1f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gamma1",
        type=float,
        default=1.0,
        help="circulation of the first vortex [m^2/s]",
    )
    parser.add_argument(
        "--gamma2",
        type=float,
        default=0.0,
        help="circulation of the second vortex; 0.0 = single-vortex case",
    )
    parser.add_argument(
        "--schemes",
        nargs="*",
        choices=SCHEMES,
        default=list(SCHEMES),
        help="diffusion schemes to run (default: all)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="number of time steps (default: run to TOTAL_TIME)",
    )
    parser.add_argument(
        "--length",
        type=float,
        default=COLUMN_LENGTH,
        help="length of the finite vortex column [m]",
    )
    parser.add_argument(
        "--spacing-factor",
        type=float,
        default=SPACING / CORE_RADIUS,
        help="particle spacing as a multiple of the core radius",
    )
    parser.add_argument(
        "--processing-unit",
        default="AUTO",
        choices=("AUTO", "CPU", "CUDA", "VULKAN", "METAL"),
        help="computation backend",
    )
    parser.add_argument(
        "--backup-frequency",
        type=int,
        default=None,
        help="backup every N steps (default: every BACKUP_PERIOD seconds)",
    )
    parser.add_argument(
        "--solution-dir",
        type=Path,
        default=SOLUTION_DIR,
        help="directory for the simulation snapshots",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_steps is not None and args.num_steps < 1:
        raise SystemExit("--num-steps must be positive")

    # This setup is prepared to run with 1 or 2 vortices
    # In the second case, in merging or dipole modes
    if abs(args.gamma2) < 1e-12:
        physics, circulations = "vortex", (args.gamma1,)
    elif args.gamma1 * args.gamma2 < 0.0:
        physics, circulations = "dipole", (args.gamma1, args.gamma2)
    else:
        physics, circulations = "merging", (args.gamma1, args.gamma2)

    print("\n===== SIMULATION =====")
    for scheme in args.schemes:
        run_case(physics, scheme, circulations, args)

    print("\n===== DONE =====")
    print("Simulation completed successfully. Run ./allplot.sh to make the figures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
