#!/usr/bin/env python3
"""Lamb--Oseen vortex benchmark: single vortex, vortex dipole, and merging pair.

The Lamb--Oseen vortex is an exact solution of the two-dimensional
Navier-Stokes equations for an isolated vortex, so the computed flow can be
compared directly with the analytic profile. This script runs the chosen
benchmark case (single vortex, vortex dipole, or merging vortex pair) with the
chosen diffusion schemes and stores the snapshots under ``solution`` while the
sampled z=L/4 velocity/vorticity fields and flow integrals are written under
``samples``.

Examples:
    python vortex_setup.py --gamma1 +1                  # single vortex, all schemes
    python vortex_setup.py --gamma1 +1 --gamma2 -1       # vortex dipole
    python vortex_setup.py --gamma1 +1 --gamma2 +1       # merging pair
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from openonda.vpm import (
    AdvectionConfig,
    LambOseenVPM,
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
BETA_RMAX = 1.12  # r(u_theta,max) / Gaussian 1/e-vorticity radius
CORE_RADIUS = 0.125  # initial velocity-peak core radius a0 [m], as in the paper
GAUSSIAN_CORE_RADIUS = CORE_RADIUS / BETA_RMAX
COLUMN_LENGTH = 50.0 * CORE_RADIUS  # length of the finite vortex column [m]

# ---- Production numerical resolution -------------------------------------
# The CS three-grid study found medium-to-fine changes below 0.5% at this
# finest level.  The field grid is kept independently finer so the velocity
# maximum and vorticity peak can be located with sub-cell interpolation.
SPACING = 0.3375 * CORE_RADIUS  # converged particle spacing near the core [m]
COLUMN_SPACING = 0.80 * CORE_RADIUS  # spacing between vortex layers [m]
PARTICLE_RADIUS = 1.50 * SPACING  # particle core radius [m]
FIELD_SPACING = 0.15 * CORE_RADIUS  # fixed diagnostic-grid spacing [m]

# ---- Run control ----------------------------------------------------------
TIME_STEP = 0.01  # conservative RK3 advection step [s]
TOTAL_TIME = 30.0  # total simulation time [s]
SAMPLE_PERIOD = 0.5  # write a snapshot every this many seconds
MERGING_SAMPLE_PERIOD = 0.25  # resolve the rapid convective merger phase
BACKUP_PERIOD = 2.5  # keep intermediate states while diagnosing runs

# ---- Established Lamb--Oseen discretisation -------------------------------
TREECODE_THETA = 0.30
TREECODE_MULTIPOLE_ORDER = 3
ADVECTION_SCHEME = "RK3"
DVH_RD_RATIO = 4
DVH_REGEN_RADIUS_RATIO = 1.25  # h/sigma=0.8, the recommended DVH overlap
DVH_THRESHOLD = 1.0e-5
DVH_MAX_NODES = 300_000  # resource guard; conservative moment recovery follows
GBD_REGEN_RADIUS_RATIO = 1.1  # retained from the validated isolated-vortex run

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
            threshold=DVH_THRESHOLD,
            threshold_mode="budget",
            max_nodes=DVH_MAX_NODES,
            cap_abs_fraction=0.99,
            regen_radius_ratio=DVH_REGEN_RADIUS_RATIO,
        )
    return ViscousConfig.gbd(
        h=spacing,
        viscosity=viscosity,
        regen_radius_ratio=GBD_REGEN_RADIUS_RATIO,
    )


def cadence_steps(period: float, time_step: float) -> int:
    return max(1, round(period / time_step))


def scheme_time_control(
    scheme: str,
    viscous: ViscousConfig,
    requested_dt: float,
    requested_total_time: float,
) -> tuple[float, int, float, float | None, int]:
    """Choose an RK3 macro step without corrupting a DVH diffusion increment.

    CS, RWM and GBD use the requested conservative macro step (slightly
    reduced only when necessary to land exactly on the requested final time).
    DVH requires its heat-kernel increment ``dt_d``; advection is therefore
    subcycled with RK3 and diffusion fires once every integer number of macro
    steps.  A DVH run ends on a completed diffusion interval, never halfway
    through one.
    """
    if requested_dt <= 0.0 or requested_total_time <= 0.0:
        raise ValueError("time step and total time must be positive")

    if scheme != "dvh":
        number_of_steps = max(1, int(np.ceil(requested_total_time / requested_dt - 1e-12)))
        time_step = requested_total_time / number_of_steps
        return time_step, number_of_steps, requested_total_time, None, 1

    # Solver.__init__ rounds dt_d to three significant figures.  Reproduce
    # that convention here so its internal cadence and this run plan agree.
    diffusion_interval = float(f"{viscous.dvh_required_dt():.3g}")
    diffusion_substeps = max(1, int(np.ceil(diffusion_interval / requested_dt)))
    time_step = diffusion_interval / diffusion_substeps
    diffusion_intervals = max(1, int(np.floor(requested_total_time / diffusion_interval + 1e-12)))
    number_of_steps = diffusion_intervals * diffusion_substeps
    effective_total_time = diffusion_intervals * diffusion_interval
    return (
        time_step,
        number_of_steps,
        effective_total_time,
        diffusion_interval,
        diffusion_substeps,
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
    *,
    spacing: float = SPACING,
    column_spacing: float = COLUMN_SPACING,
    field_spacing: float = FIELD_SPACING,
    requested_time_step: float = TIME_STEP,
    sample_period: float = SAMPLE_PERIOD,
    total_time: float = TOTAL_TIME,
    output_root: Path = TUTORIAL_DIR,
    sample_plane_fraction: float = 0.25,
    processing_unit: str = "AUTO",
    random_seed: int = 42,
    treecode_theta: float = TREECODE_THETA,
    treecode_multipole_order: int = TREECODE_MULTIPOLE_ORDER,
    advection_scheme: str = ADVECTION_SCHEME,
) -> None:
    """Run one benchmark case."""
    started_at = perf_counter()
    case_name = f"{physics}_{scheme}"

    gamma = abs(circulations[0])
    viscosity = gamma / RE_GAMMA
    particle_radius = 1.50 * spacing
    viscous = viscous_config(scheme, viscosity, spacing)
    (
        time_step,
        number_of_steps,
        effective_total_time,
        dvh_diffusion_interval,
        dvh_substeps,
    ) = scheme_time_control(scheme, viscous, requested_time_step, total_time)

    sample_steps = cadence_steps(sample_period, time_step)
    backup_steps = cadence_steps(BACKUP_PERIOD, time_step)

    y_positions = (0.0,) if physics == "vortex" else (SEPARATION / 2, -SEPARATION / 2)
    initial_half_width = max(abs(y) for y in y_positions) + 7.0 * CORE_RADIUS
    column_half_length = COLUMN_LENGTH / 2.0

    # GPU DVH/GBD must pre-allocate its diffusion grid once for the whole domain
    final_core_radius = BETA_RMAX * np.sqrt(GAUSSIAN_CORE_RADIUS**2 + 4.0 * viscosity * total_time)
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

    solution_dir = output_root / "solution"

    field_sampler = SurfaceSampler(
        point=[0, 0, sample_plane_fraction * COLUMN_LENGTH],
        normal=[0, 0, 1],
        bounds=domain_bounds[:4],
        spacing=field_spacing,
        file_name=f"{case_name}_zq",
    )
    scheduled_samplers = [field_sampler]
    final_samplers = [field_sampler]

    solver = Solver(
        setup=VPMSetup.viscous_flow_simulation(
            time_step_size=time_step,
            viscous=viscous,
            advection=AdvectionConfig(scheme=advection_scheme),
            velocity=VelocityConfig.treecode(
                theta=treecode_theta,
                multipole_order=treecode_multipole_order,
                sort_particle_targets=True,
            ),
            logging_frequency=sample_steps,
            backup_frequency=backup_steps,
            backup_file_name=case_name,
            backup_directory=str(solution_dir),
            sample_subdirectory=case_name,
            samplers=scheduled_samplers,
            final_samplers=final_samplers,
            vpm_domain_bounds=domain_bounds,
            processing_unit=processing_unit,
            random_seed=random_seed,
        )
    )

    # We need metadate with some constants in order to make the plots later.
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
        "spacing_reference_core_definition": "radius_of_maximum_azimuthal_velocity",
        "field_core_radius_definition": "radius_of_maximum_azimuthal_velocity",
        "field_center_definition": "subgrid_peak_vorticity",
        "separation": SEPARATION,
        "column_half_length": column_half_length,
        "requested_total_time": total_time,
        "total_time": effective_total_time,
        "requested_time_step": requested_time_step,
        "time_step": time_step,
        "advection_scheme": advection_scheme,
        "time_integration": "FRACTIONAL",
        "treecode_theta": treecode_theta,
        "treecode_multipole_order": treecode_multipole_order,
        "treecode_sort_particle_targets": True,
        "in_plane_spacing": spacing,
        "column_spacing": column_spacing,
        "particle_radius": particle_radius,
        "particle_replicas": 1,
        "diffusion_grid_spacing": spacing,
        "dvh_rd_ratio": DVH_RD_RATIO if scheme == "dvh" else None,
        "dvh_diffusion_interval": dvh_diffusion_interval,
        "dvh_advection_substeps_per_diffusion": dvh_substeps,
        "dvh_threshold": DVH_THRESHOLD if scheme == "dvh" else None,
        "dvh_threshold_mode": "budget" if scheme == "dvh" else None,
        "dvh_max_nodes": DVH_MAX_NODES if scheme == "dvh" else None,
        "regen_radius_ratio": (
            DVH_REGEN_RADIUS_RATIO
            if scheme == "dvh"
            else GBD_REGEN_RADIUS_RATIO
            if scheme == "gbd"
            else None
        ),
        "field_spacing": field_spacing,
        "sample_plane_z": sample_plane_fraction * COLUMN_LENGTH,
        "sample_plane_fraction": sample_plane_fraction,
        "processing_unit": processing_unit,
        "random_seed": random_seed,
        "strength_cutoff_relative": 0.01,
        "circulation_normalization": "per_vortex_after_strength_cutoff",
        "requested_sample_interval": sample_period,
        "sample_interval": sample_steps * time_step,
        "raw_backup_interval": backup_steps * time_step,
        "status": "running",
        "completed": False,
    }
    metadata_path = samples_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Add the vortex particles
    vortex_age = GAUSSIAN_CORE_RADIUS**2 / (4.0 * viscosity)
    initial_particle_count = 0
    raw_retained_circulation_per_length = []
    circulation_normalization_factors = []
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
        initial_particle_count += int(np.count_nonzero(keep))
        retained_circulation, raw_per_length, normalization_factor = normalize_retained_circulation(
            particle_circulation,
            keep,
            circulation,
            COLUMN_LENGTH,
        )
        raw_retained_circulation_per_length.append(raw_per_length)
        circulation_normalization_factors.append(normalization_factor)
        solver.add_vortex_particles(
            positions[keep],
            velocity[keep],
            retained_circulation,
            radii[keep],
            volumes[keep],
            group_id=np.full(np.count_nonzero(keep), group_id, dtype=np.int32),
        )

    metadata["initial_particle_count"] = initial_particle_count
    metadata["raw_retained_circulation_per_length"] = raw_retained_circulation_per_length
    metadata["raw_retained_circulation_fraction"] = [
        retained / requested
        for retained, requested in zip(
            raw_retained_circulation_per_length, circulations, strict=True
        )
    ]
    metadata["circulation_normalization_factors"] = circulation_normalization_factors
    metadata["retained_circulation_per_length"] = list(circulations)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    try:
        solver.execute_final_samplers()  # Record the exact t=0 state.
        for _ in range(number_of_steps):
            solver.update_state()
        solver.execute_final_samplers()
    except BaseException:
        metadata["wall_time_seconds"] = perf_counter() - started_at
        metadata["status"] = "failed"
        metadata["completed"] = False
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        raise
    finally:
        solver.reset_gpu()

    metadata["wall_time_seconds"] = perf_counter() - started_at
    metadata["status"] = "complete"
    metadata["completed"] = True
    metadata["final_time"] = number_of_steps * solver.time_step_size
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
        nargs="+",
        choices=SCHEMES,
        default=list(SCHEMES),
        help="viscous schemes to run (default: all)",
    )
    parser.add_argument(
        "--spacing-ratio",
        type=float,
        default=SPACING / CORE_RADIUS,
        help="in-plane particle spacing divided by the initial core radius",
    )
    parser.add_argument(
        "--field-spacing-ratio",
        type=float,
        default=FIELD_SPACING / CORE_RADIUS,
        help="sampling-grid spacing divided by the initial core radius",
    )
    parser.add_argument(
        "--total-time",
        type=float,
        default=TOTAL_TIME,
        help="simulation end time [s]",
    )
    parser.add_argument(
        "--time-step",
        type=float,
        default=TIME_STEP,
        help="maximum RK3 macro time step [s] (DVH uses integer substeps)",
    )
    parser.add_argument(
        "--sample-period",
        type=float,
        default=None,
        help="requested field/diagnostic sampling period [s]",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=TUTORIAL_DIR,
        help="root containing solution/ and samples/ (default: tutorial directory)",
    )
    parser.add_argument(
        "--sample-plane-fraction",
        type=float,
        default=0.25,
        help="sampling plane z/L (default: 0.25)",
    )
    parser.add_argument(
        "--processing-unit",
        choices=("AUTO", "CPU", "VULKAN", "CUDA", "METAL"),
        default="AUTO",
        help="Taichi processing backend (default: AUTO)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Taichi random seed, relevant to RWM (default: 42)",
    )
    parser.add_argument(
        "--treecode-theta",
        type=float,
        default=TREECODE_THETA,
        help="Barnes-Hut opening angle (default: 0.30)",
    )
    parser.add_argument(
        "--treecode-multipole-order",
        type=int,
        choices=(1, 2, 3),
        default=TREECODE_MULTIPOLE_ORDER,
        help="treecode multipole order (default: 3)",
    )
    parser.add_argument(
        "--advection-scheme",
        choices=("RK2", "RK3", "RK4"),
        default=ADVECTION_SCHEME,
        help="particle-advection integrator (default: RK3)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # This setup is prepared to run with 1 or 2 vortices
    # In the second case, in merging or dipole modes
    if abs(args.gamma2) < 1e-12:
        physics, circulations = "vortex", (args.gamma1,)
    elif args.gamma1 * args.gamma2 < 0.0:
        physics, circulations = "dipole", (args.gamma1, args.gamma2)
    else:
        physics, circulations = "merging", (args.gamma1, args.gamma2)

    if args.spacing_ratio <= 0.0 or args.field_spacing_ratio <= 0.0:
        raise ValueError("spacing ratios must be positive")
    if args.total_time <= 0.0 or args.time_step <= 0.0:
        raise ValueError("total time and time step must be positive")
    if not -0.5 < args.sample_plane_fraction < 0.5:
        raise ValueError("sample-plane-fraction must lie strictly inside (-0.5, 0.5)")

    spacing = args.spacing_ratio * CORE_RADIUS
    # Refine the axial lattice with the in-plane lattice so a convergence
    # study changes all particle dimensions by the same factor.
    column_spacing = COLUMN_SPACING * spacing / SPACING
    field_spacing = args.field_spacing_ratio * CORE_RADIUS
    sample_period = (
        args.sample_period
        if args.sample_period is not None
        else MERGING_SAMPLE_PERIOD
        if physics == "merging"
        else SAMPLE_PERIOD
    )
    if sample_period <= 0.0:
        raise ValueError("sample period must be positive")

    print(
        "\n===== PRODUCTION CONFIGURATION =====\n"
        f"h/a0={args.spacing_ratio:g}, field spacing/a0={args.field_spacing_ratio:g}, "
        f"dt<={args.time_step:g} s, {args.advection_scheme}, "
        f"treecode theta={args.treecode_theta:g}, p={args.treecode_multipole_order}, "
        f"sample period={sample_period:g} s"
    )

    print("\n===== SIMULATION =====")
    for scheme in args.schemes:
        run_case(
            physics,
            scheme,
            circulations,
            spacing=spacing,
            column_spacing=column_spacing,
            field_spacing=field_spacing,
            requested_time_step=args.time_step,
            sample_period=sample_period,
            total_time=args.total_time,
            output_root=args.output_root.resolve(),
            sample_plane_fraction=args.sample_plane_fraction,
            processing_unit=args.processing_unit,
            random_seed=args.random_seed,
            treecode_theta=args.treecode_theta,
            treecode_multipole_order=args.treecode_multipole_order,
            advection_scheme=args.advection_scheme,
        )

    print("\n===== DONE =====")
    print("Simulation completed successfully. Run ./allplot.sh to make the figures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
