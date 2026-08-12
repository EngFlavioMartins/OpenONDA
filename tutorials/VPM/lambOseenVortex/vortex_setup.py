#!/usr/bin/env python3
"""Run one Lamb--Oseen case selected by ``allrun.sh``.

Usage: ``python vortex_setup.py CASE SCHEME SAMPLE_PERIOD BACKUP_PERIOD``
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from assets.pair_diagnostics import PairDiagnosticsSampler
from assets.rwm_ensemble import average_final_samples, average_pair_diagnostics
from openonda.vpm import (
    AdvectionConfig,
    LambOseenVPM,
    LineSampler,
    ParticleDistributor,
    Solver,
    SurfaceSampler,
    StretchingConfig,
    VelocityConfig,
    ViscousConfig,
    VPMSetup,
)
from source.solvers.VPM.utils.field_samplers import resolve_samples_dir


TUTORIAL_DIR = Path(__file__).resolve().parent
SOLUTION_DIR = TUTORIAL_DIR / "solution"

# Physics: Cerretelli & Williamson (2003), Re_Gamma = Gamma / nu.
GAMMA = 1.0
REYNOLDS_NUMBER = 530.0
CORE_RADIUS = 0.125  # Radius of peak azimuthal velocity [m].
SEPARATION = 1.0  # Initial centre-to-centre distance for the two-vortex cases [m].
COLUMN_LENGTH = 50.0  # Vortex-column length in core radii.
TIME_STEP = 0.05
RWM_TIME_STEP = 0.40
RWM_PAIR_TIME_STEP = 0.20
GBD_TIME_STEP = 0.30
PAIR_GBD_TIME_STEP = 0.45

# The columns are uniform in z, so their axial quadrature can be lighter than
# the in-plane vortex-core resolution without shortening the pseudo-2-D domain.
SPACING = 0.48 * CORE_RADIUS
PAIR_GRID_SPACING = 0.60 * CORE_RADIUS
COLUMN_SPACING = 0.80 * CORE_RADIUS
PARTICLE_RADIUS = 1.50 * SPACING
FIELD_SPACING = 0.30 * CORE_RADIUS
PROFILE_SPACING = 0.10 * CORE_RADIUS
DVH_RD_RATIO = 3
GRID_NODE_LIMITS = {
    "vortex": {"dvh": 30_000, "gbd": 30_000},
    "dipole": {"dvh": 22_000, "gbd": 25_000},
    "merging": {"dvh": 22_000, "gbd": 25_000},
}

RWM_REALIZATIONS = 4
RWM_PARTICLE_REPLICAS = 16
RWM_FIRST_SEED = 42

# The peak-velocity radius of a Lamb--Oseen vortex is BETA_RMAX * sigma.
BETA_RMAX = 1.12


@dataclass(frozen=True)
class Case:
    """One physical arrangement of equal-strength vortices."""

    name: str
    circulations: tuple[float, ...]
    y_positions: tuple[float, ...]
    downstream_length: float = 0.0
    total_time: float = 20.0


CASES = {
    "vortex": Case("vortex", (GAMMA,), (0.0,)),
    "dipole": Case(
        "dipole",
        (GAMMA, -GAMMA),
        (SEPARATION / 2, -SEPARATION / 2),
        downstream_length=8.0,
        total_time=40.0,
    ),
    "merging": Case(
        "merging",
        (GAMMA, GAMMA),
        (SEPARATION / 2, -SEPARATION / 2),
        total_time=40.0,
    ),
}


def viscous_setup(
    scheme: str,
    viscosity: float,
    max_grid_nodes: int | None,
    spacing: float = SPACING,
) -> ViscousConfig:
    """Build the selected diffusion model."""

    match scheme:
        case "cs":
            return ViscousConfig.cs(
                viscosity=viscosity,
                characteristic_distance=spacing,
            )
        case "rwm":
            return ViscousConfig.rwm(
                viscosity=viscosity,
                characteristic_distance=spacing,
            )
        case "dvh":
            return ViscousConfig.dvh(
                h=spacing,
                viscosity=viscosity,
                threshold=1.0e-5,
                threshold_mode="budget",
                dvh_rd_ratio=DVH_RD_RATIO,
                max_nodes=max_grid_nodes,
                regen_radius_ratio=1.5,
            )
        case "gbd":
            return ViscousConfig.gbd(
                h=spacing,
                viscosity=viscosity,
                threshold=1.0e-5,
                threshold_mode="budget",
                max_nodes=max_grid_nodes,
                regen_radius_ratio=1.5,
            )
        case _:
            raise ValueError(f"Unknown diffusion scheme: {scheme}")


def cadence_steps(interval: float, time_step: float) -> int:
    """Convert a physical output cadence to the nearest whole solver step."""
    return max(1, round(interval / time_step))


def column_distribution(
    bounds: list[float],
    particle_replicas: int = 1,
    spacing: float = SPACING,
    particle_radius: float = PARTICLE_RADIUS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extrude a triangular core lattice through a uniformly loaded column."""

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

    if particle_replicas > 1:
        positions = np.repeat(positions, particle_replicas, axis=0)
        volumes = np.repeat(volumes / particle_replicas, particle_replicas)
        radii = np.repeat(radii, particle_replicas)
    return positions, volumes, radii


def write_run_metadata(
    samples_dir: Path,
    case: Case,
    scheme: str,
    viscosity: float,
    time_step: float,
    sample_steps: int,
    backup_steps: int,
    particle_replicas: int,
    particle_spacing: float,
    particle_radius: float,
    diffusion_spacing: float,
) -> None:
    """Keep the physical constants next to the sampled results, not in a backup."""

    samples_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "case": case.name,
        "scheme": scheme,
        "circulations": case.circulations,
        "viscosity": viscosity,
        "core_radius": CORE_RADIUS,
        "separation": SEPARATION,
        "column_half_length": COLUMN_LENGTH * CORE_RADIUS / 2,
        "total_time": case.total_time,
        "time_step": time_step,
        "in_plane_spacing": particle_spacing,
        "column_spacing": COLUMN_SPACING,
        "particle_radius": particle_radius,
        "particle_replicas": particle_replicas,
        "diffusion_grid_spacing": diffusion_spacing,
        "field_spacing": FIELD_SPACING,
        "sample_interval": sample_steps * time_step,
        "raw_backup_interval": backup_steps * time_step,
    }
    (samples_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def record_runtime(samples_dir: Path, wall_time_seconds: float) -> None:
    """Add the measured wall time to an existing run metadata file."""

    path = samples_dir / "run_metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["wall_time_seconds"] = wall_time_seconds
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run_case(
    case: Case,
    scheme: str,
    sample_period: float,
    backup_period: float,
    *,
    solution_dir: Path = SOLUTION_DIR,
    random_seed: int = RWM_FIRST_SEED,
    particle_replicas: int = 1,
) -> None:
    """Run one physical case with one diffusion method."""

    started_at = perf_counter()
    viscosity = GAMMA / REYNOLDS_NUMBER
    particle_spacing = SPACING
    particle_radius = 1.50 * particle_spacing
    diffusion_spacing = (
        PAIR_GRID_SPACING
        if case.name != "vortex" and scheme in {"dvh", "gbd"}
        else particle_spacing
    )
    viscous = viscous_setup(
        scheme,
        viscosity,
        GRID_NODE_LIMITS[case.name].get(scheme),
        spacing=diffusion_spacing,
    )
    if scheme == "dvh":
        time_step = float(f"{viscous.dvh_required_dt():.3g}")
    elif scheme == "gbd":
        time_step = GBD_TIME_STEP if case.name == "vortex" else PAIR_GBD_TIME_STEP
    elif scheme == "rwm":
        time_step = RWM_TIME_STEP if case.name == "vortex" else RWM_PAIR_TIME_STEP
    else:
        time_step = TIME_STEP
    sample_steps = cadence_steps(sample_period, time_step)
    backup_steps = cadence_steps(min(backup_period, case.total_time), time_step)

    initial_half_width = max(abs(y) for y in case.y_positions) + 7.0 * CORE_RADIUS
    final_core_radius = np.sqrt(
        CORE_RADIUS**2 + 4.0 * BETA_RMAX**2 * viscosity * case.total_time
    )
    if case.name == "vortex":
        lateral_half_width = initial_half_width
        axial_half_length = COLUMN_LENGTH * CORE_RADIUS / 2
    else:
        diffusion_padding = 4.0 * final_core_radius
        lateral_half_width = max(abs(y) for y in case.y_positions) + diffusion_padding
        axial_half_length = COLUMN_LENGTH * CORE_RADIUS / 2 + diffusion_padding

    domain_bounds = [
        -lateral_half_width,
        lateral_half_width + case.downstream_length * SEPARATION,
        -lateral_half_width,
        lateral_half_width,
        -axial_half_length,
        axial_half_length,
    ]
    case_name = f"{case.name}_{scheme}"

    initial_bounds = domain_bounds.copy()
    initial_bounds[:4] = [
        -initial_half_width,
        initial_half_width,
        -initial_half_width,
        initial_half_width,
    ]
    initial_bounds[4:] = [
        -COLUMN_LENGTH * CORE_RADIUS / 2,
        COLUMN_LENGTH * CORE_RADIUS / 2,
    ]
    positions, volumes, radii = column_distribution(
        initial_bounds,
        particle_replicas,
        spacing=particle_spacing,
        particle_radius=particle_radius,
    )
    vortex_age = (CORE_RADIUS / BETA_RMAX) ** 2 / (4.0 * viscosity)

    final_samplers = []
    scheduled_samplers = []
    if case.name == "vortex":
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
            case.name,
            SEPARATION,
            COLUMN_LENGTH * CORE_RADIUS,
            slab_half_width=1.5 * COLUMN_SPACING,
        )
        scheduled_samplers.append(pair_sampler)
        final_samplers.append(pair_sampler)

    solver = Solver(
        setup=VPMSetup.dns_simulation(
            time_step_size=time_step,
            processing_unit="VULKAN",
            random_seed=random_seed,
            advection=AdvectionConfig(scheme="NONE" if case.name == "vortex" else "RK3"),
            viscous=viscous,
            stretching=StretchingConfig.disabled(),
            velocity=VelocityConfig.treecode(
                theta=0.7, multipole_order=3, sort_particle_targets=True
            ),
            logging_frequency=sample_steps,
            backup_frequency=backup_steps,
            backup_file_name=case_name,
            backup_directory=str(solution_dir),
            sample_subdirectory=case_name,
            samplers=scheduled_samplers,
            final_samplers=final_samplers,
            clean=False,
            vpm_domain_bounds=domain_bounds,
        )
    )
    write_run_metadata(
        resolve_samples_dir(solution_dir, case_name),
        case,
        scheme,
        viscosity,
        time_step,
        sample_steps,
        backup_steps,
        particle_replicas,
        particle_spacing,
        particle_radius,
        diffusion_spacing,
    )

    for group_id, (circulation, y_position) in enumerate(
        zip(case.circulations, case.y_positions, strict=True)
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

    if case.name != "vortex":
        solver.execute_final_samplers()
    number_of_steps = round(case.total_time / time_step)

    for _ in range(solver.time_step, number_of_steps):
        solver.update_state()

    solver.execute_final_samplers()
    solver.reset_gpu()
    record_runtime(
        resolve_samples_dir(solution_dir, case_name),
        perf_counter() - started_at,
    )


def run_rwm_vortex(case: Case, sample_period: float, backup_period: float) -> None:
    """Run and average the independent realizations required by RWM."""

    started_at = perf_counter()
    run_case(
        case,
        "rwm",
        sample_period,
        backup_period,
        particle_replicas=RWM_PARTICLE_REPLICAS,
    )
    target_dir = resolve_samples_dir(SOLUTION_DIR, "vortex_rwm")
    member_dirs = [target_dir]

    with TemporaryDirectory(prefix="openonda-rwm-") as temporary_directory:
        temporary_root = Path(temporary_directory)
        for seed in range(RWM_FIRST_SEED + 1, RWM_FIRST_SEED + RWM_REALIZATIONS):
            member_solution = temporary_root / f"member-{seed}" / "solution"
            run_case(
                case,
                "rwm",
                case.total_time + RWM_TIME_STEP,
                case.total_time + RWM_TIME_STEP,
                solution_dir=member_solution,
                random_seed=seed,
                particle_replicas=RWM_PARTICLE_REPLICAS,
            )
            member_dirs.append(resolve_samples_dir(member_solution, "vortex_rwm"))

        average_final_samples(
            target_dir,
            member_dirs,
            realizations=RWM_REALIZATIONS,
            particle_replicas=RWM_PARTICLE_REPLICAS,
        )
    record_runtime(target_dir, perf_counter() - started_at)


def run_rwm_pair(case: Case, sample_period: float, backup_period: float) -> None:
    """Run a vortex pair as an ensemble of independent RWM realizations."""

    started_at = perf_counter()
    run_case(case, "rwm", sample_period, backup_period)
    case_name = f"{case.name}_rwm"
    target_dir = resolve_samples_dir(SOLUTION_DIR, case_name)
    member_dirs = [target_dir]

    with TemporaryDirectory(prefix="openonda-rwm-") as temporary_directory:
        temporary_root = Path(temporary_directory)
        for seed in range(RWM_FIRST_SEED + 1, RWM_FIRST_SEED + RWM_REALIZATIONS):
            member_solution = temporary_root / f"member-{seed}" / "solution"
            run_case(
                case,
                "rwm",
                sample_period,
                case.total_time + RWM_PAIR_TIME_STEP,
                solution_dir=member_solution,
                random_seed=seed,
            )
            member_dirs.append(resolve_samples_dir(member_solution, case_name))

        average_pair_diagnostics(
            target_dir,
            member_dirs,
            realizations=RWM_REALIZATIONS,
        )
    record_runtime(target_dir, perf_counter() - started_at)


def main(arguments: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if arguments is None else arguments
    case_name, scheme, sample_period, backup_period = arguments
    case = CASES[case_name]
    if scheme == "rwm":
        if case_name == "vortex":
            run_rwm_vortex(case, float(sample_period), float(backup_period))
        else:
            run_rwm_pair(case, float(sample_period), float(backup_period))
    else:
        run_case(
            case,
            scheme,
            float(sample_period),
            float(backup_period),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
