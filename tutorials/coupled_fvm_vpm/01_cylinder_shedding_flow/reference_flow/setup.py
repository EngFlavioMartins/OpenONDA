#!/usr/bin/env python3
"""Two-dimensional body-fitted cylinder flow at Re = 150 and unit span."""

import argparse
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import openonda.fvm as fvm
import openonda.fvm.mesher as msh

from .assets.initial_conditions import cylinder_initial_velocity

# Physical problem
START_FROM = "latest"  # allrun.sh preserves outputs; allclean.sh is explicit.

DIAMETER = 1.0
FREESTREAM_VELOCITY = 1.0
STARTUP_FREESTREAM_VELOCITY = (1.0, 0.1, 0.0)
STARTUP_DURATION = 2.0
STARTUP_TRANSITION_DURATION = 1.0
INITIAL_PERTURBATION = {"amplitude": 1e-3, "radius": 0.5, "centre": 0.65}
DENSITY = 1.0
REYNOLDS_NUMBER = 150.0
KINEMATIC_VISCOSITY = FREESTREAM_VELOCITY * DIAMETER / REYNOLDS_NUMBER

# Domain and mesh refinement
SPAN = 1.0
DOMAIN = (-8.0, 24.0, -10.0, 10.0, -0.5 * SPAN, 0.5 * SPAN)
BACKGROUND_CELL_SIZE_RATIO = 16.0
NEAR_WAKE_CELL_SIZE_RATIO = 2.0
WAKE_CELL_SIZE_RATIO = 4.0
NEAR_BODY = (-1.5, 1.5, -1.5, 1.5)
NEAR_WAKE = (-2.0, 6.0, -2.0, 2.0)
WAKE = (-2.5, 12.0, -2.5, 2.5)

# Time, output and sampling
DEFAULT_NAME = "phase_h004"
DEFAULT_H = 0.04
GRID_SPACINGS = (0.09, 0.06, DEFAULT_H)  # Refine only the in-plane mesh, by 1.5.
CORES = 1
END_TIME = 100.0
STATISTICS_START = 40.0
TIME_STEP_SIZE = 0.008
EXCHANGE_TIME_STEP_SIZE = 0.04

# Physical output intervals
FORCES = 0.04
PROFILES = 0.2
SLICES = 0.4
BACKUPS = 1.0
VOLUMES = 4.0

CASE_DIR = Path(__file__).resolve().parent
VELOCITY = [FREESTREAM_VELOCITY, 0.0, 0.0]


def phase_lines(outer=False):
    """Identical coordinates in the reference and corresponding coupled region."""
    lo, hi, count = (1.6, 8.0, 65) if outer else (0.6, 1.4, 9)
    region = "wake" if outer else "near"
    return [
        (f"phase_{region}_{side}", [lo, y, 0.0], [hi, y, 0.0], count)
        for side, y in (("upper", 0.6), ("lower", -0.6))
    ]


def field_lines(reference):
    lines = [("centreline", [0.6, 0.0, 0.0], [1.4, 0.0, 0.0], 11)]
    lines.append(("span_middle", [1.0, -1.2, 0.0], [1.0, 1.2, 0.0], 31))
    if reference:
        for x in (2, 4):
            lines.append((f"transverse_x{x}", [x, -2.0, 0.0], [x, 2.0, 0.0], 51))
    return lines


def sampling_plan(period, *, exchange_dt, fvm_time_step):
    """Sampling periods expressed on the FVM and particle clocks."""
    substeps = round(exchange_dt / fvm_time_step)
    sample_steps = max(1, round(period / exchange_dt))
    return SimpleNamespace(
        sample_steps=sample_steps,
        fvm_sample_steps=sample_steps * substeps,
        fvm_slice_steps=max(1, round(SLICES / exchange_dt)) * substeps,
        fvm_output_steps=max(1, round(VOLUMES / exchange_dt)) * substeps,
    )


def fvm_samplers(reference, *, span, freestream_speed, diameter, exchange_dt, fvm_time_step):
    clocks = {"exchange_dt": exchange_dt, "fvm_time_step": fvm_time_step}
    force, profile = sampling_plan(FORCES, **clocks), sampling_plan(PROFILES, **clocks)
    fast = fvm.RunSchedule(every_n_steps=force.fvm_sample_steps)
    slow = fvm.RunSchedule(every_n_steps=profile.fvm_sample_steps)
    samplers = [
        fvm.ForceSampler(
            patch_names=["cylinder"],
            reference_velocity=freestream_speed,
            reference_area=diameter * span,
            reference_length=diameter,
            file_name="forces_history",
            schedule=fast,
        )
    ]
    for name, start, end, count in phase_lines() + (phase_lines(True) if reference else []):
        samplers.append(
            fvm.LineSampler(
                start=start,
                end=end,
                n_points=count,
                k=12,
                reconstruction="affine",
                file_name=name,
                schedule=fast,
            )
        )
    for name, start, end, count in field_lines(reference):
        samplers.append(
            fvm.LineSampler(
                start=start,
                end=end,
                n_points=count,
                k=12,
                reconstruction="affine",
                file_name=name,
                schedule=slow,
            )
        )
    samplers.append(
        fvm.SurfaceSampler(
            point=[0.0, 0.0, 0.0],
            normal=[0.0, 0.0, 1.0],
            bounds=[
                -1.6,
                8.0 if reference else 1.6,
                -2.0 if reference else -1.6,
                2.0 if reference else 1.6,
            ],
            spacing=0.1,
            file_name="midspan",
            schedule=fvm.RunSchedule(every_n_steps=profile.fvm_slice_steps),
            body_bounds=[-0.5, 0.5, -0.5, 0.5, -6.0, 6.0],
            body_geometry="cylinder_z",
        )
    )
    return tuple(samplers)


def build_case(
    name: str,
    h: float,
    *,
    end_time: float | None = None,
    cores: int | None = None,
):
    """Return the reference configuration and mesh without allocating a solver."""
    refinement_request = 4.0 * h / 3.0
    source_half_span = 16.0 * refinement_request
    source_domain = (*DOMAIN[:4], -source_half_span, source_half_span)
    patches = msh.BoxPatches(
        xmin="inlet",
        xmax="outlet",
        ymin="ymin",
        ymax="ymax",
        zmin="zmin",
        zmax="zmax",
    )
    mesh = msh.ExtrudedCartesianMesher(
        source=msh.CartesianMesher(
            domain=msh.BoxDomain(bounds=source_domain, patches=patches),
            surfaces=(msh.STLSurface(CASE_DIR / "assets/cylinder_long.stl", patch="cylinder"),),
            max_cell_size=BACKGROUND_CELL_SIZE_RATIO * h,
            refinements=(
                msh.BoxRefinement(
                    "nearBody",
                    (*NEAR_BODY, -source_half_span, source_half_span),
                    refinement_request,
                ),
                msh.BoxRefinement(
                    "nearWake",
                    (*NEAR_WAKE, -source_half_span, source_half_span),
                    NEAR_WAKE_CELL_SIZE_RATIO * refinement_request,
                ),
                msh.BoxRefinement(
                    "wake",
                    (*WAKE, -source_half_span, source_half_span),
                    WAKE_CELL_SIZE_RATIO * refinement_request,
                ),
            ),
            patch_refinements=(msh.PatchRefinement("cylinder", refinement_request),),
            surface_may_cross_domain_boundary=True,
        ),
        domain=msh.BoxDomain(bounds=DOMAIN, patches=patches),
        levels=(DOMAIN[4], DOMAIN[5]),
    )

    physical_end = END_TIME if end_time is None else end_time
    clocks = dict(exchange_dt=EXCHANGE_TIME_STEP_SIZE, fvm_time_step=TIME_STEP_SIZE)
    sampling = sampling_plan(PROFILES, **clocks)
    backups = sampling_plan(BACKUPS, **clocks)
    setup = fvm.FVMSetup(
        backup=fvm.BackupConfig(
            schedule=fvm.RunSchedule(every_n_steps=backups.fvm_sample_steps), write_at_end=True
        ),
        case_name=name,
        cores=CORES if cores is None else cores,
        time=fvm.TimeConfig(
            time_step_size=TIME_STEP_SIZE,
            end_time=physical_end,
            output_schedule=fvm.RunSchedule(every_n_steps=sampling.fvm_output_steps),
        ),
        schemes=fvm.DiscretizationConfig(
            convection_scheme="limitedLinear", gradient_scheme="lsq", time_scheme="backward"
        ),
        linear=fvm.LinearSolverConfig(
            pressure_solver="amg",
            pressure_tolerance=1.0e-7,
            pressure_relative_tolerance=0.005,
            momentum_tolerance=1.0e-6,
            momentum_relative_tolerance=0.05,
        ),
        pimple=fvm.PimpleControl(
            n_outer_correctors=2, n_correctors=2, velocity_relaxation=0.7, pressure_relaxation=0.3
        ),
        samplers=fvm_samplers(
            True, span=SPAN, freestream_speed=FREESTREAM_VELOCITY, diameter=DIAMETER, **clocks
        ),
        transport=fvm.TransportConfig(
            density=DENSITY,
            kinematic_viscosity=KINEMATIC_VISCOSITY,
        ),
        turbulence=fvm.TurbulenceConfig.none(),
        boundaries=[
            fvm.BoundaryConfig.inlet("inlet", list(STARTUP_FREESTREAM_VELOCITY)),
            fvm.BoundaryConfig.outlet("outlet", kinematic_pressure=0.0),
            fvm.BoundaryConfig.slip("ymin"),
            fvm.BoundaryConfig.slip("ymax"),
            fvm.BoundaryConfig.cyclic("zmin", "zmax"),
            fvm.BoundaryConfig.cyclic("zmax", "zmin"),
            fvm.BoundaryConfig.wall("cylinder"),
        ],
        velocity_boundaries=(
            fvm.VelocityBoundary(
                patches=("inlet",),
                velocity=fvm.VelocityRamp(
                    initial=STARTUP_FREESTREAM_VELOCITY,
                    final=tuple(VELOCITY),
                    start_time=STARTUP_DURATION - STARTUP_TRANSITION_DURATION,
                    end_time=STARTUP_DURATION,
                ),
            ),
            fvm.VelocityBoundary(
                patches=("ymin", "ymax"),
                velocity=fvm.VelocityRamp(
                    initial=STARTUP_FREESTREAM_VELOCITY,
                    final=tuple(VELOCITY),
                    start_time=STARTUP_DURATION - STARTUP_TRANSITION_DURATION,
                    end_time=STARTUP_DURATION,
                ),
                normal_only=True,
            ),
        ),
        initial_velocity=list(STARTUP_FREESTREAM_VELOCITY),
    )
    return setup, mesh


def create_solver(
    name: str,
    h: float,
    *,
    output_root: Path | None = None,
    end_time: float | None = None,
    cores: int | None = None,
    solution_dir: Path | None = None,
    samples_dir: Path | None = None,
) -> fvm.FVMSolver:
    setup, mesh = build_case(name, h, end_time=end_time, cores=cores)
    output_directory = CASE_DIR if output_root is None else Path(output_root)
    return fvm.create_fvm_solver(
        setup,
        case_dir=output_directory,
        solution_dir=output_directory / "solution" if solution_dir is None else solution_dir,
        samples_dir=output_directory / "samples" if samples_dir is None else samples_dir,
        mesh=mesh,
    )


def run_solver(solver: fvm.FVMSolver, *, start_from=START_FROM) -> None:
    """Apply the coupled case's startup schedule through native FVM evolution."""
    solver.run(
        start_from=start_from,
        initial_velocity=partial(
            cylinder_initial_velocity,
            freestream_velocity=STARTUP_FREESTREAM_VELOCITY,
            **INITIAL_PERTURBATION,
        ),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--help", action="help", help="Show this help message and exit.")
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("-h", type=float, default=DEFAULT_H)
    arguments = parser.parse_args(argv)
    with create_solver(arguments.name, arguments.h) as solver:
        run_solver(solver)


if __name__ == "__main__":
    main()
