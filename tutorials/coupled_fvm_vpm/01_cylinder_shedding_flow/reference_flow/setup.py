#!/usr/bin/env python3
"""Body-fitted cylinder flow at Re = 150."""

import argparse
import math
from pathlib import Path

import openonda.fvm as fvm
import openonda.fvm.mesher as msh
from openonda.cylinder_campaign import initialize_cylinder_perturbation
from openonda.cylinder_case import DEFAULT_CYLINDER_CASE
from openonda.tutorial_support import cylinder_sampling as observations

# Physical problem
START_FROM = "latest"  # allrun.sh preserves outputs; allclean.sh is explicit.

DIAMETER = 1.0
FREESTREAM_VELOCITY = 1.0
DENSITY = 1.0
REYNOLDS_NUMBER = 150.0
KINEMATIC_VISCOSITY = FREESTREAM_VELOCITY * DIAMETER / REYNOLDS_NUMBER

# Domain and mesh refinement
SPAN = DEFAULT_CYLINDER_CASE.resolved_span
DOMAIN = (-8.0, 24.0, -10.0, 10.0, -0.5 * SPAN, 0.5 * SPAN)
BACKGROUND_CELL_SIZE_RATIO = 16.0
NEAR_WAKE_CELL_SIZE_RATIO = 2.0
WAKE_CELL_SIZE_RATIO = 4.0
NEAR_BODY = (-1.5, 1.5, -1.5, 1.5)
NEAR_WAKE = (-2.0, 6.0, -2.0, 2.0)
WAKE = (-2.5, 12.0, -2.5, 2.5)

# Time, output and sampling
CORES = 6
END_TIME = DEFAULT_CYLINDER_CASE.reference_end_time
TIME_STEP_SIZE = 0.008
EXCHANGE_TIME_STEP_SIZE = 0.04

CASE_DIR = Path(__file__).resolve().parent
VELOCITY = [FREESTREAM_VELOCITY, 0.0, 0.0]


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
    span_layers = max(4, math.ceil(SPAN / h))
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
        levels=tuple(DOMAIN[4] + layer * SPAN / span_layers for layer in range(span_layers + 1)),
    )

    physical_end = END_TIME if end_time is None else end_time
    clocks = dict(
        end=physical_end, exchange_dt=EXCHANGE_TIME_STEP_SIZE, fvm_time_step=TIME_STEP_SIZE
    )
    sampling = observations.sampling_plan(observations.PROFILES, **clocks)
    backups = observations.sampling_plan(observations.BACKUPS, **clocks)
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
        samplers=observations.fvm_samplers(
            True, span=SPAN, freestream_speed=FREESTREAM_VELOCITY, diameter=DIAMETER, **clocks
        ),
        transport=fvm.TransportConfig(
            density=DENSITY,
            kinematic_viscosity=KINEMATIC_VISCOSITY,
        ),
        turbulence=fvm.TurbulenceConfig.none(),
        boundaries=[
            fvm.BoundaryConfig.inlet("inlet", VELOCITY),
            fvm.BoundaryConfig.outlet("outlet", kinematic_pressure=0.0),
            fvm.BoundaryConfig.slip("ymin"),
            fvm.BoundaryConfig.slip("ymax"),
            fvm.BoundaryConfig.slip("zmin"),
            fvm.BoundaryConfig.slip("zmax"),
            fvm.BoundaryConfig.wall("cylinder"),
        ],
        initial_velocity=VELOCITY,
    )
    return setup, mesh


def create_solver(
    name: str,
    h: float,
    *,
    output_root: Path | None = None,
    end_time: float | None = None,
    cores: int | None = None,
) -> fvm.FVMSolver:
    setup, mesh = build_case(name, h, end_time=end_time, cores=cores)
    artifact_root = CASE_DIR if output_root is None else Path(output_root)
    return fvm.create_fvm_solver(
        setup,
        case_dir=artifact_root,
        solution_dir=artifact_root / "solution",
        samples_dir=artifact_root / "samples",
        mesh=mesh,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--help", action="help", help="Show this help message and exit.")
    parser.add_argument("--name", default="phase_h004")
    parser.add_argument("-h", type=float, default=0.04)
    arguments = parser.parse_args()

    with create_solver(arguments.name, arguments.h) as solver:
        initialize_cylinder_perturbation(solver, SPAN)
        solver.run(start_from=START_FROM)


if __name__ == "__main__":
    main()
