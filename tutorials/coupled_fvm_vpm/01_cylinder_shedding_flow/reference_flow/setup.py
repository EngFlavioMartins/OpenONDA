#!/usr/bin/env python3
"""Body-fitted cylinder flow at Re = 150."""

import argparse
import math
from pathlib import Path

import openonda.fvm as fvm
import openonda.fvm.mesher as msh
from openonda.cylinder_case import DEFAULT_CYLINDER_CASE

# Physical problem
START_FROM = "latest"  # Resume the latest backup; ./allrun.sh cleans first.

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
TIME_STEP_SIZE = 0.001
MAXIMUM_TIME_STEP_SIZE = 0.01
MAXIMUM_COURANT_NUMBER = 0.7
OUTPUT_INTERVAL = 5.0
FORCE_SAMPLE_INTERVAL = 0.04
PROFILE_SAMPLE_INTERVAL = 0.1
PROFILE_SPACING = 0.08

CASE_DIR = Path(__file__).resolve().parent
VELOCITY = [FREESTREAM_VELOCITY, 0.0, 0.0]


def create_solver(
    name: str,
    h: float,
    *,
    output_root: Path | None = None,
    end_time: float | None = None,
    cores: int | None = None,
) -> fvm.FVMSolver:
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

    setup = fvm.FVMSetup(
        backup=fvm.BackupConfig(schedule=fvm.RunSchedule(every_time=OUTPUT_INTERVAL), write_at_end=True),
        case_name=name,
        cores=CORES if cores is None else cores,
        time=fvm.TimeConfig(
            time_step_size=TIME_STEP_SIZE,
            end_time=END_TIME if end_time is None else end_time,
            output_schedule=fvm.RunSchedule(every_time=OUTPUT_INTERVAL),
            adjustment=fvm.MaximumCourantTimeStep(
                maximum=MAXIMUM_COURANT_NUMBER,
                maximum_time_step_size=MAXIMUM_TIME_STEP_SIZE,
            ),
        ),
        schemes=fvm.DiscretizationConfig(
            convection_scheme="limitedLinear",
            gradient_scheme="lsq",
            time_scheme="euler_implicit",
        ),
        linear=fvm.LinearSolverConfig(
            pressure_solver="amg",
            pressure_tolerance=1.0e-7,
            pressure_relative_tolerance=0.005,
            momentum_tolerance=1.0e-6,
            momentum_relative_tolerance=0.05,
        ),
        pimple=fvm.PimpleControl(algorithm="PISO", n_correctors=2),
        samplers=(
            fvm.ForceSampler(
                patch_names=["cylinder"],
                reference_velocity=FREESTREAM_VELOCITY,
                reference_area=DIAMETER * SPAN,
                reference_length=DIAMETER,
                file_name="forces_history",
                schedule=fvm.RunSchedule(every_time=FORCE_SAMPLE_INTERVAL),
            ),
            fvm.LineSampler(
                start=[-2.0, 0.0, 0.0],
                end=[12.0, 0.0, 0.0],
                spacing=PROFILE_SPACING,
                k=12,
                reconstruction="affine",
                file_name="centreline",
                schedule=fvm.RunSchedule(every_time=PROFILE_SAMPLE_INTERVAL),
            ),
            *(
                fvm.LineSampler(
                    start=[1.0, -1.0, z],
                    end=[1.0, 1.0, z],
                    spacing=PROFILE_SPACING,
                    k=12,
                    reconstruction="affine",
                    file_name=name,
                    schedule=fvm.RunSchedule(every_time=PROFILE_SAMPLE_INTERVAL),
                )
                for name, z in (
                    ("span_lower", -SPAN / 4),
                    ("span_middle", 0.0),
                    ("span_upper", SPAN / 4),
                )
            ),
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
    artifact_root = CASE_DIR if output_root is None else Path(output_root)
    return fvm.create_fvm_solver(
        setup,
        case_dir=artifact_root,
        solution_dir=artifact_root / "solution" / name,
        samples_dir=artifact_root / "samples" / name,
        mesh=mesh,
    )


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--name", default="grid_h004")
    parser.add_argument("-h", type=float, default=0.04)
    arguments = parser.parse_args()

    with create_solver(arguments.name, arguments.h) as solver:
        solver.run(start_from=START_FROM)
        fvm.update_grid_study(solver, arguments.h, profiles=("centreline",))


if __name__ == "__main__":
    main()
