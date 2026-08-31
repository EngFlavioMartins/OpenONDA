#!/usr/bin/env python3
"""Body-fitted Re=150 cylinder reference used by the grid study.

Example:
    python -u reference_flow_setup.py --dx 0.041666666666666664 \
        --case-name coarse
"""

from __future__ import annotations

import argparse
from pathlib import Path

import openonda.fvm as fvm

CASE_DIR = Path(__file__).resolve().parent
CYLINDER_STL = CASE_DIR.parent / "assets" / "cylinder_long.stl"

# ---- Physics -------------------------------------------------------------
DIAMETER = 1.0
CYLINDER_LENGTH = 4.0
REYNOLDS_NUMBER = 150.0
FREESTREAM_VELOCITY = [1.0, 0.0, 0.0]
KINEMATIC_VISCOSITY = 1.0 / REYNOLDS_NUMBER
DOMAIN = (-8.0, 20.0, -8.0, 8.0, -2.0, 2.0)

# ---- Time and output -----------------------------------------------------
TIME_STEP_SIZE = 0.001
TOTAL_TIME = 60.0
FORCE_INTERVAL_TIME = 0.02
LINE_INTERVAL_TIME = 0.1
SLICE_INTERVAL_TIME = 0.5
FIELD_INTERVAL_TIME = 2.5
SPANWISE_CELL_SIZE = 0.5
NUMBER_OF_CORES = 6


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dx", required=True, type=float, help="cylinder-wall cell size in D")
    parser.add_argument("--case-name", required=True, help="solution/ and samples/ subdirectory")
    return parser.parse_args()


def interval_steps(interval: float) -> int:
    steps = round(interval / TIME_STEP_SIZE)
    if abs(steps * TIME_STEP_SIZE - interval) > 1.0e-12:
        raise ValueError(f"Output interval {interval:g} is not divisible by dt={TIME_STEP_SIZE:g}")
    return steps


def grid_mesh(dx: float) -> fvm.ExplicitCylinderGridMesher:
    """Return the exact wall=dx, near=2dx, wake=4dx, far=12dx mesh."""
    return fvm.ExplicitCylinderGridMesher(
        domain=DOMAIN,
        surface_file=CYLINDER_STL,
        wall_patch_name="cylinder",
        wall_cell_size=dx,
        near_body_half_width=2.0,
        wake_half_width=4.0,
        wake_xmin=-4.0,
        interface_half_width=2.0 / 3.0,
        spanwise_cell_size=SPANWISE_CELL_SIZE,
    )


def samplers(dx: float) -> tuple:
    force_schedule = fvm.SamplingSchedule(every_n_steps=interval_steps(FORCE_INTERVAL_TIME))
    line_schedule = fvm.SamplingSchedule(every_n_steps=interval_steps(LINE_INTERVAL_TIME))
    slice_schedule = fvm.SamplingSchedule(every_n_steps=interval_steps(SLICE_INTERVAL_TIME))
    sample_spacing = min(0.125, 2.0 * dx)
    return (
        fvm.ForceSampler(
            patch_names=["cylinder"],
            reference_velocity=1.0,
            reference_area=DIAMETER * CYLINDER_LENGTH,
            reference_length=DIAMETER,
            moment_centre=[0.0, 0.0, 0.0],
            file_name="forces_history",
            schedule=force_schedule,
        ),
        fvm.LineSampler(
            start=[1.5, 0.0, 0.0],
            end=[1.5, 0.0, 0.0],
            n_points=1,
            k=12,
            reconstruction="affine",
            file_name="midspan_probe",
            schedule=force_schedule,
        ),
        fvm.LineSampler(
            start=[-2.0, 0.0, 0.0],
            end=[12.0, 0.0, 0.0],
            spacing=sample_spacing,
            k=12,
            reconstruction="affine",
            file_name="centreline",
            schedule=line_schedule,
        ),
        fvm.LineSampler(
            start=[1.0, -3.0, 0.0],
            end=[1.0, 3.0, 0.0],
            spacing=sample_spacing,
            k=12,
            reconstruction="affine",
            file_name="transverse_x1",
            schedule=line_schedule,
        ),
        fvm.LineSampler(
            start=[2.0, -3.0, 0.0],
            end=[2.0, 3.0, 0.0],
            spacing=sample_spacing,
            k=12,
            reconstruction="affine",
            file_name="transverse_x2",
            schedule=line_schedule,
        ),
        fvm.LineSampler(
            start=[4.0, -3.0, 0.0],
            end=[4.0, 3.0, 0.0],
            spacing=sample_spacing,
            k=12,
            reconstruction="affine",
            file_name="transverse_x4",
            schedule=line_schedule,
        ),
        fvm.SurfaceSampler(
            point=[0.0, 0.0, 0.0],
            normal=[0.0, 0.0, 1.0],
            bounds=[DOMAIN[0], DOMAIN[1], DOMAIN[2], DOMAIN[3]],
            spacing=sample_spacing,
            k=12,
            reconstruction="affine",
            file_name="midspan",
            schedule=slice_schedule,
            body_bounds=[-0.5, 0.5, -0.5, 0.5, -6.0, 6.0],
            body_geometry="cylinder_z",
        ),
    )


def solver_setup(case_name: str, dx: float) -> fvm.FVMSetup:
    return fvm.FVMSetup(
        case_name=case_name,
        cores=NUMBER_OF_CORES,
        execution=fvm.ComputeConfig(operator_backend="numba"),
        output=fvm.OutputConfig(
            format="vtk_xml",
            data_location="cell",
            encoding="appended",
            compression="lz4",
            precision="f32",
            asynchronous=False,
            ghost_layers=0,
        ),
        time=fvm.TimeConfig(
            time_step_size=TIME_STEP_SIZE,
            start_time=0.0,
            end_time=TOTAL_TIME,
            output_interval_steps=interval_steps(FIELD_INTERVAL_TIME),
            adjust_time_step=False,
        ),
        schemes=fvm.DiscretizationConfig(
            convection_scheme="limitedLinear",
            gradient_scheme="lsq",
            time_scheme="euler",
        ),
        linear=fvm.LinearSolverConfig(
            linear_solver="bicgstab",
            pressure_solver="amg",
            pressure_tolerance=1.0e-7,
            pressure_relative_tolerance=0.005,
            momentum_tolerance=1.0e-6,
            momentum_relative_tolerance=0.05,
        ),
        pimple=fvm.PimpleControl(
            n_correctors=2,
            n_outer_correctors=2,
            n_orthogonal_correctors=1,
            velocity_relaxation=0.7,
            pressure_relaxation=0.3,
        ),
        samplers=samplers(dx),
        transport=fvm.TransportConfig(
            density=1.0,
            kinematic_viscosity=KINEMATIC_VISCOSITY,
        ),
        turbulence=fvm.TurbulenceConfig.none(),
        boundaries=[
            fvm.BoundaryConfig.inlet("inlet", FREESTREAM_VELOCITY),
            fvm.BoundaryConfig.outlet("outlet", kinematic_pressure=0.0),
            fvm.BoundaryConfig.slip("ymin"),
            fvm.BoundaryConfig.slip("ymax"),
            fvm.BoundaryConfig.slip("zmin"),
            fvm.BoundaryConfig.slip("zmax"),
            fvm.BoundaryConfig.wall("cylinder"),
        ],
        initial_velocity=FREESTREAM_VELOCITY,
        initial_kinematic_pressure=0.0,
    )


def main() -> None:
    arguments = parse_arguments()
    if arguments.dx <= 0.0:
        raise ValueError("--dx must be positive")
    if "/" in arguments.case_name or "\\" in arguments.case_name:
        raise ValueError("--case-name must be a simple directory name")

    solution_dir = CASE_DIR / "solution" / arguments.case_name
    samples_dir = CASE_DIR / "samples" / arguments.case_name
    solver = fvm.create_fvm_solver(
        solver_setup(arguments.case_name, arguments.dx),
        case_dir=CASE_DIR,
        solution_dir=solution_dir,
        samples_dir=samples_dir,
        mesh=grid_mesh(arguments.dx),
    )
    try:
        solver.write_run_manifest()
        solver.write_vtk()
        number_of_steps = round(TOTAL_TIME / TIME_STEP_SIZE)
        checkpoint_interval = interval_steps(FIELD_INTERVAL_TIME)
        for _ in range(number_of_steps):
            solver.advance()
            if solver.step % checkpoint_interval == 0:
                solver.save_state(solution_dir / "checkpoint")
        if solver.step % checkpoint_interval:
            solver.write_vtk()
            solver.save_state(solution_dir / "checkpoint")
    finally:
        solver.close()


if __name__ == "__main__":
    main()
