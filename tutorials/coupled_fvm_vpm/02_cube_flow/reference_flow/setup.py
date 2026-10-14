#!/usr/bin/env python3
"""Body-fitted FVM reference for the coupled cube flow at Re = 1000.

The same vortex behind the cube as in the coupled case breaks the wake symmetry.
Distant slip walls and inlet approximate the unbounded stream.
"""

import argparse
from pathlib import Path

import numpy as np

import openonda.fvm as fvm
import openonda.fvm.mesher as msh

# Physical problem
START_FROM = "latest"  # Resume the latest backup; ./allrun.sh cleans first.

CUBE_SIDE = 1.0
FREESTREAM_VELOCITY = 1.0
DENSITY = 1.0
REYNOLDS_NUMBER = 1000.0
KINEMATIC_VISCOSITY = FREESTREAM_VELOCITY * CUBE_SIDE / REYNOLDS_NUMBER

# Startup: the same vortex behind the cube as in the coupled case
DISTURBANCE_AMPLITUDE = 5.0e-2  # m^2/s, vector-potential peak
DISTURBANCE_RADIUS = 0.5  # m
DISTURBANCE_CENTRE = (1.0, 0.0, 0.0)  # m, half a side behind the rear face
DISTURBANCE_AXIS = (0.0, -1.0, 2.0)

# Domain and mesh refinement
DOMAIN = (-6.48, 12.96, -6.48, 6.48, -6.48, 6.48)
BACKGROUND_CELL_SIZE_RATIO = 8.0
WAKE_CELL_SIZE_RATIO = 2.0
NEAR_BODY = (-1.8, 4.3, -1.8, 1.8, -1.8, 1.8)  # the coupled FVM box, at the finest spacing
WAKE = (-2.0, 12.0, -2.0, 2.0, -2.0, 2.0)  # 2h over the compared wake, to x = 12 D

# Time, output and sampling
CORES = 4
END_TIME = 100.0
TIME_STEP_SIZE = 0.005
MAX_COURANT_NUMBER = 0.5
OUTPUT_INTERVAL = 2.0  # s, FVM fields
BACKUP_INTERVAL = 1.0  # s
FORCE_SAMPLE_INTERVAL = 0.05
PROFILE_SAMPLE_INTERVAL = 0.25
PROFILE_SPACING = 0.06

CASE_DIR = Path(__file__).resolve().parent
VELOCITY = [FREESTREAM_VELOCITY, 0.0, 0.0]


def initial_velocity(position):
    """Freestream plus the curl of A (1 - r^2/R^2)^4 along the disturbance axis."""
    offset = position - np.asarray(DISTURBANCE_CENTRE)
    support = np.maximum(1.0 - np.sum(offset**2, axis=1) / DISTURBANCE_RADIUS**2, 0.0)
    swirl = 8.0 * DISTURBANCE_AMPLITUDE / DISTURBANCE_RADIUS**2 * support**3
    axis = np.asarray(DISTURBANCE_AXIS) / np.linalg.norm(DISTURBANCE_AXIS)
    return np.asarray(VELOCITY) + swirl[:, None] * np.cross(axis, offset)


def create_solver(name: str, h: float) -> fvm.FVMSolver:
    mesh = msh.CartesianMesher(
        domain=msh.BoxDomain(
            bounds=DOMAIN,
            patches=msh.BoxPatches(
                xmin="inlet",
                xmax="outlet",
                ymin="ymin",
                ymax="ymax",
                zmin="zmin",
                zmax="zmax",
            ),
        ),
        surfaces=(msh.STLSurface(CASE_DIR / "assets/cube.stl", patch="cube"),),
        max_cell_size=BACKGROUND_CELL_SIZE_RATIO * h,
        cell_size_anchor=h,
        refinements=(
            msh.BoxRefinement("nearBody", NEAR_BODY, h),
            msh.BoxRefinement("wake", WAKE, WAKE_CELL_SIZE_RATIO * h),
        ),
        patch_refinements=(msh.PatchRefinement("cube", h),),
    )
    setup = fvm.FVMSetup(
        backup=fvm.BackupConfig(
            schedule=fvm.RunSchedule(every_time=BACKUP_INTERVAL), write_at_end=True
        ),
        case_name=name,
        cores=CORES,
        execution=fvm.ComputeConfig(operator_backend="numba"),
        time=fvm.TimeConfig(
            time_step_size=TIME_STEP_SIZE,
            end_time=END_TIME,
            output_schedule=fvm.RunSchedule(every_time=OUTPUT_INTERVAL),
            adjustment=fvm.MaximumCourantTimeStep(
                maximum=MAX_COURANT_NUMBER,
                maximum_time_step_size=TIME_STEP_SIZE,
            ),
        ),
        schemes=fvm.DiscretizationConfig(
            convection_scheme="linearUpwind",
            gradient_scheme="gauss",
            time_scheme="backward",
        ),
        linear=fvm.LinearSolverConfig(
            pressure_solver="amg",
            pressure_tolerance=1.0e-7,
            pressure_relative_tolerance=0.005,
        ),
        pimple=fvm.PimpleControl(
            n_outer_correctors=2,
            n_nonorthogonal_correctors=1,
            velocity_relaxation=0.7,
            pressure_relaxation=0.3,
        ),
        samplers=(
            fvm.ForceSampler(
                patch_names=["cube"],
                reference_velocity=FREESTREAM_VELOCITY,
                reference_area=CUBE_SIDE**2,
                reference_length=CUBE_SIDE,
                schedule=fvm.RunSchedule(every_time=FORCE_SAMPLE_INTERVAL),
            ),
            fvm.LineSampler(
                start=[DOMAIN[0], 0.0, 0.0],
                end=[DOMAIN[1], 0.0, 0.0],
                spacing=PROFILE_SPACING,
                k=12,
                reconstruction="affine",
                file_name="centreline",
                schedule=fvm.RunSchedule(every_time=PROFILE_SAMPLE_INTERVAL),
            ),
            fvm.LineSampler(
                start=[DOMAIN[0], 0.75, 0.0],
                end=[DOMAIN[1], 0.75, 0.0],
                spacing=PROFILE_SPACING,
                k=12,
                reconstruction="affine",
                file_name="offaxis_y075",
                schedule=fvm.RunSchedule(every_time=PROFILE_SAMPLE_INTERVAL),
            ),
        ),
        transport=fvm.TransportConfig(
            density=DENSITY,
            kinematic_viscosity=KINEMATIC_VISCOSITY,
        ),
        turbulence=fvm.TurbulenceConfig.equilibrium_smagorinsky(
            subgrid_kinetic_energy_coefficient=0.094,
            subgrid_dissipation_coefficient=1.048,
        ),
        boundaries=[
            fvm.BoundaryConfig.inlet("inlet", VELOCITY),
            fvm.BoundaryConfig.outlet("outlet", kinematic_pressure=0.0),
            fvm.BoundaryConfig.slip("ymin"),
            fvm.BoundaryConfig.slip("ymax"),
            fvm.BoundaryConfig.slip("zmin"),
            fvm.BoundaryConfig.slip("zmax"),
            fvm.BoundaryConfig.wall("cube"),
        ],
        initial_velocity=VELOCITY,
    )
    return fvm.create_fvm_solver(
        setup,
        case_dir=CASE_DIR,
        solution_dir=CASE_DIR / "solution" / name,
        samples_dir=CASE_DIR / "samples" / name,
        mesh=mesh,
    )


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--name", default="grid_h0045")
    parser.add_argument("-h", type=float, default=0.045)
    arguments = parser.parse_args()

    with create_solver(arguments.name, arguments.h) as solver:
        solver.run(start_from=START_FROM, initial_velocity=initial_velocity)
        fvm.update_grid_study(
            solver,
            arguments.h,
            profiles=("centreline", "offaxis_y075"),
        )


if __name__ == "__main__":
    main()
