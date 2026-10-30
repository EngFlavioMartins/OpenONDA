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

# Restart mode
START_FROM = "latest"

# Physical problem
CUBE_SIDE = 1.0
FREESTREAM_VELOCITY = 1.0
DENSITY = 1.0
REYNOLDS_NUMBER = 1000.0
KINEMATIC_VISCOSITY = FREESTREAM_VELOCITY * CUBE_SIDE / REYNOLDS_NUMBER

# Startup vortex, matching the coupled case
DISTURBANCE_AMPLITUDE = 5.0e-2  # m^2/s, vector-potential peak
DISTURBANCE_RADIUS = 0.5  # m
DISTURBANCE_CENTRE = (1.0, 0.0, 0.0)  # m, half a side behind the rear face
DISTURBANCE_AXIS = (0.0, -1.0, 2.0)

# Outer domain and mesh refinement
DOMAIN = (-6.48, 12.96, -6.48, 6.48, -6.48, 6.48)
BACKGROUND_CELL_SIZE_RATIO = 8.0
WAKE_CELL_SIZE_RATIO = 2.0
NEAR_BODY = (-1.8, 4.3, -1.8, 1.8, -1.8, 1.8)  # m, finest spacing
WAKE = (-2.0, 12.0, -2.0, 2.0, -2.0, 2.0)  # m, wake refinement

# Time integration and output
CORES = 4
END_TIME = 100.0
TIME_STEP_SIZE = 0.005
MAX_COURANT_NUMBER = 0.5
OUTPUT_INTERVAL = 2.0  # s, FVM fields
BACKUP_INTERVAL = 1.0  # s
FORCE_SAMPLE_INTERVAL = 0.04  # s
PROFILE_SAMPLE_INTERVAL = 0.24  # s
PROFILE_SPACING = 0.06
SLICE_INTERVAL = 1.0  # s
SLICE_WINDOW = (-1.8, 12.0, -2.0, 2.0)  # m, mid-plane bounds
SLICE_SPACING = 0.09  # m

CASE_DIR = Path(__file__).resolve().parent
VELOCITY = (FREESTREAM_VELOCITY, 0.0, 0.0)


def initial_velocity(position):
    """Freestream plus the curl of A (1 - r^2/R^2)^4 along the disturbance axis."""
    offset = position - np.asarray(DISTURBANCE_CENTRE)
    support = np.maximum(1.0 - np.sum(offset**2, axis=1) / DISTURBANCE_RADIUS**2, 0.0)
    swirl = 8.0 * DISTURBANCE_AMPLITUDE / DISTURBANCE_RADIUS**2 * support**3
    axis = np.asarray(DISTURBANCE_AXIS) / np.linalg.norm(DISTURBANCE_AXIS)
    return np.asarray(VELOCITY) + swirl[:, None] * np.cross(axis, offset)


def create_solver(name: str, h: float, time_step: float | None = None) -> fvm.FVMSolver:
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
    sample_profiles = fvm.EveryTime(PROFILE_SAMPLE_INTERVAL)
    samplers = (
        fvm.ForceSampler(
            patch_names=["cube"],
            reference_velocity=FREESTREAM_VELOCITY,
            reference_area=CUBE_SIDE**2,
            reference_length=CUBE_SIDE,
            schedule=fvm.EveryTime(FORCE_SAMPLE_INTERVAL),
        ),
        fvm.LineSampler(
            (DOMAIN[0], 0.0, 0.0),
            (DOMAIN[1], 0.0, 0.0),
            spacing=PROFILE_SPACING,
            n_neighbours=12,
            reconstruction="affine",
            file_name="centreline",
            schedule=sample_profiles,
        ),
        fvm.LineSampler(
            (DOMAIN[0], 0.75, 0.0),
            (DOMAIN[1], 0.75, 0.0),
            spacing=PROFILE_SPACING,
            n_neighbours=12,
            reconstruction="affine",
            file_name="offaxis_y075",
            schedule=sample_profiles,
        ),
        fvm.SurfaceSampler(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            SLICE_WINDOW,
            SLICE_SPACING,
            n_neighbours=12,
            reconstruction="affine",
            body_bounds=(-0.5 * CUBE_SIDE, 0.5 * CUBE_SIDE) * 3,
            file_name="slice_z0",
            schedule=fvm.EveryTime(SLICE_INTERVAL),
        ),
    )
    numerics = fvm.Numerics(
        # A fixed step matches the coupled FVM step; otherwise the Courant number sets it
        time_step_size=time_step or TIME_STEP_SIZE,
        time_step_control=None
        if time_step
        else fvm.MaximumCourantTimeStep(
            maximum=MAX_COURANT_NUMBER, maximum_time_step_size=TIME_STEP_SIZE
        ),
        transport=fvm.TransportConfig(density=DENSITY, kinematic_viscosity=KINEMATIC_VISCOSITY),
        schemes=fvm.DiscretizationConfig(
            convection_scheme="linear_upwind", gradient_scheme="gauss", time_scheme="backward"
        ),
        linear=fvm.LinearSolverConfig(
            pressure_solver="amg", pressure_tolerance=1.0e-7, pressure_relative_tolerance=0.005
        ),
        pimple=fvm.PimpleConfig(
            n_outer_correctors=2,
            n_nonorthogonal_correctors=1,
            velocity_relaxation=0.7,
            pressure_relaxation=0.3,
        ),
        execution=fvm.ComputeConfig(operator_backend="numba"),
        turbulence=fvm.TurbulenceConfig.equilibrium_smagorinsky(
            subgrid_kinetic_energy_coefficient=0.094, subgrid_dissipation_coefficient=1.048
        ),
    )
    case = fvm.FVMCase(
        name=name,
        directory=CASE_DIR,
        variant=name,
        numerics=numerics,
        initial_conditions=fvm.InitialFields(velocity=VELOCITY),
        run=fvm.RunPlan(end_time=END_TIME),
        output=fvm.Output(schedule=fvm.EveryTime(OUTPUT_INTERVAL)),
        samplers=fvm.Samplers(samplers),
        backup=fvm.Backup(schedule=fvm.EveryTime(BACKUP_INTERVAL)),
        logging=fvm.LoggingConfig(detailed_timing=True),  # wall time per step
        mesh=mesh,
        boundaries=(
            fvm.BoundaryConfig.inlet("inlet", VELOCITY),
            fvm.BoundaryConfig.outlet("outlet", kinematic_pressure=0.0),
            fvm.BoundaryConfig.slip("ymin"),
            fvm.BoundaryConfig.slip("ymax"),
            fvm.BoundaryConfig.slip("zmin"),
            fvm.BoundaryConfig.slip("zmax"),
            fvm.BoundaryConfig.wall("cube"),
        ),
        cores=CORES,
    )
    return fvm.FVMSolver(case)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--name", default="grid_h0045")
    parser.add_argument("-h", type=float, default=0.045)
    parser.add_argument("--time-step", type=float)
    arguments = parser.parse_args()

    with create_solver(arguments.name, arguments.h, arguments.time_step) as solver:
        # The startup vortex behind the cube; a restart replaces it with the backup
        cell_centres = solver.geo_data["cell_centre"][: solver.mesh_data["n_cells"]]
        solver.set_initial_velocity(initial_velocity(cell_centres))
        solver.run(start_from=START_FROM)
        if arguments.time_step:
            return
        fvm.update_grid_study(
            solver,
            arguments.h,
            profiles=("centreline", "offaxis_y075"),
        )


if __name__ == "__main__":
    main()
