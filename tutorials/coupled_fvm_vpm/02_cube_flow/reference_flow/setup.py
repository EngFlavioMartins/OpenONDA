#!/usr/bin/env python3
"""Body-fitted FVM reference for the coupled cube flow at Re = 1000.

The same short transverse startup as the coupled case breaks the wake symmetry.
Distant slip walls and inlet approximate the unbounded stream.
"""

import argparse
from pathlib import Path

import openonda.fvm as fvm
import openonda.fvm.mesher as msh

# Physical problem
START_FROM = "latest"  # Resume the latest backup; ./allrun.sh cleans first.

CUBE_SIDE = 1.0
FREESTREAM_VELOCITY = 1.0
DENSITY = 1.0
REYNOLDS_NUMBER = 1000.0
KINEMATIC_VISCOSITY = FREESTREAM_VELOCITY * CUBE_SIDE / REYNOLDS_NUMBER

# Startup: the same transverse freestream as the coupled case, decaying smoothly
STARTUP_VELOCITY = (1.0, 0.1, 0.05)  # m/s
TAPER_START = 1.0  # s, the transverse components then decay smoothly ...
TAPER_END = 2.0  # s, ... to the freestream

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
VELOCITY = (FREESTREAM_VELOCITY, 0.0, 0.0)


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
    startup = fvm.VelocityRamp(
        initial=STARTUP_VELOCITY,
        final=VELOCITY,
        start_time=TAPER_START,
        end_time=TAPER_END,
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
    )
    numerics = fvm.Numerics(
        time_step_size=TIME_STEP_SIZE,
        time_step_control=fvm.MaximumCourantTimeStep(
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
        initial_conditions=fvm.InitialFields(velocity=STARTUP_VELOCITY),
        run=fvm.RunPlan(end_time=END_TIME),
        output=fvm.Output(schedule=fvm.EveryTime(OUTPUT_INTERVAL)),
        samplers=fvm.Samplers(samplers),
        backup=fvm.Backup(schedule=fvm.EveryTime(BACKUP_INTERVAL)),
        mesh=mesh,
        boundaries=(
            fvm.BoundaryConfig.inlet("inlet", STARTUP_VELOCITY),
            fvm.BoundaryConfig.outlet("outlet", kinematic_pressure=0.0),
            fvm.BoundaryConfig.slip("ymin"),
            fvm.BoundaryConfig.slip("ymax"),
            fvm.BoundaryConfig.slip("zmin"),
            fvm.BoundaryConfig.slip("zmax"),
            fvm.BoundaryConfig.wall("cube"),
        ),
        # The startup transverse velocity enters at the inlet and crosses the lateral walls
        velocity_boundaries=(
            fvm.VelocityBoundary(patches=("inlet",), velocity=startup),
            fvm.VelocityBoundary(
                patches=("ymin", "ymax", "zmin", "zmax"), velocity=startup, normal_only=True
            ),
        ),
        cores=CORES,
    )
    return fvm.FVMSolver(case)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--name", default="grid_h0045")
    parser.add_argument("-h", type=float, default=0.045)
    arguments = parser.parse_args()

    with create_solver(arguments.name, arguments.h) as solver:
        solver.run(start_from=START_FROM)
        fvm.update_grid_study(
            solver,
            arguments.h,
            profiles=("centreline", "offaxis_y075"),
        )


if __name__ == "__main__":
    main()
