#!/usr/bin/env python3
"""Von Karman vortex street behind a square cylinder at Re = 100 (FVM).

The quasi-two-dimensional body-fitted case targets St = 0.140--0.150 and
mean Cd = 1.45--1.58 at 5% blockage.

Run with ``python setup.py``.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import openonda.fvm as fvm

from assets.mesh_square import square_cylinder_mesh

# Case definition
START_FROM = "latest"  # Resume the latest backup; ./allrun.sh cleans first.

CASE_NAME = "cube_flow"
SIDE = 1.0  # side length of the square cylinder [m]
FREESTREAM_VELOCITY = 1.0  # inflow speed [m/s]
DENSITY = 1.0  # fluid density [kg/m^3]
REYNOLDS_NUMBER = 100.0
FINAL_TIME = 120.0  # simulation duration [s]

# Mesh
X_BOUNDS = (-10.0, 25.0)  # domain bounds / side length
Y_BOUNDS = (-10.0, 10.0)
CORE_X = (-2.0, 8.0)  # uniform core bounds / side length
CORE_Y = (-2.0, 2.0)
MESH_STRETCHING = 1.10
SPACING = 0.0625  # core grid spacing next to the cylinder [m]

# Time stepping and numerics
TIME_STEP_SIZE = 0.02  # initial time step [s]
MAX_TIME_STEP_SIZE = 0.05  # upper bound on the adapted time step [s]
OUTPUT_INTERVAL_TIME = 5.0  # save a snapshot every this many seconds


def create_fvm_case(depth: float) -> fvm.FVMCase:
    """Build the FVM case of the square cylinder."""
    kinematic_viscosity = FREESTREAM_VELOCITY * SIDE / REYNOLDS_NUMBER
    snapshots = fvm.EveryTime(OUTPUT_INTERVAL_TIME)

    forces = fvm.ForceSampler(
        patch_names=["cube"],
        reference_velocity=FREESTREAM_VELOCITY,
        reference_area=SIDE * depth,  # frontal area of the extruded square
        reference_length=SIDE,
        moment_centre=[0.0, 0.0, 0.5 * depth],
    )
    mesh = partial(
        square_cylinder_mesh,
        grid_spacing=SPACING,
        side_length=SIDE,
        x_bounds=X_BOUNDS,
        y_bounds=Y_BOUNDS,
        core_x=CORE_X,
        core_y=CORE_Y,
        ratio=MESH_STRETCHING,
    )

    return fvm.FVMCase(
        name=CASE_NAME,
        directory=Path(__file__).parent,
        numerics=fvm.Numerics(
            time_step_size=TIME_STEP_SIZE,
            time_step_control=fvm.MaximumCourantTimeStep(maximum_time_step_size=MAX_TIME_STEP_SIZE),
            transport=fvm.TransportConfig(density=DENSITY, kinematic_viscosity=kinematic_viscosity),
            schemes=fvm.DiscretizationConfig(gradient_scheme="gauss"),
        ),
        # A small transverse perturbation seeds asymmetric shedding.
        initial_conditions=fvm.InitialFields(
            velocity=(FREESTREAM_VELOCITY, 0.05 * FREESTREAM_VELOCITY, 0.0)
        ),
        run=fvm.RunPlan(end_time=FINAL_TIME),
        output=fvm.Output(schedule=snapshots),
        samplers=fvm.Samplers((forces,)),
        backup=fvm.Backup(schedule=snapshots),
        mesh=mesh,
        boundaries=(
            fvm.BoundaryConfig.inlet("inlet", (FREESTREAM_VELOCITY, 0.0, 0.0)),
            fvm.BoundaryConfig.outlet("outlet", kinematic_pressure=0.0),
            # Slip lateral boundaries: 20 D apart (5% blockage), no wall boundary layer.
            fvm.BoundaryConfig(name="bottom", velocity_type="slip"),
            fvm.BoundaryConfig(name="top", velocity_type="slip"),
            fvm.BoundaryConfig.wall("cube"),
            fvm.BoundaryConfig.empty("front"),
            fvm.BoundaryConfig.empty("back"),
        ),
    )


def main() -> None:
    with fvm.FVMSolver(create_fvm_case(SPACING)) as solver:
        solver.run(start_from=START_FROM)


if __name__ == "__main__":
    main()
