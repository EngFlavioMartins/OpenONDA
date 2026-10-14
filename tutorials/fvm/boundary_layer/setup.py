#!/usr/bin/env python3
"""Laminar flat-plate boundary layer at Re_L = 10,000 (FVM).

Velocity profiles and skin friction are compared with the Blasius solution.

Run with ``python setup.py``.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import openonda.fvm as fvm

from assets.mesh_plate import flat_plate_mesh
from assets.profiles import write_profiles

# Case definition
START_FROM = "latest"  # Resume the latest backup; ./allrun.sh cleans first.

CASE_NAME = "boundary_layer"
PLATE_LENGTH = 1.0  # plate length [m]
FREESTREAM_VELOCITY = 1.0  # inflow speed [m/s]
DENSITY = 1.0  # fluid density [kg/m^3]
REYNOLDS_NUMBER = 1.0e4
FINAL_TIME = 8.0  # [s]
STATIONS = (0.25, 0.5, 0.75)  # x/L positions of the Blasius profiles

# Mesh
UPSTREAM_LIMIT = -0.25  # start of the frictionless run-in [m]
N_UPSTREAM = 10
N_PLATE = 72  # cells along the plate
DOMAIN_HEIGHT = 0.35  # height of the domain [m]
WALL_CELL_HEIGHT = 0.0015  # height of the first cell next to the wall [m]
WALL_STRETCHING = 1.12  # wall-normal growth factor (1.0 = uniform)

# Time stepping and numerics
TIME_STEP_SIZE = 0.005  # initial time step [s]
MAX_TIME_STEP_SIZE = 0.02  # upper bound on the adapted time step [s]
OUTPUT_INTERVAL_TIME = 2.0  # save a snapshot every this many seconds
GRADIENT_SCHEME = "gauss"


def create_fvm_case(kinematic_viscosity: float) -> fvm.FVMCase:
    """Build the FVM case of the flat plate."""
    snapshots = fvm.EveryTime(OUTPUT_INTERVAL_TIME)

    forces = fvm.ForceSampler(
        patch_names=["plate"],
        reference_velocity=FREESTREAM_VELOCITY,
        reference_length=PLATE_LENGTH,
    )
    mesh = partial(
        flat_plate_mesh,
        x_up=UPSTREAM_LIMIT,
        n_up=N_UPSTREAM,
        plate_length=PLATE_LENGTH,
        height=DOMAIN_HEIGHT,
        n_plate=N_PLATE,
        dy_wall=WALL_CELL_HEIGHT,
        ratio=WALL_STRETCHING,
    )

    return fvm.FVMCase(
        name=CASE_NAME,
        directory=Path(__file__).parent,
        numerics=fvm.Numerics(
            time_step_size=TIME_STEP_SIZE,
            time_step_control=fvm.MaximumCourantTimeStep(maximum_time_step_size=MAX_TIME_STEP_SIZE),
            transport=fvm.TransportConfig(density=DENSITY, kinematic_viscosity=kinematic_viscosity),
            schemes=fvm.DiscretizationConfig(gradient_scheme=GRADIENT_SCHEME),
        ),
        initial_conditions=fvm.InitialFields(velocity=(FREESTREAM_VELOCITY, 0.0, 0.0)),
        run=fvm.RunPlan(end_time=FINAL_TIME),
        output=fvm.Output(schedule=snapshots),
        samplers=fvm.Samplers((forces,)),
        backup=fvm.Backup(schedule=snapshots),
        mesh=mesh,
        boundaries=(
            fvm.BoundaryConfig.inlet("inlet", (FREESTREAM_VELOCITY, 0.0, 0.0)),
            fvm.BoundaryConfig.outlet("outlet", kinematic_pressure=0.0),
            fvm.BoundaryConfig(name="floor", velocity_type="slip"),
            fvm.BoundaryConfig.wall("plate"),
            fvm.BoundaryConfig(name="top", velocity_type="slip"),
            fvm.BoundaryConfig.empty("front"),
            fvm.BoundaryConfig.empty("back"),
        ),
    )


def main() -> None:
    kinematic_viscosity = FREESTREAM_VELOCITY * PLATE_LENGTH / REYNOLDS_NUMBER
    with fvm.FVMSolver(create_fvm_case(kinematic_viscosity)) as solver:
        solver.run(start_from=START_FROM)
        solver.evaluate(
            write_profiles,
            solver.solution_dir,
            kinematic_viscosity,
            FREESTREAM_VELOCITY,
            STATIONS,
        )


if __name__ == "__main__":
    main()
