#!/usr/bin/env python3
"""Laminar flat-plate boundary layer at Re_L = 10,000 (FVM).

Velocity profiles and skin friction are compared with the Blasius solution.

Run with ``python setup.py``.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import openonda.fvm as fvm
from openonda.tutorial_runner import case_package

__package__ = case_package(Path(__file__).parent)
from .assets.mesh_plate import flat_plate_mesh
from .assets.profiles import write_profiles

# Case definition
CASE_NAME = "boundary_layer"
PLATE_LENGTH = 1.0  # plate length [m]
FREESTREAM_VELOCITY = 1.0  # inflow speed [m/s]
DENSITY = 1.0  # fluid density [kg/m^3]
REYNOLDS_NUMBER = 1.0e4
FINAL_TIME = 8.0  # [s]
STATIONS = (0.25, 0.5, 0.75)  # x/L positions of the Blasius profiles

# Mesh
N_PLATE = 72  # cells along the plate
DOMAIN_HEIGHT = 0.35  # height of the domain [m]
WALL_CELL_HEIGHT = 0.0015  # height of the first cell next to the wall [m]
WALL_STRETCHING = 1.12  # wall-normal growth factor (1.0 = uniform)

# Time stepping and numerics
TIME_STEP_SIZE = 0.005  # initial time step [s]
MAX_COURANT_NUMBER = 0.9  # target maximum Courant number
MAX_TIME_STEP_SIZE = 0.02  # upper bound on the adapted time step [s]
OUTPUT_INTERVAL_TIME = 2.0  # save a snapshot every this many seconds
PISO_CORRECTORS = 2
OUTER_CORRECTORS = 1
CONVECTION_SCHEME = "limitedLinear"
GRADIENT_SCHEME = "gauss"
LINEAR_SOLVER = "bicgstab"


def create_fvm_setup(kinematic_viscosity: float) -> fvm.FVMSetup:
    """Build the FVM setup for the flat-plate case."""
    schemes = fvm.DiscretizationConfig(
        convection_scheme=CONVECTION_SCHEME,
        gradient_scheme=GRADIENT_SCHEME,
    )
    linear = fvm.LinearSolverConfig(linear_solver=LINEAR_SOLVER)
    pimple = fvm.PimpleControl(
        n_correctors=PISO_CORRECTORS,
        n_outer_correctors=OUTER_CORRECTORS,
    )
    forces = [
        fvm.ForceSampler(
            patch_names=["plate"],
            reference_velocity=FREESTREAM_VELOCITY,
            reference_length=PLATE_LENGTH,
        )
    ]

    return fvm.FVMSetup(
        case_name=CASE_NAME,
        time=fvm.TimeConfig(
            time_step_size=TIME_STEP_SIZE,
            end_time=FINAL_TIME,
            output_schedule=fvm.RunSchedule(every_time=OUTPUT_INTERVAL_TIME),
            adjustment=fvm.MaximumCourantTimeStep(
                maximum=MAX_COURANT_NUMBER,
                maximum_time_step_size=MAX_TIME_STEP_SIZE,
            ),
        ),
        schemes=schemes,
        linear=linear,
        pimple=pimple,
        samplers=forces,
        transport=fvm.TransportConfig(density=DENSITY, kinematic_viscosity=kinematic_viscosity),
        boundaries=[
            fvm.BoundaryConfig.inlet("inlet", [FREESTREAM_VELOCITY, 0.0, 0.0]),
            fvm.BoundaryConfig.outlet("outlet", kinematic_pressure=0.0),
            fvm.BoundaryConfig(name="floor", velocity_type="slip", pressure_type="zeroGradient"),
            fvm.BoundaryConfig.wall("plate"),
            fvm.BoundaryConfig(name="top", velocity_type="slip", pressure_type="zeroGradient"),
            fvm.BoundaryConfig.empty("front"),
            fvm.BoundaryConfig.empty("back"),
        ],
        initial_velocity=[FREESTREAM_VELOCITY, 0.0, 0.0],
    )


def main() -> None:
    case_dir = Path(__file__).parent
    kinematic_viscosity = FREESTREAM_VELOCITY * PLATE_LENGTH / REYNOLDS_NUMBER
    mesh = partial(
        flat_plate_mesh,
        plate_length=PLATE_LENGTH,
        height=DOMAIN_HEIGHT,
        n_plate=N_PLATE,
        dy_wall=WALL_CELL_HEIGHT,
        ratio=WALL_STRETCHING,
    )
    solver = fvm.create_fvm_solver(
        create_fvm_setup(kinematic_viscosity), case_dir=case_dir, mesh=mesh
    )
    solver.run()
    solver.evaluate(
        write_profiles, case_dir / "solution", kinematic_viscosity, FREESTREAM_VELOCITY, STATIONS
    )


if __name__ == "__main__":
    main()
