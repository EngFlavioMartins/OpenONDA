#!/usr/bin/env python3
"""Von Karman vortex street behind a square cylinder at Re = 100 (FVM).

The quasi-two-dimensional body-fitted case targets St = 0.140--0.150 and
mean Cd = 1.45--1.58 at 5% blockage.

Run with ``python setup.py``.
"""

from __future__ import annotations

from pathlib import Path

import openonda.fvm as fvm
from openonda.tutorial_runner import case_package

__package__ = case_package(Path(__file__).parent)
from .assets.mesh_square import square_cylinder_mesh

# Case definition
CASE_NAME = "cube_flow"
SIDE = 1.0  # side length of the square cylinder [m]
FREESTREAM_VELOCITY = 1.0  # inflow speed [m/s]
DENSITY = 1.0  # fluid density [kg/m^3]
REYNOLDS_NUMBER = 100.0
FINAL_TIME = 120.0  # simulation duration [s]

# Mesh
SPACING = 0.0625  # core grid spacing next to the cylinder [m]

# Time stepping and numerics
TIME_STEP_SIZE = 0.02  # initial time step [s]
MAX_COURANT_NUMBER = 0.9  # target maximum Courant number
MAX_TIME_STEP_SIZE = 0.05  # upper bound on the adapted time step [s]
OUTPUT_INTERVAL_TIME = 5.0  # save a snapshot every this many seconds
PISO_CORRECTORS = 2
OUTER_CORRECTORS = 1
CONVECTION_SCHEME = "limitedLinear"
LINEAR_SOLVER = "bicgstab"


def create_fvm_setup(depth: float) -> fvm.FVMSetup:
    """Build the FVM setup for the square-cylinder case."""
    kinematic_viscosity = FREESTREAM_VELOCITY * SIDE / REYNOLDS_NUMBER

    schemes = fvm.DiscretizationConfig(
        convection_scheme=CONVECTION_SCHEME,
        gradient_scheme="gauss",
    )
    linear = fvm.LinearSolverConfig(linear_solver=LINEAR_SOLVER)
    pimple = fvm.PimpleControl(
        n_correctors=PISO_CORRECTORS,
        n_outer_correctors=OUTER_CORRECTORS,
    )
    forces = [
        fvm.ForceSampler(
            patch_names=["cube"],
            reference_velocity=FREESTREAM_VELOCITY,
            reference_area=SIDE * depth,  # frontal area of the extruded square
            reference_length=SIDE,
            moment_centre=[0.0, 0.0, 0.5 * depth],
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
            # Slip lateral boundaries: 20 D apart (5% blockage), no wall BL.
            fvm.BoundaryConfig(
                name="bottom",
                velocity_type="slip",
                pressure_type="zeroGradient",
                eddy_viscosity_type="zeroGradient",
            ),
            fvm.BoundaryConfig(
                name="top",
                velocity_type="slip",
                pressure_type="zeroGradient",
                eddy_viscosity_type="zeroGradient",
            ),
            fvm.BoundaryConfig.wall("cube"),
            fvm.BoundaryConfig.empty("front"),
            fvm.BoundaryConfig.empty("back"),
        ],
        # A tiny cross-stream component lets the vortex street start right
        # away instead of waiting for numerical round-off to break symmetry.
        initial_velocity=[FREESTREAM_VELOCITY, 0.05 * FREESTREAM_VELOCITY, 0.0],
    )


def main() -> None:
    mesh, depth = square_cylinder_mesh(grid_spacing=SPACING, side_length=SIDE)
    solver = fvm.create_fvm_solver(
        create_fvm_setup(depth), case_dir=Path(__file__).parent, mesh=mesh
    )
    solver.run()


if __name__ == "__main__":
    main()
