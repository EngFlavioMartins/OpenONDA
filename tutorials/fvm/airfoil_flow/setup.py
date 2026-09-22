#!/usr/bin/env python3
"""Laminar flow past a NACA 0012 airfoil on a body-fitted FVM mesh.

At zero angle of attack, lift and upper/lower pressure asymmetry should vanish.

Run with ``python setup.py``.
"""

from __future__ import annotations

import math
from pathlib import Path

import openonda.fvm as fvm
import openonda.fvm.mesher as msh
from openonda.tutorial_runner import case_package

__package__ = case_package(Path(__file__).parent)
from .assets.generate_surface import create_airfoil_surface
from .assets.surface_pressure import write_surface_cp

# Case definition
START_FROM = "latest"  # Resume the latest backup; ./allrun.sh cleans first.

CASE_NAME = "airfoil_flow"
CHORD = 1.0  # airfoil chord length [m]
DEPTH = 0.8  # finite-span extrusion depth [m]
FREESTREAM_VELOCITY = 1.0  # inflow speed [m/s]
DENSITY = 1.0  # fluid density [kg/m^3]
REYNOLDS_NUMBER = 1000.0
ANGLE_OF_ATTACK_DEGREES = 0.0
FINAL_TIME = 25.0  # [s]

# Time stepping and numerics
TIME_STEP_SIZE = 0.005  # initial time step [s]
MAX_COURANT_NUMBER = 0.9  # target maximum Courant number
MAX_TIME_STEP_SIZE = 4 * TIME_STEP_SIZE  # upper bound on the adapted time step [s]
OUTPUT_INTERVAL_TIME = 5.0  # save a snapshot every this many seconds
PISO_CORRECTORS = 2
OUTER_CORRECTORS = 1
ORTHOGONAL_CORRECTORS = 1
CONVECTION_SCHEME = "limitedLinear"
LINEAR_SOLVER = "bicgstab"
DOMAIN = (-5.0, 15.0, -5.0, 5.0, -0.5, 0.5)
AIRFOIL_STL = Path(__file__).resolve().parent / "assets" / "airfoil.stl"


def create_fvm_mesh() -> msh.CartesianMesher:
    """Declare the native surface-driven mesh for the finite wing."""
    create_airfoil_surface(AIRFOIL_STL, CHORD, DEPTH)
    return msh.CartesianMesher(
        domain=msh.BoxDomain(
            bounds=DOMAIN,
            patches=msh.BoxPatches(
                xmin="inlet",
                xmax="outlet",
                ymin="walls",
                ymax="walls",
                zmin="frontAndBack",
                zmax="frontAndBack",
            ),
        ),
        surfaces=(msh.STLSurface(AIRFOIL_STL, patch="airfoil"),),
        max_cell_size=1.0,
        boundary_cell_size=0.03125,
        min_cell_size=0.03125,
        refinements=(
            msh.BoxRefinement(
                name="near_airfoil",
                bounds=(-1.0, 3.0, -1.0, 1.0, -0.4, 0.4),
                cell_size=0.125,
            ),
        ),
    )


def create_fvm_setup(u_vec: list[float]) -> fvm.FVMSetup:
    """Build the FVM setup for the airfoil case."""
    kinematic_viscosity = FREESTREAM_VELOCITY * CHORD / REYNOLDS_NUMBER

    schemes = fvm.DiscretizationConfig(convection_scheme=CONVECTION_SCHEME)
    linear = fvm.LinearSolverConfig(linear_solver=LINEAR_SOLVER)
    pimple = fvm.PimpleControl(
        n_correctors=PISO_CORRECTORS,
        n_outer_correctors=OUTER_CORRECTORS,
        n_nonorthogonal_correctors=ORTHOGONAL_CORRECTORS,
    )
    forces = [
        fvm.ForceSampler(
            patch_names=["airfoil"],
            reference_velocity=FREESTREAM_VELOCITY,
            reference_area=CHORD * DEPTH,
            reference_length=CHORD,
            moment_centre=[0.25 * CHORD, 0.0, 0.0],
        )
    ]

    time = fvm.TimeConfig(
        time_step_size=TIME_STEP_SIZE,
        end_time=FINAL_TIME,
        output_schedule=fvm.RunSchedule(every_time=OUTPUT_INTERVAL_TIME),
        adjustment=fvm.MaximumCourantTimeStep(
            maximum=MAX_COURANT_NUMBER,
            maximum_time_step_size=MAX_TIME_STEP_SIZE,
        ),
    )

    return fvm.FVMSetup(
        backup=fvm.BackupConfig(schedule=time.output_schedule, write_at_end=True),
        case_name=CASE_NAME,
        time=time,
        schemes=schemes,
        linear=linear,
        pimple=pimple,
        samplers=forces,
        transport=fvm.TransportConfig(density=DENSITY, kinematic_viscosity=kinematic_viscosity),
        boundaries=[
            fvm.BoundaryConfig.inlet("inlet", u_vec),
            fvm.BoundaryConfig.outlet("outlet", kinematic_pressure=0.0),
            fvm.BoundaryConfig.freestream("walls", u_vec),
            fvm.BoundaryConfig.wall("airfoil"),
            fvm.BoundaryConfig.empty("frontAndBack"),
        ],
        initial_velocity=u_vec,
    )


def main() -> None:
    case_dir = Path(__file__).parent
    angle = math.radians(ANGLE_OF_ATTACK_DEGREES)
    velocity = [FREESTREAM_VELOCITY * math.cos(angle), FREESTREAM_VELOCITY * math.sin(angle), 0.0]
    solver = fvm.create_fvm_solver(
        create_fvm_setup(velocity), case_dir=case_dir, mesh=create_fvm_mesh
    )
    solver.run(start_from=START_FROM)
    solver.evaluate(write_surface_cp, case_dir / "solution", CHORD, FREESTREAM_VELOCITY)


if __name__ == "__main__":
    main()
