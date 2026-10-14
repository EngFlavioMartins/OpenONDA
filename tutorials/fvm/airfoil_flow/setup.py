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

from assets.generate_surface import create_airfoil_surface
from assets.surface_pressure import write_surface_cp

# Case definition
START_FROM = "latest"  # Resume the latest backup; ./allrun.sh cleans first.

CASE_NAME = "airfoil_flow"
CHORD = 1.0  # airfoil chord length [m]
CHORD_COUNT = 80  # surface geometry resolution
THICKNESS_RATIO = 0.12
DEPTH = 0.8  # finite-span extrusion depth [m]
FREESTREAM_VELOCITY = 1.0  # inflow speed [m/s]
DENSITY = 1.0  # fluid density [kg/m^3]
REYNOLDS_NUMBER = 1000.0
ANGLE_OF_ATTACK_DEGREES = 0.0
FINAL_TIME = 25.0  # [s]

# Time stepping and numerics
TIME_STEP_SIZE = 0.005  # initial time step [s]
MAX_TIME_STEP_SIZE = 4 * TIME_STEP_SIZE  # upper bound on the adapted time step [s]
OUTPUT_INTERVAL_TIME = 5.0  # save a snapshot every this many seconds
ORTHOGONAL_CORRECTORS = 1
DOMAIN = (-5.0, 15.0, -5.0, 5.0, -0.5, 0.5)
AIRFOIL_STL = Path(__file__).resolve().parent / "assets" / "airfoil.stl"


def create_fvm_mesh() -> msh.CartesianMesher:
    """Declare the native surface-driven mesh for the finite wing."""
    create_airfoil_surface(AIRFOIL_STL, CHORD, DEPTH, CHORD_COUNT, THICKNESS_RATIO)
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
        min_cell_size=0.03125,
        patch_refinements=(msh.PatchRefinement(patch="airfoil", cell_size=0.03125),),
        refinements=(
            msh.BoxRefinement(
                name="near_airfoil",
                bounds=(-1.0, 3.0, -1.0, 1.0, -0.4, 0.4),
                cell_size=0.125,
            ),
        ),
    )


def create_fvm_case(velocity: tuple[float, float, float]) -> fvm.FVMCase:
    """Build the FVM case of the finite wing."""
    kinematic_viscosity = FREESTREAM_VELOCITY * CHORD / REYNOLDS_NUMBER
    snapshots = fvm.EveryTime(OUTPUT_INTERVAL_TIME)

    forces = fvm.ForceSampler(
        patch_names=["airfoil"],
        reference_velocity=FREESTREAM_VELOCITY,
        reference_area=CHORD * DEPTH,
        reference_length=CHORD,
        moment_centre=[0.25 * CHORD, 0.0, 0.0],
    )

    return fvm.FVMCase(
        name=CASE_NAME,
        directory=Path(__file__).parent,
        numerics=fvm.Numerics(
            time_step_size=TIME_STEP_SIZE,
            time_step_control=fvm.MaximumCourantTimeStep(maximum_time_step_size=MAX_TIME_STEP_SIZE),
            transport=fvm.TransportConfig(density=DENSITY, kinematic_viscosity=kinematic_viscosity),
            pimple=fvm.PimpleConfig(n_nonorthogonal_correctors=ORTHOGONAL_CORRECTORS),
        ),
        initial_conditions=fvm.InitialFields(velocity=velocity),
        run=fvm.RunPlan(end_time=FINAL_TIME),
        output=fvm.Output(schedule=snapshots),
        samplers=fvm.Samplers((forces,)),
        backup=fvm.Backup(schedule=snapshots),
        mesh=create_fvm_mesh,
        boundaries=(
            fvm.BoundaryConfig.inlet("inlet", velocity),
            fvm.BoundaryConfig.outlet("outlet", kinematic_pressure=0.0),
            fvm.BoundaryConfig.freestream("walls", velocity),
            fvm.BoundaryConfig.wall("airfoil"),
            fvm.BoundaryConfig.empty("frontAndBack"),
        ),
    )


def main() -> None:
    angle = math.radians(ANGLE_OF_ATTACK_DEGREES)
    velocity = (FREESTREAM_VELOCITY * math.cos(angle), FREESTREAM_VELOCITY * math.sin(angle), 0.0)
    case = create_fvm_case(velocity)
    with fvm.FVMSolver(case) as solver:
        solver.run(start_from=START_FROM)
        solver.evaluate(write_surface_cp, solver.solution_dir, CHORD, FREESTREAM_VELOCITY)


if __name__ == "__main__":
    main()
