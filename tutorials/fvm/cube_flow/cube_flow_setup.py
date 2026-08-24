#!/usr/bin/env python3
"""Von Karman vortex street behind a square cylinder at Re = 100 (FVM).

The quasi-two-dimensional body-fitted case targets St = 0.140--0.150 and
mean Cd = 1.45--1.58 at 5% blockage.

Usage:
    python cube_flow_setup.py --Re 100 --end-time 120
"""

from __future__ import annotations

import argparse
import os

from assets.mesh_square import square_cylinder_mesh

import openonda.fvm as fvm

# ---- Case definition -----------------------------------------------------
CASE_NAME = "cube_flow"
SIDE = 1.0  # side length of the square cylinder [m]
FREESTREAM_VELOCITY = 1.0  # inflow speed [m/s]
DENSITY = 1.0  # fluid density [kg/m^3]

# ---- Mesh ----------------------------------------------------------------
SPACING = 0.0625  # core grid spacing next to the cylinder [m]

# ---- Time stepping and numerics ------------------------------------------
TIME_STEP_SIZE = 0.02  # initial time step [s]
MAX_COURANT_NUMBER = 0.9  # target maximum Courant number
MAX_TIME_STEP_SIZE = 0.05  # upper bound on the adapted time step [s]
MIN_TIME_STEP_SIZE = 1e-5  # lower bound on the adapted time step [s]
OUTPUT_INTERVAL_TIME = 5.0  # save a snapshot every this many seconds
PISO_CORRECTORS = 2
OUTER_CORRECTORS = 1
CONVECTION_SCHEME = "limitedLinear"
LINEAR_SOLVER = "bicgstab"


def create_fvm_setup(reynolds: float, end_time: float, depth: float) -> fvm.FVMSetup:
    """Build the FVM setup for the square-cylinder case."""
    kinematic_viscosity = FREESTREAM_VELOCITY * SIDE / reynolds

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
            start_time=0.0,
            end_time=end_time,
            output_interval_steps=10**9,
            output_interval_time=OUTPUT_INTERVAL_TIME,
            adjust_time_step=True,
            max_courant_number=MAX_COURANT_NUMBER,
            max_time_step_size=MAX_TIME_STEP_SIZE,
            min_time_step_size=MIN_TIME_STEP_SIZE,
        ),
        schemes=schemes,
        linear=linear,
        pimple=pimple,
        samplers=forces,
        transport=fvm.TransportConfig(density=DENSITY, kinematic_viscosity=kinematic_viscosity),
        turbulence=None,  # laminar validation case
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
        initial_kinematic_pressure=0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--Re",
        type=float,
        default=100.0,
        help="Reynolds number based on lid speed, cavity width, and kinematic viscosity",
    )
    parser.add_argument("--end-time", type=float, default=120.0, help="simulation end time [s]")
    args = parser.parse_args()

    case_dir = os.path.dirname(os.path.abspath(__file__))

    print("\n===== MESH =====")
    print("---- Generating the square-cylinder mesh ----")
    mesh_data, depth = square_cylinder_mesh(grid_spacing=SPACING, side_length=SIDE)
    print(
        f"  cells: {mesh_data['n_cells']}, core spacing {SPACING} "
        f"(D/h = {SIDE / SPACING:.0f}), blockage 5%"
    )

    print("\n===== SIMULATION =====")
    fvm_setup = create_fvm_setup(args.Re, args.end_time, depth)
    fvm_solver = fvm.FVMSolver(fvm_setup, case_dir, mesh_data=mesh_data)

    fvm_solver.write_vtk()
    while fvm_solver.time < fvm_setup.time.end_time:
        fvm_solver.advance()

    print("\n===== DONE =====")
    print("Simulation completed successfully. Run ./plot_all.sh to make the figures.")
    print("Reference values at Re = 100 (square cylinder, 5% blockage):")
    print("  St = 0.140-0.150   (Okajima 1982; Sohankar et al. 1998; Sen et al. 2011)")
    print("  mean Cd = 1.45-1.58")


if __name__ == "__main__":
    main()
