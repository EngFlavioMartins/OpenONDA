#!/usr/bin/env python3
"""Flow past a circular cylinder via the Immersed Boundary Method (FVM).

Validation case for the discrete direct-forcing IBM (Pinelli et al. 2010, as
implemented for finite-volume PISO solvers by Constant et al.; see
docs/literature/Constant2016.pdf). The cylinder is represented by Lagrangian
markers on a Cartesian mesh, so no body-fitted grid is needed.

Quality monitors (written under ``solution``, plotted by ``allplot.sh``):
  * Re = 30 (steady):        Cd -> 1.74-1.80, recirculation length
                             L/D -> 1.55-1.70   (Constant et al., Table 2)
  * Re = 100 (unsteady):     mean Cd -> 1.35-1.38, Strouhal -> 0.164-0.165
                             (Constant et al., Table 3)
  * marker no-slip error (IBM-specific quality signal), logged every step.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from assets.mesh_rectilinear import cylinder_ibm_mesh
import openonda.fvm as fvm

# ---- Case definition -----------------------------------------------------
CASE_NAME = "cylinder_ibm"
DIAMETER = 1.0  # cylinder diameter [m]
FREESTREAM_VELOCITY = 1.0  # inflow speed [m/s]
DENSITY = 1.0  # fluid density [kg/m^3]

# ---- Mesh and IBM markers ------------------------------------------------
SPACING = 0.0625  # uniform grid spacing next to the cylinder [m]
MARKER_ALPHA = 1.0  # marker spacing / grid spacing ratio

# ---- Time stepping and numerics ------------------------------------------
TIME_STEP_SIZE = 0.01  # initial time step [s]
MAX_CFL = 0.5  # target maximum Courant number (see below)
MAX_TIME_STEP_SIZE = 0.03  # upper bound on the adapted time step [s]
MIN_TIME_STEP_SIZE = 1e-5  # lower bound on the adapted time step [s]
MAX_FORCING_FOURIER = 0.1  # Fo = nu*dt/h^2 stability cap for the IBM
OUTPUT_INTERVAL_TIME = 5.0  # save a snapshot every this many seconds
PISO_CORRECTORS = 2
OUTER_CORRECTORS = 1
CONVECTION_SCHEME = "limitedLinear"
LINEAR_SOLVER = "spsolve"


def create_fvm_setup(
    reynolds: float, end_time: float, depth: float, time_step: float, max_time_step: float
) -> fvm.FVMSetup:
    """Build the FVM setup for the IBM cylinder case."""
    kinematic_viscosity = FREESTREAM_VELOCITY * DIAMETER / reynolds

    schemes = fvm.DiscretizationConfig(
        convection_scheme=CONVECTION_SCHEME,
        gradient_scheme="gauss",
    )
    linear = fvm.LinearSolverConfig(linear_solver=LINEAR_SOLVER)
    pimple = fvm.PimpleControl(
        n_correctors=PISO_CORRECTORS,
        n_outer_correctors=OUTER_CORRECTORS,
    )

    return fvm.FVMSetup(
        case_name=CASE_NAME,
        time=fvm.TimeConfig(
            time_step_size=time_step,
            start_time=0.0,
            end_time=end_time,
            output_interval_steps=10**9,
            output_interval_time=OUTPUT_INTERVAL_TIME,
            adjust_timestep=True,
            max_cfl=MAX_CFL,
            max_time_step_size=max_time_step,
            min_time_step_size=MIN_TIME_STEP_SIZE,
        ),
        schemes=schemes,
        linear=linear,
        pimple=pimple,
        transport=fvm.TransportConfig(density=DENSITY, kinematic_viscosity=kinematic_viscosity),
        turbulence=None,  # laminar validation case
        boundaries=[
            fvm.BoundaryConfig.inlet("inlet", [FREESTREAM_VELOCITY, 0.0, 0.0]),
            fvm.BoundaryConfig.outlet("outlet", p=0.0),
            fvm.BoundaryConfig.freestream("bottom", [FREESTREAM_VELOCITY, 0.0, 0.0]),
            fvm.BoundaryConfig.freestream("top", [FREESTREAM_VELOCITY, 0.0, 0.0]),
            fvm.BoundaryConfig.empty("front"),
            fvm.BoundaryConfig.empty("back"),
        ],
        initial_velocity=[FREESTREAM_VELOCITY, 0.0, 0.0],
        initial_kinematic_pressure=0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--Re", type=float, default=30.0, help="Reynolds number Re = U D / nu")
    parser.add_argument("--end-time", type=float, default=60.0, help="simulation end time [s]")
    args = parser.parse_args()

    case_dir = os.path.dirname(os.path.abspath(__file__))

    # The direct-forcing feedback loop is stable only for Fo = nu*dt/h^2 <~ 0.1
    # (in addition to Co <= 0.5); above it a slow sawtooth develops in Cd and
    # in the marker slip error. Cap dt accordingly (this binds at low Re).
    kinematic_viscosity = FREESTREAM_VELOCITY * DIAMETER / args.Re
    max_time_step = MAX_TIME_STEP_SIZE
    time_step = TIME_STEP_SIZE
    fourier_time_step_size_limit = MAX_FORCING_FOURIER * SPACING**2 / kinematic_viscosity
    if fourier_time_step_size_limit < max_time_step:
        print(
            f"  [IBM] capping max dt to {fourier_time_step_size_limit:.4g} s (Fo = nu*dt/h^2 <= {MAX_FORCING_FOURIER})"
        )
        max_time_step = fourier_time_step_size_limit
        time_step = min(time_step, fourier_time_step_size_limit)

    print("\n===== MESH =====")
    print("---- Generating the rectilinear IBM mesh ----")
    mesh_data, depth = cylinder_ibm_mesh(h=SPACING, D=DIAMETER)
    print(
        f"  cells: {mesh_data['n_cells']}, core spacing h = {SPACING} "
        f"(D/h = {DIAMETER / SPACING:.0f})"
    )

    print("\n===== SIMULATION =====")
    fvm_setup = create_fvm_setup(args.Re, args.end_time, depth, time_step, max_time_step)
    fvm_solver = fvm.FVMSolver(fvm_setup, case_dir, mesh_data=mesh_data)

    print("---- Setting the immersed cylinder ----")
    body = fvm.ImmersedBody.cylinder_z(
        centre=[0.0, 0.0, 0.5 * depth],
        diameter=DIAMETER,
        h=SPACING,
        alpha=MARKER_ALPHA,
        name="cylinder",
    )
    fvm_solver.set_immersed_bodies(body, h=SPACING)

    # Save the marker cloud so the plotting scripts can draw the cylinder.
    sol_dir = os.path.join(case_dir, "solution")
    os.makedirs(sol_dir, exist_ok=True)
    np.savetxt(
        os.path.join(sol_dir, "ibm_markers.csv"), body.X, delimiter=",", header="x,y,z", comments=""
    )

    fvm_solver.write_vtk()
    while fvm_solver.time < fvm_setup.time.end_time:
        fvm_solver.advance()

    print("\n===== DONE =====")
    print("Simulation completed successfully. Run ./allplot.sh to make the figures.")
    print("Reference values (Constant et al. 2017):")
    if abs(args.Re - 30.0) < 1e-9:
        print("  Re=30 steady:  Cd = 1.74-1.80, recirculation L/D = 1.55-1.70")
    elif abs(args.Re - 100.0) < 1e-9:
        print("  Re=100 unsteady:  mean Cd = 1.35-1.38, St = 0.164-0.165")


if __name__ == "__main__":
    main()
