#!/usr/bin/env python3
"""Von Karman vortex street behind a square cylinder (body-fitted FVM).

A square-section cylinder (an extruded cube, side D) sits in a uniform stream
at Re = U D / nu. Above Re ~ 50 the wake becomes globally unstable and sheds a
periodic von Karman street whose frequency and mean drag are classical
validation data:

  Re = 100, blockage 5%:
    Strouhal St = f D / U  ->  0.140-0.150
    mean drag  Cd          ->  1.45-1.58
  (Okajima, J. Fluid Mech. 123, 1982; Sohankar et al., Int. J. Numer. Meth.
   Fluids 26, 1998; Sen, Mittal & Biswas, Int. J. Numer. Meth. Fluids 67, 2011)

The case is quasi-two-dimensional (one cell thick) on a rectilinear mesh built
in memory (assets/mesh_square.py). Cd(t) and Cl(t) are logged to
samples/forces_history.csv; ``allplot.sh`` extracts the Strouhal number from
the lift signal and compares it with the bands above.
"""

from __future__ import annotations

import argparse
import os

from assets.mesh_square import square_cylinder_mesh

from openonda.fvm import (
    BoundaryConfig,
    FVMSetup,
    ForceSampler,
    Solver,
    LinearSolverConfig,
    PimpleControl,
    SchemesConfig,
    TimeConfig,
    TransportConfig,
)

# ---- Case definition -----------------------------------------------------
CASE_NAME = "cubeFlow"
SIDE = 1.0  # side length of the square cylinder [m]
FREESTREAM_VELOCITY = 1.0  # inflow speed [m/s]
DENSITY = 1.0  # fluid density [kg/m^3]

# ---- Mesh ----------------------------------------------------------------
SPACING = 0.0625  # core grid spacing next to the cylinder [m]

# ---- Time stepping and numerics ------------------------------------------
TIME_STEP = 0.02  # initial time step [s]
MAX_CFL = 0.9  # target maximum Courant number
MAX_TIME_STEP = 0.05  # upper bound on the adapted time step [s]
MIN_TIME_STEP = 1e-5  # lower bound on the adapted time step [s]
WRITE_INTERVAL_TIME = 5.0  # save a snapshot every this many seconds
PISO_CORRECTORS = 2
OUTER_CORRECTORS = 1
CONVECTION_SCHEME = "limitedLinear"
LINEAR_SOLVER = "bicgstab"


def build_config(reynolds: float, end_time: float, depth: float) -> FVMSetup:
    """Build the FVM setup for the square-cylinder case."""
    nu = FREESTREAM_VELOCITY * SIDE / reynolds

    schemes = SchemesConfig(
        convection_scheme=CONVECTION_SCHEME,
        gradient_scheme="gauss",
    )
    linear = LinearSolverConfig(linear_solver=LINEAR_SOLVER)
    pimple = PimpleControl(
        n_correctors=PISO_CORRECTORS,
        n_outer_correctors=OUTER_CORRECTORS,
    )
    forces = [
        ForceSampler(
            patch_names=["cube"],
            ref_velocity=FREESTREAM_VELOCITY,
            ref_area=SIDE * depth,  # frontal area of the extruded square
            ref_length=SIDE,
            moment_centre=[0.0, 0.0, 0.5 * depth],
        )
    ]

    return FVMSetup(
        case_name=CASE_NAME,
        time=TimeConfig(
            delta_t=TIME_STEP,
            start_time=0.0,
            end_time=end_time,
            write_interval=10**9,
            write_interval_time=WRITE_INTERVAL_TIME,
            adjust_timestep=True,
            max_cfl=MAX_CFL,
            max_delta_t=MAX_TIME_STEP,
            min_delta_t=MIN_TIME_STEP,
        ),
        schemes=schemes,
        linear=linear,
        pimple=pimple,
        samplers=forces,
        transport=TransportConfig(density=DENSITY, nu=nu),
        turbulence=None,  # laminar validation case
        boundaries=[
            BoundaryConfig.inlet("inlet", [FREESTREAM_VELOCITY, 0.0, 0.0]),
            BoundaryConfig.outlet("outlet", p=0.0),
            # Slip lateral boundaries: 20 D apart (5% blockage), no wall BL.
            BoundaryConfig(
                name="bottom", type_U="slip", type_p="zeroGradient", type_nut="zeroGradient"
            ),
            BoundaryConfig(
                name="top", type_U="slip", type_p="zeroGradient", type_nut="zeroGradient"
            ),
            BoundaryConfig.wall("cube"),
            BoundaryConfig.empty("front"),
            BoundaryConfig.empty("back"),
        ],
        # A tiny cross-stream component lets the vortex street start right
        # away instead of waiting for numerical round-off to break symmetry.
        initial_U=[FREESTREAM_VELOCITY, 0.05 * FREESTREAM_VELOCITY, 0.0],
        initial_p=0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--Re", type=float, default=100.0, help="Reynolds number Re = U D / nu")
    parser.add_argument("--end-time", type=float, default=120.0, help="simulation end time [s]")
    args = parser.parse_args()

    case_dir = os.path.dirname(os.path.abspath(__file__))

    print("\n===== MESH =====")
    print("---- Generating the square-cylinder mesh ----")
    mesh_data, depth = square_cylinder_mesh(h=SPACING, D=SIDE)
    print(
        f"  cells: {mesh_data['n_elements']}, core spacing {SPACING} "
        f"(D/h = {SIDE / SPACING:.0f}), blockage 5%"
    )

    print("\n===== SIMULATION =====")
    config = build_config(args.Re, args.end_time, depth)
    solver = Solver(config, case_dir, mesh_data=mesh_data)

    solver.write_vtk()
    while solver.flow_time < config.time.end_time:
        solver.evolve()

    print("\n===== DONE =====")
    print("Simulation completed successfully. Run ./allplot.sh to make the figures.")
    print("Reference values at Re = 100 (square cylinder, 5% blockage):")
    print("  St = 0.140-0.150   (Okajima 1982; Sohankar et al. 1998; Sen et al. 2011)")
    print("  mean Cd = 1.45-1.58")


if __name__ == "__main__":
    main()
