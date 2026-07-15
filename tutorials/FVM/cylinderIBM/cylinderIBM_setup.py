#!/usr/bin/env python3
"""Flow past a circular cylinder via the Immersed Boundary Method (FVM).

Validation case for the discrete direct-forcing IBM (Pinelli et al. 2010, as
implemented for finite-volume PISO solvers by Constant et al. — see
docs/literature/Constant2016.pdf).
The cylinder is represented by Lagrangian markers on a Cartesian mesh; no
body-fitted grid is used.

Quality monitors (written to solution/, plotted by allplot.sh):
  * Re = 30 (default, steady):  Cd -> 1.74-1.80, recirculation length
    L/D -> 1.55-1.70  (Constant et al. Table 2)
  * Re = 100 (unsteady):        mean Cd -> 1.35-1.38, Strouhal -> 0.164-0.165
    (Constant et al. Table 3)
  * marker no-slip error (IBM-specific quality signal), logged every step.

Usage:
    python cylinderIBM_setup.py --Re 30 --end-time 60
    python cylinderIBM_setup.py --Re 100 --end-time 150 --h 0.05
"""

import argparse
import os
import sys

import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

from source.solvers.FVM import (  # noqa: E402
    BoundaryConfig,
    FVMConfig,
    Solver,
    SolverParams,
    TimeConfig,
    TransportConfig,
)
from source.solvers.FVM.immersed_boundary import ImmersedBody  # noqa: E402

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets"))
from mesh_rectilinear import cylinder_ibm_mesh  # noqa: E402


def build_config(args, depth):
    """FVM configuration for the IBM cylinder case."""
    nu = args.u_inf * args.diameter / args.Re
    solver = SolverParams.pimple(
        n_correctors=args.n_correctors,
        n_outer=args.n_outer,
        linear_solver=args.linear_solver,
        convection_scheme=args.convection_scheme,
        gradient_scheme="gauss",  # exact on this orthogonal rectilinear mesh
    )
    solver.ref_velocity = args.u_inf
    solver.ref_area = args.diameter * depth  # frontal area of the extruded cylinder
    solver.ref_length = args.diameter

    return FVMConfig(
        case_name="cylinderIBM",
        time=TimeConfig(
            delta_t=args.initial_dt,
            start_time=0.0,
            end_time=args.end_time,
            write_interval=10**9,
            write_interval_time=args.write_interval_time,
            adjust_timestep=True,
            max_cfl=args.max_cfl,
            max_delta_t=args.max_dt,
            min_delta_t=1e-5,
        ),
        solver=solver,
        transport=TransportConfig(density=args.rho, nu=nu),
        turbulence=None,  # laminar validation case
        boundaries=[
            BoundaryConfig.inlet("inlet", [args.u_inf, 0.0, 0.0]),
            BoundaryConfig.outlet("outlet", p=0.0),
            BoundaryConfig.freestream("bottom", [args.u_inf, 0.0, 0.0]),
            BoundaryConfig.freestream("top", [args.u_inf, 0.0, 0.0]),
            BoundaryConfig.empty("front"),
            BoundaryConfig.empty("back"),
        ],
        initial_U=[args.u_inf, 0.0, 0.0],
        initial_p=0.0,
    )


def main():
    parser = argparse.ArgumentParser(description="Run the cylinderIBM tutorial")
    parser.add_argument("--Re", type=float, default=30.0, help="Reynolds number U*D/nu")
    parser.add_argument("--end-time", type=float, default=60.0, help="End time [s]")
    parser.add_argument(
        "--h",
        type=float,
        default=0.0625,
        help="Uniform grid spacing near the cylinder (in D units)",
    )
    parser.add_argument("--diameter", type=float, default=1.0, help="Cylinder diameter")
    parser.add_argument("--u-inf", type=float, default=1.0, help="Freestream velocity")
    parser.add_argument("--rho", type=float, default=1.0, help="Density")
    parser.add_argument("--initial-dt", type=float, default=0.01, help="Initial dt [s]")
    # CFL 0.5 follows Constant et al.; the direct-forcing feedback loop
    # becomes underdamped at larger Courant numbers.
    parser.add_argument("--max-cfl", type=float, default=0.5, help="Target max Courant")
    parser.add_argument("--max-dt", type=float, default=0.03, help="Max dt cap [s]")
    parser.add_argument(
        "--max-fo",
        type=float,
        default=0.1,
        help="Max forcing Fourier number nu*dt/h^2 (stability cap)",
    )
    parser.add_argument(
        "--write-interval-time",
        type=float,
        default=5.0,
        help="Write VTK every N seconds of flow time",
    )
    parser.add_argument("--n-correctors", type=int, default=2, help="PISO correctors")
    parser.add_argument(
        "--n-outer",
        type=int,
        default=1,
        help="PIMPLE outer correctors (= IBM/pressure sub-iterations)",
    )
    parser.add_argument("--linear-solver", type=str, default="spsolve")
    parser.add_argument("--convection-scheme", type=str, default="limitedLinear")
    parser.add_argument(
        "--marker-alpha", type=float, default=1.0, help="Marker spacing / grid spacing ratio"
    )
    args = parser.parse_args()

    case_dir = os.path.dirname(os.path.abspath(__file__))

    # --- IBM forcing stability: Fourier-number cap on dt --------------------
    # The direct-forcing feedback loop is stable only for
    # Fo = nu*dt/h^2 <~ 0.1 (in addition to Co <= 0.5); above it a slow
    # sawtooth develops in Cd/slip.  Cap dt accordingly (binds at low Re).
    nu = args.u_inf * args.diameter / args.Re
    dt_fo = args.max_fo * args.h**2 / nu
    if dt_fo < args.max_dt:
        print(f"  [IBM] capping max dt to {dt_fo:.4g} s (Fo = nu*dt/h^2 <= {args.max_fo})")
        args.max_dt = dt_fo
        args.initial_dt = min(args.initial_dt, dt_fo)

    # --- Mesh: uniform h core around the cylinder, stretched far field -----
    print("\n--- Mesh Generation (rectilinear, in-memory) ---")
    mesh_data, depth = cylinder_ibm_mesh(h=args.h, D=args.diameter)
    print(
        f"  cells: {mesh_data['n_elements']}, core spacing h = {args.h} "
        f"(D/h = {args.diameter / args.h:.0f})"
    )

    config = build_config(args, depth)
    solver = Solver(config, case_dir, mesh_data=mesh_data)

    # --- Immersed cylinder --------------------------------------------------
    body = ImmersedBody.cylinder_z(
        centre=[0.0, 0.0, 0.5 * depth],
        diameter=args.diameter,
        h=args.h,
        alpha=args.marker_alpha,
        name="cylinder",
    )
    solver.set_immersed_bodies(body, h=args.h)

    # Save the marker cloud for the plotting scripts.
    sol_dir = os.path.join(case_dir, "solution")
    os.makedirs(sol_dir, exist_ok=True)
    np.savetxt(
        os.path.join(sol_dir, "ibm_markers.csv"), body.X, delimiter=",", header="x,y,z", comments=""
    )

    solver.write_vtk()
    while solver.flow_time < config.time.end_time:
        solver.evolve()

    print("\nSimulation completed successfully.")
    print("Reference values (Constant et al. 2017):")
    if abs(args.Re - 30.0) < 1e-9:
        print("  Re=30 steady:  Cd = 1.74-1.80, recirculation L/D = 1.55-1.70")
    elif abs(args.Re - 100.0) < 1e-9:
        print("  Re=100 unsteady:  mean Cd = 1.35-1.38, St = 0.164-0.165")


if __name__ == "__main__":
    main()
