#!/usr/bin/env python3
"""Run cubeFlow using the Pure Python OpenONDA stack.

Loads a pre-generated mesh (use assets/mesh_cube.py first) and runs
an FVM simulation of flow past a cube at Re = Uinf*L/nu.

Usage:
    python assets/mesh_cube.py --mesh-size 20 --output cubeFlow.msh
    python cubeFlow_setup.py --end-time 30.0 --Re 300 --max-cfl 1.0
"""

import argparse
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

from source.solvers.FVM import (
    Solver,
    FVMConfig,
    TimeConfig,
    SolverParams,
    TransportConfig,
    TurbulenceConfig,
    BoundaryConfig,
)
from source.solvers.FVM.mesh.gmsh_importer import GmshImporter


def build_config(args):
    """Construct the FVM configuration object from CLI arguments."""
    nu = args.u_inf * args.length / args.Re

    return FVMConfig(
        case_name=args.case_name,
        time=TimeConfig(
            delta_t=args.initial_dt,
            start_time=0.0,
            end_time=args.end_time,
            write_interval=args.write_interval,
            write_interval_time=args.write_interval_time,
            adjust_timestep=args.max_cfl is not None,
            max_cfl=args.max_cfl or 0.9,
            max_delta_t=args.max_dt or 0.1,
            min_delta_t=1e-4,
        ),
        solver=SolverParams.pimple(
            n_correctors=args.n_correctors,
            n_outer=args.n_outer,
            linear_solver=args.linear_solver,
        ),
        transport=TransportConfig(density=args.rho, nu=nu),
        turbulence=TurbulenceConfig.smagorinsky(Cs=args.Cs, dynamic=False),
        boundaries=[
            BoundaryConfig.inlet("inlet", [args.u_inf, 0, 0]),
            BoundaryConfig.outlet("outlet", p=0.0),
            BoundaryConfig.freestream("walls", [args.u_inf, 0, 0]),
            BoundaryConfig.wall("cube"),
        ],
        initial_U=[args.u_inf, 0, 0],
        initial_p=0.0,
    )


def build_config_with_forces(args):
    """Build config with force computation on the cube patch."""
    config = build_config(args)
    config.solver.force_patches = ["cube"]
    config.solver.ref_velocity = args.u_inf
    config.solver.ref_area = args.length ** 2
    config.solver.ref_length = args.length
    return config


def main():
    parser = argparse.ArgumentParser(description="Run cubeFlow Simulation")
    parser.add_argument("--end-time", type=float, default=30.0, help="End time [s]")
    parser.add_argument("--initial-dt", type=float, default=0.01, help="Initial time step [s]")
    parser.add_argument("--max-cfl", type=float, default=None, help="Target max Courant number (enables adaptive dt)")
    parser.add_argument("--max-dt", type=float, default=None, help="Max time step cap [s]")
    parser.add_argument("--write-interval", type=int, default=5, help="Write interval [steps] (ignored if --write-interval-time set)")
    parser.add_argument("--write-interval-time", type=float, default=2.0, help="Write VTK every N seconds of flow time")
    parser.add_argument("--Re", type=float, default=300.0, help="Reynolds number")
    parser.add_argument("--u-inf", type=float, default=1.0, help="Inlet velocity magnitude [m/s]")
    parser.add_argument("--length", type=float, default=1.0, help="Reference length [m]")
    parser.add_argument("--rho", type=float, default=1.0, help="Fluid density [kg/m^3]")
    parser.add_argument("--msh", type=str, default="./assets/cubeFlow.msh", help="Path to mesh file")
    parser.add_argument("--linear-solver", type=str, default="bicgstab", help="Linear solver")
    parser.add_argument("--n-correctors", type=int, default=2, help="PIMPLE correctors")
    parser.add_argument("--n-outer", type=int, default=1, help="PIMPLE outer iterations")
    parser.add_argument("--Cs", type=float, default=0.17, help="Smagorinsky constant")
    parser.add_argument("--case-name", type=str, default="cubeFlow", help="Case name")
    args = parser.parse_args()

    case_dir = os.path.dirname(os.path.abspath(__file__))
    msh_path = os.path.join(case_dir, args.msh)

    print("\n--- Mesh Import ---")
    importer = GmshImporter()
    importer.load_mesh(msh_path)
    mesh_data = importer.get_mesh_data()
    importer.finalize()

    config = build_config_with_forces(args)
    solver = Solver(config, case_dir, mesh_data=mesh_data)

    solver.write_vtk()

    while solver.flow_time < config.time.end_time:
        solver.evolve()

    print("\nSimulation completed successfully.")


if __name__ == "__main__":
    main()
