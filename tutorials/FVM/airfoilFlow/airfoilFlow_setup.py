#!/usr/bin/env python3
"""Run airfoilFlow (NACA0012) using the Modernized OpenONDA API."""

import argparse
import math
import os

import assets.mesh_airfoil as mesher
from source.solvers.FVM import (
    Solver,
    FVMConfig,
    TimeConfig,
    SolverParams,
    TransportConfig,
    BoundaryConfig,
    TurbulenceConfig,
)
from source.solvers.FVM.mesh.gmsh_importer import GmshImporter


def main():
    parser = argparse.ArgumentParser(description="Run airfoilFlow (NACA0012).")
    parser.add_argument("--end-time", type=float, default=30.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--write-interval", type=int, default=50)
    parser.add_argument("--Re", type=float, default=1000.0)
    parser.add_argument("--u-inf", type=float, default=1.0)
    parser.add_argument("--angle", type=float, default=23.0)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--linear-solver", type=str, default="bicgstab")
    parser.add_argument("--n-correctors", type=int, default=3)
    parser.add_argument("--n-outer", type=int, default=1)
    parser.add_argument("--case-name", type=str, default="airfoilFlow")
    parser.add_argument("--solution-dir", type=str, default="solution")
    args = parser.parse_args()

    case_dir = os.path.dirname(os.path.abspath(__file__))
    nu = args.u_inf * 1.0 / args.Re
    angle_rad = math.radians(args.angle)
    U_vec = [args.u_inf * math.cos(angle_rad), args.u_inf * math.sin(angle_rad), 0.0]

    solver_params = SolverParams.pimple(
        n_correctors=args.n_correctors,
        n_outer=args.n_outer,
        linear_solver=args.linear_solver,
        convection_scheme="deferred",
    )
    solver_params.force_patches = ["airfoil"]
    solver_params.ref_velocity = args.u_inf
    solver_params.ref_area = 1.0
    solver_params.ref_length = 1.0
    solver_params.n_orthogonal_correctors = 1

    tc = TimeConfig.transient(
        dt=args.dt, duration=args.end_time, write_interval=args.write_interval
    )
    tc.adjust_timestep = True
    tc.max_cfl = 2.0
    tc.max_delta_t = args.dt * 2

    config = FVMConfig(
        case_name=args.case_name,
        time=tc,
        solver=solver_params,
        transport=TransportConfig(density=args.rho, nu=nu),
        turbulence=TurbulenceConfig(model="none"),
        boundaries=[
            BoundaryConfig.inlet("inlet", U_vec),
            BoundaryConfig.outlet("outlet", p=0.0),
            BoundaryConfig.freestream("walls", U_vec),
            BoundaryConfig.wall("airfoil"),
            BoundaryConfig.empty("frontAndBack"),
        ],
        initial_U=U_vec,
        initial_p=0.0,
    )

    msh_path = os.path.join(case_dir, "assets/airfoil.msh")
    mesher.generate_mesh(msh_path)

    importer = GmshImporter()
    importer.load_mesh(msh_path)
    mesh_data = importer.get_mesh_data()
    importer.finalize()

    solver = Solver(config, case_dir, mesh_data=mesh_data)

    solver.write_vtk()
    while solver.flow_time < config.time.end_time:
        solver.evolve()


if __name__ == "__main__":
    main()
