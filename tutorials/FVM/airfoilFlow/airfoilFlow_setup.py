#!/usr/bin/env python3
"""Laminar flow past a body-fitted NACA 0012 airfoil.

The quasi-2D Gmsh mesh uses empty front/back patches. Force histories and the
final surface-pressure distribution are written under ``solution``. At zero
angle of attack, lift and upper/lower pressure asymmetry provide symmetry
checks; no drag-reference band is claimed for this coarse tutorial mesh.
"""

import argparse
import csv
import math
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets"))

import mesh_airfoil as mesher  # noqa: E402

from source.solvers.FVM import (  # noqa: E402
    BoundaryConfig,
    FVMConfig,
    Solver,
    ForcesConfig,
    LinearSolverConfig,
    PimpleControl,
    SchemesConfig,
    TimeConfig,
    TransportConfig,
)
from source.solvers.FVM.mesh.gmsh_importer import GmshImporter  # noqa: E402


def build_config(args, u_vec):
    """FVM configuration for the airfoil case."""
    nu = args.u_inf * args.chord / args.Re

    solver_params_schemes = SchemesConfig(convection_scheme=args.convection_scheme)
    solver_params_linear = LinearSolverConfig(linear_solver=args.linear_solver)
    solver_params_pimple = PimpleControl(
        n_correctors=args.n_correctors,
        n_outer_correctors=args.n_outer,
        n_orthogonal_correctors=1,
    )
    solver_params_forces = ForcesConfig(
        force_patches=["airfoil"],
        ref_velocity=args.u_inf,
        ref_area=args.chord * mesher.DEPTH,
        ref_length=args.chord,
        moment_centre=[0.25 * args.chord, 0.0, 0.0],
        force_log_interval=1,
    )

    tc = TimeConfig(
        delta_t=args.dt,
        start_time=0.0,
        end_time=args.end_time,
        write_interval=10**9,
        write_interval_time=args.write_interval_time,
        adjust_timestep=True,
        max_cfl=args.max_cfl,
        max_delta_t=args.dt * 4,
        min_delta_t=1e-5,
    )

    return FVMConfig(
        case_name=args.case_name,
        time=tc,
        schemes=solver_params_schemes,
        linear=solver_params_linear,
        pimple=solver_params_pimple,
        forces=solver_params_forces,
        transport=TransportConfig(density=args.rho, nu=nu),
        turbulence=None,
        boundaries=[
            BoundaryConfig.inlet("inlet", u_vec),
            BoundaryConfig.outlet("outlet", p=0.0),
            BoundaryConfig.freestream("walls", u_vec),
            BoundaryConfig.wall("airfoil"),
            BoundaryConfig.empty("frontAndBack"),
        ],
        initial_U=u_vec,
        initial_p=0.0,
    )


def write_surface_cp(solver, sol_dir, args):
    """Surface pressure coefficient on the airfoil patch -> surface_cp.csv."""
    n = solver.mesh_data["n_elements"]
    n_interior = solver.mesh_data["n_interior_faces"]
    q = 0.5 * args.u_inf**2  # kinematic pressure, rho folded out
    rows = []
    for patch in solver.boundaries:
        if patch["name"] != "airfoil":
            continue
        start, nf = patch["startFace"], patch["nFaces"]
        centres = solver.geo_data["face_centroids"][start : start + nf]
        ghost = n + (start - n_interior)
        p_face = solver.p[ghost : ghost + nf]
        for (x, y, _z), p_i in zip(centres, p_face, strict=True):
            rows.append((x / args.chord, y / args.chord, p_i / q))
    rows.sort()
    path = os.path.join(sol_dir, "surface_cp.csv")
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["x_c", "y_c", "Cp"])
        writer.writerows(rows)
    print(f"  Surface Cp written: {path}")


def main():
    parser = argparse.ArgumentParser(description="Flow past a NACA 0012 airfoil")
    parser.add_argument("--Re", type=float, default=1000.0, help="Chord Reynolds number")
    parser.add_argument("--angle", type=float, default=0.0, help="Angle of attack [deg]")
    parser.add_argument("--end-time", type=float, default=25.0, help="End time [s]")
    parser.add_argument("--dt", type=float, default=0.005, help="Initial time step [s]")
    parser.add_argument("--max-cfl", type=float, default=1.0, help="Target max Courant")
    parser.add_argument("--chord", type=float, default=1.0, help="Chord length")
    parser.add_argument("--u-inf", type=float, default=1.0, help="Freestream velocity")
    parser.add_argument("--rho", type=float, default=1.0, help="Density")
    parser.add_argument(
        "--write-interval-time",
        type=float,
        default=5.0,
        help="Write VTK every N seconds of flow time",
    )
    parser.add_argument("--n-correctors", type=int, default=2, help="PISO correctors")
    parser.add_argument("--n-outer", type=int, default=1, help="PIMPLE outer correctors")
    parser.add_argument("--linear-solver", type=str, default="bicgstab")
    parser.add_argument("--convection-scheme", type=str, default="limitedLinear")
    parser.add_argument("--case-name", type=str, default="airfoilFlow")
    args = parser.parse_args()

    case_dir = os.path.dirname(os.path.abspath(__file__))
    angle_rad = math.radians(args.angle)
    u_vec = [
        args.u_inf * math.cos(angle_rad),
        args.u_inf * math.sin(angle_rad),
        0.0,
    ]

    print("\n--- Mesh Generation (gmsh) ---")
    msh_path = os.path.join(case_dir, "assets", "airfoil.msh")
    mesher.generate_mesh(msh_path)

    print("\n--- Mesh Import ---")
    importer = GmshImporter()
    importer.load_mesh(msh_path)
    mesh_data = importer.get_mesh_data()
    importer.finalize()

    config = build_config(args, u_vec)
    solver = Solver(config, case_dir, mesh_data=mesh_data)

    solver.write_vtk()
    while solver.flow_time < config.time.end_time:
        solver.evolve()

    sol_dir = os.path.join(case_dir, "solution")
    os.makedirs(sol_dir, exist_ok=True)
    write_surface_cp(solver, sol_dir, args)

    print("\nSimulation completed successfully.")
    if abs(args.angle) < 1e-9:
        print("Zero-angle check: mean lift and upper/lower Cp asymmetry should approach zero.")


if __name__ == "__main__":
    main()
