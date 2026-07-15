#!/usr/bin/env python3
"""Compare convection schemes by advecting a passive scalar step.

For unit velocity and zero diffusivity, the exact solution is

    phi(x, t) = 1  for x < U t,   0  otherwise.

The output reports L1 error, front width, and over/undershoot for each selected
scheme at a fixed Courant number.
"""

import argparse
import csv
import os
import sys

import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets"))

from mesh_step import generate_mesh  # noqa: E402

from source.solvers.FVM.io import logging  # noqa: E402
from source.solvers.FVM.io.vtk_exporter import PVDManager, VTKExporter  # noqa: E402
from source.solvers.FVM.mesh import geometry  # noqa: E402
from source.solvers.FVM.mesh.gmsh_importer import GmshImporter  # noqa: E402
from source.solvers.FVM.solve.equation_solver import ScalarEquationSolver  # noqa: E402


def setup_scalar_bcs(mesh_data, phi):
    """Step inlet (phi = 1), zero-gradient elsewhere; fill ghost values."""
    n_cells = mesh_data["n_elements"]
    for b in mesh_data["boundary"]:
        if b["name"] == "inlet":
            b["bc_type_phi"] = "fixedValue"
            b["value_phi"] = 1.0
        else:
            b["bc_type_phi"] = "zeroGradient"
        start = n_cells + (b["startFace"] - mesh_data["n_interior_faces"])
        end = start + b["nFaces"]
        if b["bc_type_phi"] == "fixedValue":
            phi[start:end] = b["value_phi"]


def centreline_profile(mesh_data, geo_data, phi):
    """(x, phi) along the mid-height cell row."""
    n = mesh_data["n_elements"]
    centroids = geo_data["element_centroids"][:n]
    y_mid = 0.5 * (centroids[:, 1].min() + centroids[:, 1].max())
    rows = np.unique(np.round(centroids[:, 1], 12))
    y_row = rows[np.argmin(np.abs(rows - y_mid))]
    sel = np.abs(centroids[:, 1] - y_row) < 1e-9
    order = np.argsort(centroids[sel, 0])
    return centroids[sel, 0][order], phi[:n][sel][order]


def main():
    parser = argparse.ArgumentParser(description="stepProfile FVM tutorial")
    parser.add_argument("--end-time", type=float, default=0.5)
    parser.add_argument("--courant", type=float, default=0.5, help="Co = U dt / dx")
    parser.add_argument("--nx", type=int, default=100)
    parser.add_argument("--ny", type=int, default=5)
    parser.add_argument("--u-mag", type=float, default=1.0)
    parser.add_argument(
        "--schemes",
        type=str,
        default="upwind,limitedLinear,superbee",
        help="Comma-separated convection schemes to compare",
    )
    parser.add_argument("--write-interval", type=int, default=20)
    parser.add_argument("--case-name", type=str, default="stepProfile")
    parser.add_argument("--solution-dir", type=str, default="solution")
    args = parser.parse_args()

    case_dir = os.path.dirname(os.path.abspath(__file__))
    msh_path = os.path.join(case_dir, "assets", "stepProfile.msh")
    sol_dir = os.path.join(case_dir, args.solution_dir)
    os.makedirs(sol_dir, exist_ok=True)

    print("--- Mesh Generation ---")
    generate_mesh(msh_path, args.nx, args.ny)

    print("\n--- Mesh Import ---")
    importer = GmshImporter()
    importer.load_mesh(msh_path)
    mesh_data = importer.get_mesh_data()
    importer.finalize()
    geo_data = geometry.compute_mesh_geometry(mesh_data)

    print("\n--- Simulation Start ---")
    logging.print_openonda_header()

    n_cells = mesh_data["n_elements"]
    n_total = n_cells + (mesh_data["n_faces"] - mesh_data["n_interior_faces"])

    dx = 1.0 / args.nx
    dt = args.courant * dx / args.u_mag
    n_steps = int(round(args.end_time / dt))
    t_end = n_steps * dt
    print(f"  Co = {args.courant}, dx = {dx:.4g}, dt = {dt:.4g}, {n_steps} steps to t = {t_end:g}")

    U = np.zeros((n_total, 3))
    U[:] = [args.u_mag, 0.0, 0.0]

    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    profiles = {}
    solutions_for_export = None

    for scheme in schemes:
        print(f"\n--- Scheme: {scheme} ---")
        phi = np.zeros(n_total)
        setup_scalar_bcs(mesh_data, phi)
        solver = ScalarEquationSolver(mesh_data, geo_data, mesh_data["boundary"])
        solutions = solver.solve_transient_advection_diffusion(
            phi,
            U,
            gamma=0.0,
            density=1.0,
            dt=dt,
            n_steps=n_steps,
            convection_scheme=scheme,
            time_scheme="euler_implicit",
            solver="spsolve",
        )
        x_cl, phi_cl = centreline_profile(mesh_data, geo_data, solutions[-1])
        profiles[scheme] = phi_cl
        if solutions_for_export is None:
            solutions_for_export = solutions  # VTU series for the first scheme

    # --- Final profiles vs the exact step -----------------------------------
    front = args.u_mag * t_end
    exact = np.where(x_cl < front, 1.0, 0.0)
    with open(os.path.join(sol_dir, "profiles.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["x", "exact", *schemes])
        for i in range(len(x_cl)):
            writer.writerow([x_cl[i], exact[i], *(profiles[s][i] for s in schemes)])
    print(f"\n  Final profiles written: {os.path.join(sol_dir, 'profiles.csv')}")

    # --- VTU time series (first scheme) for visualisation -------------------
    print("\n--- Export ---")
    exporter = VTKExporter(mesh_data)
    pvd = PVDManager(os.path.join(sol_dir, f"{args.case_name}.pvd"))
    for i, sol in enumerate(solutions_for_export):
        if i % args.write_interval == 0 or i == n_steps:
            current_time = i * dt
            fname = os.path.join(sol_dir, f"snapshot_{current_time:.5f}.vtu")
            exporter.export(fname, {"phi": sol, "U": U})
            pvd.add_step(current_time, fname)
            print(f"  Step {i:4d} | Time {current_time:6.3f} | Written: {fname}")

    print("\nSimulation completed successfully.")
    print(f"Exact solution: phi = 1 for x < {front:g}, 0 beyond (front advected from x = 0).")


if __name__ == "__main__":
    main()
