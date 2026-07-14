#!/usr/bin/env python3
import argparse
import os
import sys

import gmsh
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

from source.solvers.FVM import FVMConfig, TimeConfig, SolverParams, TransportConfig, BoundaryConfig
from source.solvers.FVM.mesh.gmsh_importer import GmshImporter
from source.solvers.FVM.mesh import geometry, topology
from source.solvers.FVM.solve.equation_solver import ScalarEquationSolver
from source.solvers.FVM.io.vtk_exporter import VTKExporter, PVDManager
from source.solvers.FVM.io import logging


def generate_mesh(msh_path, nx, ny):
    L = 1.0
    W = 0.1
    D = 0.01

    if not gmsh.isInitialized():
        gmsh.initialize()
    gmsh.model.add("stepProfile")

    p1 = gmsh.model.occ.addPoint(0, 0, 0)
    p3 = gmsh.model.occ.addPoint(L, 0, 0)
    p4 = gmsh.model.occ.addPoint(L, W, 0)
    p5 = gmsh.model.occ.addPoint(0, W, 0)

    l1 = gmsh.model.occ.addLine(p1, p3)
    l2 = gmsh.model.occ.addLine(p3, p4)
    l3 = gmsh.model.occ.addLine(p4, p5)
    l4 = gmsh.model.occ.addLine(p5, p1)

    loop = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
    surf = gmsh.model.occ.addPlaneSurface([loop])
    gmsh.model.occ.extrude([(2, surf)], 0, 0, D, numElements=[1], heights=[1], recombine=True)
    gmsh.model.occ.synchronize()

    tol_e = 1e-6
    for c_dim, c_tag in gmsh.model.getEntities(1):
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(c_dim, c_tag)
        dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
        if dx > tol_e and dy < tol_e and dz < tol_e:
            gmsh.model.mesh.setTransfiniteCurve(c_tag, nx + 1)
        elif dy > tol_e and dx < tol_e and dz < tol_e:
            gmsh.model.mesh.setTransfiniteCurve(c_tag, ny + 1)
        else:
            gmsh.model.mesh.setTransfiniteCurve(c_tag, 2)

    for s_dim, s_tag in gmsh.model.getEntities(2):
        gmsh.model.mesh.setTransfiniteSurface(s_tag)
        gmsh.model.mesh.setRecombine(s_dim, s_tag)

    for v_dim, v_tag in gmsh.model.getEntities(3):
        gmsh.model.mesh.setTransfiniteVolume(v_tag)

    tol = 1e-6
    boundary_surfaces = gmsh.model.getBoundary(gmsh.model.getEntities(3), oriented=False)
    inlet, outlet, walls, frontAndBack = [], [], [], []

    for dim, tag in boundary_surfaces:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)

        if abs(zmin) < tol and abs(zmax) < tol:
            frontAndBack.append(tag)
        elif abs(zmin - D) < tol and abs(zmax - D) < tol:
            frontAndBack.append(tag)
        elif abs(xmin) < tol and abs(xmax) < tol:
            inlet.append(tag)
        elif abs(xmax - L) < tol and abs(xmin - L) < tol:
            outlet.append(tag)
        else:
            walls.append(tag)

    gmsh.model.addPhysicalGroup(2, inlet, 1, "inlet")
    gmsh.model.addPhysicalGroup(2, outlet, 2, "outlet")
    gmsh.model.addPhysicalGroup(2, walls, 3, "walls")
    gmsh.model.addPhysicalGroup(2, frontAndBack, 4, "frontAndBack")
    gmsh.model.addPhysicalGroup(3, [v[1] for v in gmsh.model.getEntities(3)], 1, "fluid")

    gmsh.model.mesh.generate(3)
    gmsh.write(msh_path)
    print(f"  Mesh written: {msh_path}")


def main():
    parser = argparse.ArgumentParser(description="stepProfile FVM tutorial")
    parser.add_argument("--end-time", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--write-interval", type=int, default=10)
    parser.add_argument("--nx", type=int, default=50)
    parser.add_argument("--ny", type=int, default=5)
    parser.add_argument("--u-mag", type=float, default=1.0)
    parser.add_argument("--v-mag", type=float, default=0.0)
    parser.add_argument("--case-name", type=str, default="stepProfile")
    parser.add_argument("--solution-dir", type=str, default="solution")
    args = parser.parse_args()

    case_dir = os.path.dirname(os.path.abspath(__file__))
    msh_path = os.path.join(case_dir, "stepProfile.msh")
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
    n_boundary_faces = mesh_data["n_faces"] - mesh_data["n_interior_faces"]
    n_total = n_cells + n_boundary_faces

    phi = np.zeros(n_total)

    boundaries = mesh_data["boundary"]
    for b in boundaries:
        if b["name"] == "inlet":
            b["bc_type_phi"] = "fixedValue"
            b["value_phi"] = 1.0
        else:
            b["bc_type_phi"] = "zeroGradient"

        start = n_cells + (b["startFace"] - mesh_data["n_interior_faces"])
        end = start + b["nFaces"]
        if b.get("bc_type_phi") == "fixedValue":
            phi[start:end] = b["value_phi"]

    U = np.zeros((n_total, 3))
    U[:] = [args.u_mag, args.v_mag, 0]

    dt = args.dt
    n_steps = int(args.end_time / dt)

    solver = ScalarEquationSolver(mesh_data, geo_data, boundaries)
    solutions = solver.solve_transient_advection_diffusion(
        phi,
        U,
        gamma=0.0,
        density=1.0,
        dt=dt,
        n_steps=n_steps,
        convection_scheme="upwind",
        time_scheme="euler_implicit",
        solver="spsolve",
    )

    print("\n--- Export ---")
    exporter = VTKExporter(mesh_data)
    pvd = PVDManager(os.path.join(sol_dir, f"{args.case_name}.pvd"))

    for i, sol in enumerate(solutions):
        if i % args.write_interval == 0 or i == n_steps:
            current_time = i * dt
            fname = os.path.join(sol_dir, f"snapshot_{current_time:.5f}.vtu")
            exporter.export(fname, {"phi": sol, "U": U})
            pvd.add_step(current_time, fname)
            print(f"  Step {i:4d} | Time {current_time:6.3f} | Written: {fname}")

    print("\nSimulation completed successfully.")


if __name__ == "__main__":
    main()
