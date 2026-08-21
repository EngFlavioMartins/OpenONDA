#!/usr/bin/env python3
"""Laminar flow past a NACA 0012 airfoil (body-fitted FVM mesh).

A quasi-two-dimensional airfoil in a uniform stream. The force history and
the final surface-pressure distribution are written under ``solution``;
``allplot.sh`` turns them into figures. At zero angle of attack the mean lift
and the upper/lower pressure asymmetry should both be zero, which makes a
nice symmetry check on a coarse tutorial mesh.
"""

from __future__ import annotations

import argparse
import csv
import math
import os

from assets import mesh_airfoil as mesher

import openonda.fvm as fvm

# ---- Case definition -----------------------------------------------------
CASE_NAME = "airfoil_flow"
CHORD = 1.0  # airfoil chord length [m]
FREESTREAM_VELOCITY = 1.0  # inflow speed [m/s]
DENSITY = 1.0  # fluid density [kg/m^3]

# ---- Time stepping and numerics ------------------------------------------
TIME_STEP_SIZE = 0.005  # initial time step [s]
MAX_CFL = 1.0  # target maximum Courant number
MAX_TIME_STEP_SIZE = 4 * TIME_STEP_SIZE  # upper bound on the adapted time step [s]
MIN_TIME_STEP_SIZE = 1e-5  # lower bound on the adapted time step [s]
OUTPUT_INTERVAL_TIME = 5.0  # save a snapshot every this many seconds
PISO_CORRECTORS = 2
OUTER_CORRECTORS = 1
ORTHOGONAL_CORRECTORS = 1
CONVECTION_SCHEME = "limitedLinear"
LINEAR_SOLVER = "bicgstab"


def create_fvm_setup(reynolds: float, end_time: float, u_vec: list[float]) -> fvm.FVMSetup:
    """Build the FVM setup for the airfoil case."""
    kinematic_viscosity = FREESTREAM_VELOCITY * CHORD / reynolds

    schemes = fvm.DiscretizationConfig(convection_scheme=CONVECTION_SCHEME)
    linear = fvm.LinearSolverConfig(linear_solver=LINEAR_SOLVER)
    pimple = fvm.PimpleControl(
        n_correctors=PISO_CORRECTORS,
        n_outer_correctors=OUTER_CORRECTORS,
        n_orthogonal_correctors=ORTHOGONAL_CORRECTORS,
    )
    forces = [
        fvm.ForceSampler(
            patch_names=["airfoil"],
            ref_velocity=FREESTREAM_VELOCITY,
            ref_area=CHORD * mesher.DEPTH,
            ref_length=CHORD,
            moment_centre=[0.25 * CHORD, 0.0, 0.0],
        )
    ]

    time = fvm.TimeConfig(
        time_step_size=TIME_STEP_SIZE,
        start_time=0.0,
        end_time=end_time,
        output_interval_steps=10**9,
        output_interval_time=OUTPUT_INTERVAL_TIME,
        adjust_time_step=True,
        max_cfl=MAX_CFL,
        max_time_step_size=MAX_TIME_STEP_SIZE,
        min_time_step_size=MIN_TIME_STEP_SIZE,
    )

    return fvm.FVMSetup(
        case_name=CASE_NAME,
        time=time,
        schemes=schemes,
        linear=linear,
        pimple=pimple,
        samplers=forces,
        transport=fvm.TransportConfig(density=DENSITY, kinematic_viscosity=kinematic_viscosity),
        turbulence=None,
        boundaries=[
            fvm.BoundaryConfig.inlet("inlet", u_vec),
            fvm.BoundaryConfig.outlet("outlet", p=0.0),
            fvm.BoundaryConfig.freestream("walls", u_vec),
            fvm.BoundaryConfig.wall("airfoil"),
            fvm.BoundaryConfig.empty("frontAndBack"),
        ],
        initial_velocity=u_vec,
        initial_kinematic_pressure=0.0,
    )


def write_surface_cp(fvm_solver, sol_dir: str) -> None:
    """Write the surface pressure coefficient to ``surface_cp.csv``."""
    n = fvm_solver.mesh_data["n_cells"]
    n_interior = fvm_solver.mesh_data["n_interior_faces"]
    q = 0.5 * FREESTREAM_VELOCITY**2  # kinematic pressure (rho folds out)
    rows = []
    for patch in fvm_solver.boundaries:
        if patch["name"] != "airfoil":
            continue
        start, nf = patch["start_face"], patch["n_faces"]
        centres = fvm_solver.geo_data["face_centroids"][start : start + nf]
        ghost = n + (start - n_interior)
        p_face = fvm_solver.kinematic_pressure[ghost : ghost + nf]
        for (x, y, _z), p_i in zip(centres, p_face, strict=True):
            rows.append((x / CHORD, y / CHORD, p_i / q))
    rows.sort()

    path = os.path.join(sol_dir, "surface_cp.csv")
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["x_c", "y_c", "Cp"])
        writer.writerows(rows)
    print(f"  Surface Cp written: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--Re", type=float, default=1000.0, help="chord Reynolds number Re = U c / nu"
    )
    parser.add_argument("--angle", type=float, default=0.0, help="angle of attack [deg]")
    parser.add_argument("--end-time", type=float, default=25.0, help="simulation end time [s]")
    args = parser.parse_args()

    case_dir = os.path.dirname(os.path.abspath(__file__))
    angle = math.radians(args.angle)
    u_vec = [
        FREESTREAM_VELOCITY * math.cos(angle),
        FREESTREAM_VELOCITY * math.sin(angle),
        0.0,
    ]

    print("\n===== MESH =====")
    print("---- Generating the airfoil mesh (gmsh) ----")
    msh_path = os.path.join(case_dir, "assets", "airfoil.msh")
    mesher.generate_mesh(msh_path)

    print("---- Importing the mesh ----")
    importer = fvm.GmshImporter()
    importer.load_mesh(msh_path)
    mesh_data = importer.get_mesh_data()
    importer.finalize()

    print("\n===== SIMULATION =====")
    fvm_setup = create_fvm_setup(args.Re, args.end_time, u_vec)
    fvm_solver = fvm.FVMSolver(fvm_setup, case_dir, mesh_data=mesh_data)

    fvm_solver.write_vtk()
    while fvm_solver.time < fvm_setup.time.end_time:
        fvm_solver.advance()

    sol_dir = os.path.join(case_dir, "solution")
    os.makedirs(sol_dir, exist_ok=True)
    write_surface_cp(fvm_solver, sol_dir)

    print("\n===== DONE =====")
    print("Simulation completed successfully. Run ./allplot.sh to make the figures.")
    if abs(args.angle) < 1e-9:
        print("Zero-angle check: mean lift and upper/lower Cp asymmetry should approach zero.")


if __name__ == "__main__":
    main()
