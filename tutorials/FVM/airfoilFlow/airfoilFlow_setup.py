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
from openonda.fvm import GmshImporter

# ---- Case definition -----------------------------------------------------
CASE_NAME = "airfoilFlow"
CHORD = 1.0  # airfoil chord length [m]
FREESTREAM_VELOCITY = 1.0  # inflow speed [m/s]
DENSITY = 1.0  # fluid density [kg/m^3]

# ---- Time stepping and numerics ------------------------------------------
TIME_STEP = 0.005  # initial time step [s]
MAX_CFL = 1.0  # target maximum Courant number
MAX_TIME_STEP = 4 * TIME_STEP  # upper bound on the adapted time step [s]
MIN_TIME_STEP = 1e-5  # lower bound on the adapted time step [s]
WRITE_INTERVAL_TIME = 5.0  # save a snapshot every this many seconds
PISO_CORRECTORS = 2
OUTER_CORRECTORS = 1
ORTHOGONAL_CORRECTORS = 1
CONVECTION_SCHEME = "limitedLinear"
LINEAR_SOLVER = "bicgstab"


def build_config(reynolds: float, end_time: float, u_vec: list[float]) -> FVMSetup:
    """Build the FVM setup for the airfoil case."""
    nu = FREESTREAM_VELOCITY * CHORD / reynolds

    schemes = SchemesConfig(convection_scheme=CONVECTION_SCHEME)
    linear = LinearSolverConfig(linear_solver=LINEAR_SOLVER)
    pimple = PimpleControl(
        n_correctors=PISO_CORRECTORS,
        n_outer_correctors=OUTER_CORRECTORS,
        n_orthogonal_correctors=ORTHOGONAL_CORRECTORS,
    )
    forces = [
        ForceSampler(
            patch_names=["airfoil"],
            ref_velocity=FREESTREAM_VELOCITY,
            ref_area=CHORD * mesher.DEPTH,
            ref_length=CHORD,
            moment_centre=[0.25 * CHORD, 0.0, 0.0],
        )
    ]

    time = TimeConfig(
        delta_t=TIME_STEP,
        start_time=0.0,
        end_time=end_time,
        write_interval=10**9,
        write_interval_time=WRITE_INTERVAL_TIME,
        adjust_timestep=True,
        max_cfl=MAX_CFL,
        max_delta_t=MAX_TIME_STEP,
        min_delta_t=MIN_TIME_STEP,
    )

    return FVMSetup(
        case_name=CASE_NAME,
        time=time,
        schemes=schemes,
        linear=linear,
        pimple=pimple,
        samplers=forces,
        transport=TransportConfig(density=DENSITY, nu=nu),
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


def write_surface_cp(solver, sol_dir: str) -> None:
    """Write the surface pressure coefficient to ``surface_cp.csv``."""
    n = solver.mesh_data["n_elements"]
    n_interior = solver.mesh_data["n_interior_faces"]
    q = 0.5 * FREESTREAM_VELOCITY**2  # kinematic pressure (rho folds out)
    rows = []
    for patch in solver.boundaries:
        if patch["name"] != "airfoil":
            continue
        start, nf = patch["startFace"], patch["nFaces"]
        centres = solver.geo_data["face_centroids"][start : start + nf]
        ghost = n + (start - n_interior)
        p_face = solver.p[ghost : ghost + nf]
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
    importer = GmshImporter()
    importer.load_mesh(msh_path)
    mesh_data = importer.get_mesh_data()
    importer.finalize()

    print("\n===== SIMULATION =====")
    config = build_config(args.Re, args.end_time, u_vec)
    solver = Solver(config, case_dir, mesh_data=mesh_data)

    solver.write_vtk()
    while solver.flow_time < config.time.end_time:
        solver.evolve()

    sol_dir = os.path.join(case_dir, "solution")
    os.makedirs(sol_dir, exist_ok=True)
    write_surface_cp(solver, sol_dir)

    print("\n===== DONE =====")
    print("Simulation completed successfully. Run ./allplot.sh to make the figures.")
    if abs(args.angle) < 1e-9:
        print("Zero-angle check: mean lift and upper/lower Cp asymmetry should approach zero.")


if __name__ == "__main__":
    main()
