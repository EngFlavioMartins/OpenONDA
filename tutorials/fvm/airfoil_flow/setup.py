#!/usr/bin/env python3
"""Laminar flow past a NACA 0012 airfoil on a body-fitted FVM mesh.

At zero angle of attack, lift and upper/lower pressure asymmetry should vanish.

Run with ``python setup.py``.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path

import openonda.fvm as fvm

# ---- Case definition -----------------------------------------------------
CASE_NAME = "airfoil_flow"
CHORD = 1.0  # airfoil chord length [m]
DEPTH = 0.8  # finite-span extrusion depth [m]
FREESTREAM_VELOCITY = 1.0  # inflow speed [m/s]
DENSITY = 1.0  # fluid density [kg/m^3]
REYNOLDS_NUMBER = 1000.0
ANGLE_OF_ATTACK_DEGREES = 0.0
FINAL_TIME = 25.0

# ---- Time stepping and numerics ------------------------------------------
TIME_STEP_SIZE = 0.005  # initial time step [s]
MAX_COURANT_NUMBER = 0.9  # target maximum Courant number
MAX_TIME_STEP_SIZE = 4 * TIME_STEP_SIZE  # upper bound on the adapted time step [s]
OUTPUT_INTERVAL_TIME = 5.0  # save a snapshot every this many seconds
PISO_CORRECTORS = 2
OUTER_CORRECTORS = 1
ORTHOGONAL_CORRECTORS = 1
CONVECTION_SCHEME = "limitedLinear"
LINEAR_SOLVER = "bicgstab"
DOMAIN = (-5.0, 15.0, -5.0, 5.0, -0.5, 0.5)
AIRFOIL_STL = Path(__file__).resolve().parent / "assets" / "airfoil.stl"


def create_fvm_mesh() -> fvm.CartesianMesher:
    """Declare the native surface-driven mesh for the finite wing."""
    return fvm.CartesianMesher(
        domain=fvm.BoxDomain(
            bounds=DOMAIN,
            patches=fvm.BoxPatches(
                xmin="inlet",
                xmax="outlet",
                ymin="walls",
                ymax="walls",
                zmin="frontAndBack",
                zmax="frontAndBack",
            ),
        ),
        surfaces=(fvm.STLSurface(AIRFOIL_STL, patch="airfoil"),),
        max_cell_size=1.0,
        boundary_cell_size=0.03125,
        min_cell_size=0.03125,
        refinements=(
            fvm.BoxRefinement(
                name="near_airfoil",
                bounds=(-1.0, 3.0, -1.0, 1.0, -0.4, 0.4),
                cell_size=0.125,
            ),
        ),
    )


def create_fvm_setup(u_vec: list[float]) -> fvm.FVMSetup:
    """Build the FVM setup for the airfoil case."""
    kinematic_viscosity = FREESTREAM_VELOCITY * CHORD / REYNOLDS_NUMBER

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
            reference_velocity=FREESTREAM_VELOCITY,
            reference_area=CHORD * DEPTH,
            reference_length=CHORD,
            moment_centre=[0.25 * CHORD, 0.0, 0.0],
        )
    ]

    time = fvm.TimeConfig(
        time_step_size=TIME_STEP_SIZE,
        start_time=0.0,
        end_time=FINAL_TIME,
        output_schedule=fvm.RunSchedule(every_time=OUTPUT_INTERVAL_TIME),
        adjustment=fvm.MaximumCourantTimeStep(
            maximum=MAX_COURANT_NUMBER,
            maximum_time_step_size=MAX_TIME_STEP_SIZE,
        ),
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
            fvm.BoundaryConfig.outlet("outlet", kinematic_pressure=0.0),
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
        centres = fvm_solver.geo_data["face_centre"][start : start + nf]
        ghost = n + (start - n_interior)
        p_face = fvm_solver.kinematic_pressure[ghost : ghost + nf]
        for (x, y, _z), p_i in zip(centres, p_face, strict=True):
            rows.append((x / CHORD, y / CHORD, p_i / q))
    rows.sort()

    path = os.path.join(sol_dir, "surface_cp.csv")
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["position_x_over_chord", "position_y_over_chord", "pressure_coefficient"])
        writer.writerows(rows)
    print(f"  Surface Cp written: {path}")


def main() -> None:
    case_dir = os.path.dirname(os.path.abspath(__file__))
    angle = math.radians(ANGLE_OF_ATTACK_DEGREES)
    u_vec = [
        FREESTREAM_VELOCITY * math.cos(angle),
        FREESTREAM_VELOCITY * math.sin(angle),
        0.0,
    ]

    print("\n===== SIMULATION =====")
    fvm_setup = create_fvm_setup(u_vec)
    fvm_solver = fvm.create_fvm_solver(
        fvm_setup,
        case_dir=case_dir,
        mesh=create_fvm_mesh(),
    )
    fvm_solver.run()

    sol_dir = os.path.join(case_dir, "solution")
    Path(sol_dir).mkdir(parents=True, exist_ok=True)
    write_surface_cp(fvm_solver, sol_dir)

    print("\n===== DONE =====")
    print("Simulation completed successfully. Run ./allplot.sh to make the figures.")
    if abs(ANGLE_OF_ATTACK_DEGREES) < 1e-9:
        print("Zero-angle check: mean lift and upper/lower Cp asymmetry should approach zero.")


if __name__ == "__main__":
    main()
