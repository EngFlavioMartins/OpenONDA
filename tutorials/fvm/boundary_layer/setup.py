#!/usr/bin/env python3
"""Laminar flat-plate boundary layer at Re_L = 10,000 (FVM).

Velocity profiles and skin friction are compared with the Blasius solution.

Run with ``python setup.py``.
"""

from __future__ import annotations

import csv
import os

import numpy as np

from assets.mesh_plate import flat_plate_mesh

import openonda.fvm as fvm

# ---- Case definition -----------------------------------------------------
CASE_NAME = "boundary_layer"
PLATE_LENGTH = 1.0  # plate length [m]
FREESTREAM_VELOCITY = 1.0  # inflow speed [m/s]
DENSITY = 1.0  # fluid density [kg/m^3]
REYNOLDS_NUMBER = 1.0e4
FINAL_TIME = 8.0
STATIONS = (0.25, 0.5, 0.75)  # x/L positions of the Blasius profiles

# ---- Mesh ----------------------------------------------------------------
N_PLATE = 72  # cells along the plate
DOMAIN_HEIGHT = 0.35  # height of the domain [m]
WALL_CELL_HEIGHT = 0.0015  # height of the first cell next to the wall [m]
WALL_STRETCHING = 1.12  # wall-normal growth factor (1.0 = uniform)

# ---- Time stepping and numerics ------------------------------------------
TIME_STEP_SIZE = 0.005  # initial time step [s]
MAX_COURANT_NUMBER = 0.9  # target maximum Courant number
MAX_TIME_STEP_SIZE = 0.02  # upper bound on the adapted time step [s]
OUTPUT_INTERVAL_TIME = 2.0  # save a snapshot every this many seconds
PISO_CORRECTORS = 2
OUTER_CORRECTORS = 1
CONVECTION_SCHEME = "limitedLinear"
GRADIENT_SCHEME = "gauss"
LINEAR_SOLVER = "bicgstab"


def create_fvm_setup(kinematic_viscosity: float) -> fvm.FVMSetup:
    """Build the FVM setup for the flat-plate case."""
    schemes = fvm.DiscretizationConfig(
        convection_scheme=CONVECTION_SCHEME,
        gradient_scheme=GRADIENT_SCHEME,
    )
    linear = fvm.LinearSolverConfig(linear_solver=LINEAR_SOLVER)
    pimple = fvm.PimpleControl(
        n_correctors=PISO_CORRECTORS,
        n_outer_correctors=OUTER_CORRECTORS,
    )
    forces = [
        fvm.ForceSampler(
            patch_names=["plate"],
            reference_velocity=FREESTREAM_VELOCITY,
            reference_length=PLATE_LENGTH,
        )
    ]

    return fvm.FVMSetup(
        case_name=CASE_NAME,
        time=fvm.TimeConfig(
            time_step_size=TIME_STEP_SIZE,
            start_time=0.0,
            end_time=FINAL_TIME,
            output_schedule=fvm.RunSchedule(every_time=OUTPUT_INTERVAL_TIME),
            adjustment=fvm.MaximumCourantTimeStep(
                maximum=MAX_COURANT_NUMBER,
                maximum_time_step_size=MAX_TIME_STEP_SIZE,
            ),
        ),
        schemes=schemes,
        linear=linear,
        pimple=pimple,
        samplers=forces,
        transport=fvm.TransportConfig(density=DENSITY, kinematic_viscosity=kinematic_viscosity),
        turbulence=None,  # laminar for Re_x <= 1e4 << 5e5
        boundaries=[
            fvm.BoundaryConfig.inlet("inlet", [FREESTREAM_VELOCITY, 0.0, 0.0]),
            fvm.BoundaryConfig.outlet("outlet", kinematic_pressure=0.0),
            fvm.BoundaryConfig(name="floor", velocity_type="slip", pressure_type="zeroGradient"),
            fvm.BoundaryConfig.wall("plate"),
            fvm.BoundaryConfig(name="top", velocity_type="slip", pressure_type="zeroGradient"),
            fvm.BoundaryConfig.empty("front"),
            fvm.BoundaryConfig.empty("back"),
        ],
        initial_velocity=[FREESTREAM_VELOCITY, 0.0, 0.0],
        initial_kinematic_pressure=0.0,
    )


def write_profiles(fvm_solver, sol_dir: str, kinematic_viscosity: float) -> None:
    """Sample u(y) at the stations and Cf(x) along the plate into CSV files."""
    n = fvm_solver.mesh_data["n_cells"]
    cell_centre = fvm_solver.geo_data["cell_centre"][:n]
    u = fvm_solver.velocity[:n]
    xc, yc = cell_centre[:, 0], cell_centre[:, 1]

    # The plate uses a uniform x grid, so the column width is easy to find.
    plate_x = np.unique(np.round(xc[xc > 0], 12))
    dx = np.min(np.diff(plate_x))

    with open(os.path.join(sol_dir, "profiles.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["station", "position_x", "position_y", "velocity_x", "velocity_y"])
        for station in STATIONS:
            x_col = plate_x[np.argmin(np.abs(plate_x - station))]
            sel = np.abs(xc - x_col) < 0.5 * dx
            order = np.argsort(yc[sel])
            for y_i, u_i, v_i in zip(
                yc[sel][order], u[sel, 0][order], u[sel, 1][order], strict=True
            ):
                writer.writerow([station, x_col, y_i, u_i, v_i])

    # Skin friction from the wall-adjacent cell row: tau_w ~ mu * u1 / y1.
    kinematic_pressure = fvm_solver.kinematic_pressure[:n]
    y1 = yc.min()
    y_top = yc.max()
    wall = (np.abs(yc - y1) < 1e-12) & (xc > 0.0)
    top = np.abs(yc - y_top) < 1e-12
    order = np.argsort(xc[wall])
    x_w = xc[wall][order]
    u_w = u[wall, 0][order]
    kinematic_pressure_wall = kinematic_pressure[wall][order]
    u_e = np.interp(x_w, np.sort(xc[top]), u[top, 0][np.argsort(xc[top])])
    cf = 2.0 * kinematic_viscosity * u_w / (y1 * FREESTREAM_VELOCITY**2)
    rex = FREESTREAM_VELOCITY * x_w / kinematic_viscosity

    with open(os.path.join(sol_dir, "cf.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "position_x",
                "reynolds_number",
                "skin_friction_coefficient",
                "skin_friction_coefficient_blasius",
                "kinematic_pressure_wall",
                "velocity_x_top",
            ]
        )
        for row in zip(
            x_w,
            rex,
            cf,
            0.664 / np.sqrt(rex),
            kinematic_pressure_wall,
            u_e,
            strict=True,
        ):
            writer.writerow(row)

    print(f"  Profiles written: {os.path.join(sol_dir, 'profiles.csv')}")
    print(f"  Skin friction written: {os.path.join(sol_dir, 'cf.csv')}")


def main() -> None:
    case_dir = os.path.dirname(os.path.abspath(__file__))
    kinematic_viscosity = FREESTREAM_VELOCITY * PLATE_LENGTH / REYNOLDS_NUMBER

    print("\n===== MESH =====")
    print("---- Generating the flat-plate mesh ----")
    mesh_data, _depth = flat_plate_mesh(
        plate_length=PLATE_LENGTH,
        height=DOMAIN_HEIGHT,
        n_plate=N_PLATE,
        dy_wall=WALL_CELL_HEIGHT,
        ratio=WALL_STRETCHING,
    )
    delta_L = 5.0 * PLATE_LENGTH / np.sqrt(REYNOLDS_NUMBER)
    print(
        f"  cells: {mesh_data['n_cells']}, first wall cell {WALL_CELL_HEIGHT} "
        f"(delta99(L)/dy_wall = {delta_L / WALL_CELL_HEIGHT:.0f})"
    )

    print("\n===== SIMULATION =====")
    fvm_setup = create_fvm_setup(kinematic_viscosity)
    fvm_solver = fvm.create_fvm_solver(fvm_setup, case_dir=case_dir, mesh=mesh_data)
    fvm_solver.run()

    sol_dir = os.path.join(case_dir, "solution")
    os.makedirs(sol_dir, exist_ok=True)
    write_profiles(fvm_solver, sol_dir, kinematic_viscosity)

    print("\n===== DONE =====")
    print("Simulation completed successfully. Run ./allplot.sh to make the figures.")
    print("Validation targets (Blasius; Schlichting, Boundary-Layer Theory):")
    print("  u/U = f'(eta) with eta = y*sqrt(U/(nu x));  Cf = 0.664/sqrt(Re_x)")


if __name__ == "__main__":
    main()
