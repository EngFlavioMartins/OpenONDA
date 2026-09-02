#!/usr/bin/env python3
"""Laminar backward-facing-step flow at Re_h = 100 (FVM, PIMPLE).

The body-fitted expansion targets a reattachment length x/h of about 4--5.

Run with ``python setup.py``.
"""

from __future__ import annotations

import csv
import os

import numpy as np

from assets.mesh_step import backward_facing_step_mesh

import openonda.fvm as fvm
import openonda.fvm.mesher as msh

# ---- Case definition -----------------------------------------------------
CASE_NAME = "step_profile"
STEP_HEIGHT = 1.0  # step height h [m]
MEAN_VELOCITY = 1.0  # bulk inlet velocity [m/s]
DENSITY = 1.0  # fluid density [kg/m^3]
REYNOLDS_NUMBER = 100.0
FINAL_TIME = 12.0

# ---- Mesh ----------------------------------------------------------------
N_UPSTREAM = 24  # cells upstream of the step (x/h < 0)
N_DOWNSTREAM = 120  # cells downstream of the step (x/h > 0)
N_HEIGHT = 16  # cells across the inlet channel height h

# ---- Time stepping and numerics ------------------------------------------
TIME_STEP_SIZE = 0.02  # initial time step [s]
MAX_COURANT_NUMBER = 0.9  # target maximum Courant number
MAX_TIME_STEP_SIZE = 0.05  # upper bound on the adapted time step [s]
OUTPUT_INTERVAL_TIME = 2.0  # save a snapshot every this many seconds
PISO_CORRECTORS = 2
OUTER_CORRECTORS = 1
CONVECTION_SCHEME = "limitedLinear"
GRADIENT_SCHEME = "gauss"
LINEAR_SOLVER = "bicgstab"


def inlet_velocity(mesh_data, geo_data):
    """Parabolic inlet profile with the requested bulk velocity."""
    patch = next(item for item in mesh_data["boundary"] if item["name"] == "inlet")
    start = patch["start_face"]
    stop = start + patch["n_faces"]
    y = geo_data["face_centre"][start:stop, 1]
    eta = np.clip((y - STEP_HEIGHT) / STEP_HEIGHT, 0.0, 1.0)
    values = np.zeros((patch["n_faces"], 3))
    values[:, 0] = 6.0 * MEAN_VELOCITY * eta * (1.0 - eta)
    return values


def initial_velocity(geo_data, n_cells):
    """Divergence-compatible profile on each side of the expansion."""
    centres = geo_data["cell_centre"][:n_cells]
    x, y = centres[:, 0], centres[:, 1]
    values = np.zeros((n_cells, 3))

    upstream = x < 0.0
    eta_up = np.clip((y[upstream] - STEP_HEIGHT) / STEP_HEIGHT, 0.0, 1.0)
    values[upstream, 0] = 6.0 * MEAN_VELOCITY * eta_up * (1.0 - eta_up)

    downstream = ~upstream
    eta_down = np.clip(y[downstream] / (2.0 * STEP_HEIGHT), 0.0, 1.0)
    values[downstream, 0] = 3.0 * MEAN_VELOCITY * eta_down * (1.0 - eta_down)
    return values


def reattachment_location(fvm_solver):
    """Estimate x/h where the first downstream cell row turns to positive u."""
    n_cells = fvm_solver.mesh_data["n_cells"]
    centres = fvm_solver.geo_data["cell_centre"][:n_cells]
    downstream = centres[:, 0] > 0.0
    y0 = np.min(centres[downstream, 1])
    near_wall = downstream & np.isclose(centres[:, 1], y0, atol=1e-10)
    order = np.argsort(centres[near_wall, 0])
    x = centres[near_wall, 0][order]
    u = fvm_solver.velocity[:n_cells, 0][near_wall][order]

    negative = np.flatnonzero(u < 0.0)
    if not len(negative):
        return np.nan, float(np.min(u))
    last_negative = negative[-1]
    if last_negative + 1 == len(u):
        return np.nan, float(np.min(u))
    x0, x1 = x[last_negative], x[last_negative + 1]
    u0, u1 = u[last_negative], u[last_negative + 1]
    x_re = x0 - u0 * (x1 - x0) / (u1 - u0)
    return float(x_re / STEP_HEIGHT), float(np.min(u))


def write_solution_tables(fvm_solver, solution_dir, history):
    """Write the cell fields and the reattachment/health history."""
    os.makedirs(solution_dir, exist_ok=True)
    n_cells = fvm_solver.mesh_data["n_cells"]
    centres = fvm_solver.geo_data["cell_centre"][:n_cells]

    fields_path = os.path.join(solution_dir, "fields.csv")
    with open(fields_path, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "position_x_over_height",
                "position_y_over_height",
                "velocity_x",
                "velocity_y",
                "kinematic_pressure",
            ]
        )
        for centre, velocity, pressure in zip(
            centres,
            fvm_solver.velocity[:n_cells],
            fvm_solver.kinematic_pressure[:n_cells],
            strict=True,
        ):
            writer.writerow(
                [
                    centre[0] / STEP_HEIGHT,
                    centre[1] / STEP_HEIGHT,
                    velocity[0],
                    velocity[1],
                    pressure,
                ]
            )

    history_path = os.path.join(solution_dir, "reattachment_history.csv")
    with open(history_path, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "time",
                "reattachment_position_over_height",
                "min_near_wall_velocity",
                "max_continuity_error",
                "max_courant_number",
            ]
        )
        writer.writerows(history)
    print(f"  Cell fields written: {fields_path}")
    print(f"  Reattachment history written: {history_path}")


def create_fvm_setup(
    reynolds: float, end_time: float, inlet_values: np.ndarray, kinematic_viscosity: float
) -> fvm.FVMSetup:
    """Build the FVM setup for the backward-facing-step case."""
    schemes = fvm.DiscretizationConfig(
        convection_scheme=CONVECTION_SCHEME,
        gradient_scheme=GRADIENT_SCHEME,
    )
    linear = fvm.LinearSolverConfig(linear_solver=LINEAR_SOLVER)
    pimple = fvm.PimpleControl(
        n_correctors=PISO_CORRECTORS,
        n_outer_correctors=OUTER_CORRECTORS,
    )

    return fvm.FVMSetup(
        case_name=CASE_NAME,
        time=fvm.TimeConfig(
            time_step_size=TIME_STEP_SIZE,
            end_time=end_time,
            output_schedule=fvm.RunSchedule(every_time=OUTPUT_INTERVAL_TIME),
            adjustment=fvm.MaximumCourantTimeStep(
                maximum=MAX_COURANT_NUMBER,
                maximum_time_step_size=MAX_TIME_STEP_SIZE,
            ),
        ),
        schemes=schemes,
        linear=linear,
        pimple=pimple,
        transport=fvm.TransportConfig(density=DENSITY, kinematic_viscosity=kinematic_viscosity),
        turbulence=None,
        boundaries=[
            fvm.BoundaryConfig.inlet("inlet", inlet_values.tolist()),
            fvm.BoundaryConfig.outlet("outlet", kinematic_pressure=0.0),
            fvm.BoundaryConfig.wall("walls"),
            fvm.BoundaryConfig.empty("front"),
            fvm.BoundaryConfig.empty("back"),
        ],
        initial_velocity=[0.0, 0.0, 0.0],
        initial_kinematic_pressure=0.0,
    )


def main() -> None:
    case_dir = os.path.dirname(os.path.abspath(__file__))
    solution_dir = os.path.join(case_dir, "solution")

    print("\n===== MESH =====")
    print("---- Generating the backward-facing-step mesh ----")
    mesh_data, depth = backward_facing_step_mesh(
        step_height=STEP_HEIGHT,
        n_upstream=N_UPSTREAM,
        n_downstream=N_DOWNSTREAM,
        n_height=N_HEIGHT,
    )
    geo_data = msh.geometry.compute_mesh_geometry(mesh_data)
    print(f"  cells: {mesh_data['n_cells']}; extrusion depth: {depth:g}")
    print("  geometry: inlet y/h=1..2, vertical step at x/h=0, outlet height=2h")

    print("\n===== SIMULATION =====")
    kinematic_viscosity = MEAN_VELOCITY * STEP_HEIGHT / REYNOLDS_NUMBER
    inlet_values = inlet_velocity(mesh_data, geo_data)
    fvm_setup = create_fvm_setup(REYNOLDS_NUMBER, FINAL_TIME, inlet_values, kinematic_viscosity)
    fvm_solver = fvm.create_fvm_solver(fvm_setup, case_dir=case_dir, mesh=mesh_data)
    fvm_solver.set_initial_velocity(initial_velocity(geo_data, mesh_data["n_cells"]))
    fvm_solver.write_vtk()

    history = []
    while fvm_solver.time < fvm_setup.time.end_time:
        fvm_solver.advance()
        x_re, min_u = reattachment_location(fvm_solver)
        diagnostics = fvm_solver.last_diagnostics
        history.append(
            [
                fvm_solver.time,
                x_re,
                min_u,
                diagnostics.max_continuity_error,
                diagnostics.max_courant_number,
            ]
        )

    write_solution_tables(fvm_solver, solution_dir, history)
    x_re, min_u = reattachment_location(fvm_solver)

    print("\n===== DONE =====")
    print("Simulation completed successfully. Run ./allplot.sh to make the figures.")
    print(f"  Re_h={REYNOLDS_NUMBER:g}; minimum downstream near-wall u={min_u:.6g}")
    print(
        f"  estimated x_reattachment/h={x_re:.6g}"
        if np.isfinite(x_re)
        else "  reattachment not resolved yet"
    )


if __name__ == "__main__":
    main()
