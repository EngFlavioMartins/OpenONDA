#!/usr/bin/env python3
"""Laminar backward-facing-step flow at Re_h = 100 (FVM, PIMPLE).

The body-fitted expansion targets a reattachment length x/h of about 4--5.

Run with ``python setup.py``.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np

import openonda.fvm as fvm

from assets.mesh_step import backward_facing_step_mesh
from assets.reattachment import history_row, write_solution_tables

# Case definition
START_FROM = "latest"  # Resume the latest backup; ./allrun.sh cleans first.

CASE_NAME = "step_profile"
STEP_HEIGHT = 1.0  # step height h [m]
MEAN_VELOCITY = 1.0  # bulk inlet velocity [m/s]
DENSITY = 1.0  # fluid density [kg/m^3]
REYNOLDS_NUMBER = 100.0
FINAL_TIME = 12.0  # [s]

# Mesh
UPSTREAM_LENGTH = 4.0  # upstream domain length / step height
DOWNSTREAM_LENGTH = 20.0  # downstream domain length / step height
N_UPSTREAM = 24  # cells upstream of the step (x/h < 0)
N_DOWNSTREAM = 120  # cells downstream of the step (x/h > 0)
N_HEIGHT = 16  # cells across the inlet channel height h

# Time stepping and numerics
TIME_STEP_SIZE = 0.02  # initial time step [s]
MAX_TIME_STEP_SIZE = 0.05  # upper bound on the adapted time step [s]
OUTPUT_INTERVAL_TIME = 2.0  # save a snapshot every this many seconds
GRADIENT_SCHEME = "gauss"


def inlet_velocity():
    """Parabolic inlet profile with the requested bulk velocity."""
    n_inlet = N_HEIGHT // 2
    eta = (np.arange(n_inlet) + 0.5) / n_inlet
    values = np.zeros((n_inlet, 3))
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


def create_fvm_case(inlet_values: np.ndarray, kinematic_viscosity: float) -> fvm.FVMCase:
    """Build the FVM case of the backward-facing step."""
    snapshots = fvm.EveryTime(OUTPUT_INTERVAL_TIME)
    mesh = partial(
        backward_facing_step_mesh,
        step_height=STEP_HEIGHT,
        upstream_length=UPSTREAM_LENGTH,
        downstream_length=DOWNSTREAM_LENGTH,
        n_upstream=N_UPSTREAM,
        n_downstream=N_DOWNSTREAM,
        n_height=N_HEIGHT,
    )

    return fvm.FVMCase(
        name=CASE_NAME,
        directory=Path(__file__).parent,
        numerics=fvm.Numerics(
            time_step_size=TIME_STEP_SIZE,
            time_step_control=fvm.MaximumCourantTimeStep(maximum_time_step_size=MAX_TIME_STEP_SIZE),
            transport=fvm.TransportConfig(density=DENSITY, kinematic_viscosity=kinematic_viscosity),
            schemes=fvm.DiscretizationConfig(gradient_scheme=GRADIENT_SCHEME),
        ),
        run=fvm.RunPlan(end_time=FINAL_TIME),
        output=fvm.Output(schedule=snapshots),
        backup=fvm.Backup(schedule=snapshots),
        mesh=mesh,
        boundaries=(
            fvm.BoundaryConfig.inlet("inlet", inlet_values),
            fvm.BoundaryConfig.outlet("outlet", kinematic_pressure=0.0),
            fvm.BoundaryConfig.wall("walls"),
            fvm.BoundaryConfig.empty("front"),
            fvm.BoundaryConfig.empty("back"),
        ),
    )


def main() -> None:
    kinematic_viscosity = MEAN_VELOCITY * STEP_HEIGHT / REYNOLDS_NUMBER
    case = create_fvm_case(inlet_velocity(), kinematic_viscosity)
    columns = (
        "time",
        "reattachment_position_over_height",
        "min_near_wall_velocity",
        "max_continuity_error",
        "max_courant_number",
    )

    with fvm.FVMSolver(case) as solver:
        solver.set_initial_velocity(initial_velocity(solver.geo_data, solver.mesh_data["n_cells"]))
        restored = solver.start_from(START_FROM)
        recorded = solver.restore_history("reattachment_history.csv")
        if not restored:
            solver.save_backup()
            solver.write_vtk()

        def record_history():
            row = solver.evaluate(history_row, STEP_HEIGHT)
            solver.write_csv("reattachment_history.csv", [row], columns=columns, append=True)

        if restored and not recorded and solver.step > 0:
            record_history()
        while solver.time < case.run.end_time:
            solver.advance()
            record_history()

        solver.evaluate(write_solution_tables, solver.solution_dir, STEP_HEIGHT)


if __name__ == "__main__":
    main()
