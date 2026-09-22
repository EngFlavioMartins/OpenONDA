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
from openonda.tutorial_runner import case_package

__package__ = case_package(Path(__file__).parent)
from .assets.mesh_step import backward_facing_step_mesh
from .assets.reattachment import history_row, write_solution_tables

# Case definition
START_FROM = "latest"  # Resume the latest backup; ./allrun.sh cleans first.

CASE_NAME = "step_profile"
STEP_HEIGHT = 1.0  # step height h [m]
MEAN_VELOCITY = 1.0  # bulk inlet velocity [m/s]
DENSITY = 1.0  # fluid density [kg/m^3]
REYNOLDS_NUMBER = 100.0
FINAL_TIME = 12.0  # [s]

# Mesh
N_UPSTREAM = 24  # cells upstream of the step (x/h < 0)
N_DOWNSTREAM = 120  # cells downstream of the step (x/h > 0)
N_HEIGHT = 16  # cells across the inlet channel height h

# Time stepping and numerics
TIME_STEP_SIZE = 0.02  # initial time step [s]
MAX_COURANT_NUMBER = 0.9  # target maximum Courant number
MAX_TIME_STEP_SIZE = 0.05  # upper bound on the adapted time step [s]
OUTPUT_INTERVAL_TIME = 2.0  # save a snapshot every this many seconds
PISO_CORRECTORS = 2
OUTER_CORRECTORS = 1
CONVECTION_SCHEME = "limitedLinear"
GRADIENT_SCHEME = "gauss"
LINEAR_SOLVER = "bicgstab"


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
        backup=fvm.BackupConfig(schedule=fvm.RunSchedule(every_time=OUTPUT_INTERVAL_TIME), write_at_end=True),
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
        boundaries=[
            fvm.BoundaryConfig.inlet("inlet", inlet_values.tolist()),
            fvm.BoundaryConfig.outlet("outlet", kinematic_pressure=0.0),
            fvm.BoundaryConfig.wall("walls"),
            fvm.BoundaryConfig.empty("front"),
            fvm.BoundaryConfig.empty("back"),
        ],
        initial_velocity=[0.0, 0.0, 0.0],
    )


def main() -> None:
    case_dir = Path(__file__).parent
    solution_dir = case_dir / "solution"

    mesh = partial(
        backward_facing_step_mesh,
        step_height=STEP_HEIGHT,
        n_upstream=N_UPSTREAM,
        n_downstream=N_DOWNSTREAM,
        n_height=N_HEIGHT,
    )

    kinematic_viscosity = MEAN_VELOCITY * STEP_HEIGHT / REYNOLDS_NUMBER
    inlet_values = inlet_velocity()
    fvm_setup = create_fvm_setup(REYNOLDS_NUMBER, FINAL_TIME, inlet_values, kinematic_viscosity)
    with fvm.create_fvm_solver(fvm_setup, case_dir=case_dir, mesh=mesh) as fvm_solver:
        fvm_solver.set_initial_velocity(
            initial_velocity(fvm_solver.geo_data, fvm_solver.mesh_data["n_cells"])
        )
        restored = fvm_solver.start_from(START_FROM)
        recorded = fvm_solver.reconcile_history("reattachment_history.csv")
        if not restored:
            fvm_solver.save_state(solution_dir / "backup")
            fvm_solver.write_vtk()
        columns = ("time", "reattachment_position_over_height", "min_near_wall_velocity",
                   "max_continuity_error", "max_courant_number")
        def record_history():
            row = fvm_solver.evaluate(history_row, STEP_HEIGHT)
            fvm_solver.write_csv("reattachment_history.csv", [row], columns=columns, append=True)

        if restored and not recorded and fvm_solver.step > 0:
            record_history()
        while fvm_solver.time < fvm_setup.time.end_time:
            fvm_solver.advance()
            record_history()

        fvm_solver.evaluate(write_solution_tables, solution_dir, None, STEP_HEIGHT)


if __name__ == "__main__":
    main()
