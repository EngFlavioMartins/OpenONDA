#!/usr/bin/env python3
"""Periodic Taylor–Green vortex with analytic viscous decay (FVM, PIMPLE).

The velocity decays as exp(-2 nu t), providing exact total_kinetic_energy and error checks.

Run with ``python setup.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import openonda.fvm as fvm
from openonda.fvm.mesher import box_mesh_3d

from assets.decay_errors import flow_integrals, history_row

START_FROM = "latest"  # Resume the latest backup; ./allrun.sh cleans first.

CASE_DIR = Path(__file__).resolve().parent


# Flow properties
DENSITY = 1.0  # fluid density [kg/m^3]
KINEMATIC_VISCOSITY = 0.1  # [m^2/s]

# Mesh and numerics
TIME_SCHEME = "backward"
LINEAR_SOLVER = "spsolve"
NUMBER_OF_CELLS = 24
TIME_STEP_SIZE = 0.005  # [s]
FINAL_TIME = 0.05  # [s]
CONVECTION_SCHEME = "central"


def exact_velocity(centres: np.ndarray, time: float, kinematic_viscosity: float) -> np.ndarray:
    """Return the analytic velocity at cell centres."""
    x = centres[:, 0]
    y = centres[:, 1]
    decay = np.exp(-2.0 * kinematic_viscosity * time)
    return np.column_stack(
        (
            decay * np.sin(x) * np.cos(y),
            -decay * np.cos(x) * np.sin(y),
            np.zeros_like(x),
        )
    )


def periodic_square_mesh() -> dict:
    """Square of side 2 pi with one cell across the span.

    The x and y patches are paired by cyclic boundaries; z-min/z-max are empty.
    """
    spacing = 2.0 * np.pi / NUMBER_OF_CELLS
    axis = np.linspace(0.0, 2.0 * np.pi, NUMBER_OF_CELLS + 1)
    mesh = box_mesh_3d(axis, axis, np.array([0.0, spacing]))
    for patch in mesh["boundary"]:
        patch["name"] = {"inlet": "xmin", "outlet": "xmax"}.get(patch["name"], patch["name"])
        if patch["name"].startswith("z"):
            patch["type"] = "empty"
    return mesh


def create_fvm_case() -> fvm.FVMCase:
    """Build the periodic Taylor-Green case."""
    final_frame = fvm.EverySteps(int(round(FINAL_TIME / TIME_STEP_SIZE)))
    return fvm.FVMCase(
        name="taylor_green",
        directory=CASE_DIR,
        numerics=fvm.Numerics(
            time_step_size=TIME_STEP_SIZE,
            # CFL control may reduce the nominal time step but never coarsens it.
            time_step_control=fvm.MaximumCourantTimeStep(maximum_time_step_size=TIME_STEP_SIZE),
            transport=fvm.TransportConfig(density=DENSITY, kinematic_viscosity=KINEMATIC_VISCOSITY),
            schemes=fvm.DiscretizationConfig(
                convection_scheme=CONVECTION_SCHEME, time_scheme=TIME_SCHEME
            ),
            linear=fvm.LinearSolverConfig(linear_solver=LINEAR_SOLVER),
        ),
        run=fvm.RunPlan(end_time=FINAL_TIME),
        output=fvm.Output(schedule=final_frame),
        backup=fvm.Backup(schedule=final_frame),
        mesh=periodic_square_mesh,
        boundaries=(
            fvm.BoundaryConfig.cyclic("xmin", "xmax"),
            fvm.BoundaryConfig.cyclic("xmax", "xmin"),
            fvm.BoundaryConfig.cyclic("ymin", "ymax"),
            fvm.BoundaryConfig.cyclic("ymax", "ymin"),
            fvm.BoundaryConfig.empty("zmin"),
            fvm.BoundaryConfig.empty("zmax"),
        ),
    )


def main() -> None:
    fields = (
        "step",
        "time",
        "total_kinetic_energy",
        "analytic_total_kinetic_energy",
        "total_kinetic_energy_relative_error",
        "velocity_l2_error",
        "total_enstrophy",
        "analytic_total_enstrophy",
        "total_enstrophy_relative_error",
        "max_continuity_error",
        "max_courant_number",
    )

    with fvm.FVMSolver(create_fvm_case()) as solver:
        centres = solver.geo_data["cell_centre"]
        solver.set_initial_velocity(exact_velocity(centres, 0.0, KINEMATIC_VISCOSITY))
        initial_total_kinetic_energy, initial_enstrophy = solver.evaluate(flow_integrals)
        restored = solver.start_from(START_FROM)
        recorded = solver.restore_history("history.csv")
        if not restored:
            solver.save_backup()
            solver.write_vtk()

        def record_history(*, append):
            row = solver.evaluate(
                history_row,
                exact_velocity,
                KINEMATIC_VISCOSITY,
                initial_total_kinetic_energy,
                initial_enstrophy,
            )
            solver.write_csv(
                "history.csv", [[row[name] for name in fields]], columns=fields, append=append
            )

        if not recorded:
            record_history(append=restored)
        while solver.time < FINAL_TIME - 1.0e-14:
            solver.advance()
            record_history(append=True)


if __name__ == "__main__":
    main()
