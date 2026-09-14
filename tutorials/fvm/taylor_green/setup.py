#!/usr/bin/env python3
"""Periodic Taylor–Green vortex with analytic viscous decay (FVM, PIMPLE).

The velocity decays as exp(-2 nu t), providing exact total_kinetic_energy and error checks.

Run with ``python setup.py``.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np

import openonda.fvm as fvm
from openonda.fvm.mesher import periodic_square_mesh
from openonda.tutorial_runner import case_package

__package__ = case_package(Path(__file__).parent)
from .assets.decay_errors import flow_integrals, history_row

CASE_DIR = Path(__file__).resolve().parent


# Flow properties
DENSITY = 1.0  # fluid density [kg/m^3]
KINEMATIC_VISCOSITY = 0.1  # [m^2/s]

# Mesh and numerics
TIME_SCHEME = "backward"
LINEAR_SOLVER = "spsolve"
PISO_CORRECTORS = 2
OUTER_CORRECTORS = 1
NUMBER_OF_CELLS = 24
TIME_STEP_SIZE = 0.005  # [s]
FINAL_TIME = 0.05  # [s]
MAX_COURANT_NUMBER = 0.9
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


def main() -> None:
    nsteps = int(round(FINAL_TIME / TIME_STEP_SIZE))

    mesh = partial(periodic_square_mesh, NUMBER_OF_CELLS)

    schemes = fvm.DiscretizationConfig(convection_scheme=CONVECTION_SCHEME, time_scheme=TIME_SCHEME)
    linear = fvm.LinearSolverConfig(linear_solver=LINEAR_SOLVER)
    pimple = fvm.PimpleControl(n_correctors=PISO_CORRECTORS, n_outer_correctors=OUTER_CORRECTORS)
    boundaries = [
        fvm.BoundaryConfig.cyclic("xmin", "xmax"),
        fvm.BoundaryConfig.cyclic("xmax", "xmin"),
        fvm.BoundaryConfig.cyclic("ymin", "ymax"),
        fvm.BoundaryConfig.cyclic("ymax", "ymin"),
        fvm.BoundaryConfig.empty("zmin"),
        fvm.BoundaryConfig.empty("zmax"),
    ]
    fvm_setup = fvm.FVMSetup(
        case_name="taylor_green",
        time=fvm.TimeConfig(
            time_step_size=TIME_STEP_SIZE,
            end_time=FINAL_TIME,
            output_schedule=fvm.RunSchedule(every_n_steps=nsteps),
            adjustment=fvm.MaximumCourantTimeStep(
                maximum=MAX_COURANT_NUMBER,
                # Preserve this verification case's nominal time resolution;
                # CFL control may reduce it but does not coarsen it.
                maximum_time_step_size=TIME_STEP_SIZE,
            ),
        ),
        schemes=schemes,
        linear=linear,
        pimple=pimple,
        transport=fvm.TransportConfig(density=DENSITY, kinematic_viscosity=KINEMATIC_VISCOSITY),
        boundaries=boundaries,
    )

    with fvm.create_fvm_solver(fvm_setup, case_dir=CASE_DIR, mesh=mesh) as fvm_solver:
        centres = fvm_solver.geo_data["cell_centre"]
        fvm_solver.set_initial_velocity(exact_velocity(centres, 0.0, KINEMATIC_VISCOSITY))
        fvm_solver.write_vtk()
        initial_total_kinetic_energy, initial_enstrophy = fvm_solver.evaluate(flow_integrals)
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

        def record_history(*, append):
            row = fvm_solver.evaluate(
                history_row,
                exact_velocity,
                KINEMATIC_VISCOSITY,
                initial_total_kinetic_energy,
                initial_enstrophy,
            )
            fvm_solver.write_csv(
                "history.csv",
                [[row[name] for name in fields]],
                columns=fields,
                append=append,
            )

        record_history(append=False)
        while fvm_solver.time < FINAL_TIME - 1.0e-14:
            fvm_solver.advance()
            record_history(append=True)


if __name__ == "__main__":
    main()
