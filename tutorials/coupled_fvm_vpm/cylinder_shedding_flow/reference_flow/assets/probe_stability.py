#!/usr/bin/env python3
"""Run a short, non-writing stability probe on a reference-flow mesh."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

import openonda.fvm as fvm

CASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_DIR))

import setup as reference  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dx", type=float, default=1.0 / 24.0)
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--end-time", type=float, default=0.012)
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--maximum-dt", type=float, default=0.004)
    parser.add_argument("--maximum-courant", type=float, default=0.9)
    parser.add_argument("--outer-correctors", type=int, default=2)
    parser.add_argument("--correctors", type=int, default=2)
    parser.add_argument("--non-orthogonal-correctors", type=int, default=1)
    parser.add_argument("--velocity-relaxation", type=float, default=0.7)
    parser.add_argument("--pressure-relaxation", type=float, default=0.3)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    base = reference.solver_setup("stability_probe", arguments.dx)
    setup = replace(
        base,
        cores=1,
        logging=fvm.LoggingConfig(
            console=False,
            schedule=fvm.RunSchedule(every_n_steps=1),
        ),
        backup=fvm.BackupConfig(schedule=None, write_at_end=False),
        time=fvm.TimeConfig(
            time_step_size=arguments.dt,
            start_time=0.0,
            end_time=arguments.end_time,
            output_schedule=fvm.RunSchedule(every_n_steps=1_000_000),
            adjustment=(
                fvm.MaximumCourantTimeStep(
                    maximum=arguments.maximum_courant,
                    maximum_time_step_size=arguments.maximum_dt,
                )
                if arguments.adaptive
                else None
            ),
        ),
        pimple=fvm.PimpleControl(
            n_correctors=arguments.correctors,
            n_outer_correctors=arguments.outer_correctors,
            n_orthogonal_correctors=arguments.non_orthogonal_correctors,
            velocity_relaxation=arguments.velocity_relaxation,
            pressure_relaxation=arguments.pressure_relaxation,
        ),
        samplers=(),
    )

    with tempfile.TemporaryDirectory(prefix="openonda-fvm-stability-") as temporary:
        temporary_path = Path(temporary)
        solver = fvm.create_fvm_solver(
            setup,
            case_dir=CASE_DIR,
            solution_dir=temporary_path / "solution",
            samples_dir=temporary_path / "samples",
            mesh=reference.grid_mesh(arguments.dx),
        )
        solver.auto_write = False
        try:
            while solver.time < arguments.end_time - 1.0e-14:
                solver.advance()
                diagnostics = solver.last_diagnostics
                velocity = np.asarray(solver.velocity[: solver.mesh_data["n_cells"]])
                pressure = np.asarray(
                    solver.kinematic_pressure[: solver.mesh_data["n_cells"]]
                )
                print(
                    json.dumps(
                        {
                            "step": solver.step,
                            "time": solver.time,
                            "dt": solver._accepted_time_step_size,
                            "cfl": diagnostics.max_courant_number,
                            "velocity_max": float(
                                np.max(np.linalg.norm(velocity, axis=1))
                            ),
                            "pressure_abs_max": float(np.max(np.abs(pressure))),
                            "continuity_max": diagnostics.max_continuity_error,
                            "velocity_residual": diagnostics.residuals.get("velocity"),
                            "pressure_residual": diagnostics.residuals.get(
                                "kinematic_pressure"
                            ),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        finally:
            solver.close()


if __name__ == "__main__":
    main()
