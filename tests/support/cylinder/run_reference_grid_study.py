#!/usr/bin/env python3
"""Mesh-convergence study of the cylinder FVM reference.

The in-plane wall spacing is refined by 1.5 while the span keeps one periodic
cell. Each grid runs in its own ``solution/`` and ``samples/`` subdirectory and
resumes from its latest native backup. Developed-shedding forces are then
compared with Richardson extrapolation and grid convergence indices.

Usage::

    python -m tests.support.cylinder.run_reference_grid_study [--run-dir DIR] [--end-time T]
"""

import argparse
from dataclasses import replace
from datetime import UTC, datetime
import importlib.util
from pathlib import Path

from .postprocess_grid_study import analyse_forces, print_report

REFERENCE = (
    Path(__file__).resolve().parents[3]
    / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/reference_flow"
)
GRID_SPACINGS = (0.09, 0.06, 0.04)  # m, wall spacing of the coarse, medium and fine grids
STATISTICS_START = 40.0  # s


def load_reference():
    """Return a fresh copy of the reference setup module."""
    spec = importlib.util.spec_from_file_location(
        "cylinder_reference_setup", REFERENCE / "setup.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def grid_name(spacing: float) -> str:
    return "grid_h" + format(spacing, ".8g").replace(".", "p")


def grid_case(directory: Path, spacing: float, end_time: float):
    """Return the reference module and its case at one wall spacing."""
    reference = load_reference()
    reference.CELL_SIZE = spacing
    name = grid_name(spacing)
    case = replace(
        reference.build_case(),
        name=name,
        directory=directory,
        variant=name,
    )
    case = replace(case, run=replace(case.run, end_time=end_time))
    return reference, case


def run_grid(directory: Path, spacing: float, end_time: float) -> None:
    reference, case = grid_case(directory, spacing, end_time)
    with reference.fvm.FVMSolver(case) as solver:
        cell_centres = solver.geo_data["cell_centre"][: solver.mesh_data["n_cells"]]
        solver.set_initial_velocity(reference.initial_velocity(cell_centres))
        solver.run(start_from="latest")
        reference.fvm.update_grid_study(solver, spacing, profiles=("centreline",))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="existing study directory to resume")
    parser.add_argument("--end-time", type=float, help="shorter physical horizon in s")
    options = parser.parse_args()

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    directory = options.run_dir or REFERENCE.parent / "study_results" / "reference_grids" / stamp
    end_time = options.end_time or load_reference().END_TIME
    for spacing in GRID_SPACINGS:
        run_grid(directory.resolve(), spacing, end_time)
    if end_time > STATISTICS_START:
        report = analyse_forces(
            samples_dir=directory / "samples",
            output_dir=directory / "figures",
            start=STATISTICS_START,
            end=end_time,
        )
        print_report(report)


if __name__ == "__main__":
    main()
