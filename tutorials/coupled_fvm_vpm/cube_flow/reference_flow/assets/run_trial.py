"""Run a short reference FVM case in a separate output directory."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys


REFERENCE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REFERENCE_DIR))

import reference_flow_setup as case  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--end-time", type=float, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    time = replace(case.FVM_SETUP.time, end_time=arguments.end_time)
    setup = replace(case.FVM_SETUP, time=time)
    solver = case.fvm.create_fvm_solver(
        setup,
        case_dir=arguments.output_directory.resolve(),
        mesh=case.FVM_MESH,
    )
    solver.write_vtk()
    final_time_tolerance = 0.5 * setup.time.time_step_size
    while solver.time < setup.time.end_time - final_time_tolerance:
        solver.advance()
    solver.close()


if __name__ == "__main__":
    main()
