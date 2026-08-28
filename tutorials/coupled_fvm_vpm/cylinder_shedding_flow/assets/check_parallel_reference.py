"""Compare one serial and one four-rank reference field cell by cell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyvista as pv


FIELDS = ("velocity", "kinematic_pressure", "courant_number", "vorticity")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("serial", type=Path, help="serial VTU field")
    parser.add_argument("parallel", type=Path, help="partitioned PVTU field")
    parser.add_argument("--output", type=Path, required=True, help="comparison JSON")
    parser.add_argument("--tolerance", type=float, default=1.0e-3)
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    serial = pv.read(arguments.serial)
    parallel = pv.read(arguments.parallel)

    global_id = np.asarray(parallel.cell_data["global_cell_id"], dtype=np.int64)
    ghost = np.asarray(parallel.cell_data["vtkGhostType"], dtype=np.uint8)
    owned = ghost == 0
    order = np.argsort(global_id[owned])
    ordered_ids = global_id[owned][order]
    expected_ids = np.arange(serial.n_cells, dtype=np.int64)
    if not np.array_equal(ordered_ids, expected_ids):
        raise ValueError("Parallel owned cells do not cover each serial global cell exactly once")

    serial_centres = np.asarray(serial.cell_centers().points)
    parallel_centres = np.asarray(parallel.cell_centers().points)[owned][order]
    centre_error = float(np.max(np.abs(parallel_centres - serial_centres)))

    comparisons = {}
    for name in FIELDS:
        serial_values = np.asarray(serial.cell_data[name], dtype=np.float64)
        parallel_values = np.asarray(parallel.cell_data[name], dtype=np.float64)[owned][order]
        difference = parallel_values - serial_values
        scale = max(float(np.max(np.abs(serial_values))), np.finfo(np.float32).tiny)
        comparisons[name] = {
            "max_absolute_difference": float(np.max(np.abs(difference))),
            "rms_difference": float(np.sqrt(np.mean(difference * difference))),
            "relative_linf_difference": float(np.max(np.abs(difference)) / scale),
        }

    maximum_relative_difference = max(
        result["relative_linf_difference"] for result in comparisons.values()
    )
    passed = (
        serial.n_cells == int(np.count_nonzero(owned))
        and centre_error <= 1.0e-12
        and maximum_relative_difference <= arguments.tolerance
    )
    result = {
        "schema": "openonda-serial-parallel-reference-check/1",
        "serial_file": str(arguments.serial.resolve()),
        "parallel_file": str(arguments.parallel.resolve()),
        "cells": serial.n_cells,
        "parallel_ghost_cells": int(np.count_nonzero(~owned)),
        "maximum_cell_centre_difference": centre_error,
        "relative_linf_tolerance": arguments.tolerance,
        "fields": comparisons,
        "maximum_relative_linf_difference": maximum_relative_difference,
        "passed": passed,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit("Serial/parallel reference comparison failed")


if __name__ == "__main__":
    main()
