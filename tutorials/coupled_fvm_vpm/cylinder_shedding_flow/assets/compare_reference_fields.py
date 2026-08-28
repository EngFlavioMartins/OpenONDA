"""Compare two serial or partitioned reference fields by global cell ID."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyvista as pv


FIELDS = ("velocity", "kinematic_pressure", "courant_number", "vorticity")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="reference VTU or PVTU field")
    parser.add_argument("candidate", type=Path, help="candidate VTU or PVTU field")
    parser.add_argument("--output", type=Path, required=True, help="comparison JSON")
    parser.add_argument("--tolerance", type=float, default=1.0e-3)
    return parser.parse_args()


def _owned_cells(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    field = pv.read(path)
    cell_data = field.cell_data
    if "global_cell_id" in cell_data:
        global_ids = np.asarray(cell_data["global_cell_id"], dtype=np.int64)
    else:
        global_ids = np.arange(field.n_cells, dtype=np.int64)
    if "vtkGhostType" in cell_data:
        owned = np.asarray(cell_data["vtkGhostType"], dtype=np.uint8) == 0
    else:
        owned = np.ones(field.n_cells, dtype=bool)
    order = np.argsort(global_ids[owned])
    return (
        global_ids[owned][order],
        np.asarray(field.cell_centers().points)[owned][order],
        {
            name: np.asarray(cell_data[name], dtype=np.float64)[owned][order]
            for name in FIELDS
        },
    )


def main() -> None:
    arguments = _arguments()
    reference_ids, reference_centres, reference_fields = _owned_cells(arguments.reference)
    candidate_ids, candidate_centres, candidate_fields = _owned_cells(arguments.candidate)
    if not np.array_equal(candidate_ids, reference_ids):
        raise ValueError("The two fields do not contain the same owned global cells")

    centre_error = float(np.max(np.abs(candidate_centres - reference_centres)))
    comparisons = {}
    for name in FIELDS:
        difference = candidate_fields[name] - reference_fields[name]
        reference_scale = max(
            float(np.max(np.abs(reference_fields[name]))), np.finfo(np.float32).tiny
        )
        comparisons[name] = {
            "max_absolute_difference": float(np.max(np.abs(difference))),
            "rms_difference": float(np.sqrt(np.mean(difference * difference))),
            "relative_linf_difference": float(np.max(np.abs(difference)) / reference_scale),
        }

    maximum_relative_difference = max(
        result["relative_linf_difference"] for result in comparisons.values()
    )
    passed = centre_error <= 1.0e-12 and maximum_relative_difference <= arguments.tolerance
    result = {
        "schema": "openonda-reference-field-comparison/1",
        "reference_file": str(arguments.reference.resolve()),
        "candidate_file": str(arguments.candidate.resolve()),
        "owned_cells": int(reference_ids.size),
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
        raise SystemExit("Reference field comparison failed")


if __name__ == "__main__":
    main()
