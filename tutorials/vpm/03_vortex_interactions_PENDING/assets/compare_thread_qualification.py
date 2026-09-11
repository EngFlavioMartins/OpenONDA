#!/usr/bin/env python3
"""Compare two CPU-thread qualification continuations.

The comparison is deliberately stricter than the qualification gate: it checks
every native particle field, both sampled CSV files, and both native VTK field
planes.  A JSON record makes the parity decision reproducible without relying
on formatted solver-log values.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


RELATIVE_RMS_LIMIT = 5.0e-5
ABSOLUTE_FLOORS = {
    "core_radius": 1.0e-12,
    "eddy_viscosity": 1.0e-14,
    "effective_viscosity": 1.0e-14,
    "kinematic_viscosity": 1.0e-14,
    "particle_volume": 1.0e-14,
    "position": 1.0e-12,
    "velocity": 1.0e-12,
    "vortex_strength": 1.0e-14,
    "vorticity": 1.0e-12,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _native_state(reference: Path, candidate: Path) -> dict[str, object]:
    fields: dict[str, object] = {}
    passed = True
    with h5py.File(reference, "r") as left, h5py.File(candidate, "r") as right:
        left_names = sorted(left["particles"])
        right_names = sorted(right["particles"])
        if left_names != right_names:
            raise ValueError("native particle-field names differ")
        for name in left_names:
            left_values = left[f"particles/{name}"][...]
            right_values = right[f"particles/{name}"][...]
            if left_values.shape != right_values.shape:
                raise ValueError(f"shape mismatch for {name}")
            discrete = np.issubdtype(left_values.dtype, np.integer)
            difference = right_values.astype(np.float64) - left_values.astype(np.float64)
            absolute_max = float(np.max(np.abs(difference), initial=0.0))
            exact = bool(np.array_equal(left_values, right_values))
            finite = bool(np.all(np.isfinite(left_values)) and np.all(np.isfinite(right_values)))
            if discrete:
                relative_rms = 0.0 if exact else float("inf")
                scale = None
            else:
                rms_reference = float(np.sqrt(np.mean(left_values.astype(np.float64) ** 2)))
                scale = max(rms_reference, ABSOLUTE_FLOORS[name])
                relative_rms = float(np.sqrt(np.mean(difference**2)) / scale)
            field_passed = finite and (exact if discrete else relative_rms <= RELATIVE_RMS_LIMIT)
            passed = passed and field_passed
            fields[name] = {
                "shape": list(left_values.shape),
                "dtype": str(left_values.dtype),
                "finite": finite,
                "exact": exact,
                "absolute_max_difference": absolute_max,
                "relative_rms_difference": relative_rms,
                "declared_scale": scale,
                "passed": field_passed,
            }
    return {
        "relative_rms_limit": RELATIVE_RMS_LIMIT,
        "fields": fields,
        "byte_identical": _sha256(reference) == _sha256(candidate),
        "passed": passed,
    }


def _csv_record(reference: Path, candidate: Path) -> dict[str, object]:
    with reference.open(newline="") as stream:
        left = list(csv.DictReader(stream))
    with candidate.open(newline="") as stream:
        right = list(csv.DictReader(stream))
    return {
        "rows": len(left),
        "byte_identical": _sha256(reference) == _sha256(candidate),
        "parsed_identical": left == right,
        "passed": left == right,
    }


def compare(
    reference_solution: Path,
    candidate_solution: Path,
    reference_samples: Path,
    candidate_samples: Path,
) -> dict[str, object]:
    native = _native_state(reference_solution, candidate_solution)
    sampled = {
        name: _csv_record(reference_samples / name, candidate_samples / name)
        for name in ("flow_integrals.csv", "ring_diagnostics.csv")
    }
    field_planes = {
        name: {
            "byte_identical": _sha256(reference_samples / name)
            == _sha256(candidate_samples / name),
        }
        for name in ("core_section_000050.vts", "cross_section_000050.vts")
    }
    for record in field_planes.values():
        record["passed"] = record["byte_identical"]
    passed = (
        bool(native["passed"])
        and all(bool(record["passed"]) for record in sampled.values())
        and all(bool(record["passed"]) for record in field_planes.values())
    )
    return {
        "qualification": "two-versus-six CPU threads from common native step-40 checkpoint",
        "native_state": native,
        "sampled_csv": sampled,
        "sampled_field_planes": field_planes,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-solution", required=True, type=Path)
    parser.add_argument("--candidate-solution", required=True, type=Path)
    parser.add_argument("--reference-samples", required=True, type=Path)
    parser.add_argument("--candidate-samples", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    record = compare(
        args.reference_solution,
        args.candidate_solution,
        args.reference_samples,
        args.candidate_samples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "passed": record["passed"]}))


if __name__ == "__main__":
    main()
