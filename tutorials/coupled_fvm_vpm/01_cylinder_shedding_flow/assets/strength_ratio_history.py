#!/usr/bin/env python3
"""Summarize transfer strength and boundary mismatch in the developed wake."""

import json
from pathlib import Path

import numpy as np

CASE_DIR = Path(__file__).resolve().parents[1]
DEVELOPED_START = 40.0  # s
DEVELOPED_END = 55.0  # s
CAP_FRACTION = 0.995
COLUMNS = (
    "run",
    "configured_kappa_maximum",
    "transfers",
    "kappa_median",
    "kappa_maximum_observed",
    "near_strength_limit_fraction",
    "mean_boundary_velocity_mismatch",
    "mean_maximum_boundary_velocity_mismatch",
    "mean_removed_vortex_strength_fraction",
    "mean_moment_correction_fraction",
)


def developed_transfer_statistics(path):
    """Statistics of coupling transfers in the shared developed-flow window."""
    metadata = json.loads((path.parent / "coupler_metadata.json").read_text())
    maximum_strength_ratio = metadata["configuration"]["max_strength_ratio"]
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records = [
        record
        for record in records
        if DEVELOPED_START <= record["time"] <= DEVELOPED_END
    ]
    strength_ratio = np.array([record["transfer"]["strength_ratio"] for record in records])
    boundary_mismatch = np.array(
        [record["fvm_boundary_data"]["mean_velocity_mismatch"] for record in records]
    )
    maximum_boundary_mismatch = np.array(
        [record["fvm_boundary_data"]["maximum_velocity_mismatch"] for record in records]
    )
    removal = np.array(
        [record["transfer"]["removed_vortex_strength_fraction"] for record in records]
    )
    moment_correction = np.array(
        [record["gbd_moment_correction"]["correction_fraction"] for record in records]
    )
    return {
        "configured_kappa_maximum": maximum_strength_ratio,
        "transfers": len(records),
        "kappa_median": np.median(strength_ratio),
        "kappa_maximum_observed": strength_ratio.max(),
        "near_strength_limit_fraction": np.mean(
            strength_ratio >= CAP_FRACTION * maximum_strength_ratio
        ),
        "mean_boundary_velocity_mismatch": boundary_mismatch.mean(),
        "mean_maximum_boundary_velocity_mismatch": maximum_boundary_mismatch.mean(),
        "mean_removed_vortex_strength_fraction": removal.mean(),
        "mean_moment_correction_fraction": moment_correction.mean(),
    }


def main():
    """Print and save transfer statistics for every sensitivity run."""
    rows = [
        {"run": path.parents[1].name, **developed_transfer_statistics(path)}
        for path in sorted(CASE_DIR.glob("sensitivity/box*/solution/coupler_diagnostics.jsonl"))
    ]
    print("Developed transfer diagnostics, 40 <= t <= 55 s")
    print("run  kappa median/max  near limit  mismatch mean/max  removed  moment")
    for row in rows:
        print(
            f"{row['run']}  {row['kappa_median']:.3f}/{row['kappa_maximum_observed']:.3f}  "
            f"{100.0 * row['near_strength_limit_fraction']:.1f}%  "
            f"{row['mean_boundary_velocity_mismatch']:.3e}/"
            f"{row['mean_maximum_boundary_velocity_mismatch']:.3e}  "
            f"{100.0 * row['mean_removed_vortex_strength_fraction']:.2f}%  "
            f"{100.0 * row['mean_moment_correction_fraction']:.2f}%"
        )
    lines = [",".join(COLUMNS)]
    lines += [
        ",".join(
            row[column] if isinstance(row[column], str) else f"{row[column]:.12g}"
            for column in COLUMNS
        )
        for row in rows
    ]
    output = CASE_DIR / "figures/auxiliary/strength_ratio_history.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
