#!/usr/bin/env python3
"""Report leapfrogging trajectory agreement with the local LBM reference."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ASSETS_DIR = Path(__file__).resolve().parent
CASE_DIR = ASSETS_DIR.parent
REFERENCE = ASSETS_DIR / "references" / "leapfrogging_lbm_trajectory.csv"
REFERENCE_ORIGIN = 2.5
REFERENCE_PASS_POSITIONS = np.array([1.292, 3.596, 4.898])
RING_CIRCULATION = np.pi


def pass_positions(diagnostics: pd.DataFrame) -> np.ndarray:
    """Return axial midpoints where the two material-ring centroids overtake."""
    centres = diagnostics.pivot_table(
        index="step",
        columns="group_id",
        values="vortex_centroid_x",
        aggfunc="last",
    ).dropna()
    if len(centres.columns) != 2:
        return np.array([])
    first, second = centres.columns
    difference = centres[first].to_numpy() - centres[second].to_numpy()
    crossings = np.flatnonzero(np.signbit(difference[1:]) != np.signbit(difference[:-1]))
    midpoint = 0.5 * (centres[first].to_numpy() + centres[second].to_numpy())
    return midpoint[crossings]


def radius_rmse(diagnostics: pd.DataFrame, reference: pd.DataFrame) -> list[float]:
    """Compare each material-ring radius along its common axial path."""
    errors = []
    mapping = {0: 2, 1: 1}
    for group_id, reference_ring in mapping.items():
        numerical = diagnostics[diagnostics.group_id == group_id].sort_values(
            "vortex_centroid_x"
        )
        target = reference[reference.ring == reference_ring].sort_values("x_over_R0")
        target_x = target.x_over_R0.to_numpy() - REFERENCE_ORIGIN
        lower = max(numerical.vortex_centroid_x.min(), target_x.min())
        upper = min(numerical.vortex_centroid_x.max(), target_x.max())
        selected = (target_x >= lower) & (target_x <= upper)
        interpolated = np.interp(
            target_x[selected],
            numerical.vortex_centroid_x,
            numerical.major_radius,
        )
        errors.append(
            float(np.sqrt(np.mean((interpolated - target.R_over_R0.to_numpy()[selected]) ** 2)))
        )
    return errors


def main() -> None:
    case_name = sys.argv[1] if len(sys.argv) > 1 else "leapfrog_les_rvpm_sfs"
    diagnostics_path = CASE_DIR / "samples" / case_name / "ring_diagnostics.csv"
    integrals_path = CASE_DIR / "samples" / case_name / "flow_integrals.csv"
    diagnostics = pd.read_csv(diagnostics_path)
    integrals = pd.read_csv(integrals_path)
    diagnostics = (
        diagnostics.sort_values("step", kind="stable")
        .drop_duplicates(["step", "group_id"], keep="last")
    )
    integrals = (
        integrals.sort_values("step", kind="stable")
        .drop_duplicates("step", keep="last")
    )
    reference = pd.read_csv(REFERENCE)

    passes = pass_positions(diagnostics)
    errors = radius_rmse(diagnostics, reference)
    last_x = float(
        diagnostics.groupby("step").vortex_centroid_x.mean().iloc[-1]
    )
    net_columns = [f"net_vortex_strength_{axis}" for axis in "xyz"]
    net_strength = integrals[net_columns].to_numpy(float)
    net_strength_drift = float(
        np.linalg.norm(net_strength[-1] - net_strength[0]) / RING_CIRCULATION
    )
    energy_growth = float(
        integrals.total_kinetic_energy.max() / integrals.total_kinetic_energy.iloc[0] - 1.0
    )
    print(f"case: {case_name}")
    print("pass positions, x/R0:", ", ".join(f"{value:.3f}" for value in passes))
    print("LBM first three, x/R0:", ", ".join(f"{value:.3f}" for value in REFERENCE_PASS_POSITIONS))
    print("radius RMSE, R/R0:", ", ".join(f"{value:.4f}" for value in errors))
    print(f"last mean position, x/R0: {last_x:.3f}")
    print(f"net vector-strength drift/Gamma0: {net_strength_drift:.3e}")
    print(f"maximum kinetic-energy growth: {energy_growth:+.2%}")

    pass_error = (
        float(np.max(np.abs(passes[:3] - REFERENCE_PASS_POSITIONS)))
        if len(passes) >= 3
        else np.inf
    )
    if (
        len(passes) < 3
        or pass_error > 0.5
        or last_x < 6.8
        or net_strength_drift > 0.10
        or energy_growth > 0.05
    ):
        raise SystemExit("leapfrogging trajectory or conservation check is incomplete")


if __name__ == "__main__":
    main()
