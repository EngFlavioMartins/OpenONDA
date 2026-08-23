#!/usr/bin/env python3
"""Certification checks for the flat-plate suite."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

AOA_TAGS = [
    "aoan10",
    "aoan05",
    "aoan02",
    "aoa00",
    "aoa02",
    "aoa05",
    "aoa08",
    "aoa10",
    "aoa12",
    "aoa15",
]


def main() -> int:
    case_dir = Path(__file__).resolve().parents[1]
    root = case_dir / "samples"
    figs = case_dir / "figures"
    failures: list[str] = []

    for frame in ("moving", "static"):
        for tag in AOA_TAGS:
            name = f"exp_{frame}_{tag}"
            path = root / name / f"{name}.csv"
            if not path.exists():
                failures.append(f"missing {path}")
                continue
            df = pd.read_csv(path)
            if (
                df.empty
                or not np.isfinite(df[["lift_coefficient", "drag_coefficient"]].to_numpy()).all()
            ):
                failures.append(f"{name}: empty or non-finite force history")
                continue
            if (
                "nondimensional_distance_travelled" not in df
                or df["nondimensional_distance_travelled"].max() < 23.5
            ):
                failures.append(f"{name}: did not reach 24 chord lengths")
                continue
            tail = df[
                df["nondimensional_distance_travelled"]
                >= df["nondimensional_distance_travelled"].max() - 5.0
            ]
            scale = max(abs(float(tail["lift_coefficient"].mean())), 1e-12)
            rel = float(tail["lift_coefficient"].max() - tail["lift_coefficient"].min()) / scale
            if rel > 2e-3:
                failures.append(f"{name}: lift_coefficient tail range {100 * rel:.3f}% > 0.2%")

    kelvin_csv = root / "exp_static_aoa08" / "exp_static_aoa08.csv"
    if kelvin_csv.exists():
        df = pd.read_csv(kelvin_csv)
        residual = df.bound_vortex_strength_y + df.wake_vortex_strength_y
        scale = max(float(np.max(np.abs(df.bound_vortex_strength_y))), 1e-15)
        closure = float(np.max(np.abs(residual)) / scale)
        print(f"Kelvin bound/wake closure: {closure:.3e}")
        if closure > 1e-4:
            failures.append(f"Kelvin closure {closure:.3e} > 1e-4")

    for name in [
        "plate_polar.png",
        "plate_staticvsmoving.png",
        "plate_spanwise.png",
        "flat_plate_kelvin.png",
    ]:
        if not (figs / name).exists():
            failures.append(f"missing figure {name}")

    if failures:
        print("\n".join(f"[FAIL] {x}" for x in failures))
        return 1
    print("[OK] flat_plate certification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
