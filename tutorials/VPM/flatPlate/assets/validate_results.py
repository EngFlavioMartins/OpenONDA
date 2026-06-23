#!/usr/bin/env python3
"""Certification checks for the flat-plate suite."""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

AOA_TAGS = ["aoan10", "aoan05", "aoan02", "aoa00", "aoa02", "aoa05", "aoa08", "aoa10", "aoa12", "aoa15"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution-dir", default="solution")
    ap.add_argument("--figures-dir", default="figures")
    args = ap.parse_args()
    root, figs = Path(args.solution_dir), Path(args.figures_dir)
    failures: list[str] = []

    for frame in ("moving", "static"):
        for tag in AOA_TAGS:
            name = f"exp_{frame}_{tag}"
            path = root / name / "samples" / f"{name}.csv"
            if not path.exists():
                failures.append(f"missing {path}")
                continue
            df = pd.read_csv(path)
            if df.empty or not np.isfinite(df[["CL", "CD"]].to_numpy()).all():
                failures.append(f"{name}: empty or non-finite force history")
                continue
            if "chords" not in df or df["chords"].max() < 29.5:
                failures.append(f"{name}: did not reach 30 chord lengths")
                continue
            tail = df[df["chords"] >= df["chords"].max() - 5.0]
            scale = max(abs(float(tail.CL.mean())), 1e-12)
            rel = float(tail.CL.max() - tail.CL.min()) / scale
            if rel > 2e-3:
                failures.append(f"{name}: CL tail range {100*rel:.3f}% > 0.2%")

    kelvin_csv = root / "exp_static_aoa08" / "samples" / "exp_static_aoa08.csv"
    if kelvin_csv.exists():
        df = pd.read_csv(kelvin_csv)
        residual = df.gamma_bound_y + df.gamma_wake_y
        scale = max(float(np.max(np.abs(df.gamma_bound_y))), 1e-15)
        closure = float(np.max(np.abs(residual)) / scale)
        print(f"Kelvin bound/wake closure: {closure:.3e}")
        if closure > 1e-4:
            failures.append(f"Kelvin closure {closure:.3e} > 1e-4")

    for name in ["plate_polar.png", "plate_staticvsmoving.png", "plate_spanwise.png", "flat_plate_kelvin.png"]:
        if not (figs / name).exists():
            failures.append(f"missing figure {name}")

    if failures:
        print("\n".join(f"[FAIL] {x}" for x in failures))
        return 1
    print("[OK] flatPlate certification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
