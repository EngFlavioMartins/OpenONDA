#!/usr/bin/env python3
"""Integrated bound/wake vortex-strength closure for the static 8-degree case.

Output: figures/flat_plate_kelvin.png
"""

from __future__ import annotations

if not __package__:
    from pathlib import Path as _CasePath
    from openonda.tutorial_runner import case_package

    __package__ = case_package(_CasePath(__file__).resolve().parents[1]) + ".assets"


import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

from ._plot_theme import (
    validation_subplots,
    validation_legend,
    CASE_DIR,
    color,
    cm,
    export_formats,
    save_fig,
)


CM = cm()


def load_budget(samples_dir: Path, name: str):
    csv = samples_dir / name / "vlm_forces.csv"
    if not csv.exists():
        print(f"  [MISSING] {csv}")
        return None
    df = pd.read_csv(csv)
    required = {"time", "bound_vortex_strength_y", "wake_vortex_strength_y"}
    missing = required.difference(df.columns)
    if missing:
        print(f"  [MISSING] {csv} lacks vector-strength columns: {sorted(missing)}")
        return None
    t = df["time"].to_numpy(float)
    bound = df["bound_vortex_strength_y"].to_numpy(float)
    wake = df["wake_vortex_strength_y"].to_numpy(float)
    valid = np.isfinite(t) & np.isfinite(bound) & np.isfinite(wake)
    if not valid.all():
        raise ValueError(f"Non-finite bound/wake strength data in {csv}")
    return t, bound, wake


def main() -> None:
    ap = argparse.ArgumentParser(description="Bound/wake vortex-strength closure.")
    ap.add_argument("--format", choices=export_formats(), default="both")
    ap.add_argument("--dpi", type=int, default=400)
    args = ap.parse_args()

    name = "exp_static_aoa08"
    angle_of_attack = 8.0
    budget = load_budget(CASE_DIR / "samples", name)
    if budget is None:
        print("  Skipping flat_plate_kelvin: budget data unavailable.")
        return
    t, bound, wake = budget
    if t.size == 0:
        raise SystemExit("No finite Kelvin-budget rows were found.")

    c_bound = color("vpm")
    c_wake = color("TUDcyan")
    residual = bound + wake
    scale = max(float(np.max(np.abs(bound))), 1e-15)
    rel = 100.0 * residual / scale
    max_rel = float(np.max(np.abs(rel)))

    fig, (ax, axr) = validation_subplots(2, height_cm=14, sharex=True, outer=0.098, top_padding_cm=0.12)
    ax.plot(t, bound, color=c_bound, lw=1.5, label=r"Bound, $\mathcal{A}_{b,y}$")
    ax.plot(
        t,
        -wake,
        "-",
        marker="s",
        markevery=80,
        ms=2.5,
        color=c_wake,
        lw=1.5,
        label=r"Wake, $-\mathcal{A}_{w,y}$",
    )
    ax.set_ylabel(r"$\mathcal{A}_y$ [m$^3$/s]")
    validation_legend(fig, ax)

    axr.axhline(0.0, color=color("reference"), ls="--", lw=1.0)
    axr.plot(t, 1e4 * rel, color=color("vpm"), lw=1.2, label="Signed span component")
    flow = pd.read_csv(CASE_DIR / "samples" / name / "flow_integrals.csv")
    coupled = flow[[f"coupled_vortex_strength_{a}" for a in "xyz"]].to_numpy()
    bound_vector = flow[[f"bound_vortex_strength_{a}" for a in "xyz"]].to_numpy()
    norm_scale = max(np.linalg.norm(bound_vector, axis=1).max(), 1e-15)
    axr.plot(
        flow.time,
        1e6 * np.linalg.norm(coupled, axis=1) / norm_scale,
        "-",
        marker="s",
        markevery=80,
        ms=2.5,
        color=color("TUDcyan"),
        label="Full vector norm",
    )
    axr.legend(frameon=False, loc="upper left")
    axr.margins(y=0.4)
    print("Vector-strength closure is necessary, not load validation")
    axr.set_xlabel("Time [s]")
    axr.set_ylabel("Closure residual [ppm]")
    axr.set_xlim(float(t.min()), float(t.max()))
    out_dir = CASE_DIR / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "flat_plate_kelvin.png"
    save_fig(fig, out, figure_format=args.format, dpi=args.dpi)
    print(f"Maximum relative closure residual: {max_rel / 100.0:.3e}")


if __name__ == "__main__":
    main()
