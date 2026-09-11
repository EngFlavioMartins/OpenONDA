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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._plot_theme import CASE_DIR, centered_subplots_adjust, color, cm, export_formats, save_fig


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
    ap.add_argument("--format", choices=export_formats(), default="png")
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
    c_wake = color("hybrid")
    residual = bound + wake
    scale = max(float(np.max(np.abs(bound))), 1e-15)
    rel = 100.0 * residual / scale
    max_rel = float(np.max(np.abs(rel)))

    fig, (ax, axr) = plt.subplots(
        2,
        1,
        figsize=(12.5 * CM, 9.5 * CM),
        sharex=True,
        gridspec_kw={"height_ratios": [1.7, 1.1]},
    )
    centered_subplots_adjust(fig, outer=0.17, bottom=0.13, top=0.90, hspace=0.13)

    ax.plot(t, bound, color=c_bound, lw=1.5, label=r"Bound, $\mathcal{A}_{b,y}$")
    ax.plot(t, -wake, "--", color=c_wake, lw=1.5, label=r"Wake, $-\mathcal{A}_{w,y}$")
    ax.set_ylabel(r"Integrated strength [m$^3$/s]")
    ax.set_title(rf"Bound--wake strength balance, $\alpha={angle_of_attack:.0f}^\circ$")
    ax.legend(loc="lower right")

    axr.axhline(0.0, color=color("reference"), ls="--", lw=1.0)
    axr.plot(t, rel, color=color("DarkText"), lw=1.2)
    axr.set_xlabel("Time [s]")
    axr.set_ylabel(r"Residual [\%]")
    axr.set_xlim(float(t.min()), float(t.max()))
    out_dir = CASE_DIR / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "flat_plate_kelvin.png"
    save_fig(fig, out, figure_format=args.format, dpi=args.dpi)
    print(f"Maximum relative closure residual: {max_rel / 100.0:.3e}")


if __name__ == "__main__":
    main()
