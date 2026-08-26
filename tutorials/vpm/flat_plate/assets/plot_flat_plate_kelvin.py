#!/usr/bin/env python3
"""Bound/wake vortex-strength closure for the static 8-degree case (Kelvin's theorem).

Output: figures/flat_plate_kelvin.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _plot_theme import CASE_DIR, SAMPLES_DIR, color, cm, save_fig, export_formats


CM = cm()


def load_budget(samples_dir: Path, name: str):
    csv = samples_dir / name / f"{name}.csv"
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
    return t[valid], bound[valid], wake[valid]


def main() -> None:
    ap = argparse.ArgumentParser(description="Bound/wake vortex-strength closure.")
    ap.add_argument("--format", choices=export_formats(), default="png")
    ap.add_argument("--dpi", type=int, default=300)
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
    scale = max(float(np.max(np.abs(bound))), float(np.max(np.abs(wake))), 1e-15)
    rel = 100.0 * residual / scale
    max_rel = float(np.max(np.abs(rel)))

    fig, (ax, axr) = plt.subplots(
        2,
        1,
        figsize=(12 * CM, 10 * CM),
        sharex=True,
        gridspec_kw={"height_ratios": [1.7, 1.1]},
    )
    fig.subplots_adjust(left=0.16, right=0.95, bottom=0.12, top=0.91, hspace=0.13)

    ax.plot(t, bound, color=c_bound, lw=1.5, label=r"Bound, $\alpha_{b,y}$")
    ax.plot(t, -wake, "--", color=c_wake, lw=1.5, label=r"Wake, $-\alpha_{w,y}$")
    ax.set_ylabel(r"Vortex strength [m$^3$/s]")
    ax.set_title(rf"Bound–wake vortex-strength closure, $\alpha={angle_of_attack:.0f}^\circ$")
    ax.legend(loc="lower right")

    axr.axhline(0.0, color=color("reference"), ls="--", lw=1.0)
    axr.plot(t, rel / 1e-4, color=color("DarkText"), lw=1.2)
    axr.set_xlabel("Time [s]")
    axr.set_ylabel(r"$\mathrm{Residual}\ [10^{-4}\,\%]$")
    axr.set_xlim(float(t.min()), float(t.max()))
    axr.text(
        0.02,
        0.94,
        rf"$\max|\Sigma\alpha_y|/\max|\alpha_y| = {max_rel:.1e}\,\%$",
        transform=axr.transAxes,
        ha="left",
        va="top",
    )

    out_dir = CASE_DIR / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "flat_plate_kelvin.png"
    save_fig(fig, out, figure_format=args.format, dpi=args.dpi)
    print(f"Maximum relative closure residual: {max_rel / 100.0:.3e}")


if __name__ == "__main__":
    main()
