#!/usr/bin/env python3
"""Discrete bound/wake vortex-strength closure for the flat-plate case.

VPM particles store vector vortex strength ``alpha = integral(omega dV)``
[m^3/s].  The comparable VLM quantity is therefore not peak sectional
circulation [m^2/s], but the oriented bound-vortex strength
``sum(Gamma_TE * bound_leg)``.  For the complete bound+wake vortex system,
the two y-components must cancel (Kelvin's circulation theorem).

Styling follows the other flat-plate figures (plot_plate_polar.py,
plot_plate_spanwise.py): shared docs/themes/matplotlib_setup.py theme.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CASE_DIR.parents[2]
THEME_PATH = REPO_ROOT / "docs" / "themes" / "matplotlib_setup.py"

# -- Theme (same pattern as the sibling flat-plate plotters) ------------------
_M = None
if not THEME_PATH.exists():
    raise FileNotFoundError(f"OpenONDA matplotlib theme not found: {THEME_PATH}")
_spec = importlib.util.spec_from_file_location("matplotlib_setup", str(THEME_PATH))
_M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_M)
_M.set_style()
CM = _M.CM


def _c(key: str) -> str:
    """Theme colour by key."""
    return _M.COLORS[key]


def load_budget(case_dir: Path, name: str):
    csv = case_dir / "samples" / f"{name}.csv"
    if not csv.exists():
        raise FileNotFoundError(csv)
    df = pd.read_csv(csv)
    required = {"time", "gamma_bound_y", "gamma_wake_y"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{csv} lacks corrected circulation columns: {sorted(missing)}")
    t = df["time"].to_numpy(float)
    bound = df["gamma_bound_y"].to_numpy(float)
    wake = df["gamma_wake_y"].to_numpy(float)
    valid = np.isfinite(t) & np.isfinite(bound) & np.isfinite(wake)
    return t[valid], bound[valid], wake[valid]


def main() -> None:
    ap = argparse.ArgumentParser(description="Bound/wake vortex-strength closure.")
    ap.add_argument("--name", default="exp_static_aoa08")
    ap.add_argument("--aoa", type=float, default=8.0, help="Angle of attack [deg] (title only).")
    ap.add_argument("--solution-dir", default=str(CASE_DIR / "solution"))
    ap.add_argument("--figures-dir", default=str(CASE_DIR / "figures"))
    ap.add_argument("--format", choices=_M.EXPORT_FORMATS, default="png")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    t, bound, wake = load_budget(Path(args.solution_dir) / args.name, args.name)
    if t.size == 0:
        raise SystemExit("No finite Kelvin-budget rows were found.")

    c_bound = _c("vpm")
    c_wake = _c("hybrid")
    residual = bound + wake
    scale = max(float(np.max(np.abs(bound))), float(np.max(np.abs(wake))), 1e-15)
    rel = 100.0 * residual / scale
    max_rel = float(np.max(np.abs(rel)))

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(12 * CM, 10 * CM), sharex=True,
        gridspec_kw={"height_ratios": [1.7, 1.1]},
    )
    fig.subplots_adjust(left=0.16, right=0.95, bottom=0.12, top=0.91, hspace=0.13)

    ax.plot(t, bound, color=c_bound, lw=1.5, label=r"Bound, $\alpha_{b,y}$")
    ax.plot(t, -wake, "--", color=c_wake, lw=1.5, label=r"Wake, $-\alpha_{w,y}$")
    ax.set_ylabel(r"Vortex strength [m$^3$/s]")
    ax.set_title(rf"Bound–wake vortex-strength closure, $\alpha={args.aoa:.0f}^\circ$")
    ax.legend(loc="lower right")

    axr.axhline(0.0, color=_c("reference"), ls="--", lw=1.0)
    axr.plot(t, rel / 1e-4, color=_c("DarkText"), lw=1.2)
    axr.set_xlabel("Time [s]")
    axr.set_ylabel(r"$\mathrm{Residual}\ [10^{-4}\,\%]$")
    axr.set_xlim(float(t.min()), float(t.max()))
    axr.text(
        0.02, 0.94, rf"$\max|\Sigma\alpha_y|/\max|\alpha_y| = {max_rel:.1e}\,\%$",
        transform=axr.transAxes, ha="left", va="top" ,
    )

    out_dir = Path(args.figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "flat_plate_kelvin.png"
    _M.save_fig(fig, out, figure_format=args.format, dpi=args.dpi)
    print(f"Maximum relative closure residual: {max_rel / 100.0:.3e}")


if __name__ == "__main__":
    main()
