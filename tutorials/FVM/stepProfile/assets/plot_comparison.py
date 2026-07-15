#!/usr/bin/env python3
"""Final scalar profiles per convection scheme vs the exact advected step.

Reads solution/profiles.csv (written by stepProfile_setup.py) and reports, for
every scheme, the L1 transport error, the front smearing width in cells, and
the peak over/undershoot (a bounded scheme must show none).
"""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _common import COLORS, build_arg_parser, figure_size, save_fig  # noqa: E402

SCHEME_COLORS = ["TUDdark", "FVMorange", "TUDcyan", "VPMpurple"]


def main():
    args = build_arg_parser().parse_args()
    csv_path = Path(args.solution_dir) / "profiles.csv"
    if not csv_path.exists():
        print(f"  WARNING: {csv_path} not found")
        return
    with open(csv_path) as fh:
        rows = list(csv.DictReader(fh))
    columns = list(rows[0].keys())
    data = {k: np.array([float(r[k]) for r in rows]) for k in columns}
    schemes = [k for k in columns if k not in ("x", "exact")]
    x, exact = data["x"], data["exact"]
    dx = float(np.median(np.diff(np.sort(x))))

    fig, ax = plt.subplots(figsize=figure_size("single"))
    ax.plot(x, exact, color=COLORS["reference"], lw=1.3, ls="--", label="Exact step")

    print(f"  {'scheme':>14} {'L1 error':>9} {'front width':>12} {'overshoot':>10}")
    for i, scheme in enumerate(schemes):
        phi = data[scheme]
        color = COLORS[SCHEME_COLORS[i % len(SCHEME_COLORS)]]
        ax.plot(x, phi, color=color, lw=1.1, label=scheme)

        l1 = float(np.mean(np.abs(phi - exact)))
        smear = (phi > 0.1) & (phi < 0.9)
        width = (x[smear].max() - x[smear].min()) / dx if smear.any() else 0.0
        overshoot = max(float(phi.max() - 1.0), float(-phi.min()))
        print(f"  {scheme:>14} {l1:9.4f} {width:9.1f} cells {overshoot:+10.4f}")

    ax.set_xlabel("x")
    ax.set_ylabel(r"$\phi$")
    ax.set_ylim(-0.1, 1.15)
    ax.set_title("Advected step: convection schemes vs exact solution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    save_fig(fig, "step_comparison.png", args.figures_dir, dpi=args.dpi, figure_format=args.format)


if __name__ == "__main__":
    main()
