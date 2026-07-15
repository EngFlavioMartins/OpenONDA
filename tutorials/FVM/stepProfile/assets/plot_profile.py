#!/usr/bin/env python3
"""Front propagation: mid-height profiles of the default scheme over time,
each compared with the exact step location x = U t."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).parent))
from _common import COLORS, build_arg_parser, figure_size, load_pvd_timesteps, save_fig  # noqa: E402


def main():
    args = build_arg_parser().parse_args()

    Path(args.figures_dir).mkdir(parents=True, exist_ok=True)

    timesteps = load_pvd_timesteps(args.solution_dir)
    if not timesteps:
        print("No PVD timesteps found.")
        return
    timesteps = sorted(timesteps, key=lambda ts: ts["time"])

    # Up to 4 snapshots, evenly spaced in time (skipping t = 0).
    later = [ts for ts in timesteps if ts["time"] > 0]
    picks = later[:: max(len(later) // 4, 1)][-4:] if later else []
    if not picks:
        print("No snapshots after t = 0.")
        return

    fig, ax = plt.subplots(figsize=figure_size("single"))
    colors = [COLORS[name] for name in ("TUDdark", "VPMpurple", "FVMorange", "TUDcyan")]

    for i, snap in enumerate(picks):
        mesh = pv.read(snap["file"])
        phi = mesh.point_data["phi"]
        pts = mesh.points
        x, y = pts[:, 0], pts[:, 1]
        mid = np.abs(y - 0.5 * (y.max() + y.min())) < 0.011
        if not mid.any():
            continue
        idx = np.argsort(x[mid])
        color = colors[i % len(colors)]
        ax.plot(x[mid][idx], phi[mid][idx], color=color, lw=1.1, label=f"t = {snap['time']:.2f} s")
        # Exact front position for this snapshot.
        ax.axvline(snap["time"], color=color, lw=0.7, ls=":", alpha=0.7)

    ax.set_xlabel("x")
    ax.set_ylabel(r"$\phi$")
    ax.set_ylim(-0.1, 1.15)
    ax.set_title("Step advection (dotted lines: exact front x = U t)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    save_fig(fig, "step_evolution.png", args.figures_dir, dpi=args.dpi, figure_format=args.format)


if __name__ == "__main__":
    main()
