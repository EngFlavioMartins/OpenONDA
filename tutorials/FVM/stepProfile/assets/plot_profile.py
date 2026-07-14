#!/usr/bin/env python3
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

sys.path.insert(0, str(Path(__file__).parent))
from _common import COLORS, build_arg_parser, figure_size, load_pvd_timesteps, save_fig


def main():
    args = build_arg_parser().parse_args()
    sol_dir = args.solution_dir
    figs_dir = args.figures_dir
    dpi = args.dpi

    Path(figs_dir).mkdir(parents=True, exist_ok=True)

    timesteps = load_pvd_timesteps(sol_dir)
    if not timesteps:
        print("No PVD timesteps found.")
        return

    target_times = [0.25, 0.5, 0.75, 1.0]
    snapshots = []
    for t in target_times:
        best = min(timesteps, key=lambda ts: abs(ts["time"] - t))
        snapshots.append(best)
        print(f"  t={t}: using {best['time']:.3f} s -> {best['file']}")

    fig, ax = plt.subplots(figsize=figure_size("single"))
    colors = [COLORS[name] for name in ("TUDdark", "VPMpurple", "FVMorange", "TUDcyan")]

    for i, snap in enumerate(snapshots):
        mesh = pv.read(snap["file"])
        phi = mesh.point_data["phi"]
        pts = mesh.points
        x = pts[:, 0]
        y = pts[:, 1]
        mid = np.abs(y - 0.5 * (y.max() - y.min())) < 0.01
        if not mid.any():
            continue
        idx = np.argsort(x[mid])
        ax.plot(
            x[mid][idx],
            phi[mid][idx],
            color=colors[i % len(colors)],
            lw=1.2,
            label=f"t = {snap['time']:.3f} s",
        )

    ax.set_xlabel("x along diagonal")
    ax.set_ylabel(r"$\phi$")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    save_fig(fig, "step_profile.png", figs_dir, dpi=dpi, figure_format=args.format)


if __name__ == "__main__":
    main()
