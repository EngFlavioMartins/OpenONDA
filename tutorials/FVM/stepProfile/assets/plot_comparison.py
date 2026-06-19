#!/usr/bin/env python3
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

sys.path.insert(0, str(Path(__file__).parent))
from _common import build_arg_parser, load_pvd_timesteps, save_fig


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

    final = max(timesteps, key=lambda ts: ts["time"])
    t_final = final["time"]
    print(f"Final snapshot at t = {t_final:.3f} s: {final['file']}")

    mesh = pv.read(final["file"])
    phi = mesh.point_data["phi"]
    pts = mesh.points
    x = pts[:, 0]
    y = pts[:, 1]
    mid = np.abs(y - 0.5 * (y.max() - y.min())) < 0.01
    if not mid.any():
        print("No mid-height cells found.")
        return
    idx = np.argsort(x[mid])
    x_cl = x[mid][idx]
    phi_num = phi[mid][idx]

    step_pos = 1.0 * t_final
    phi_exact = np.where(x_cl < step_pos, 1.0, 0.0)

    smearing_mask = (phi_num > 0.1) & (phi_num < 0.9)
    if smearing_mask.any():
        x_low = x_cl[smearing_mask].min()
        x_high = x_cl[smearing_mask].max()
        smearing_width_cells = (x_high - x_low) / (x_cl[1] - x_cl[0])
    else:
        smearing_width_cells = 0.0

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_cl, phi_num, color="#1A6B9A", lw=1.2, label="Numerical")
    ax.plot(x_cl, phi_exact, color="black", lw=1.0, ls="--", label="Exact")
    ax.fill_between(x_cl, 0, 1, where=smearing_mask, color="gray", alpha=0.15,
                     label=f"Smearing = {smearing_width_cells:.1f} cells")
    ax.set_xlabel("x along diagonal")
    ax.set_ylabel(r"$\phi$")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    save_fig(fig, "step_comparison.png", figs_dir, dpi=dpi)


if __name__ == "__main__":
    main()
