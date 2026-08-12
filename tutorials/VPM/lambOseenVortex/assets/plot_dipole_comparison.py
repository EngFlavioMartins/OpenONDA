#!/usr/bin/env python3
"""Counter-rotating vortex dipole — core trajectory and radius comparison.

Reads compact mid-plane diagnostics from each viscous scheme and plots:
  - core x-position  xc / b0  vs  ν t / b0²
  - core radius       a_c / a_{c,0}  vs  ν t / b0²

Saves: figures/dipole_comparison.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__:
    from ._common import (
        SCHEMES,
        build_arg_parser,
        build_style_map,
        load_theme,
        publication_size,
        resolve_runtime_physics,
        save_publication_figure,
    )
else:
    from _common import (
        SCHEMES,
        build_arg_parser,
        build_style_map,
        load_theme,
        publication_size,
        resolve_runtime_physics,
        save_publication_figure,
    )


def extract_dipole_timeseries(
    samples_dir: Path,
    scheme: str,
) -> dict | None:
    path = samples_dir / f"dipole_{scheme}" / "pair_diagnostics.csv"
    if not path.is_file():
        return None
    data = pd.read_csv(path, on_bad_lines="skip").dropna(subset=["flow_time", "time_step"])
    data = data.sort_values("time_step").drop_duplicates("time_step", keep="last")
    if data.empty:
        return None
    return {
        "t": data["flow_time"].to_numpy(float),
        "x_core": data["x_core"].to_numpy(float),
        "a_c": data["core_radius"].to_numpy(float),
        "total_gamma": data["surface_circulation"].to_numpy(float),
    }


# =============================================================
# Plot
# =============================================================


def plot_dipole_case(args) -> int:
    samples_dir = Path(args.samples_dir)
    fmt = getattr(args, "format", "png")
    out = Path(args.figures_dir) / f"dipole_comparison.{fmt}"
    out.parent.mkdir(parents=True, exist_ok=True)

    runtime = resolve_runtime_physics(samples_dir, args.gamma, args.nu, args.b0, args.a0_over_b0)
    run_nu = runtime["nu"]
    a0 = runtime["ac0"]
    colors, _ = load_theme()
    style_map = build_style_map(colors)

    fig, axes = plt.subplots(1, 2, figsize=publication_size(7.5))
    fig.subplots_adjust(wspace=0.43, bottom=0.34, top=0.88, left=0.13, right=0.97)

    plotted_schemes = []
    for scheme in SCHEMES:
        ts = extract_dipole_timeseries(samples_dir, scheme)
        if ts is None:
            print(f"  [dipole] skipping {scheme!r} — no data")
            continue
        t = ts["t"]
        xc = ts["x_core"]
        a_c = ts["a_c"]
        mask = np.isfinite(xc) & np.isfinite(a_c)
        t, xc, a_c = t[mask], xc[mask], a_c[mask]
        if len(t) == 0:
            continue
        tau = run_nu * t / (a0**2)
        st = style_map[scheme]
        plot_kw = {
            "color": st["color"],
            "label": st["label"],
            "marker": st["marker"],
            "markersize": 2.2,
            "linestyle": "None",
            "linewidth": 1.0,
        }
        axes[0].plot(tau, xc / args.b0, **plot_kw)
        axes[1].plot(tau, a_c / a0, **plot_kw)
        plotted_schemes.append(scheme)

    if not plotted_schemes:
        plt.close(fig)
        out.unlink(missing_ok=True)
        print("  [dipole] no sampled trajectories; figure not generated")
        return 0

    print(f"  [dipole] plotting {len(plotted_schemes)}/{len(SCHEMES)} methods")

    axes[0].set_xlabel(r"$\nu t / a_{c,0}^2$")
    axes[0].set_ylabel(r"$x_c / b_0$")
    axes[0].set_title("Core trajectory over time")
    axes[0].set_xlim([0.0, 5.0])
    axes[0].set_ylim([0.0, 5.5])
    axes[1].set_xlabel(r"$\nu t / a_{c,0}^2$")
    axes[1].set_ylabel(r"$a_c / a_{c,0}$")
    axes[1].set_title(r"Core radius over time")
    axes[1].set_xlim([0.0, 5.0])
    axes[1].set_ylim([0.8, 7.5])

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(handles),
            bbox_to_anchor=(0.5, 0.05),
        )
    save_publication_figure(fig, out, args.dpi)
    return 0


def main() -> int:
    p = build_arg_parser("Counter-rotating dipole trajectory and core-radius comparison.")
    return plot_dipole_case(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
