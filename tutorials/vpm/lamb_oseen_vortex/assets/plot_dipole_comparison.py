#!/usr/bin/env python3
"""Counter-rotating vortex dipole — core trajectory and radius comparison.

Reads the field-based vortex diagnostics (``field_diagnostics.csv``, from the
z=L/4 velocity/vorticity plane) for each viscous scheme and plots:
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
    from .plot_style import build_arg_parser, build_style_map, figure_size, load_theme, save_fig
    from .vortex_diagnostics import SCHEMES, resolve_runtime_physics
else:
    from plot_style import build_arg_parser, build_style_map, figure_size, load_theme, save_fig
    from vortex_diagnostics import SCHEMES, resolve_runtime_physics


def extract_dipole_timeseries(
    samples_dir: Path,
    scheme: str,
) -> dict | None:
    path = samples_dir / f"dipole_{scheme}" / "field_diagnostics.csv"
    if not path.is_file():
        return None
    try:
        data = pd.read_csv(path, on_bad_lines="skip").dropna(subset=["time", "step"])
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        print(f"  [dipole] skipping unreadable live CSV for {scheme!r}: {exc}")
        return None
    data = data.sort_values("step").drop_duplicates("step", keep="last")
    if data.empty:
        return None
    return {
        "t": data["time"].to_numpy(float),
        "x_core": data["vortex_centre_0_x"].to_numpy(float),
        "a_c": data["core_radius_0"].to_numpy(float),
    }


# =============================================================
# Plot
# =============================================================


def plot_dipole_case(args) -> int:
    samples_dir = Path(args.samples_dir)
    fmt = getattr(args, "format", "png")
    out = Path(args.figures_dir) / f"dipole_comparison.{fmt}"
    out.parent.mkdir(parents=True, exist_ok=True)

    runtime = resolve_runtime_physics(
        samples_dir,
        args.circulation,
        args.kinematic_viscosity,
        args.b0,
        args.a0_over_b0,
        prefix="dipole",
    )
    run_kinematic_viscosity = runtime["kinematic_viscosity"]
    a0 = runtime["velocity_peak_radius0"]
    colors, _ = load_theme()
    style_map = build_style_map(colors)

    fig, axes = plt.subplots(1, 2, figsize=figure_size("trajectory"))
    fig.subplots_adjust(wspace=0.20, bottom=0.25, top=0.92, left=0.08, right=0.98)

    plotted_schemes = []
    for scheme in SCHEMES:
        ts = extract_dipole_timeseries(samples_dir, scheme)
        if ts is None:
            print(f"  [dipole] skipping {scheme!r} — no data")
            continue
        t = ts["t"]
        xc = ts["x_core"]
        a_c = ts["a_c"]
        trajectory_mask = np.isfinite(t) & np.isfinite(xc)
        core_mask = np.isfinite(t) & np.isfinite(a_c)
        if not trajectory_mask.any() and not core_mask.any():
            continue
        st = style_map[scheme]
        plot_kw = {
            "color": st["color"],
            "label": st["label"],
            "marker": st["marker"],
            "markersize": 2.2,
            "linestyle": "None",
            "linewidth": 1.0,
        }
        if trajectory_mask.any():
            tau = run_kinematic_viscosity * t[trajectory_mask] / (a0**2)
            axes[0].plot(tau, xc[trajectory_mask] / a0, **plot_kw)
        if core_mask.any():
            tau = run_kinematic_viscosity * t[core_mask] / (a0**2)
            axes[1].plot(tau, a_c[core_mask] / a0, **plot_kw)
        plotted_schemes.append(scheme)

    if not plotted_schemes:
        plt.close(fig)
        out.unlink(missing_ok=True)
        print("  [dipole] no sampled trajectories; figure not generated")
        return 0

    print(f"  [dipole] plotting {len(plotted_schemes)}/{len(SCHEMES)} methods")

    axes[0].set_xlabel(r"$\nu t / a_{c,0}^2$")
    axes[0].set_ylabel(r"$x_c / a_{c,0}$")
    axes[0].set_title("Core trajectory over time")
    axes[0].set_xlim([0.0, 3.8])
    axes[0].set_ylim([0.0, 33.0])
    axes[1].set_xlabel(r"$\nu t / a_{c,0}^2$")
    axes[1].set_ylabel(r"$a_c / a_{c,0}$")
    axes[1].set_title(r"Core radius over time")
    axes[1].set_xlim([0.0, 3.8])
    axes[1].set_ylim([0.8, 4.5])

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(handles),
            bbox_to_anchor=(0.5, 0.00),
        )
    save_fig(fig, out, args.dpi)
    return 0


def main() -> int:
    p = build_arg_parser("Counter-rotating dipole trajectory and core-radius comparison.")
    return plot_dipole_case(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
