#!/usr/bin/env python3
"""Counter-rotating vortex dipole — core trajectory and radius comparison.

Reads the field-based vortex diagnostics (``field_diagnostics.csv``, from the
z=L/4 velocity/vorticity plane) for each viscous scheme and plots:
  - core x-position  xc / a_{c,0}  vs  ν t / a_{c,0}²
  - core radius       a_c / a_{c,0}  vs  ν t / a_{c,0}²

The trajectory panel also includes the analytical translation of the finite
Lamb--Oseen vortex filaments.  The reference uses the fixed initial spacing
between the two filaments and the plane's finite-filament endpoint factor.

Saves: figures/dipole_comparison.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__:
    from .postprocess import (
        SCHEME_DRAW_ORDER,
        SCHEMES,
        build_arg_parser,
        build_style_map,
        extract_dipole_timeseries,
        figure_size,
        load_theme,
        resolve_runtime_physics,
        save_fig,
        scheme_zorder,
        theoretical_dipole_trajectory,
    )
else:
    from postprocess import (
        SCHEME_DRAW_ORDER,
        SCHEMES,
        build_arg_parser,
        build_style_map,
        extract_dipole_timeseries,
        figure_size,
        load_theme,
        resolve_runtime_physics,
        save_fig,
        scheme_zorder,
        theoretical_dipole_trajectory,
    )


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
    colors, theme = load_theme()
    style_map = build_style_map(colors)

    fig, axes = plt.subplots(1, 2, figsize=figure_size("trajectory"))
    fig.subplots_adjust(wspace=0.20, bottom=0.27, top=0.92, left=0.08, right=0.92)

    plotted_schemes = []
    for scheme in SCHEME_DRAW_ORDER:
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
            "zorder": 150 if scheme == "rwm" else scheme_zorder(scheme),
        }
        if trajectory_mask.any():
            tau = run_kinematic_viscosity * t[trajectory_mask] / (a0**2)
            axes[0].plot(tau, xc[trajectory_mask] / a0, **plot_kw)
            if scheme == "rwm":
                lower = ts["x_core_ci_lower"][trajectory_mask] / a0
                upper = ts["x_core_ci_upper"][trajectory_mask] / a0
                finite_interval = np.isfinite(lower) & np.isfinite(upper)
                axes[0].fill_between(
                    tau[finite_interval],
                    lower[finite_interval],
                    upper[finite_interval],
                    color=st["color"],
                    alpha=0.18,
                    linewidth=0,
                    zorder=scheme_zorder(scheme) - 1,
                )
        if core_mask.any():
            tau = run_kinematic_viscosity * t[core_mask] / (a0**2)
            axes[1].plot(tau, a_c[core_mask] / a0, **plot_kw)
            if scheme == "rwm":
                lower = ts["a_c_ci_lower"][core_mask] / a0
                upper = ts["a_c_ci_upper"][core_mask] / a0
                finite_interval = np.isfinite(lower) & np.isfinite(upper)
                axes[1].fill_between(
                    tau[finite_interval],
                    lower[finite_interval],
                    upper[finite_interval],
                    color=st["color"],
                    alpha=0.18,
                    linewidth=0,
                    zorder=scheme_zorder(scheme) - 1,
                )
        plotted_schemes.append(scheme)

    if not plotted_schemes:
        plt.close(fig)
        out.unlink(missing_ok=True)
        print("  [dipole] no sampled trajectories; figure not generated")
        return 0

    print(f"  [dipole] plotting {len(plotted_schemes)}/{len(SCHEMES)} methods")

    # The pair is initialized as two finite columns.  Evaluate the theory on
    tau_ref = np.linspace(0.0, 3.8, 400)
    time_ref = tau_ref * a0**2 / run_kinematic_viscosity
    x_ref = theoretical_dipole_trajectory(
        time_ref,
        runtime["circulation"],
        runtime["vortex_separation"],
        run_kinematic_viscosity,
        runtime["t0"],
        runtime["column_length"],
    )
    reference_options = dict(theme.REFERENCE_STYLE)
    reference_options.update(label=r"$\int_0^t U_b\,dt'$", zorder=100)
    axes[0].plot(tau_ref, x_ref / a0, **reference_options)

    axes[0].set_xlabel(r"$\nu t / a_{c,0}^2$")
    axes[0].set_ylabel(r"$x_c / a_{c,0}$")
    axes[0].set_title("Core trajectory over time")
    axes[1].set_xlabel(r"$\nu t / a_{c,0}^2$")
    axes[1].set_ylabel(r"$a_c / a_{c,0}$")
    axes[1].set_title(r"Core radius over time")

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
