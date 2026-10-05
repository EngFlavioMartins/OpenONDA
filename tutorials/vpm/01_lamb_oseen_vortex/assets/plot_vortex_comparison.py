"""Lamb-Oseen single vortex — centreline profile comparison.

Reads the last cross-sectional field from each viscous scheme
(z=L/4 for deterministic schemes, column projection for RWM),
slices the grid row nearest y=0, and plots:
  - signed y-velocity  uy / U_{c,0}
  - z-vorticity         ωz / ω_{c,0}
  - velocity gradient   (∂uy/∂x) · a_{c,0} / U_{c,0}

Saves: figures/vortex_comparison.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from .postprocess import (
    SCHEMES,
    build_arg_parser,
    build_style_map,
    centered_subplots_adjust,
    figure_size,
    lamb_oseen_profile,
    latest_common_time,
    load_profile,
    load_theme,
    resolve_runtime_physics,
    save_fig,
    scheme_zorder,
)


def plot_vortex_case(args) -> int:
    samples_dir = Path(args.samples_dir)
    fmt = args.format
    out = Path(args.figures_dir) / f"vortex_comparison.{fmt}"
    colors, _ = load_theme()
    style_map = build_style_map(colors)
    runtime = resolve_runtime_physics(samples_dir)
    run_kinematic_viscosity = runtime["kinematic_viscosity"]
    run_t0 = runtime["t0"]
    ac0 = runtime["velocity_peak_radius0"]
    run_circulation = runtime["circulation"]
    uc_ref = run_circulation / (2.0 * np.pi * ac0)
    wc_ref = run_circulation / (np.pi * ac0**2)
    gc_ref = uc_ref / ac0
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=figure_size("stacked_tall"))
    centered_subplots_adjust(fig, outer=0.117, hspace=0.12, top=0.953, bottom=0.195)
    time_scale = run_kinematic_viscosity / ac0**2
    comparison_time = latest_common_time(samples_dir)
    for scheme in SCHEMES:
        profile = load_profile(
            samples_dir, scheme, comparison_time, include_uncertainty=scheme == "rwm"
        )
        x, uy, oz, _time = profile[:4]
        dvx = np.gradient(uy, x)
        st = style_map[scheme]
        plot_kw = {
            "color": st["color"],
            "label": st["label"],
            "marker": st["marker"],
            "markersize": 2.0,
            "markevery": 1,
            "linestyle": "None",
            "linewidth": 1.0,
            "zorder": scheme_zorder(scheme),
        }
        axes[0].plot(x / ac0, uy / uc_ref, **plot_kw)
        axes[1].plot(x / ac0, oz / wc_ref, **plot_kw)
        axes[2].plot(x / ac0, dvx / gc_ref, **plot_kw)
        if scheme == "rwm":
            velocity_se, vorticity_se, gradient_se, multiplier = profile[4:]
            for axis, mean, standard_error, scale in (
                (axes[0], uy, velocity_se, uc_ref),
                (axes[1], oz, vorticity_se, wc_ref),
                (axes[2], dvx, gradient_se, gc_ref),
            ):
                lower = (mean - multiplier * standard_error) / scale
                upper = (mean + multiplier * standard_error) / scale
                finite_interval = np.isfinite(lower) & np.isfinite(upper)
                axis.fill_between(
                    x[finite_interval] / ac0,
                    lower[finite_interval],
                    upper[finite_interval],
                    color=st["color"],
                    alpha=0.18,
                    linewidth=0,
                    zorder=scheme_zorder(scheme) - 1,
                )
    elapsed_time = comparison_time
    print(
        f"  [vortex] plotting {len(SCHEMES)} methods at common nu*t/a_c0^2={elapsed_time * time_scale:.2g}"
    )
    r_line = np.linspace(-10.0 * ac0, 10.0 * ac0, 400)
    ref_kw = {"color": colors["reference"], "lw": 1.1, "zorder": 100, "linestyle": "--"}
    theory_t = run_t0 + elapsed_time
    tv, to, _ = lamb_oseen_profile(r_line, theory_t, run_circulation, run_kinematic_viscosity)
    tg = np.gradient(tv, r_line)
    axes[0].plot(r_line / ac0, tv / uc_ref, label="Theory", **ref_kw)
    axes[1].plot(r_line / ac0, to / wc_ref, **ref_kw)
    axes[2].plot(r_line / ac0, tg / gc_ref, **ref_kw)
    axes[0].set_title("Single vortex characteristics")
    axes[0].set_ylabel("$u_y / U_{c,0}$")
    axes[0].set_xlim([-5.5, 5.5])
    axes[1].set_ylabel("$\\omega_z / \\omega_{c,0}$")
    axes[2].set_xlabel("$x / a_{c,0}$")
    axes[2].set_ylabel("$(\\partial u_y / \\partial x)\\,a_{c,0} / U_{c,0}$")
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    save_fig(fig, out, args.dpi)
    return 0


def main() -> int:
    p = build_arg_parser("Lamb-Oseen single-vortex centreline profile comparison.")
    return plot_vortex_case(p.parse_args())


if __name__ == "__main__":
    main()
