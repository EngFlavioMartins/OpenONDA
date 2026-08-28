#!/usr/bin/env python3
"""Lamb-Oseen single vortex — radial profile comparison.

Reads the last z=L/4 surface-field snapshot from each viscous scheme,
slices the grid row nearest y=0, and plots:
  - azimuthal velocity  uθ / U_{c,0}
  - z-vorticity         ωz / ω_{c,0}
  - velocity gradient   (∂uy/∂x) · a_{c,0} / U_{c,0}

Saves: figures/vortex_comparison.png
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
        TOTAL_TIME,
        build_arg_parser,
        build_style_map,
        figure_size,
        lamb_oseen_gradient,
        lamb_oseen_profile,
        latest_common_time,
        load_profile,
        load_theme,
        resolve_runtime_physics,
        save_fig,
        scheme_zorder,
    )
else:
    from postprocess import (
        SCHEME_DRAW_ORDER,
        SCHEMES,
        TOTAL_TIME,
        build_arg_parser,
        build_style_map,
        figure_size,
        lamb_oseen_gradient,
        lamb_oseen_profile,
        latest_common_time,
        load_profile,
        load_theme,
        resolve_runtime_physics,
        save_fig,
        scheme_zorder,
    )

from matplotlib.ticker import FormatStrFormatter


# =============================================================
# Plot
# =============================================================


def plot_vortex_case(args) -> int:
    samples_dir = Path(args.samples_dir)
    fmt = getattr(args, "format", "png")
    out = Path(args.figures_dir) / f"vortex_comparison.{fmt}"
    out.parent.mkdir(parents=True, exist_ok=True)

    colors, _ = load_theme()
    style_map = build_style_map(colors)
    runtime = resolve_runtime_physics(
        samples_dir, args.circulation, args.kinematic_viscosity, args.b0, args.a0_over_b0
    )
    run_kinematic_viscosity = runtime["kinematic_viscosity"]
    run_t0 = runtime["t0"]
    # a_c is the radius of maximum azimuthal velocity.
    ac0 = runtime["velocity_peak_radius0"]
    run_circulation = runtime["circulation"]

    uc_ref = run_circulation / (2.0 * np.pi * ac0)
    wc_ref = run_circulation / (np.pi * ac0**2)
    gc_ref = uc_ref / ac0

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=figure_size("stacked_tall"))
    fig.subplots_adjust(hspace=0.12, top=0.95, bottom=0.19, left=0.12, right=0.88)

    comparison_time = latest_common_time(samples_dir)
    scheme_data: list[tuple[str, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for scheme in SCHEMES:
        profile = load_profile(samples_dir, scheme, comparison_time, include_uncertainty=True)
        if profile is None:
            continue
        x, uy, oz, t, velocity_se, vorticity_se, gradient_se, multiplier = profile
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
        scheme_data.append((scheme, t, x, uy, oz, dvx))

    if scheme_data:
        sample_times = [scheme[1] for scheme in scheme_data]
        elapsed_time = (
            comparison_time if comparison_time is not None else float(np.median(sample_times))
        )
        print(
            f"  [vortex] plotting {len(scheme_data)}/{len(SCHEMES)} methods "
            f"at common t={elapsed_time:.3g}s "
            f"(selected samples {min(sample_times):.3g}–{max(sample_times):.3g}s)"
        )
    else:
        elapsed_time = TOTAL_TIME
        print(f"  [vortex] no sampled profiles; plotting reference only at t={elapsed_time:.3g}s")

    r_line = np.linspace(-10.0 * ac0, 10.0 * ac0, 400)
    ref_kw = {"color": colors["reference"], "lw": 1.1, "zorder": 100, "linestyle": "-"}
    theory_t = run_t0 + elapsed_time
    tv, to, _ = lamb_oseen_profile(r_line, theory_t, run_circulation, run_kinematic_viscosity)
    tg = np.gradient(tv, r_line)
    axes[0].plot(r_line / ac0, tv / uc_ref, label="Theory", **ref_kw)
    axes[1].plot(r_line / ac0, to / wc_ref, **ref_kw)
    axes[2].plot(r_line / ac0, tg / gc_ref, **ref_kw)

    axes[0].set_title(r"Single vortex characteristics")
    axes[0].set_ylabel(r"$u_\theta / U_{c,0}$")
    axes[0].set_xlim([-5.5, 5.5])

    axes[1].set_ylabel(r"$\omega_z / \omega_{c,0}$")

    axes[2].set_xlabel(r"$r / a_{c,0}$")
    axes[2].set_ylabel(r"$(\partial u_y / \partial x)\,a_{c,0} / U_{c,0}$")

    handles, labels = axes[0].get_legend_handles_labels()

    for ax in axes:
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    save_fig(fig, out, args.dpi)
    return 0


def main() -> int:
    p = build_arg_parser("Lamb-Oseen single-vortex radial profile comparison.")
    return plot_vortex_case(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
