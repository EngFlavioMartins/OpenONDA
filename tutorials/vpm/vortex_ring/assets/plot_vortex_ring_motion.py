#!/usr/bin/env python3
"""
Self-induced velocity U_ring/U_ref,0 versus normalized time t Gamma/R0^2.

Compares DNS and LES vortex-ring self-induced velocity
against the analytical Saffman model with Gaussian core diffusion.

Saves: figures/vortex_ring_motion.png
"""

import numpy as np
import matplotlib.pyplot as plt

from tutorials.vpm.vortex_ring.assets.ring_metrics import (
    FIGURES_DIR,
    SAMPLES_DIR,
    REFERENCE_TIME,
    REFERENCE_VELOCITY,
    VARIANT_LABEL,
    VARIANT_STYLE,
    build_arg_parser,
    figure_size,
    load_sampled_ring_speed,
    load_theme,
    plot_variants,
    reference_style,
    saffman_speed,
    saffman_valid_time_limit,
    save_fig,
    with_sample_gaps,
)


def main() -> None:
    args = build_arg_parser(
        "Self-induced velocity U_ring/U_ref,0 versus normalized time t Gamma/R0^2."
    ).parse_args()
    figs = FIGURES_DIR
    figs.mkdir(parents=True, exist_ok=True)
    n_skip = 7  # plot every n-th marker

    load_theme()

    fig, (ax, comparison) = plt.subplots(
        1,
        2,
        figsize=(125 / 25.4, 90 / 25.4),
        gridspec_kw={"width_ratios": (1.0, 1.0)},
    )
    fig.subplots_adjust(wspace=0.55, left=0.12, right=0.97, top=0.88, bottom=0.38)
    curves = []
    plotted_values = []

    # -- Ring speed — all available variants ---------------------------------
    for variant in plot_variants():
        st = VARIANT_STYLE[variant]
        csv_path = SAMPLES_DIR / variant / "ring_diagnostics.csv"
        nondimensional_time, nondimensional_velocity = load_sampled_ring_speed(csv_path)
        if nondimensional_time.size == 0:
            print(f"  (no ring speed data for {variant})")
            continue
        label = VARIANT_LABEL[variant]
        print(f"  {variant}: {len(nondimensional_time)} samples")
        line_kwargs = {
            "linestyle": st["linestyle"],
            "color": st["color"],
            "lw": st["linewidth"],
            "marker": st["marker"],
            "mew": st["markeredgewidth"],
        }
        ax.plot(
            *with_sample_gaps(nondimensional_time, nondimensional_velocity),
            ms=st["markersize"],
            markevery=n_skip,
            label=label,
            **line_kwargs,
        )
        curves.append((nondimensional_time, nondimensional_velocity, line_kwargs))
        plotted_values.append(nondimensional_velocity)

    # -- Analytical Saffman solution in its thin-core range ------------------
    theory_end_physical = saffman_valid_time_limit()
    theory_end = theory_end_physical / REFERENCE_TIME
    t_phys = np.linspace(0.0, theory_end_physical, 300)
    saffman_nondimensional_velocity = saffman_speed(t_phys) / REFERENCE_VELOCITY
    saffman_t = t_phys / REFERENCE_TIME
    ax.plot(
        saffman_t,
        saffman_nondimensional_velocity,
        **reference_style(),
        zorder=5,
        label=r"Saffman--Archer",
    )
    plotted_values.append(saffman_nondimensional_velocity)

    relative_errors = []
    for nondimensional_time, nondimensional_velocity, line_kwargs in curves:
        valid = nondimensional_time <= theory_end
        reference = saffman_speed(nondimensional_time[valid] * REFERENCE_TIME) / REFERENCE_VELOCITY
        relative_error = (nondimensional_velocity[valid] - reference) / reference
        relative_errors.append(relative_error)
        comparison.plot(
            nondimensional_time[valid],
            100.0 * relative_error,
            markevery=n_skip,
            ms=2.0,
            **line_kwargs,
        )
    comparison.axhline(0.0, color="0.45", linewidth=0.8, linestyle=":")
    comparison.set_xlim(0.0, theory_end)
    comparison.set_xlabel(r"$t\,\Gamma_0/R_0^2$")
    comparison.set_ylabel(r"$(U_{\rm ring}-U_{\rm S})/U_{\rm S}$ [\%]")
    comparison.set_title(r"(b) Speed difference")
    if relative_errors:
        lower = min(0.0, 100.0 * min(float(np.min(error)) for error in relative_errors))
        upper = max(0.0, 100.0 * max(float(np.max(error)) for error in relative_errors))
        padding = max(0.5, 0.10 * (upper - lower))
        comparison.set_ylim(lower - padding, upper + padding)

    ax.set_title(r"(a) Self-induced speed")
    ax.set_xlabel(r"$t\,\Gamma_0/R_0^2$")
    ax.set_ylabel(r" $U_{\rm ring}/U_{\rm ref,0}$")
    if plotted_values:
        lower = min(float(np.min(values)) for values in plotted_values)
        upper = max(float(np.max(values)) for values in plotted_values)
        padding = 0.04 * (upper - lower)
        ax.set_ylim(lower - padding, upper + padding)
    if curves:
        ax.set_xlim(0.0, 1.01 * max(float(time[-1]) for time, _, _ in curves))
    fig.legend(
        *ax.get_legend_handles_labels(),
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        fontsize=10.95,
    )
    save_fig(fig, figs / "vortex_ring_motion.png", dpi=args.dpi, figure_format=args.format)


if __name__ == "__main__":
    main()
