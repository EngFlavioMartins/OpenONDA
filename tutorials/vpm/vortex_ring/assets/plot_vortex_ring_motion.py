#!/usr/bin/env python3
"""
Self-induced velocity U/U₀ vs t*.

Compares DNS and LES (transposed stretching) ring self-induced velocity
against the analytical Saffman model with Gaussian core diffusion.

Saves: figures/vortex_ring_motion.png
"""

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from plot_style import (
    VARIANT_LABEL,
    VARIANT_STYLE,
    build_arg_parser,
    figure_size,
    load_theme,
    reference_style,
    save_fig,
)
from ring_metrics import (
    FIGURES_DIR,
    SAMPLES_DIR,
    REFERENCE_TIME,
    REFERENCE_VELOCITY,
    load_sampled_ring_speed,
    saffman_speed,
)


def main() -> None:
    args = build_arg_parser("Self-induced velocity U/U₀ vs t*.").parse_args()
    figs = FIGURES_DIR
    figs.mkdir(parents=True, exist_ok=True)
    n_skip = 7  # plot every n-th marker
    plot_end = 185

    load_theme()

    fig, ax = plt.subplots(figsize=figure_size("single_tall"))
    fig.subplots_adjust(wspace=0.10, hspace=0.10, left=0.10, right=0.97, top=0.95, bottom=0.13)

    # -- Ring speed — all available variants ---------------------------------
    for variant, st in VARIANT_STYLE.items():
        csv_path = SAMPLES_DIR / variant / "ring_diagnostics.csv"
        nondimensional_time, nondimensional_velocity = load_sampled_ring_speed(csv_path)
        if nondimensional_time.size == 0:
            print(f"  (no ring speed data for {variant})")
            continue
        label = VARIANT_LABEL[variant]
        print(f"  {variant}: {len(nondimensional_time)} samples")
        ax.plot(
            nondimensional_time,
            nondimensional_velocity,
            linestyle=st["linestyle"],
            color=st["color"],
            lw=st["linewidth"],
            marker=st["marker"],
            ms=st["markersize"],
            markevery=n_skip,
            mew=st["markeredgewidth"],
            label=label,
        )

    # -- Analytical Saffman solution ------------------------------------------
    t_phys = np.linspace(0.0, plot_end * REFERENCE_TIME, 500)
    saffman_nondimensional_velocity = saffman_speed(t_phys) / REFERENCE_VELOCITY
    ax.plot(
        t_phys / REFERENCE_TIME,
        saffman_nondimensional_velocity,
        **reference_style(),
        zorder=5,
        label="Saffman (analytical)",
    )

    ax.set_xlabel(r"Normalized time, $t\,\Gamma / R_0^2$")
    ax.set_ylabel(r"Self-induced speed, $U_\Gamma / U_{\text{ref},0}$")
    ax.set_ylim(0.4, 1.0)
    ax.set_xlim(0, plot_end)
    ax.legend(ncol=1, loc="lower left")
    save_fig(fig, figs / "vortex_ring_motion.png", dpi=args.dpi, figure_format=args.format)


if __name__ == "__main__":
    main()
