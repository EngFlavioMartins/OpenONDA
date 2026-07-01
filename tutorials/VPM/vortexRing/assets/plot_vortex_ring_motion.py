#!/usr/bin/env python3
"""
Self-induced velocity U/U₀ vs t*.

Compares DNS and LES (transposed stretching) ring self-induced velocity
against the analytical Saffman model with Gaussian core diffusion.

Saves: figures/vortex_ring_motion.png
"""

import sys
import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    build_arg_parser,
    load_theme,
    load_ring_speed,
    saffman_speed,
    save_fig,
    T_REF,
    U_REF,
    VARIANT_LABEL,
    VARIANT_STYLE,
    CM,
)


def main() -> None:
    args = build_arg_parser("Self-induced velocity U/U₀ vs t*.").parse_args()
    sol = Path(args.solution_dir)
    figs = Path(args.figures_dir)
    figs.mkdir(parents=True, exist_ok=True)

    load_theme()

    fig, ax = plt.subplots(figsize=(12.8 * CM, 8 * CM))

    # -- Ring speed — all available variants ---------------------------------
    for variant, st in VARIANT_STYLE.items():
        h5_files = sorted(glob.glob(str(sol / variant / f"vpm_{variant}_*.h5")))
        if not h5_files:
            continue
        t_star, U_norm = load_ring_speed(h5_files)
        if t_star.size == 0:
            print(f"  (no ring speed data for {variant})")
            continue
        ls = "--" if variant.startswith("DNS") else "-"
        label = VARIANT_LABEL[variant]
        print(f"  {variant}: {len(h5_files)} files")
        ax.plot(
            t_star,
            U_norm,
            ls,
            color=st["color"],
            lw=1.1,
            marker=st["marker"],
            ms=3,
            markevery=3,
            mew=0.4,
            label=label,
        )

    # -- Analytical Saffman solution ------------------------------------------
    t_phys = np.linspace(0.0, 38 * T_REF, 500)
    U_saffman = saffman_speed(t_phys) / U_REF
    ax.plot(
        t_phys / T_REF,
        U_saffman,
        "--",
        color="gray",
        lw=1.0,
        zorder=5,
        label="Saffman (analytical)",
    )

    ax.set_xlabel(r"Normalized time, $t\,\Gamma / R_0^2$")
    ax.set_ylabel(r"Self-induced speed, $U_\Gamma / U_{\text{ref},0}$")
    ax.set_ylim(0.6, 1.)
    ax.set_xlim(0, 38)
    ax.legend(fontsize=10, ncol=1, loc='lower left')
    save_fig(fig, figs / "vortex_ring_motion.png", dpi=args.dpi)


if __name__ == "__main__":
    main()
