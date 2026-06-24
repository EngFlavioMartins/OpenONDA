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
    VARIANT_STYLE,
    CM,
)


def main() -> None:
    args = build_arg_parser("Self-induced velocity U/U₀ vs t*.").parse_args()
    sol = Path(args.solution_dir)
    figs = Path(args.figures_dir)
    figs.mkdir(parents=True, exist_ok=True)

    load_theme()

    dns_st = VARIANT_STYLE["DNS_transposed"]
    les_st = VARIANT_STYLE["LES_rvpm"]

    dns_files = sorted(glob.glob(str(sol / "DNS_transposed" / "vpm_DNS_transposed_*.h5")))
    les_files = sorted(glob.glob(str(sol / "LES_rvpm" / "vpm_LES_rvpm_*.h5")))
    print(f"DNS: {len(dns_files)} files,  LES (rVPM): {len(les_files)} files")

    fig, ax = plt.subplots(figsize=(12 * CM, 7 * CM))

    # ── Numerical results ────────────────────────────────────────────────────
    for h5_files, st, ls, label in [
        (dns_files, dns_st, "--", "DNS"),
        (les_files, les_st, "-", "In-house LES"),
    ]:
        t_star, U_norm = load_ring_speed(h5_files)
        if t_star.size == 0:
            print(f"  (no data for {label})")
            continue
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

    # ── Analytical Saffman solution ──────────────────────────────────────────
    t_phys = np.linspace(0.0, 38 * T_REF, 500)
    U_saffman = saffman_speed(t_phys) / U_REF
    ax.plot(
        t_phys / T_REF,
        U_saffman,
        ":",
        color="black",
        lw=1.1,
        zorder=5,
        label="Saffman (analytical)",
    )

    ax.set_xlabel(r"Normalized time, $t\,\Gamma / R_0^2$")
    ax.set_ylabel(r"Self-induced speed, $U_\Gamma / U_{\Gamma,0}$")
    ax.set_ylim(0.5, 1.05)
    ax.set_xlim(0, 38)
    ax.legend(fontsize=10, ncol=1)
    save_fig(fig, figs / "vortex_ring_motion.png", dpi=args.dpi)


if __name__ == "__main__":
    main()
