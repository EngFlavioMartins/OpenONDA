#!/usr/bin/env python3
"""Skin-friction distribution along the plate vs Cf = 0.664 / sqrt(Re_x)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _common import (  # noqa: E402
    COLORS,
    FIGURES_DIR,
    SOLUTION_DIR,
    build_arg_parser,
    figure_size,
    load_csv_columns,
    save_fig,
)


def main():
    args = build_arg_parser().parse_args()
    data = load_csv_columns(Path(SOLUTION_DIR) / "cf.csv")
    if not data:
        return

    x = data["position_x"]
    cf = data["skin_friction_coefficient"]
    cf_ref = data["skin_friction_coefficient_blasius"]

    fig, ax = plt.subplots(figsize=figure_size("single"))
    ax.plot(
        x, cf_ref, color=COLORS["reference"], linewidth=1.4, label=r"Blasius $0.664/\sqrt{Re_x}$"
    )
    ax.plot(x, cf, "o", color=COLORS["TUDdark"], markersize=3, linestyle="none", label="FVM")
    ax.set_xlabel("x / L")
    ax.set_ylabel("$C_f$")
    ax.set_title(f"Skin friction (Re$_L$ = {args.Re:g})")
    ax.set_ylim(0, min(0.06, 1.5 * float(np.max(cf_ref))))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    save_fig(fig, "skin_friction.png", FIGURES_DIR, dpi=args.dpi, figure_format=args.format)

    # Deviation away from the leading edge (the LE cell itself is singular).
    sel = (x > 0.2) & (x < 0.95)
    if sel.any():
        dev = np.abs(cf[sel] / cf_ref[sel] - 1.0)
        print(
            f"  Cf deviation from Blasius on 0.2 < x/L < 0.95: "
            f"mean {100 * dev.mean():.1f}%, max {100 * dev.max():.1f}%"
            f"  [{'OK' if dev.mean() < 0.05 else 'OUT OF BAND'} — target mean < 5%]"
        )


if __name__ == "__main__":
    main()
