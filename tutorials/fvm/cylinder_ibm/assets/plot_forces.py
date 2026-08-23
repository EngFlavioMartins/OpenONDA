#!/usr/bin/env python3
"""Plot drag/lift coefficients, marker slip error, and the Strouhal number
from samples/ibm_forces_history.csv, with reference bands from
Constant et al. 2017."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _common import (  # noqa: E402
    COLORS,
    REFERENCES,
    build_arg_parser,
    figure_size,
    load_ibm_forces_csv,
    save_fig,
)


def strouhal_from_lift(t, cl):
    """Dominant lift frequency (Hz) from the second half of the signal."""
    n = len(t)
    if n < 32:
        return None
    t2, cl2 = t[n // 2 :], cl[n // 2 :]
    if np.ptp(cl2) < 1e-6:
        return None
    tu = np.linspace(t2[0], t2[-1], len(t2))
    clu = np.interp(tu, t2, cl2)
    clu -= clu.mean()
    freqs = np.fft.rfftfreq(len(tu), tu[1] - tu[0])
    amp = np.abs(np.fft.rfft(clu))
    if amp[1:].max() < 1e-8:
        return None
    return float(freqs[1:][np.argmax(amp[1:])])


def main():
    args = build_arg_parser().parse_args()
    ref = REFERENCES.get(args.Re, {})
    data = load_ibm_forces_csv(args.solution_dir)
    if not data:
        print("  No IBM force data to plot.")
        return

    for name, d in data.items():
        t = d["time"]
        drag_coefficient = d["drag_coefficient"]
        lift_coefficient = d["lift_coefficient"]
        slip_error = d["slip_error"]
        # Statistics over the settled part (last third).
        i0 = 2 * len(t) // 3
        drag_coefficient_mean = float(np.mean(drag_coefficient[i0:]))
        lift_coefficient_rms = float(
            np.sqrt(np.mean((lift_coefficient[i0:] - np.mean(lift_coefficient[i0:])) ** 2))
        )

        fig, axes = plt.subplots(3, 1, figsize=figure_size("stacked"), sharex=True)

        ax = axes[0]
        ax.plot(t, drag_coefficient, color=COLORS["TUDdark"], linewidth=0.9)
        if "drag_coefficient" in ref:
            ax.axhspan(
                *ref["drag_coefficient"],
                color=COLORS["reference"],
                alpha=0.25,
                label=f"Constant et al.: {ref['drag_coefficient'][0]:.2f}-{ref['drag_coefficient'][1]:.2f}",
            )
            ax.legend(loc="upper right", fontsize=8)
        ax.set_ylabel("$C_d$")
        ax.set_title(f"IBM cylinder forces — {name} (Re = {args.Re:g})")
        ax.text(
            0.02,
            0.06,
            f"mean drag coefficient (last 1/3) = {drag_coefficient_mean:.4f}",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(t, lift_coefficient, color=COLORS["TUDdark"], linewidth=0.9)
        strouhal_number = strouhal_from_lift(t, lift_coefficient)
        label = f"lift coefficient rms = {lift_coefficient_rms:.4f}"
        if strouhal_number is not None and "strouhal_number" in ref:
            label += f",  strouhal_number = {strouhal_number:.4f} (ref {ref['strouhal_number'][0]:.3f}-{ref['strouhal_number'][1]:.3f})"
        ax.text(0.02, 0.06, label, transform=ax.transAxes, fontsize=8)
        ax.set_ylabel("$C_l$")
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        ax.semilogy(t, np.maximum(slip_error, 1e-16), color=COLORS["TUDdark"], linewidth=0.9)
        ax.set_ylabel(r"marker slip error $\max_s |u(X_s)|$")
        ax.set_xlabel("t [s]")
        ax.grid(True, alpha=0.3, which="both")

        fig.tight_layout()
        save_fig(
            fig, f"forces_{name}.png", args.figures_dir, dpi=args.dpi, figure_format=args.format
        )

        print(
            f"  {name}: mean drag_coefficient = {drag_coefficient_mean:.4f}",
            end="",
        )
        if "drag_coefficient" in ref:
            lo, hi = ref["drag_coefficient"]
            status = "OK" if lo * 0.95 <= drag_coefficient_mean <= hi * 1.05 else "OUT OF BAND"
            print(f"  [reference {lo:.2f}-{hi:.2f}: {status}]", end="")
        if strouhal_number is not None:
            print(f", strouhal_number = {strouhal_number:.4f}", end="")
        print(f", final slip_error = {slip_error[-1]:.2e}")


if __name__ == "__main__":
    main()
