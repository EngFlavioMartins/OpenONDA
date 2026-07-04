#!/usr/bin/env python3
"""Plot Cd(t), Cl(t), marker no-slip error, and (unsteady) the Strouhal number
from solution/ibm_forces_history.csv, with reference bands from
Constant et al. 2017."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _common import (  # noqa: E402
    COLORS, REFERENCES, build_arg_parser, figure_size, load_ibm_forces_csv, save_fig,
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
        t, cd, cl, slip = d["time"], d["Cd"], d["Cl"], d["slip"]
        # Statistics over the settled part (last third).
        i0 = 2 * len(t) // 3
        cd_mean = float(np.mean(cd[i0:]))
        cl_rms = float(np.sqrt(np.mean((cl[i0:] - np.mean(cl[i0:])) ** 2)))

        fig, axes = plt.subplots(3, 1, figsize=figure_size("stacked"), sharex=True)

        ax = axes[0]
        ax.plot(t, cd, color=COLORS["TUDdark"], linewidth=0.9)
        if "Cd" in ref:
            ax.axhspan(*ref["Cd"], color=COLORS["reference"], alpha=0.25,
                       label=f"Constant et al.: {ref['Cd'][0]:.2f}-{ref['Cd'][1]:.2f}")
            ax.legend(loc="upper right", fontsize=8)
        ax.set_ylabel("$C_d$")
        ax.set_title(f"IBM cylinder forces — {name} (Re = {args.Re:g})")
        ax.text(0.02, 0.06, f"mean $C_d$ (last 1/3) = {cd_mean:.4f}",
                transform=ax.transAxes, fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(t, cl, color=COLORS["TUDdark"], linewidth=0.9)
        st = strouhal_from_lift(t, cl)
        label = f"$C_l$ rms = {cl_rms:.4f}"
        if st is not None and "St" in ref:
            label += f",  St = {st:.4f} (ref {ref['St'][0]:.3f}-{ref['St'][1]:.3f})"
        ax.text(0.02, 0.06, label, transform=ax.transAxes, fontsize=8)
        ax.set_ylabel("$C_l$")
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        ax.semilogy(t, np.maximum(slip, 1e-16), color=COLORS["TUDdark"], linewidth=0.9)
        ax.set_ylabel(r"marker slip $\max_s |u(X_s)|$")
        ax.set_xlabel("t [s]")
        ax.grid(True, alpha=0.3, which="both")

        fig.tight_layout()
        save_fig(fig, f"forces_{name}.png", args.figures_dir, dpi=args.dpi)

        print(f"  {name}: mean Cd = {cd_mean:.4f}", end="")
        if "Cd" in ref:
            lo, hi = ref["Cd"]
            status = "OK" if lo * 0.95 <= cd_mean <= hi * 1.05 else "OUT OF BAND"
            print(f"  [reference {lo:.2f}-{hi:.2f}: {status}]", end="")
        if st is not None:
            print(f", St = {st:.4f}", end="")
        print(f", final slip = {slip[-1]:.2e}")


if __name__ == "__main__":
    main()
