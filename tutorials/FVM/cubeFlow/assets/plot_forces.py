#!/usr/bin/env python3
"""Plot Cd(t), Cl(t), and FFT of Cd(t) from forces_history.csv."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _common import build_arg_parser, load_forces_csv, save_fig, U_INF, L_REF, RE, NU


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    solution_dir = args.solution_dir
    figures_dir = args.figures_dir
    dpi = args.dpi

    data = load_forces_csv(solution_dir)
    if not data:
        print("  No force data to plot.")
        return

    for pname, d in data.items():
        t = d["time"]
        cd = d["Cd"]
        cl = d["Cl"]

        fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=False)

        ax = axes[0]
        ax.plot(t, cd, "b-", linewidth=0.8)
        cd_mean = np.mean(cd)
        ax.axhline(cd_mean, color="r", linestyle="--", linewidth=0.6)
        ax.text(
            0.97, 0.92, f"mean Cd = {cd_mean:.4f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8),
        )
        ax.set_ylabel("$C_d$")
        ax.set_title(f"Drag coefficient — {pname}")
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(t, cl, "g-", linewidth=0.8)
        ax.set_ylabel("$C_l$")
        ax.set_xlabel("Time [s]")
        ax.set_title(f"Lift coefficient — {pname}")
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        dt = t[1] - t[0] if len(t) > 1 else 0.1
        n = len(cd)
        cd_detrend = cd - np.mean(cd)
        fft_vals = np.fft.rfft(cd_detrend)
        fft_freq = np.fft.rfftfreq(n, d=dt)
        fft_mag = np.abs(fft_vals)
        mask = fft_freq > 0
        ax.plot(fft_freq[mask], fft_mag[mask], "purple", linewidth=0.8)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Magnitude")
        ax.set_title("FFT of $C_d$ (detrended)")
        ax.set_xlim(0, 2.0)
        ax.grid(True, alpha=0.3)

        if len(fft_mag[mask]) > 0:
            peak_idx = np.argmax(fft_mag[mask])
            if peak_idx < len(fft_freq[mask]):
                st = fft_freq[mask][peak_idx] * L_REF / U_INF
                ax.text(
                    0.97, 0.92, f"St = {st:.3f}  (f = {fft_freq[mask][peak_idx]:.3f} Hz)",
                    transform=ax.transAxes, ha="right", va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8),
                )

        plt.tight_layout()
        save_fig(fig, "cube_forces.png", figures_dir, dpi=dpi)


if __name__ == "__main__":
    main()
