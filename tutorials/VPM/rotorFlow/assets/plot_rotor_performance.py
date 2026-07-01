#!/usr/bin/env python3
"""Rotor performance — Ct / Cp time history and actuator-disk theory comparison.

Reads ``solution/samples/vlm_forces.csv`` and produces two subplots:

1. Thrust coefficient (Ct) and power coefficient (Cp) versus time,
   with Betz-limit reference lines.
2. Cp–Ct operating trajectory overlaid on the actuator-disk momentum
   theory envelope ``Cp = 0.5·Ct·(1 + sqrt(1-Ct))``.

Saves: ``figures/rotor_performance.png``
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _common import CM, build_arg_parser, load_theme, rotor_styles, save_figure


# ── Physics helpers ──────────────────────────────────────────────────────────


def actuator_disk_cp(ct: np.ndarray) -> np.ndarray:
    """Actuator-disk momentum theory: Cp as a function of Ct."""
    ct = np.asarray(ct)
    cp = np.zeros_like(ct, dtype=float)
    mask = (ct >= 0) & (ct <= 1.0)
    cp[mask] = 0.5 * ct[mask] * (1.0 + np.sqrt(1.0 - ct[mask]))
    return cp


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_rotor_performance(args) -> int:
    solution_dir = Path(args.solution_dir)
    csv_path = solution_dir / "samples" / "vlm_forces.csv"
    if not csv_path.exists():
        print(f"[rotor] CSV not found: {csv_path}")
        return 1

    df = pd.read_csv(csv_path)
    if df.empty:
        print("[rotor] CSV is empty.")
        return 1

    # Physical constants
    rho = args.rho
    U = args.freestream_velocity
    R = args.rotor_radius
    omega = args.tip_speed_ratio * U / R
    A = np.pi * R**2
    q = 0.5 * rho * U**2
    qA = q * A
    qAU = q * A * U

    # Coefficients
    # U_inf is +x and omega_vec is -omega*xhat.
    ct = df["Fx"].to_numpy() / qA
    cp = (-df["Mx"].to_numpy() * omega) / qAU
    time = df["time"].to_numpy()
    rotations = time * omega / (2.0 * np.pi)

    colors, _ = load_theme()
    styles = rotor_styles(colors)

    fig, axes = plt.subplots(2, 1, figsize=(12.8 * CM, 12.8 * CM))
    fig.subplots_adjust(hspace=0.34, top=0.95, bottom=0.13, left=0.14, right=0.96)

    # ── Subplot 1: Ct & Cp vs time ─────────────────────────────────────────
    ax1 = axes[0]
    color_ct = colors.get("hybrid", "#772953")
    color_cp = colors.get("dvhr", "#0E8A85")

    ax1.plot(rotations, ct, color=color_ct, lw=1.0, label=r"$C_T$")
    ax1.plot(rotations, cp, color=color_cp, lw=1.0, label=r"$C_P$")

    # Betz-limit references
    ct_betz = 8.0 / 9.0
    cp_betz = 16.0 / 27.0
    ax1.axhline(
        ct_betz,
        color=color_ct,
        ls="--",
        lw=0.8,
        alpha=0.75,
        label=r"$C_T$ Betz",
    )
    ax1.axhline(
        cp_betz,
        color=color_cp,
        ls="--",
        lw=0.8,
        alpha=0.75,
        label=r"$C_P$ Betz",
    )

    ax1.set_xlabel(r"Rotor rotations")
    ax1.set_ylabel(r"Coefficient")
    ax1.set_xlim([0, rotations[-1]])
    ax1.set_ylim([0, 1.1])
    ax1.legend(loc="upper right", ncol=2, handlelength=2.2, columnspacing=1.0)
    ax1.set_title(r"Rotor performance coefficients")

    # ── Subplot 2: Cp vs Ct with theory envelope ───────────────────────────
    ax2 = axes[1]
    ct_theory = np.linspace(0.0, 1.0, 300)
    cp_theory = actuator_disk_cp(ct_theory)

    ax2.plot(
        ct_theory,
        cp_theory,
        label=r"Actuator-disk theory",
        zorder=0,
        **styles["theory"],
    )
    ax2.plot(
        ct,
        cp,
        color=styles["vpm"]["color"],
        lw=1.0,
        marker=styles["vpm"]["marker"],
        markersize=styles["vpm"]["markersize"],
        markevery=max(1, len(ct) // 24),
        zorder=1,
        label="VLM-VPM",
    )

    # Betz point
    ax2.scatter(
        [ct_betz],
        [cp_betz],
        color=colors.get("DarkText", "#2E3D46"),
        marker="*",
        s=36,
        zorder=2,
        label="Betz limit",
    )

    ax2.set_xlabel(r"$C_T$")
    ax2.set_ylabel(r"$C_P$")
    ax2.set_xlim([0, 1.0])
    ax2.set_ylim([0, max(0.7, float(np.nanmax(cp)) * 1.08, cp_betz * 1.08)])
    ax2.legend(loc="lower right")
    ax2.set_title(r"Operating trajectory")

    # Save
    out = Path(args.figures_dir) / f"rotor_performance.{args.format}"
    save_figure(fig, out, args.dpi, args.format)
    plt.close(fig)
    print(f"  Saved: {out}")
    return 0


def main() -> int:
    p = build_arg_parser("Rotor performance Ct/Cp plotting.")
    p.add_argument(
        "--rho",
        type=float,
        default=1.225,
        help="Fluid density [kg/m^3].",
    )
    p.add_argument("--freestream-velocity", type=float, default=7.0)
    p.add_argument("--rotor-radius", type=float, default=6.0, help="Rotor radius [m].")
    p.add_argument("--tip-speed-ratio", type=float, default=7.0, help="Tip-speed ratio TSR.")
    return plot_rotor_performance(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
