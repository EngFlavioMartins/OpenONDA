#!/usr/bin/env python3
"""Rotor performance — Ct / Cp time history and actuator-disk theory comparison.

Reads ``samples/rotor/vlm_forces.csv`` and produces two subplots:

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
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    build_arg_parser,
    build_rotor_style_map,
    DENSITY,
    FIGURES_DIR,
    FREESTREAM_VELOCITY,
    load_theme,
    ROTOR_RADIUS,
    SAMPLES_DIR,
    TIP_SPEED_RATIO,
)

# ==============================================================================
# Physics helpers
# ==============================================================================


def actuator_disk_cp(ct: np.ndarray) -> np.ndarray:
    """Actuator-disk momentum theory: Cp as a function of Ct."""
    ct = np.asarray(ct)
    cp = np.zeros_like(ct, dtype=float)
    mask = (ct >= 0) & (ct <= 1.0)
    cp[mask] = 0.5 * ct[mask] * (1.0 + np.sqrt(1.0 - ct[mask]))
    return cp


# ==============================================================================
# Plot
# ==============================================================================


def plot_rotor_performance(args) -> int:
    fmt = getattr(args, "format", "png")
    out = FIGURES_DIR / f"rotor_performance.{fmt}"
    out.parent.mkdir(parents=True, exist_ok=True)

    csv_path = SAMPLES_DIR / "vlm_forces.csv"
    if not csv_path.exists():
        print(f"[rotor] CSV not found: {csv_path}")
        return 1

    df = pd.read_csv(csv_path)
    if df.empty:
        print("[rotor] CSV is empty.")
        return 1

    # Physical constants
    rho = DENSITY
    U = FREESTREAM_VELOCITY
    R = ROTOR_RADIUS
    omega = TIP_SPEED_RATIO * U / R
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
    styles = build_rotor_style_map(colors)
    s_ref = styles["reference"]
    ct_betz = 8.0 / 9.0
    cp_betz = 16.0 / 27.0

    fig, axes = plt.subplots(1, 2, figsize=(12.8 / 2.54, 6.0 / 2.54))
    fig.subplots_adjust(wspace=0.23, top=0.95, bottom=0.13, left=0.14, right=0.96)

    # -- Ct & Cp vs time -------------------------------------------------
    ax1 = axes[0]
    s_ct = styles["ct"]
    s_cp = styles["cp"]

    ax1.plot(
        rotations,
        ct,
        color=s_ct["color"],
        marker=s_ct["marker"],
        markersize=s_ct["markersize"],
        lw=s_ct["linewidth"],
        label=s_ct["label"],
        markevery=5,
    )
    ax1.plot(
        rotations,
        cp,
        color=s_cp["color"],
        marker=s_cp["marker"],
        markersize=s_cp["markersize"],
        lw=s_cp["linewidth"],
        label=s_cp["label"],
        markevery=5,
    )

    ax1.axhline(ct_betz, color=s_ref["color"], ls=s_ref["linestyle"], lw=s_ref["linewidth"])
    ax1.axhline(cp_betz, color=s_ref["color"], ls=s_ref["linestyle"], lw=s_ref["linewidth"])

    ax1.text(
        7, 1.01 * ct_betz, r"$C_{T,\mathrm{Betz}}$", color=s_ref["color"], va="bottom", ha="left"
    )
    ax1.text(
        7, 1.01 * cp_betz, r"$C_{P,\mathrm{Betz}}$", color=s_ref["color"], va="bottom", ha="left"
    )

    ax1.set_xlabel(r"Revolutions")
    ax1.set_ylabel(r"Coefficient")
    ax1.set_xlim([0, rotations[-1]])
    ax1.set_ylim([0, 1.1])
    ax1.legend(loc="lower center", ncol=2)
    ax1.set_title(r"Rotor performance coefficients")

    # -- Cp vs Ct with theory envelope -----------------------------------
    ax2 = axes[1]
    ct_theory = np.linspace(0.0, 1.0, 300)
    cp_theory = actuator_disk_cp(ct_theory)

    ax2.plot(
        ct_theory,
        cp_theory,
        zorder=0,
        **styles["theory"],
    )

    trajectory_color = styles["vpm"]["color"]
    if len(ct) > 1:
        points = np.column_stack([ct, cp]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        segment_alpha = np.linspace(0.05, 1.0, len(ct))[1:]
        segment_colors = [to_rgba(trajectory_color, alpha=a) for a in segment_alpha]
        trajectory = LineCollection(
            segments,
            colors=segment_colors,
            linewidths=1.0,
            zorder=1,
        )
        ax2.add_collection(trajectory)
    else:
        segment_alpha = np.array([1.0])

    markevery = max(1, len(ct) // 24)
    marker_idx = np.arange(0, len(ct), markevery)
    if marker_idx[-1] != len(ct) - 1:
        marker_idx = np.append(marker_idx, len(ct) - 1)
    marker_alpha = np.linspace(0.0, 1.0, len(ct))[marker_idx]
    marker_colors = [to_rgba(trajectory_color, alpha=a) for a in marker_alpha]
    ax2.scatter(
        ct[marker_idx],
        cp[marker_idx],
        color=marker_colors,
        marker="s",
        s=styles["vpm"]["markersize"] ** 2,
        zorder=2,
    )

    ax2.scatter(
        [ct_betz],
        [cp_betz],
        color=s_ref["color"],
        marker="*",
        s=24,
        zorder=2,
    )

    ax2.set_xlabel(r"$C_T$")
    ax2.set_ylabel(r"$C_P$")
    ax2.set_xlim([0, 1.0])
    ax2.set_ylim([0, max(0.7, float(np.nanmax(cp)) * 1.08, cp_betz * 1.08)])

    from matplotlib.lines import Line2D

    ax2.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=trajectory_color,
                lw=1.0,
                marker="s",
                markersize=3,
                label="VLM-VPM time trajectory",
            ),
            Line2D([0], [0], color=s_ref["color"], ls="--", lw=1.0, label="Actuator-disk theory"),
            Line2D([0], [0], color=s_ref["color"], marker="*", ms=6, lw=0, label="Betz limit"),
        ],
        loc="lower center",
    )
    ax2.set_title(r"Operating trajectory")

    save_kw: dict = {"bbox_inches": "tight"}
    if fmt == "png":
        save_kw["dpi"] = args.dpi
    plt.savefig(out, **save_kw)
    plt.close(fig)
    print(f"  Saved: {out}")
    return 0


def main() -> int:
    return plot_rotor_performance(
        build_arg_parser("Rotor performance Ct/Cp plotting.").parse_args()
    )


if __name__ == "__main__":
    raise SystemExit(main())
