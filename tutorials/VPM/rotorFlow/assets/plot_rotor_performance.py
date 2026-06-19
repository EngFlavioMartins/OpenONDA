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

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Theme loading (same mechanism as lambOseenVortex) ────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
TUTORIAL_DIR = SCRIPT_DIR.parent
THEME_PATH = SCRIPT_DIR.parents[2] / "docs" / "themes" / "matplotlib_setup.py"
FONT_PATH = SCRIPT_DIR.parents[2] / "docs" / "themes" / "DejaVuSerif.ttf"


def _load_theme() -> tuple[dict[str, str], object | None]:
    import matplotlib.font_manager as fm

    theme = None
    if THEME_PATH.exists():
        spec = importlib.util.spec_from_file_location("mpl_setup", THEME_PATH)
        theme = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(theme)
        try:
            theme.set_style()
        except Exception:
            pass

    if FONT_PATH.exists():
        fm.fontManager.addfont(str(FONT_PATH))
        plt.rcParams["font.family"] = "DejaVu Serif"

    if theme is not None and hasattr(theme, "COLORS"):
        return dict(theme.COLORS), theme
    return {}, theme


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
    # Fx is negative for thrust (pointing upstream); take absolute value
    ct = (-df["Fx"].to_numpy()) / qA
    cp = (df["Mx"].to_numpy() * omega) / qAU
    time = df["time"].to_numpy()
    rotations = time * omega / (2.0 * np.pi)

    colors, _ = _load_theme()

    fig, axes = plt.subplots(2, 1, figsize=(12.8 / 2.54, 14.0 / 2.54))
    fig.subplots_adjust(hspace=0.35, top=0.95, bottom=0.12, left=0.14, right=0.86)

    # ── Subplot 1: Ct & Cp vs time ─────────────────────────────────────────
    ax1 = axes[0]
    color_ct = colors.get("hybrid", "#DB5400")
    color_cp = colors.get("dvhr", "#1079A9")

    ax1.plot(rotations, ct, color=color_ct, lw=1.0, label=r"$C_t$")
    ax1.plot(rotations, cp, color=color_cp, lw=1.0, label=r"$C_p$")

    # Betz-limit references
    ct_betz = 8.0 / 9.0
    cp_betz = 16.0 / 27.0
    ax1.axhline(ct_betz, color=color_ct, ls="--", lw=0.8, alpha=0.6)
    ax1.axhline(cp_betz, color=color_cp, ls="--", lw=0.8, alpha=0.6)
    ax1.text(
        rotations[-1] * 1.01,
        ct_betz,
        r" $C_{t}^{\mathrm{Betz}}$",
        color=color_ct,
        va="center",
        fontsize=8,
    )
    ax1.text(
        rotations[-1] * 1.01,
        cp_betz,
        r" $C_{p}^{\mathrm{Betz}}$",
        color=color_cp,
        va="center",
        fontsize=8,
    )

    ax1.set_xlabel(r"Rotation")
    ax1.set_ylabel(r"Coefficient")
    ax1.set_xlim([0, rotations[-1] * 1.08])
    ax1.set_ylim([0, 1.1])
    ax1.legend(loc="upper right", ncol=2)
    ax1.set_title("Rotor thrust and power coefficients")

    # ── Subplot 2: Cp vs Ct with theory envelope ───────────────────────────
    ax2 = axes[1]
    ct_theory = np.linspace(0.0, 1.0, 300)
    cp_theory = actuator_disk_cp(ct_theory)

    ax2.plot(ct_theory, cp_theory, "k-", lw=1.2, zorder=0, label="Actuator-disk theory")
    ax2.plot(ct, cp, color=color_ct, lw=1.0, zorder=1, label="Simulation")

    # Betz point
    ax2.scatter([ct_betz], [cp_betz], color="k", marker="*", s=80, zorder=2, label="Betz limit")

    ax2.set_xlabel(r"$C_t$")
    ax2.set_ylabel(r"$C_p$")
    ax2.set_xlim([0, 1.0])
    ax2.set_ylim([0, 0.7])
    ax2.legend(loc="lower right")
    ax2.set_title(r"$C_p$ vs $C_t$ — momentum theory envelope")

    # Save
    out = Path(args.figures_dir) / f"rotor_performance.{args.format}"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_kw: dict = {"bbox_inches": "tight"}
    if args.format == "png":
        save_kw["dpi"] = args.dpi
    plt.savefig(out, **save_kw)
    plt.close(fig)
    print(f"  Saved: {out}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Rotor performance Ct/Cp plotting.")
    p.add_argument(
        "--solution-dir", default=str(TUTORIAL_DIR / "solution"), help="Solution directory."
    )
    p.add_argument("--figures-dir", default=str(TUTORIAL_DIR / "figures"), help="Output directory.")
    p.add_argument("--format", choices=["png", "svg"], default="png")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--rho", type=float, default=1.225, help="Fluid density [kg/m³].")
    p.add_argument(
        "--freestream-velocity", type=float, default=7.0, help="Freestream velocity [m/s]."
    )
    p.add_argument("--rotor-radius", type=float, default=6.0, help="Rotor radius [m].")
    p.add_argument("--tip-speed-ratio", type=float, default=7.0, help="Tip-speed ratio TSR.")
    return plot_rotor_performance(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
