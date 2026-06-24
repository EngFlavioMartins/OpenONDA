#!/usr/bin/env python3
"""Diagnostics for the two-wing wake-crossing Delta Wing tutorial.

Figures
-------
1. delta_wing_forces.png   — 2 panels: (top) lift on the front vs rear wing over
   time; (bottom) the two wings' plunge (z) trajectories, showing the rear wing
   crossing up/down through the front wing's wake.
2. delta_wing_circulation_history.png — total |Γ| carried by the VPM wake.

(The old particle-count figure was removed: particle count is governed by the
shedding cadence + wake-bounding adaptation and carries no physical insight.)
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re

import h5py
import matplotlib.pyplot as plt
import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

CASE_DIR = Path(__file__).resolve().parents[1]
SOLUTION_DIR = CASE_DIR / "solution" / "delta_wing"
FIGURES_DIR = CASE_DIR / "figures"
THEME_PATH = CASE_DIR.parents[2] / "docs" / "themes" / "matplotlib_setup.py"
FONT_PATH = CASE_DIR.parents[2] / "docs" / "themes" / "DejaVuSerif.ttf"


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


_COLORS, _theme = _load_theme()


def _step(path: Path) -> int:
    m = re.search(r"_(\d{6})\.h5$", path.name)
    return int(m.group(1)) if m else -1


# ----------------------------------------------------------------------------
# Figure 1: per-wing forces + plunge trajectories
# ----------------------------------------------------------------------------
def _wing_lift_history(samples_dir: Path, surface: str):
    """(time, lift) for one wing from its spanwise loading CSV."""
    if pd is None:
        return np.array([]), np.array([])
    csv = samples_dir / f"vlm_spanwise_{surface}.csv"
    if not csv.exists():
        return np.array([]), np.array([])
    df = pd.read_csv(csv)
    if "surface" in df.columns:
        df = df[df["surface"] == surface]
    col = "Fz_sec" if "Fz_sec" in df.columns else ("L_prime" if "L_prime" in df.columns else None)
    if col is None or "step" not in df.columns:
        return np.array([]), np.array([])
    rows = []
    for step, g in df.groupby("step"):
        t = float(g["time"].iloc[0]) if "time" in g else float(step)
        val = float((g[col] * g["dy"]).sum()) if col == "L_prime" and "dy" in g else float(g[col].sum())
        rows.append((t, val))
    rows.sort()
    a = np.asarray(rows)
    return (a[:, 0], a[:, 1]) if a.size else (np.array([]), np.array([]))


def plot_forces(solution_dir: Path, figures_dir: Path) -> None:
    samples = solution_dir / "samples"
    meta_path = solution_dir / "motion_params.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    fig, (ax_f, ax_z) = plt.subplots(2, 1, figsize=(12.8 / 2.54, 10.0 / 2.54), sharex=True)

    # Top: per-wing lift
    plotted = False
    c_front = _COLORS.get("TUDcyan", "#0E8A85")
    c_rear = _COLORS.get("AccentRed", "#9C2F50")
    for surf, color, lbl in [("front_wing", c_front, "Front wing"),
                             ("rear_wing", c_rear, "Rear wing")]:
        t, lift = _wing_lift_history(samples, surf)
        if t.size:
            ax_f.plot(t, lift, "-", color=color, lw=1.3, label=lbl)
            plotted = True
    ax_f.axhline(0, color="grey", lw=0.5, alpha=0.5)
    ax_f.set_ylabel("Lift [N]")
    ax_f.set_title("Forces on front vs rear delta wing")
    if plotted:
        ax_f.legend()

    # Bottom: plunge trajectories z(t) = A(1 - cos(ωt + φ))
    if meta:
        A = meta["A"]; omega = meta["omega"]; dt = meta["dt"]; n = meta["num_steps"]
        t = np.arange(n) * dt
        for surf, color, lbl in [("front_wing", c_front, "Front wing"),
                                 ("rear_wing", c_rear, "Rear wing")]:
            phase = meta.get("wings", {}).get(surf, 0.0)
            z = A * (1.0 - np.cos(omega * t + phase))
            ax_z.plot(t, z, "-", color=color, lw=1.3, label=f"{lbl} $z(t)$")
        ax_z.legend()
    ax_z.set_xlabel("Time [s]")
    ax_z.set_ylabel("Plunge position $z$ [m]")
    fig.tight_layout()
    out = figures_dir / "delta_wing_forces.png"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ----------------------------------------------------------------------------
# Figure 2: total wake circulation history
# ----------------------------------------------------------------------------
def plot_circulation(solution_dir: Path, figures_dir: Path, pattern: str) -> None:
    files = sorted(solution_dir.glob(pattern), key=_step)
    if not files:
        files = sorted(solution_dir.glob("vpm_*.h5"), key=_step)
    if not files:
        print("  [WARNING] no backups for circulation history.")
        return
    times, circ_l1 = [], []
    for f in files:
        with h5py.File(f, "r") as h:
            times.append(float(h["solver"].attrs.get("flow_time", 0.0)))
            if "particles" in h and "circulation" in h["particles"]:
                c = h["particles"]["circulation"][:]
                circ_l1.append(float(np.linalg.norm(c, axis=1).sum()))
            else:
                circ_l1.append(0.0)
    fig, ax = plt.subplots(figsize=(12.8 / 2.54, 7.0 / 2.54))
    ax.plot(times, circ_l1, "-o", ms=3, lw=1.2)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$\sum |\Gamma|$ [m$^2$/s]")  # fixed: single-backslash mathtext
    ax.set_title("Delta wing: wake circulation magnitude history")
    fig.tight_layout()
    out = figures_dir / "delta_wing_circulation_history.png"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
