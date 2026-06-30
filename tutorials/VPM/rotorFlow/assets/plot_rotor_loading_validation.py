#!/usr/bin/env python3
"""
Spanwise loading validation — VLM bound circulation vs BEM prediction.
======================================================================
Reads ``solution/rotor/samples/vlm_spanwise_blade_0.csv`` produced by the
loading-distribution sampler and compares the time-averaged bound circulation
Γ(r/R) against the BEM prediction for the same Betz-optimal blade geometry
(TSR = 7, three blades, Selig–Betz twist from hub to tip).

A second panel shows the spanwise sectional lift coefficient Cl(r/R), which
provides an independent check that the VLM is operating at the design angle of
attack (~5° once the wake induction has fully developed).

Saves: ``figures/rotor_loading_validation.png``
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

SCRIPT_DIR = Path(__file__).resolve().parent
TUTORIAL_DIR = SCRIPT_DIR.parent
THEME_PATH = TUTORIAL_DIR.parents[2] / "docs" / "themes" / "matplotlib_setup.py"
FONT_PATH = TUTORIAL_DIR.parents[2] / "docs" / "themes" / "DejaVuSerif.ttf"


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


def _read_vlm_spanwise(
    samples_dir: Path,
    surface: str,
    tail_fraction: float,
    hub_r: float,
    tip_r: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return (r, Gamma, cl) averaged over the tail of the simulation.

    Uses the ``orig`` half only (not the mirror, which doesn't exist for a
    rotor blade without symmetry) and restricts radial positions to the
    physical blade span [hub_r, tip_r].
    """
    csv_path = samples_dir / f"vlm_spanwise_{surface}.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    if df.empty:
        return None

    if "half" in df.columns:
        df = df[df["half"] == "orig"]

    t_max = df["time"].max()
    t_cut = t_max * (1.0 - tail_fraction)
    tail = df[df["time"] >= t_cut]

    gp = tail.groupby("y")[["Gamma", "cl"]].mean().reset_index()
    r = gp["y"].to_numpy()
    mask = (r >= hub_r - 0.1) & (r <= tip_r + 0.1)
    r = r[mask]
    gamma = gp["Gamma"].to_numpy()[mask]
    cl = gp["cl"].to_numpy()[mask]
    return r, gamma, cl


def main() -> int:
    ap = argparse.ArgumentParser(description="Spanwise loading validation: VLM vs BEM")
    ap.add_argument(
        "--solution-dir",
        default=str(TUTORIAL_DIR / "solution" / "rotor"),
        help="Directory containing VPM/VLM backup files.",
    )
    ap.add_argument(
        "--figures-dir",
        default=str(TUTORIAL_DIR / "figures"),
        help="Output directory for figures.",
    )
    ap.add_argument(
        "--tail-fraction",
        type=float,
        default=0.25,
        help="Fraction of simulation tail used for time-averaging (default 0.25 = last 25%%).",
    )
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--format", choices=["png", "svg"], default="png")
    args = ap.parse_args()

    sys.path.insert(0, str(SCRIPT_DIR))
    from rotor_theory import bem_solve
    from generate_openvsp_blade import RotorBladeDesign, design_schedule

    samples = Path(args.solution_dir) / "samples"
    figs = Path(args.figures_dir)
    figs.mkdir(parents=True, exist_ok=True)

    # ── BEM reference ────────────────────────────────────────────────────────
    design = RotorBladeDesign()
    sched = design_schedule(design)
    bem = bem_solve(
        r=sched["r"],
        chord=sched["chord"],
        twist_rad=np.radians(sched["theta_deg"]),
        B=3,
        R=design.radius,
        U_inf=design.freestream_velocity,
        omega=design.omega,
    )
    bem_ct = bem.attrs["Ct"]
    bem_cp = bem.attrs["Cp"]

    # ── VLM time-averaged spanwise data ──────────────────────────────────────
    # blade_0 is at azimuth=0 → span axis = +Y → y_station = physical radius r
    result = _read_vlm_spanwise(
        samples,
        surface="blade_0",
        tail_fraction=args.tail_fraction,
        hub_r=design.hub_radius,
        tip_r=design.radius,
    )
    if result is None:
        print(
            f"  [WARNING] vlm_spanwise_blade_0.csv not found in {samples} — "
            "run rotorFlow first."
        )
        return 0

    r_vlm, gamma_vlm, cl_vlm = result

    # ── Plot ─────────────────────────────────────────────────────────────────
    colors, _ = _load_theme()
    color_vlm = colors.get("vpm", "#1b9e77")
    color_bem = colors.get("hybrid", "#DB5400")

    fig, axes = plt.subplots(1, 2, figsize=(19.0 / 2.54, 7.0 / 2.54))
    fig.subplots_adjust(wspace=0.35, left=0.10, right=0.97, top=0.93, bottom=0.14)

    # ── Γ(r/R) ───────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(
        r_vlm / design.radius,
        gamma_vlm,
        "-",
        color=color_vlm,
        lw=1.4,
        label=r"VLM (tail avg.)",
    )
    ax.plot(
        bem["r_over_R"],
        bem["Gamma"],
        "--",
        color=color_bem,
        lw=1.4,
        label=r"BEM reference",
    )
    ax.set_xlabel(r"$r/R$")
    ax.set_ylabel(r"$\Gamma$ [m²/s]")
    ax.set_title(r"Bound circulation $\Gamma(r/R)$")
    ax.set_xlim([0, 1])
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)

    # ── Cl(r/R) ──────────────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(
        r_vlm / design.radius,
        cl_vlm,
        "-",
        color=color_vlm,
        lw=1.4,
        label=r"VLM (tail avg.)",
    )
    ax2.plot(
        bem["r_over_R"],
        bem["Cl"],
        "--",
        color=color_bem,
        lw=1.4,
        label=r"BEM reference",
    )
    ax2.set_xlabel(r"$r/R$")
    ax2.set_ylabel(r"$C_l$")
    ax2.set_title(r"Sectional lift coefficient $C_l(r/R)$")
    ax2.set_xlim([0, 1])
    ax2.legend(fontsize=8)

    # ── Annotation: integrated Ct / Cp from BEM ──────────────────────────────
    fig.text(
        0.5,
        0.01,
        f"BEM: $C_t = {bem_ct:.3f}$,  $C_p = {bem_cp:.3f}$"
        f"   (Betz: $C_t = {8/9:.3f}$, $C_p = {16/27:.3f}$)",
        ha="center",
        fontsize=7,
        color="grey",
    )

    out = figs / f"rotor_loading_validation.{args.format}"
    save_kw: dict = {"bbox_inches": "tight"}
    if args.format == "png":
        save_kw["dpi"] = args.dpi
    fig.savefig(out, **save_kw)
    plt.close(fig)
    print(f"  Saved: {out}")
    print(
        f"  BEM Ct={bem_ct:.4f}, Cp={bem_cp:.4f}"
        f" (Betz: Ct={8/9:.4f}, Cp={16/27:.4f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
