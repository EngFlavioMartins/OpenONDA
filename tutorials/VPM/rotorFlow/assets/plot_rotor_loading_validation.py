#!/usr/bin/env python3
"""Spanwise loading validation — VLM bound circulation vs BEM prediction.

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

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from _common import build_arg_parser, build_rotor_style_map, load_theme

# ==============================================================================
# Data loader
# ==============================================================================


def _read_vlm_spanwise(
    samples_dir: Path,
    surface: str,
    tail_fraction: float,
    averaging_rotations: float,
    omega: float,
    hub_r: float,
    tip_r: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray] | None:
    """Return (r, |Gamma|, chord, Cl, last_times) averaged over the tail.

    The blade rotates, so signed span coordinates are not stable across
    snapshots.  The sampler already exports one row per physical span station;
    group by that station identity and use ``abs(y)`` as the radial coordinate.
    This avoids row-order artifacts when the blade azimuth changes.  The CSV
    ``cl`` column is a fixed wind-axis quantity and is not a sectional
    rotor-blade coefficient, so the plot reconstructs an equivalent sectional
    coefficient from bound circulation.
    """
    csv_path = samples_dir / f"vlm_spanwise_{surface}.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    if df.empty:
        return None

    if "half" in df.columns:
        df = df[df["half"] == "orig"]

    t_max = float(df["time"].max())
    if averaging_rotations > 0.0 and omega > 0.0:
        t_cut = max(float(df["time"].min()), t_max - averaging_rotations * 2.0 * np.pi / omega)
    else:
        t_cut = t_max * (1.0 - tail_fraction)
    tail = df[df["time"] >= t_cut].copy()
    if "span_coordinate_abs" in tail.columns:
        tail["r_abs"] = tail["span_coordinate_abs"]
    elif "r" in tail.columns:
        tail["r_abs"] = tail["r"].abs()
    else:
        tail["r_abs"] = tail["y"].abs()

    if "Gamma_abs" not in tail.columns:
        tail["Gamma_abs"] = tail["Gamma"].abs()

    station_cols = [col for col in ("station_id", "wing_uid", "segment_uid", "span_index") if col in tail.columns]
    agg_spec = {
        "r": ("r_abs", "mean"),
        "Gamma_abs": ("Gamma_abs", "mean"),
        "chord_local": ("chord_local", "mean"),
    }
    if "cl_from_gamma" in tail.columns:
        agg_spec["cl_from_gamma"] = ("cl_from_gamma", "mean")

    if station_cols:
        gp = (
            tail.groupby(station_cols, sort=False)
            .agg(**agg_spec)
            .sort_values("r")
            .reset_index(drop=True)
        )
    else:
        tail = tail.sort_values(["time", "r_abs"], kind="mergesort").copy()
        tail["station"] = tail.groupby("time").cumcount()
        gp = (
            tail.groupby("station")
            .agg(**agg_spec)
            .sort_values("r")
            .reset_index(drop=True)
        )

    r = gp["r"].clip(lower=hub_r, upper=tip_r).to_numpy()
    gamma = gp["Gamma_abs"].to_numpy()
    chord = gp["chord_local"].to_numpy()
    cl = gp["cl_from_gamma"].to_numpy() if "cl_from_gamma" in gp.columns else None
    return r, gamma, chord, cl, tail["time"].to_numpy()


# ==============================================================================
# Plot
# ==============================================================================


def plot_loading_validation(args) -> int:
    samples = Path(args.solution_dir) / "samples"
    figs = Path(args.figures_dir)
    fmt = getattr(args, "format", "png")
    out = figs / f"rotor_loading_validation.{fmt}"
    out.parent.mkdir(parents=True, exist_ok=True)

    # -- BEM reference ---------------------------------------------------
    sys.path.insert(0, str(Path(__file__).parent))
    from rotor_theory import bem_solve
    from generate_openvsp_blade import RotorBladeDesign, design_schedule

    design = RotorBladeDesign(n_stations=23, chord_stations=7)
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

    # -- VLM time-averaged spanwise data ---------------------------------
    result = _read_vlm_spanwise(
        samples,
        surface="blade_0",
        tail_fraction=args.tail_fraction,
        averaging_rotations=args.averaging_rotations,
        omega=design.omega,
        hub_r=design.hub_radius,
        tip_r=design.radius,
    )
    if result is None:
        print(
            f"  [WARNING] vlm_spanwise_blade_0.csv not found in {samples} — "
            "run rotorFlow first."
        )
        return 0

    r_vlm, gamma_vlm, chord_vlm, cl_vlm_csv, _ = result
    if cl_vlm_csv is not None:
        cl_vlm = cl_vlm_csv
    else:
        vrel_geom = np.sqrt(design.freestream_velocity**2 + (design.omega * r_vlm) ** 2)
        cl_vlm = 2.0 * gamma_vlm / (chord_vlm * vrel_geom)

    # Normalised circulation reference: Γ* = Γ / (U∞ R)
    gamma_ref = design.freestream_velocity * design.radius
    gamma_vlm_norm = gamma_vlm / gamma_ref
    bem_gamma_norm = bem["Gamma"].to_numpy() / gamma_ref

    colors, _ = load_theme()
    styles = build_rotor_style_map(colors)
    s_vpm = styles["vpm"]
    s_bem = styles["bem"]

    fig, axes = plt.subplots(1, 2, figsize=(12.8 / 2.54, 7.4 / 2.54), sharex=True)
    fig.subplots_adjust(wspace=0.27, left=0.12, right=0.96, top=0.87, bottom=0.27)

    # -- Gamma*(r/R) -----------------------------------------------------
    ax = axes[0]
    vpm_kw = {
        "color": s_vpm["color"],
        "marker": s_vpm["marker"],
        "markersize": s_vpm["markersize"],
        "lw": s_vpm["linewidth"],
        "label": s_vpm["label"],
    }
    bem_kw = {
        "color": s_bem["color"],
        "ls": s_bem["linestyle"],
        "lw": s_bem["linewidth"],
        "label": s_bem["label"],
    }
    ax.plot(r_vlm / design.radius, gamma_vlm_norm, **vpm_kw)
    ax.plot(bem["r_over_R"], bem_gamma_norm, **bem_kw)
    ax.set_xlabel(r"$r/R$")
    ax.set_ylabel(r"$\Gamma\,/\,(U_\infty R)$")
    ax.set_title(r"Normalized bound circulation")
    ax.set_xlim([0, 1])
    gamma_top = max(float(np.nanmax(gamma_vlm_norm)), float(np.nanmax(bem_gamma_norm))) * 1.08
    ax.set_ylim([0, gamma_top])

    # -- Cl(r/R) ---------------------------------------------------------
    ax2 = axes[1]
    ax2.plot(r_vlm / design.radius, cl_vlm, **vpm_kw)
    ax2.plot(bem["r_over_R"], bem["Cl"], **bem_kw)
    ax2.set_xlabel(r"$r/R$")
    ax2.set_ylabel(r"$C_l$")
    ax2.set_title(r"Sectional lift coefficient")
    ax2.set_xlim([0, 1])
    cl_top = max(float(np.nanmax(cl_vlm)), float(np.nanmax(bem["Cl"]))) * 1.08
    ax2.set_ylim([0, cl_top])

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.03),
        columnspacing=1.1,
        handlelength=2.2,
    )

    save_kw: dict = {"bbox_inches": "tight"}
    if fmt == "png":
        save_kw["dpi"] = args.dpi
    plt.savefig(out, **save_kw)
    plt.close(fig)
    print(f"  Saved: {out}")
    print(
        f"  BEM Ct={bem_ct:.4f}, Cp={bem_cp:.4f}"
        f" (Betz: Ct={8/9:.4f}, Cp={16/27:.4f})"
    )
    return 0


def main() -> int:
    p = build_arg_parser("Spanwise loading validation: VLM vs BEM")
    p.add_argument(
        "--tail-fraction",
        type=float,
        default=0.25,
        help="Fraction of simulation tail used for time-averaging (default 0.25 = last 25%%).",
    )
    p.add_argument(
        "--averaging-rotations",
        type=float,
        default=3.0,
        help="Number of final rotor rotations to average (default: 3).",
    )
    return plot_loading_validation(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
