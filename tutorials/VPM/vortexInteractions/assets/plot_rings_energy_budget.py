#!/usr/bin/env python3
"""Energy-budget audit for vortex-ring interaction tests.

The target identity for an unbounded viscous incompressible flow is

    dE/dt = -nu * integral(|omega|^2 dV)

for the energy and enstrophy conventions used by the VPM diagnostics.  Some
texts define enstrophy as 0.5*integral(|omega|^2), which is the same statement
written as dE/dt = -2*nu*Enstrophy.  This script reports both the code-native
``neg_nu_enstrophy`` balance and the literal ``-2*enstrophy`` balance so a
normalization mismatch is impossible to miss.

It reads the solver log's ``FLOW DIAGNOSTICS`` sections.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ASSETS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ASSETS_DIR))
from _common import (
    T_REF,
    build_arg_parser,
    case_style,
    compact_case_legend_handles,
    compounded_discarded_fraction,
    discover_cases,
    figure_size,
    load_theme,
    mark_every,
    read_integrals,
    reference_fill_style,
    save_fig,
    secondary_line_style,
)


def read_budget(case_dir: Path) -> tuple[pd.DataFrame | None, str]:
    """Return a monotone log-derived time series."""
    df = read_integrals(case_dir)
    if df is None:
        return None, ""
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["time", "kinetic_energy"])
    if len(df) < 3:
        return None, ""
    return df.reset_index(drop=True), "log"


def local_poly_derivative(t: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    """Derivative from a moving polynomial fit using the actual timestamps."""
    n = len(t)
    if n < 2:
        return np.full_like(y, np.nan, dtype=float)
    if window < 3:
        window = 3
    if window % 2 == 0:
        window += 1
    window = min(window, n if n % 2 == 1 else n - 1)
    if window < 3:
        return np.gradient(y, t, edge_order=1)

    half = window // 2
    dydt = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        if hi - lo < 3:
            if lo == 0:
                hi = min(n, 3)
            else:
                lo = max(0, n - 3)
        tt = t[lo:hi]
        yy = y[lo:hi]
        deg = min(3, len(tt) - 1)
        tau = tt - t[i]
        coeff = np.polyfit(tau, yy, deg)
        dcoeff = np.polyder(coeff)
        dydt[i] = np.polyval(dcoeff, 0.0)
    return dydt


def trapz(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.trapezoid(y, x))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else np.nan


def balance_metrics(
    t: np.ndarray,
    energy: np.ndarray,
    dE_dt: np.ndarray,
    sink: np.ndarray,
) -> dict[str, float]:
    """Compare dE/dt against a target sink time series."""
    valid = np.isfinite(dE_dt) & np.isfinite(sink)
    if valid.sum() < 2:
        return {
            "point_rel_l2": np.nan,
            "point_bias": np.nan,
            "integrated_ratio": np.nan,
            "integrated_residual_E0": np.nan,
            "fit_slope": np.nan,
            "fit_r2": np.nan,
        }

    tv = t[valid]
    ev = energy[valid]
    dv = dE_dt[valid]
    sv = sink[valid]
    diff = dv - sv
    denom = rms(sv)
    point_rel_l2 = rms(diff) / denom if denom > 0.0 else np.nan
    point_bias = float(np.mean(diff) / (np.mean(np.abs(sv)) + 1e-30))

    dE_total = float(ev[-1] - ev[0])
    sink_int = trapz(sv, tv)
    residual_int = dE_total - sink_int
    integrated_ratio = dE_total / sink_int if abs(sink_int) > 1e-30 else np.nan

    cumulative = np.zeros_like(tv)
    if len(tv) > 1:
        increments = 0.5 * (sv[1:] + sv[:-1]) * (tv[1:] - tv[:-1])
        cumulative[1:] = np.cumsum(increments)
    delta_e = ev - ev[0]
    if np.std(cumulative) > 0.0:
        slope, intercept = np.polyfit(cumulative, delta_e, 1)
        pred = slope * cumulative + intercept
        ss_res = float(np.sum((delta_e - pred) ** 2))
        ss_tot = float(np.sum((delta_e - np.mean(delta_e)) ** 2))
        fit_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else np.nan
    else:
        slope, fit_r2 = np.nan, np.nan

    e0 = float(energy[0])
    return {
        "point_rel_l2": point_rel_l2,
        "point_bias": point_bias,
        "integrated_ratio": integrated_ratio,
        "integrated_residual_E0": residual_int / e0 if abs(e0) > 1e-30 else np.nan,
        "fit_slope": float(slope),
        "fit_r2": fit_r2,
    }


def summarize_case(case_dir: Path, window: int) -> tuple[pd.DataFrame | None, dict | None]:
    df, source = read_budget(case_dir)
    if df is None:
        return None, None

    t = df["time"].to_numpy(float)
    energy = df["kinetic_energy"].to_numpy(float)
    dE_poly = local_poly_derivative(t, energy, window)
    dE_grad = np.gradient(energy, t, edge_order=2 if len(t) > 2 else 1)

    sink_nu = (
        df["neg_nu_enstrophy"].to_numpy(float)
        if "neg_nu_enstrophy" in df.columns
        else np.full_like(t, np.nan)
    )
    sink_minus2 = (
        -2.0 * df["enstrophy"].to_numpy(float)
        if "enstrophy" in df.columns
        else np.full_like(t, np.nan)
    )
    enstrophy = (
        df["enstrophy"].to_numpy(float) if "enstrophy" in df.columns else np.full_like(t, np.nan)
    )

    e0 = float(energy[0]) if abs(float(energy[0])) > 1e-30 else 1.0

    # Objective-aligned diagnostics.  For an unbounded viscous flow energy can
    # only decay, so any dE/dt > 0 is spurious injection.  The cumulative
    # spurious energy ∫₀ᵗ max(dE/dt, 0) dt / E₀ is a monotone curve that stays
    # flat at zero for a run that never violates dE/dt ≤ 0.
    spurious_rate = np.clip(dE_poly, 0.0, None)
    cum_spurious = np.zeros_like(t)
    if len(t) > 1:
        inc = 0.5 * (spurious_rate[1:] + spurious_rate[:-1]) * np.diff(t)
        cum_spurious[1:] = np.cumsum(inc)

    out = df.copy()
    out.insert(0, "case", case_dir.name)
    out["source"] = source
    out["t_star"] = t / T_REF
    out["dE_dt_poly"] = dE_poly
    out["dE_dt_gradient"] = dE_grad
    out["sink_minus2_enstrophy"] = sink_minus2
    out["residual_vs_neg_nu_enstrophy"] = dE_poly - sink_nu
    out["residual_vs_minus2_enstrophy"] = dE_poly - sink_minus2
    out["cum_spurious_E0"] = cum_spurious / e0

    m_nu = balance_metrics(t, energy, dE_poly, sink_nu)
    m_m2 = balance_metrics(t, energy, dE_poly, sink_minus2)

    valid = np.isfinite(dE_poly) & np.isfinite(enstrophy) & (enstrophy > 0.0)
    fitted_c = np.nan
    if valid.sum() >= 2:
        # Best positive factor c in dE/dt ~= -c * enstrophy.
        fitted_c = -float(np.dot(dE_poly[valid], enstrophy[valid])) / float(
            np.dot(enstrophy[valid], enstrophy[valid])
        )

    e0 = float(energy[0])

    # Impulse drift — the physically conserved invariant for this flow.
    imp_cols = [f"impulse_{axis}" for axis in "xyz"]
    if all(col in df.columns for col in imp_cols):
        imp = df[imp_cols].to_numpy(float)
        imp0 = float(np.linalg.norm(imp[0]))
        impulse_drift = float(np.linalg.norm(imp[-1] - imp[0]) / imp0) if imp0 > 1e-30 else np.nan
    else:
        impulse_drift = np.nan

    npart = df["n_particles"].to_numpy(float) if "n_particles" in df.columns else np.array([np.nan])
    dE_valid = dE_poly[np.isfinite(dE_poly)]
    summary = {
        "case": case_dir.name,
        "source": source,
        "n_samples": len(df),
        "dt_sample_mean": float(np.mean(np.diff(t))) if len(t) > 1 else np.nan,
        "t_final": float(t[-1]),
        "E_ratio": float(energy[-1] / e0) if abs(e0) > 1e-30 else np.nan,
        "dE_total_E0": float((energy[-1] - energy[0]) / e0) if abs(e0) > 1e-30 else np.nan,
        "best_c_for_minus_c_enstrophy": fitted_c,
        # --- objective verdict: dE/dt ≤ 0 and physics preserved -------------
        "frac_dEdt_pos": float(np.mean(dE_valid > 0.0)) if dE_valid.size else np.nan,
        "max_dEdt_E0": float(dE_valid.max() / e0) if dE_valid.size and abs(e0) > 1e-30 else np.nan,
        "spurious_E_in_E0": float(cum_spurious[-1] / e0) if abs(e0) > 1e-30 else np.nan,
        "destroyed_circ": compounded_discarded_fraction(case_dir),
        "impulse_drift": impulse_drift,
        "N_ratio": float(npart[-1] / npart[0]) if npart.size and npart[0] > 0 else np.nan,
    }
    for prefix, metrics in (("nu", m_nu), ("minus2", m_m2)):
        for key, value in metrics.items():
            summary[f"{prefix}_{key}"] = value
    if "strength_magnitude" in df.columns and df["strength_magnitude"].iloc[0] != 0.0:
        summary["strength_ratio"] = float(
            df["strength_magnitude"].iloc[-1] / df["strength_magnitude"].iloc[0]
        )
    elif "sum_gamma_magnitude" in df.columns and df["sum_gamma_magnitude"].iloc[0] != 0.0:
        summary["strength_ratio"] = float(
            df["sum_gamma_magnitude"].iloc[-1] / df["sum_gamma_magnitude"].iloc[0]
        )
    else:
        summary["strength_ratio"] = np.nan

    return out, summary


def make_figure(
    timeseries: pd.DataFrame,
    summary: pd.DataFrame,
    figures_dir: Path,
    dpi: int,
    figure_format: str = "png",
) -> None:
    load_theme()
    fig, (ax_rate, ax_spur) = plt.subplots(2, 1, figsize=figure_size("wide_stacked"), sharex=True)
    rate_values: list[np.ndarray] = []
    spurious_values: list[np.ndarray] = []

    for case, df in timeseries.groupby("case", sort=True):
        st = case_style(case)
        t = df["t_star"].to_numpy(float)
        e0 = float(df["kinetic_energy"].iloc[0])
        scale = abs(e0) if abs(e0) > 1e-30 else 1.0
        common = dict(
            color=st["color"],
            linestyle=st["linestyle"],
            lw=st["linewidth"],
            marker=st["marker"],
            ms=st["markersize"],
            markevery=mark_every(),
            mew=st["markeredgewidth"],
        )

        rate = df["dE_dt_poly"].to_numpy(float) / scale
        rate_values.append(rate)
        ax_rate.plot(t, rate, label=st["label"], **common)
        if "neg_nu_enstrophy" in df:
            ax_rate.plot(
                t,
                df["neg_nu_enstrophy"] / scale,
                color=st["color"],
                **secondary_line_style(),
            )

        # Cumulative spurious energy: monotone, stays at 0 for a run that never
        # violates dE/dt ≤ 0.  This is the primary pass/fail signal.
        spur = df["cum_spurious_E0"].to_numpy(float)
        spurious_values.append(spur)
        ax_spur.plot(t, spur, **common)

    def padded_limits(values: list[np.ndarray]) -> tuple[float, float]:
        finite = np.concatenate([array[np.isfinite(array)] for array in values])
        lower = min(-0.05, float(finite.min()))
        upper = max(0.05, float(finite.max()))
        padding = 0.06 * (upper - lower)
        return lower - padding, upper + padding

    # Energy may only decay: dE/dt > 0 (shaded) is spurious injection.
    ax_rate.axhspan(0, 2.0, **reference_fill_style())
    ax_rate.axhline(0.0, color="0.55", linestyle=":", linewidth=0.8)
    ax_rate.set_ylabel(r"$E_0^{-1}\,dE/dt$")
    ax_rate.set_title(r"Energy budget — target $dE/dt \leq 0$ everywhere")
    ax_rate.set_ylim(padded_limits(rate_values))

    ax_spur.set_xlabel(r"Normalized time, $t\Gamma_0/R_0^2$")
    ax_spur.set_ylabel(r"$E_0^{-1}\!\int_0^t\max(dE/dt,0)\,dt$")
    upper = max(1e-3, float(np.nanmax([np.nanmax(v) for v in spurious_values])))
    ax_spur.set_ylim(-0.02 * upper, 1.06 * upper)

    fig.legend(
        handles=compact_case_legend_handles(),
        ncol=5,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
    )
    save_fig(
        fig,
        figures_dir / "rings_energy_budget.png",
        dpi=dpi,
        figure_format=figure_format,
        tight_rect=(0.0, 0.16, 1.0, 1.0),
    )


def main() -> None:
    parser = build_arg_parser("Audit dE/dt against viscous enstrophy dissipation.")
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Odd moving polynomial window for dE/dt. Use 3 for minimal smoothing.",
    )
    args = parser.parse_args()

    solution_dir = Path(args.solution_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_series: list[pd.DataFrame] = []
    rows: list[dict] = []
    for case_dir in discover_cases(solution_dir):
        ts, summary = summarize_case(case_dir, args.window)
        if ts is None or summary is None:
            continue
        all_series.append(ts)
        rows.append(summary)

    if not rows:
        raise SystemExit(f"No energy diagnostics found under {solution_dir}")

    timeseries = pd.concat(all_series, ignore_index=True)
    summary = pd.DataFrame(rows).sort_values("case")

    make_figure(timeseries, summary, figures_dir, args.dpi, args.format)

    # Objective verdict: a method succeeds only if it never injects energy
    # (frac_dEdt_pos == 0) AND preserves the physics — no destroyed circulation,
    # negligible impulse drift.  Ranked worst-violation first.
    verdict = summary.copy()
    verdict["PASS"] = (
        (verdict["frac_dEdt_pos"] <= 0.0)
        & (verdict["destroyed_circ"] <= 1e-3)
        & (verdict["impulse_drift"].abs() <= 1e-2)
    )
    cols = [
        "case",
        "PASS",
        "frac_dEdt_pos",
        "max_dEdt_E0",
        "spurious_E_in_E0",
        "destroyed_circ",
        "impulse_drift",
        "E_ratio",
        "N_ratio",
    ]
    verdict = verdict.sort_values(
        ["frac_dEdt_pos", "spurious_E_in_E0", "destroyed_circ"], ascending=False
    )
    print("\n=== OBJECTIVE VERDICT (dE/dt ≤ 0 always, physics preserved) ===")
    print(verdict[cols].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print(
        "\nPASS requires: frac_dEdt_pos=0, destroyed_circ≤1e-3, |impulse_drift|≤1e-2.\n"
        "frac_dEdt_pos = fraction of samples with dE/dt>0 (spurious energy gain);\n"
        "spurious_E_in_E0 = total injected energy ∫max(dE/dt,0)dt / E0;\n"
        "destroyed_circ = circulation thrown away by capped/thresholded remeshing."
    )


if __name__ == "__main__":
    main()
